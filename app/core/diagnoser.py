import ast
import json
import re
import structlog
from app.models.schemas import Diagnosis, TaskDomain
from app.services.model_engine import generate
from app.config import get_settings

logger = structlog.get_logger()
settings = get_settings()

_DIAGNOSER_MAX_TOKENS = 120

DIAGNOSER_PROMPTS = {
    TaskDomain.MATH: """You are a strict mathematical verifier. Assume the response may contain errors.
Independently verify the answer by working through the problem yourself.
Check: arithmetic correctness, unit conversions, formula usage, and final answer.
Be skeptical — do not just agree with the response.
Reply with ONLY this JSON and nothing else:
{"confidence_score": <float 0.0-1.0>, "is_correct": <true|false>, "reasoning": "<one sentence stating what you verified>"}
Set confidence_score below 0.85 and is_correct to false if you find ANY error.""",

    TaskDomain.CODE: """You are a strict code reviewer. Assume the code may contain bugs.
Trace through the logic manually. Check: correctness for edge cases (empty list, duplicates,
negative numbers), off-by-one errors, wrong algorithm, missing return values.
Be skeptical — do not just agree with the response.
Reply with ONLY this JSON and nothing else:
{"confidence_score": <float 0.0-1.0>, "is_correct": <true|false>, "reasoning": "<one sentence stating what you checked>"}
Set confidence_score below 0.85 and is_correct to false if you find ANY bug.""",

    TaskDomain.COMMONSENSE: """You are a critical reasoning auditor. Assume the response may be flawed.
Check for: false premises, logical contradictions, overlooked constraints, faulty conclusions.
Be skeptical — do not just agree with the response.
Reply with ONLY this JSON and nothing else:
{"confidence_score": <float 0.0-1.0>, "is_correct": <true|false>, "reasoning": "<one sentence stating what you checked>"}
Set confidence_score below 0.85 and is_correct to false if you find ANY flaw.""",

    TaskDomain.QA: """You are a strict fact-checker. Assume the answer may be wrong or incomplete.
Verify the key facts independently. Check for: wrong names, wrong dates, hallucinated details,
missing important information.
Be skeptical — do not just agree with the response.
Reply with ONLY this JSON and nothing else:
{"confidence_score": <float 0.0-1.0>, "is_correct": <true|false>, "reasoning": "<one sentence stating what you verified>"}
Set confidence_score below 0.85 and is_correct to false if you find ANY factual error.""",

    TaskDomain.GENERAL: """You are a strict evaluator. Assume the response may be wrong or incomplete.
Critically examine: accuracy, completeness, logical consistency, and any unsupported claims.
Be skeptical — do not just agree with the response.
Reply with ONLY this JSON and nothing else:
{"confidence_score": <float 0.0-1.0>, "is_correct": <true|false>, "reasoning": "<one sentence stating what you checked>"}
Set confidence_score below 0.85 and is_correct to false if you find ANY issue.""",
}

# Sentinel to distinguish "field was absent" from "field was 0.0 / false"
_MISSING = object()


def _clean_raw(raw: str) -> str:
    raw = re.sub(r"```(?:json|python)?", "", raw).strip().rstrip("`").strip()
    for sentinel in ['"""', "'''", "\nimport ", "\ndef ", "@pytest", "\nfrom "]:
        idx = raw.find(sentinel)
        if idx != -1:
            raw = raw[:idx].strip()
    raw = re.sub(r':\s*True\b', ': true', raw)
    raw = re.sub(r':\s*False\b', ': false', raw)
    return raw.strip()


def _try_ast(raw: str) -> dict | None:
    """
    Parse a Python-style dict (single-quoted keys/values) that phi-2 emits
    instead of valid JSON. ast.literal_eval needs True/False, not true/false.
    """
    candidate = re.sub(r':\s*true\b',  ': True',  raw)
    candidate = re.sub(r':\s*false\b', ': False', candidate)
    try:
        result = ast.literal_eval(candidate)
        if isinstance(result, dict):
            logger.debug("diagnoser_ast_parse_success")
            return result
    except (ValueError, SyntaxError):
        pass
    return None


def _extract_json(raw: str) -> dict:
    """
    Robustly extract the diagnosis dict from phi-2 output.

    Parse order
    -----------
    1. json.loads on full cleaned string        (double-quoted JSON)
    2. ast.literal_eval on full cleaned string  (single-quoted Python dict)  <- NEW
    3. json.loads / ast on first {...} block
    4. Field-by-field regex (accepts ' or " around keys/values)              <- UPDATED
    5. Total failure -> _MISSING sentinels
    """
    raw = _clean_raw(raw)

    # 1. Standard JSON
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Python-style dict (phi-2's most common non-JSON output)
    result = _try_ast(raw)
    if result is not None:
        return result

    # 3. First complete {...} block
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        block = _clean_raw(match.group(0))
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            pass
        result = _try_ast(block)
        if result is not None:
            return result

    # 4. Field-by-field partial recovery — accept ' or " around keys/values
    partial = re.search(r"\{.*", raw, re.DOTALL)
    if partial:
        fragment = partial.group(0)
        score_m   = re.search(r"""['"]confidence_score['"]\s*:\s*([0-9.]+)""", fragment)
        correct_m = re.search(r"""['"]is_correct['"]\s*:\s*(true|false|True|False)""", fragment)
        reason_m  = re.search(r"""['"]reasoning['"]\s*:\s*['"]([^'"]*)""", fragment)

        if score_m or correct_m:
            score      = float(score_m.group(1)) if score_m else _MISSING
            is_correct = (correct_m.group(1).lower() == "true") if correct_m else _MISSING
            reasoning  = reason_m.group(1) if reason_m else "Partial parse."
            logger.warning("diagnoser_json_partial_recovery", score=score)
            return {"confidence_score": score, "is_correct": is_correct, "reasoning": reasoning}

    # 5. Total parse failure
    logger.warning("diagnoser_json_parse_failed", raw=raw[:300])
    return {
        "confidence_score": _MISSING,
        "is_correct": _MISSING,
        "reasoning": "Parse failed — treating as incorrect.",
    }


def _resolve_fields(parsed: dict, threshold: float) -> tuple[float, bool]:
    """
    Cross-infer confidence_score and is_correct when either field is absent.

    Both present    -> use as-is.
    Score missing   -> infer: True->0.90, False->0.20
    Correct missing -> infer: score >= threshold -> True
    Both missing    -> (0.0, False)
    """
    raw_score   = parsed.get("confidence_score", _MISSING)
    raw_correct = parsed.get("is_correct",       _MISSING)

    score_absent   = raw_score   is _MISSING
    correct_absent = raw_correct is _MISSING

    if score_absent and correct_absent:
        logger.warning("diagnoser_both_fields_missing")
        return 0.0, False

    if score_absent:
        model_says_correct = str(raw_correct).lower() == "true"
        inferred_score = 0.90 if model_says_correct else 0.20
        logger.warning("diagnoser_confidence_score_missing",
                       is_correct=model_says_correct, inferred_score=inferred_score)
        confidence = inferred_score
        return confidence, model_says_correct and (confidence >= threshold)

    if correct_absent:
        confidence = max(0.0, min(1.0, float(raw_score)))
        inferred_correct = confidence >= threshold
        logger.warning("diagnoser_is_correct_missing",
                       confidence=confidence, inferred_correct=inferred_correct)
        return confidence, inferred_correct

    # Both present — normal path
    confidence         = max(0.0, min(1.0, float(raw_score)))
    model_says_correct = str(raw_correct).lower() == "true"
    return confidence, model_says_correct and (confidence >= threshold)


class Diagnoser:
    """Calibrated uncertainty estimation. Outputs a confidence score and correctness flag."""

    def evaluate(
        self,
        original_prompt: str,
        initial_output: str,
        domain: TaskDomain = TaskDomain.GENERAL,
    ) -> Diagnosis:
        system_instruction = DIAGNOSER_PROMPTS[domain]
        truncated_output = initial_output[:400] + ("..." if len(initial_output) > 400 else "")

        user_prompt = (
            f"Problem:\n{original_prompt}\n\n"
            f"Response to verify (may contain errors):\n{truncated_output}\n\n"
            f"Verify independently and output only the JSON."
        )

        logger.info("diagnoser_start", domain=domain)
        raw = generate(
            prompt=user_prompt,
            system_instruction=system_instruction,
            temperature=0.1,
            max_new_tokens=_DIAGNOSER_MAX_TOKENS,
            adapter_name="diagnoser"
        )

        logger.debug("diagnoser_raw_output", raw=raw[:300])
        parsed = _extract_json(raw)

        confidence, is_correct = _resolve_fields(
            parsed, threshold=settings.diagnoser_confidence_threshold
        )

        diagnosis = Diagnosis(
            confidence_score=confidence,
            is_correct=is_correct,
            reasoning=parsed.get("reasoning", ""),
        )
        logger.info(
            "diagnoser_done",
            confidence=confidence,
            confidence_pct=round(confidence * 100, 1),
            is_correct=is_correct,
        )
        return diagnosis