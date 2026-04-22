import structlog
from app.models.schemas import TaskDomain
from app.services.model_engine import generate

logger = structlog.get_logger()

# ── Per-domain token caps ─────────────────────────────────────────────────────
# Phi-2 fills all available tokens when confused, causing repetition loops.
# Math/QA answers are short — cap them tightly. Code needs more room.
_CORRECTOR_MAX_TOKENS = {
    TaskDomain.MATH:        180,
    TaskDomain.CODE:        400,
    TaskDomain.COMMONSENSE: 220,
    TaskDomain.QA:          220,
    TaskDomain.GENERAL:     280,
}

# ── Per-domain repetition penalties ──────────────────────────────────────────
# Math symbolic expressions (2^100 % 3) are especially prone to looping.
# Higher penalty for math/code where token repetition is most damaging.
_REPETITION_PENALTY = {
    TaskDomain.MATH:        1.4,
    TaskDomain.CODE:        1.3,
    TaskDomain.COMMONSENSE: 1.2,
    TaskDomain.QA:          1.2,
    TaskDomain.GENERAL:     1.2,
}

CORRECTOR_PROMPTS = {
    TaskDomain.MATH: """You are an expert mathematician correcting a wrong answer.
The previous attempt was incorrect. Use the diagnostic rationale to find and fix the error.
Write a clean step-by-step solution. Do not repeat the original wrong answer.
Stop after stating the final answer — do not continue writing.""",

    TaskDomain.CODE: """You are a senior software engineer fixing buggy code.
The previous attempt had bugs. Use the diagnostic rationale to identify and fix them.
Write the complete corrected function only. Do not repeat the buggy version.
Stop after the closing brace — do not continue writing.""",

    TaskDomain.COMMONSENSE: """You are a careful reasoner correcting a flawed answer.
The previous attempt had logical errors. Use the diagnostic rationale to fix them.
Write a concise, logically sound answer. Do not repeat the flawed reasoning.
Stop after your conclusion — do not continue writing.""",

    TaskDomain.QA: """You are an expert researcher correcting a wrong answer.
The previous attempt had factual errors. Use the diagnostic rationale to fix them.
Write a concise, accurate answer. Do not repeat the wrong information.
Stop after answering the question — do not continue writing.""",

    TaskDomain.GENERAL: """You are an expert assistant correcting an incorrect response.
The previous attempt had errors. Use the diagnostic rationale to fix them.
Write a clear, accurate response. Do not repeat the errors.
Stop after completing your answer — do not continue writing.""",
}


def _detect_loop(text: str, min_phrase_len: int = 8, repeat_threshold: int = 3) -> bool:
    """
    Detect if the generated text has fallen into a repetition loop.
    Looks for any phrase of min_phrase_len chars repeated repeat_threshold+ times.
    """
    if len(text) < min_phrase_len * repeat_threshold:
        return False
    for length in [12, 20, 30]:
        for start in range(0, len(text) - length * repeat_threshold, 4):
            phrase = text[start:start + length]
            if text.count(phrase) >= repeat_threshold:
                return True
    return False


class Corrector:
    """Produces refined outputs conditioned on the Explainer's diagnostic rationale."""

    def refine(
        self,
        original_prompt: str,
        initial_output: str,
        rationale: str,
        domain: TaskDomain = TaskDomain.GENERAL,
    ) -> str:
        system_instruction = CORRECTOR_PROMPTS[domain]
        user_prompt = (
            f"Problem:\n{original_prompt}\n\n"
            f"Why the previous answer was wrong:\n{rationale}\n\n"
            f"Write the corrected answer now."
        )

        logger.info("corrector_start", domain=domain)
        refined = generate(
            prompt=user_prompt,
            system_instruction=system_instruction,
            temperature=0.3,
            max_new_tokens=_CORRECTOR_MAX_TOKENS[domain],
            repetition_penalty=_REPETITION_PENALTY[domain],
            adapter_name="corrector"
        )
        refined = refined.strip()

        # Safety net: if a loop slipped through, truncate at first repeated phrase
        if _detect_loop(refined):
            logger.warning("corrector_loop_detected", domain=domain, length=len(refined))
            # Find where the loop starts and cut there
            for length in [12, 20, 30]:
                for start in range(0, len(refined) - length * 3, 4):
                    phrase = refined[start:start + length]
                    second = refined.find(phrase, start + length)
                    if second != -1:
                        refined = refined[:second].strip()
                        break
                else:
                    continue
                break

        logger.info("corrector_done", length=len(refined))
        return refined

    def _get_system_prompt(self, domain: TaskDomain) -> str:
        return CORRECTOR_PROMPTS[domain]