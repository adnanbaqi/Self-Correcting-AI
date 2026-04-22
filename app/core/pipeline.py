import asyncio
import threading
import structlog
import unicodedata
from typing import AsyncGenerator


from app.config import get_settings
from app.core.diagnoser import Diagnoser
from app.core.explainer import Explainer
from app.core.corrector import Corrector
from app.models.schemas import (
    InferenceRequest, TaskDomain, StageStartEvent,
    StageTokenEvent, StageDoneEvent, DiagnosisEvent, PipelineDoneEvent
)
from app.services.model_engine import generate, generate_streaming_with_probs

logger = structlog.get_logger()
settings = get_settings()

GENERATOR_PROMPTS = {
    TaskDomain.MATH:        "You are a precise mathematician. Solve the problem step by step, showing all working.",
    TaskDomain.CODE:        "You are an expert software engineer. Write correct, clean, well-commented code.",
    TaskDomain.COMMONSENSE: "You are a thoughtful reasoner. Answer with clear, logical reasoning.",
    TaskDomain.QA:          "You are a knowledgeable assistant. Answer accurately and completely.",
    TaskDomain.GENERAL:     "You are a helpful and accurate assistant. Answer thoroughly.",
}

# ── Keyword-based fast router ────────────────────────────────────────────────
# Used as a fallback when the LLM router returns an unparseable response.
# Checked before the LLM call to avoid an unnecessary generate() for obvious cases.
_DOMAIN_KEYWORDS = {
    TaskDomain.CODE: [
        "write a", "function", "code", "program", "script", "class", "algorithm",
        "implement", "debug", "python", "java", "javascript", "sql", "def ", "return",
        "loop", "array", "list", "sort", "recursion", "bug", "error in code",
    ],
    TaskDomain.MATH: [
        "calculate", "compute", "solve", "equation", "integral", "derivative",
        "remainder", "divisible", "percentage", "%", "square root", "prime",
        "sum of", "product of", "how many", "proof", "theorem", "formula",
        "arithmetic", "algebra", "geometry", "probability", "matrix",
    ],
    TaskDomain.COMMONSENSE: [
        "why do", "what would happen", "is it possible", "makes sense",
        "logical", "reasonable", "common sense", "everyday", "real life",
        "if someone", "should i", "is it normal",
    ],
    TaskDomain.QA: [
        "who is", "who was", "when did", "where is", "what is the capital",
        "how many countries", "which country", "name the", "tell me about",
        "history of", "founded by", "invented by",
    ],
}


def _keyword_route(prompt: str) -> TaskDomain | None:
    """Fast keyword scan. Returns a domain if confident, else None."""
    lower = prompt.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return domain
    return None


def _parse_domain(raw: str) -> TaskDomain | None:
    """
    Extract a valid TaskDomain from raw LLM output.
    Phi-2 often returns 'Output: code' or 'the category is math' instead of
    just the word — so we scan for any valid domain name in the response.
    """
    clean = raw.strip().lower()
    for domain in TaskDomain:
        if domain.value in clean:
            return domain
    return None


class InferencePipeline:
    def __init__(self):
        self.diagnoser = Diagnoser()
        self.explainer = Explainer()
        self.corrector = Corrector()

    async def _auto_detect_domain(self, prompt: str) -> TaskDomain:
        """
        Two-stage domain routing:
        1. Fast keyword scan — avoids a generate() call for obvious cases
        2. LLM routing — for ambiguous prompts, with robust output parsing
        """
        # Stage 1: keyword fast-path
        keyword_domain = _keyword_route(prompt)
        if keyword_domain is not None:
            logger.info("domain_keyword_routed", domain=keyword_domain.value)
            return keyword_domain

        # Stage 2: LLM routing with robust parsing
        routing_prompt = (
            f"Classify this prompt into one category: math, code, commonsense, qa, or general.\n"
            f"Prompt: {prompt}\n"
            f"Answer with one word only."
        )
        try:
            detected = await asyncio.to_thread(
                generate,
                prompt=routing_prompt,
                system_instruction="Output only one word from: math, code, commonsense, qa, general.",
                temperature=0.0,
                max_new_tokens=10,
            )
            domain = _parse_domain(detected)
            if domain is not None:
                logger.info("domain_llm_routed", domain=domain.value, raw=detected.strip())
                return domain
            logger.warning("domain_llm_unparseable", raw=detected.strip(), fallback="general")
        except Exception as e:
            logger.warning("domain_auto_detect_failed", error=str(e), fallback="general")

        return TaskDomain.GENERAL

    async def _async_generate_stream(
        self, prompt: str, system_instruction: str, temperature: float
    ) -> AsyncGenerator[tuple[str, float], None]:
        """Bridges synchronous PyTorch generation to an async generator via thread-safe queue."""
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _producer():
            try:
                for chunk, logprob in generate_streaming_with_probs(
                    prompt=prompt, system_instruction=system_instruction, temperature=temperature
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, (chunk, logprob))
                loop.call_soon_threadsafe(queue.put_nowait, (None, None))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, (e, None))

        threading.Thread(target=_producer, daemon=True).start()

        while True:
            chunk, logprob = await queue.get()
            if isinstance(chunk, Exception):
                raise chunk
            if chunk is None:
                break
            yield chunk, logprob

    async def stream_pipeline(self, request: InferenceRequest) -> AsyncGenerator[object, None]:

        normalized_prompt = unicodedata.normalize("NFKC", request.prompt)
        request.prompt = normalized_prompt
        # ── Stage 0: Domain Resolution ─────────────────────────────────────
        if getattr(request, "domain", None) is None:
            yield StageStartEvent(stage="routing", label="Auto-detecting task domain…")
            domain = await self._auto_detect_domain(request.prompt)
            yield StageDoneEvent(stage="routing", content=f"Domain detected: {domain.value}")
        else:
            domain = request.domain

        # ── Stage 1: Initial Output ───────────────────────────────
        yield StageStartEvent(stage="generating", label=f"Generating & Analyzing ({domain.value})…")

        output_buffer = []
        sys_prompt = GENERATOR_PROMPTS[domain]

        async for chunk, logprob in self._async_generate_stream(request.prompt, sys_prompt, 0.5):
            output_buffer.append(chunk)
            yield {"event": "token", "stage": "generating", "token": chunk, "logprob": logprob}

            if logprob is not None and logprob < -1.5 and len(output_buffer) > 10:
                yield {"event": "diagnostic_alert", "status": f"Anomaly detected on token: '{chunk}'"}

        initial_output = "".join(output_buffer)
        yield StageDoneEvent(stage="generating", content=initial_output)

        # ── Stage 2: Diagnoser ─────────────────────────────────────────────
        yield StageStartEvent(stage="diagnosing", label="Diagnosing the response…")
        diagnosis = await asyncio.to_thread(
            self.diagnoser.evaluate, request.prompt, initial_output, domain
        )

        yield DiagnosisEvent(
            confidence_score=diagnosis.confidence_score,
            is_correct=diagnosis.is_correct,
            reasoning=diagnosis.reasoning,
            will_refine=not diagnosis.is_correct,
        )

        if diagnosis.is_correct:
            yield PipelineDoneEvent(was_refined=False, pipeline_stages_executed=2)
            return

        # ── Stage 3: Explainer ─────────────────────────────────────────────
        yield StageStartEvent(stage="explaining", label="Diagnosing the specific error…")

        rationale_buffer = []
        explainer_prompt = (
            f"Original Question:\n{request.prompt}\n\n"
            f"Flagged Response:\n{initial_output}\n\n"
            f"Diagnose the errors."
        )
        explainer_sys = self.explainer._get_system_prompt(domain)

        async for chunk, _ in self._async_generate_stream(explainer_prompt, explainer_sys, 0.2):
            rationale_buffer.append(chunk)
            yield {"event": "token", "stage": "explaining", "token": chunk}

        rationale = "".join(rationale_buffer).strip()
        yield StageDoneEvent(stage="explaining", content=rationale)

        # ── Stage 4: Corrector ─────────────────────────────────────────────
        yield StageStartEvent(stage="correcting", label="Refining response…")

        corrector_buffer = []
        corrector_prompt = (
            f"Original Problem:\n{request.prompt}\n\n"
            f"Initial (Incorrect) Response:\n{initial_output}\n\n"
            f"Diagnostic Rationale:\n{rationale}\n\n"
            f"Provide the fully corrected response."
        )
        corrector_sys = self.corrector._get_system_prompt(domain)

        async for chunk, _ in self._async_generate_stream(corrector_prompt, corrector_sys, 0.3):
            corrector_buffer.append(chunk)
            yield {"event": "token", "stage": "correcting", "token": chunk}

        final_output = "".join(corrector_buffer).strip()
        yield StageDoneEvent(stage="correcting", content=final_output)

        yield PipelineDoneEvent(was_refined=True, pipeline_stages_executed=4)