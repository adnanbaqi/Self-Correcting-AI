import structlog
from app.models.schemas import TaskDomain
from app.services.model_engine import generate

logger = structlog.get_logger()

_EXPLAINER_MAX_TOKENS = 150

# ── Phi-2 compatible explainer prompts ───────────────────────────────────────
# Key changes from original:
# 1. Removed "Do NOT provide the correct answer" — phi-2 ignores this instruction
#    and then hallucinates a wrong "correct answer" anyway, misleading the Corrector.
# 2. Instead we redirect focus: "describe only what is wrong with the METHOD",
#    which keeps phi-2 talking about the error rather than guessing the answer.
# 3. Added "2-3 sentences maximum" — shorter outputs = less drift into hallucination.
# 4. Removed bullet points — phi-2 often treats bullet point instructions as content
#    to reproduce, not as formatting directives.
EXPLAINER_PROMPTS = {
    TaskDomain.MATH: """You are a mathematics professor reviewing a wrong answer.
Describe specifically what is wrong with the method or calculation shown.
Focus on the error in reasoning or arithmetic — not on what the answer should be.
Keep your diagnosis to 2-3 sentences.""",

    TaskDomain.CODE: """You are a senior engineer reviewing buggy code.
Describe specifically what type of bug or logic error is present.
Name the error pattern (off-by-one, wrong condition, missing edge case, etc.) and where it occurs.
Keep your diagnosis to 2-3 sentences.""",

    TaskDomain.COMMONSENSE: """You are a critical thinking expert reviewing a flawed answer.
Describe specifically where the reasoning breaks down or what false assumption is made.
Focus on the logical flaw — not on what the correct conclusion should be.
Keep your diagnosis to 2-3 sentences.""",

    TaskDomain.QA: """You are a fact-checker reviewing a potentially wrong answer.
Describe specifically what claim appears to be factually incorrect or missing.
Focus on what is wrong with the answer as given — not on what the correct answer should be.
Keep your diagnosis to 2-3 sentences.""",

    TaskDomain.GENERAL: """You are an expert reviewer examining an incorrect response.
Describe specifically what is wrong or incomplete about the response.
Focus on the error — not on what the correct response should be.
Keep your diagnosis to 2-3 sentences.""",
}


class Explainer:
    """Generates targeted diagnostic rationales without leaking the correct answer."""

    def generate_rationale(
        self,
        original_prompt: str,
        initial_output: str,
        domain: TaskDomain = TaskDomain.GENERAL,
    ) -> str:
        system_instruction = EXPLAINER_PROMPTS[domain]
        user_prompt = (
            f"Question:\n{original_prompt}\n\n"
            f"Wrong answer given:\n{initial_output}\n\n"
            f"Describe what is wrong with this answer."
        )

        logger.info("explainer_start", domain=domain)
        rationale = generate(
            prompt=user_prompt,
            system_instruction=system_instruction,
            temperature=0.2,
            max_new_tokens=_EXPLAINER_MAX_TOKENS,
            adapter_name="explainer"
        )
        logger.info("explainer_done", length=len(rationale))
        return rationale.strip()

    def _get_system_prompt(self, domain: TaskDomain) -> str:
        return EXPLAINER_PROMPTS[domain]