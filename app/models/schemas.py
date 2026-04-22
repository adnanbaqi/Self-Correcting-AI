from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal
from enum import Enum


class TaskDomain(str, Enum):
    MATH = "math"
    COMMONSENSE = "commonsense"
    QA = "qa"
    CODE = "code"
    GENERAL = "general"


class InferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The input prompt to process")
    domain: Optional[TaskDomain] = Field(
        default=None, 
        description="Optional task domain override. If omitted, the system auto-detects the domain."
    )


class Diagnosis(BaseModel):
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    is_correct: bool
    reasoning: Optional[str] = Field(None)


class InferenceResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    original_prompt: str
    initial_output: str
    diagnosis: Diagnosis
    was_refined: bool
    rationale: Optional[str] = None
    final_output: str
    model_id: str
    pipeline_stages_executed: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_id: str
    model_pt_path: str
    device: str
    pipeline_components: list[str]


# ── SSE streaming event types ─────────────────────────────────────────────────
# Every event sent over /infer/stream is one of these JSON objects.
# The `event` field is the discriminator — clients switch on it.

class StageStartEvent(BaseModel):
    """Emitted when a pipeline stage begins. Lets the UI show a spinner."""
    event: Literal["stage_start"] = "stage_start"
    stage: str       # "generating" | "diagnosing" | "explaining" | "correcting"
    label: str       # Human-readable, e.g. "Generating initial response…"

class StageTokenEvent(BaseModel):
    """One decoded token chunk streamed live from the model."""
    event: Literal["token"] = "token"
    stage: str
    token: str

class StageDoneEvent(BaseModel):
    """Stage finished. Carries the complete text output for that stage."""
    event: Literal["stage_done"] = "stage_done"
    stage: str
    content: str

class DiagnosisEvent(BaseModel):
    """Structured diagnosis result — emitted after the Diagnoser stage."""
    event: Literal["diagnosis"] = "diagnosis"
    confidence_score: float
    is_correct: bool
    reasoning: Optional[str]
    will_refine: bool   # True when Explainer + Corrector will fire next

class PipelineDoneEvent(BaseModel):
    """Final event. Signals the stream is complete."""
    event: Literal["pipeline_done"] = "pipeline_done"
    was_refined: bool
    pipeline_stages_executed: int

class ErrorEvent(BaseModel):
    """Emitted if a stage raises an unrecoverable error."""
    event: Literal["error"] = "error"
    message: str