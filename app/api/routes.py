import json
import structlog
import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import (
    InferenceRequest, InferenceResponse, HealthResponse,
    StageStartEvent, StageTokenEvent, StageDoneEvent,
    DiagnosisEvent, PipelineDoneEvent, ErrorEvent,
)
from app.core.pipeline import InferencePipeline
from app.services.model_engine import ensure_loaded, is_model_loaded
from app.config import get_settings

router = APIRouter()
logger = structlog.get_logger()
settings = get_settings()

pipeline = InferencePipeline()


# ── Blocking endpoint (original) ─────────────────────────────────────────────

@router.post("/infer", response_model=InferenceResponse, summary="Run the Self Correcting inference pipeline")
async def run_inference(request: InferenceRequest) -> InferenceResponse:
    """
    Executes the full Self Correcting pipeline and returns when complete.

    - **Stage 1**: Single-pass baseline generation
    - **Stage 2**: Diagnoser — calibrated uncertainty estimation
    - **Stage 3** *(conditional)*: Explainer → Corrector, triggered only when confidence < threshold

    For real-time stage-by-stage output, use **POST /infer/stream** instead.
    """
    if not request.prompt.strip():
        raise HTTPException(status_code=422, detail="Prompt cannot be empty.")
    logger.info("api_infer", domain=request.domain)
    return pipeline.run(request)


# ── Streaming SSE endpoint ────────────────────────────────────────────────────

def _sse_line(event_obj) -> str:
    """
    Format one Pydantic event object (or dict) as a Server-Sent Event line.
    """
    # Check if it's already a dictionary (like our fast token events)
    if isinstance(event_obj, dict):
        payload = event_obj
    else:
        # Otherwise, it's a Pydantic model, so dump it to a dict
        payload = event_obj.model_dump()
        
    event_name = payload.get("event", "message")
    return f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"

async def _stream_generator(request: InferenceRequest):
    """
    Async generator that drives the (sync) pipeline generator and yields
    SSE-formatted byte strings.

    The pipeline generator is synchronous (model.generate() blocks a thread)
    but that's fine here — FastAPI runs streaming responses in a threadpool
    automatically when we use StreamingResponse with a sync iterator.
    """
    async for event in pipeline.stream_pipeline(request):
        yield _sse_line(event).encode()


@router.post(
    "/infer/stream",
    summary="Stream the Self Correcting pipeline stage by stage (SSE)",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-Sent Events stream",
            "content": {"text/event-stream": {}},
        }
    },
)
async def run_inference_stream(request: InferenceRequest):
    """
    Same pipeline as **POST /infer** but streams each stage's output
    as Server-Sent Events the moment it is produced.

    ## Event sequence

    | Event | When | Key fields |
    |---|---|---|
    | `stage_start` | Stage begins | `stage`, `label` |
    | `token` | Each model token | `stage`, `token` |
    | `stage_done` | Stage finishes | `stage`, `content` |
    | `diagnosis` | After Diagnoser | `confidence_score`, `is_correct`, `will_refine` |
    | `pipeline_done` | All done | `was_refined`, `pipeline_stages_executed` |
    | `error` | On failure | `message` |

    ## Stages
    - `generating`  — initial single-pass response
    - `diagnosing`  — correctness evaluation (no token stream)
    - `explaining`  — error rationale *(only if diagnoser flags error)*
    - `correcting`  — refined response *(only if diagnoser flags error)*

    ## JavaScript example
    ```js
    const es = new EventSource('/api/v1/infer/stream', {method: 'POST'});
    // Use fetch + ReadableStream for POST with body:
    const resp = await fetch('/api/v1/infer/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ prompt: '...', domain: 'math' }),
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const lines = decoder.decode(value).split('\\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const evt = JSON.parse(line.slice(6));
          console.log(evt.event, evt);
        }
      }
    }
    ```
    """
    if not request.prompt.strip():
        raise HTTPException(status_code=422, detail="Prompt cannot be empty.")

    logger.info("api_infer_stream", domain=request.domain)

    return StreamingResponse(
        _stream_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if behind a proxy
        },
    )


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, summary="System health check")
async def health_check() -> HealthResponse:
    """Returns model load status, device info, and pipeline component list."""
    loaded = is_model_loaded()

    if torch.cuda.is_available():
        device = f"cuda ({torch.cuda.get_device_name(0)})"
    else:
        device = "cpu"

    return HealthResponse(
        status="healthy" if loaded else "loading",
        model_loaded=loaded,
        model_id=settings.hf_model_id,
        model_pt_path=settings.model_pt_path,
        device=device,
        pipeline_components=["Diagnoser", "Explainer", "Corrector"],
    )