from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.config import get_settings
from app.services.model_engine import ensure_loaded
import structlog

logger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "server_startup",
        environment=settings.environment,
        model_id=settings.hf_model_id,
        model_pt_path=settings.model_pt_path,
        confidence_threshold=settings.diagnoser_confidence_threshold,
    )
    # Load model weights into memory before accepting requests.
    # This blocks startup until the model is ready — intentional.
    ensure_loaded()
    yield
    logger.info("server_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Self Correcting Neural Systems API",
        description=(
            "Inference-time autonomous error detection and refinement. "
            "Pipeline: Generator → Diagnoser → [Explainer → Corrector]."
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()
