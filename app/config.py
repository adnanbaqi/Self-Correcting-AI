from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", protected_namespaces=())

    # ── Model ───────────────────────────────────────────────────────────────
    # HuggingFace model ID used for first-time download AND as the tokenizer
    # source when loading from a .pt file.
    # Change this to your custom model's HF ID or local path at training time.
    hf_model_id: str = "microsoft/Phi-3-mini-4k-instruct"

    # Path where the model weights are saved/loaded as a .pt file.
    # Run scripts/download_and_save.py once to create this file.
    model_pt_path: str = "./model/phi3"

    # "auto"  → CUDA if available, else CPU
    # "cuda"  → force GPU
    # "cpu"   → force CPU
    device: str = "auto"

    # Generation limits
    max_new_tokens: int = 512

    # ── Pipeline ────────────────────────────────────────────────────────────
    diagnoser_confidence_threshold: float = 0.85

    # ── Server ──────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    environment: str = "development"


@lru_cache()
def get_settings() -> Settings:
    return Settings()