import os
from threading import Lock
from typing import Optional, Iterator, Tuple
from contextlib import nullcontext

import structlog
from fastapi import HTTPException
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from peft import PeftModel

logger = structlog.get_logger()

_model = None
_tokenizer = None
_device = None
_lock = Lock()

_STOP_BOUNDARIES = [
    "Instruct:",
    "Output:",
    "\nUser:",
    "\nHuman:",
    "\nAssistant:",
    "###",
]

_LOOP_WINDOW     = 120
_LOOP_PHRASE_LEN = 10
_LOOP_REPEATS    = 3


def _is_looping(buffer: str) -> bool:
    tail = buffer[-_LOOP_WINDOW:] if len(buffer) > _LOOP_WINDOW else buffer
    for length in [_LOOP_PHRASE_LEN, 16, 24]:
        for start in range(0, len(tail) - length * _LOOP_REPEATS, 3):
            phrase = tail[start:start + length]
            if tail.count(phrase) >= _LOOP_REPEATS:
                return True
    return False


def _build_prompt(prompt: str, system_instruction: str = "") -> str:
    if system_instruction.strip():
        instruct_block = f"{system_instruction.strip()}\n{prompt.strip()}"
    else:
        instruct_block = prompt.strip()
    return f"Instruct: {instruct_block}\nOutput:"


def _trim_response(response: str) -> str:
    for boundary in _STOP_BOUNDARIES:
        if boundary in response:
            response = response.split(boundary)[0]
    return response.strip()


def _resolve_device():
    from app.config import get_settings
    cfg = get_settings().device
    if cfg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("device_selected", device="cuda", gpu=torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")
            logger.info("device_selected", device="cpu")
        return device
    return torch.device(cfg)


def _get_settings():
    from app.config import get_settings
    return get_settings()


def _load() -> None:
    global _model, _tokenizer, _device

    _device = _resolve_device()

    local_model_path = "./model/phi3/"
    lora_base_path = os.path.join(local_model_path, "loras")

    logger.info("model_loading_start", path=local_model_path)

    # 1. Load Tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # 2. 4-bit quantization config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # 3. Fix RoPE config bug
    logger.info("intercepting_model_config_for_rope_fix")
    config = AutoConfig.from_pretrained(
        local_model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    config.rope_scaling = None

    # 4. Load base model
    #    - No device_map: bitsandbytes places the model on GPU automatically
    #      when load_in_4bit=True. Passing device_map triggers accelerate's
    #      dispatch_model which calls .to() on an already-placed 4-bit model
    #      and raises a ValueError on recent accelerate versions.
    #    - No torch_dtype: dtype is owned by bnb_4bit_compute_dtype inside
    #      BitsAndBytesConfig; setting it at the top level conflicts.
    base_model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        config=config,
        quantization_config=quant_config,
        local_files_only=True,
        trust_remote_code=True,
    )

    # 5. Attach LoRA adapters if present
    diagnoser_path = os.path.join(lora_base_path, "diagnoser")
    if os.path.exists(diagnoser_path):
        logger.info("attaching_lora_adapters")
        _model = PeftModel.from_pretrained(
            base_model,
            diagnoser_path,
            adapter_name="diagnoser",
            local_files_only=True,
        )
        for adapter in ["explainer", "corrector"]:
            adapter_path = os.path.join(lora_base_path, adapter)
            if os.path.exists(adapter_path):
                _model.load_adapter(
                    adapter_path,
                    adapter_name=adapter,
                    local_files_only=True,
                )
    else:
        logger.warning("no_loras_found_using_base_model_only", path=lora_base_path)
        _model = base_model

    _model.eval()
    logger.info("model_ready", device=str(_device))


def ensure_loaded() -> None:
    global _model
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        _load()


def is_model_loaded() -> bool:
    return _model is not None


def _manage_adapters(adapter_name: Optional[str]):
    """
    Safely switches between LoRA adapters.
    If no adapters are loaded in the model, it gracefully does nothing.
    """
    has_peft = hasattr(_model, "peft_config") and len(getattr(_model, "peft_config", {})) > 0

    if not has_peft:
        return nullcontext()

    if adapter_name:
        try:
            if adapter_name in _model.peft_config:
                _model.set_adapter(adapter_name)
                return nullcontext()
            else:
                logger.warning(f"adapter_{adapter_name}_not_found_locally")
                return _model.disable_adapters()
        except Exception as e:
            logger.error("adapter_switch_failed", error=str(e))
            return _model.disable_adapters()
    else:
        return _model.disable_adapters()


def generate(
    prompt: str,
    system_instruction: str = "",
    temperature: float = 0.3,
    max_new_tokens: Optional[int] = None,
    repetition_penalty: float = 1.15,
    adapter_name: Optional[str] = None,
) -> str:
    ensure_loaded()
    settings = _get_settings()
    max_tokens = max_new_tokens or settings.max_new_tokens
    full_prompt = _build_prompt(prompt, system_instruction)

    try:
        inputs = _tokenizer(
            full_prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(_device)
        input_len = inputs["input_ids"].shape[1]

        with _manage_adapters(adapter_name):
            with torch.no_grad():
                output_ids = _model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    pad_token_id=_tokenizer.pad_token_id,
                    eos_token_id=_tokenizer.eos_token_id,
                    repetition_penalty=repetition_penalty,
                )

        new_tokens = output_ids[0][input_len:]
        response = _tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _trim_response(response)

    except Exception as e:
        logger.error("generation_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Model generation failed: {e}")


def generate_streaming_with_probs(
    prompt: str,
    system_instruction: str = "",
    temperature: float = 0.3,
    max_new_tokens: Optional[int] = None,
    repetition_penalty: float = 1.3,
    adapter_name: Optional[str] = None,
) -> Iterator[Tuple[str, float]]:
    ensure_loaded()
    settings = _get_settings()
    max_tokens = max_new_tokens or settings.max_new_tokens
    full_prompt = _build_prompt(prompt, system_instruction)

    inputs = _tokenizer(full_prompt, return_tensors="pt").to(_device)
    input_ids = inputs["input_ids"]

    generated_ids = []
    buffer = ""
    prev_decoded = ""   # tracks the full decoded string so we can diff each step

    with _manage_adapters(adapter_name):
        for step in range(max_tokens):
            with torch.no_grad():
                # use_cache=False: avoids the DynamicCache API mismatch between
                # the cached modeling_phi3.py and the installed transformers version.
                outputs = _model(input_ids, use_cache=False)
                next_token_logits = outputs.logits[:, -1, :].clone()

                if repetition_penalty != 1.0 and generated_ids:
                    for token_id in set(generated_ids):
                        if next_token_logits[0, token_id] < 0:
                            next_token_logits[0, token_id] *= repetition_penalty
                        else:
                            next_token_logits[0, token_id] /= repetition_penalty

                if temperature > 0:
                    scaled = next_token_logits / temperature
                    probs = torch.nn.functional.softmax(scaled, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

                logprob = torch.log(probs[0, next_token[0, 0]]).item()

            token_id = next_token.item()
            if token_id == _tokenizer.eos_token_id:
                break

            generated_ids.append(token_id)

            # Grow input_ids for next step (no KV cache, so we feed full sequence)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # ── Decode full sequence and diff to get the new chunk ─────────
            # Decoding one token at a time with skip_special_tokens=True drops
            # the leading space that SentencePiece encodes into word-starting
            # tokens, causing words to run together ("HowcanIhelp").
            # Decoding the full sequence and slicing the new suffix fixes this.
            full_decoded = _tokenizer.decode(generated_ids, skip_special_tokens=True)
            chunk = full_decoded[len(prev_decoded):]
            prev_decoded = full_decoded
            # ──────────────────────────────────────────────────────────────

            if not chunk:
                continue

            buffer += chunk

            # Stop boundary detection
            hit_boundary = False
            for boundary in _STOP_BOUNDARIES:
                if boundary in buffer:
                    clean = buffer.split(boundary)[0]
                    if clean:
                        yield clean, logprob
                    hit_boundary = True
                    break

            if hit_boundary:
                break

            # Repetition loop detection
            if _is_looping(buffer):
                tail = buffer[-_LOOP_WINDOW:]
                for length in [_LOOP_PHRASE_LEN, 16, 24]:
                    for start in range(0, len(tail) - length * _LOOP_REPEATS, 3):
                        phrase = tail[start:start + length]
                        if tail.count(phrase) >= _LOOP_REPEATS:
                            loop_start = buffer.rfind(phrase, 0, len(buffer) - len(tail) + start)
                            if loop_start > 0:
                                clean_prefix = buffer[:loop_start].strip()
                                if clean_prefix:
                                    yield clean_prefix, logprob
                            break
                    else:
                        continue
                    break
                break

            yield chunk, logprob