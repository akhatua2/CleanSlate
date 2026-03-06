"""Model and tokenizer setup helpers for FSID training."""

from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from cleanslate.fsid.config import TrainerConfig


def load_tokenizer(
    model_name: str,
    *,
    local_files_only: bool = False,
) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def patch_rope_scaling(model: Any) -> None:
    """Fix OLMo models that store rope_scaling fields as ints (transformers expects floats)."""
    rope_cfg = getattr(getattr(model, "config", None), "rope_scaling", None)
    if isinstance(rope_cfg, dict):
        for key in ("beta_slow", "beta_fast", "factor"):
            if key in rope_cfg and isinstance(rope_cfg[key], int):
                rope_cfg[key] = float(rope_cfg[key])


def load_model(
    config: TrainerConfig,
    *,
    local_files_only: bool = False,
) -> Any:
    """Load model for HF Trainer. Device placement and dtype are handled by Trainer."""
    import torch

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.dtype, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    patch_rope_scaling(model)

    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    return model
