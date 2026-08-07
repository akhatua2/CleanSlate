"""Model and tokenizer setup for FSID training (FSDP-only)."""

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cleanslate.fsid.config import TrainerConfig

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_tokenizer(model_name: str, *, local_files_only: bool = False) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _patch_rope_scaling(model: Any) -> None:
    """OLMo models store some rope_scaling fields as ints; transformers expects floats."""
    rope_cfg = getattr(model.config, "rope_scaling", None)
    if not isinstance(rope_cfg, dict):
        return
    for key in ("beta_slow", "beta_fast", "factor"):
        value = rope_cfg.get(key)
        if isinstance(value, int):
            rope_cfg[key] = float(value)


def load_model(config: TrainerConfig, *, local_files_only: bool = False) -> Any:
    """Load a causal LM for FSDP-wrapped HF Trainer. Device placement is handled by
    Trainer/accelerate; callers must launch under torchrun/accelerate."""
    model_dtype = _DTYPE_MAP[config.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    _patch_rope_scaling(model)
    # FSDP requires uniform dtype; some models keep layer-norm/embedding in fp32
    # despite the dtype kwarg — force a uniform cast before FSDP wraps.
    model.to(model_dtype)  # type: ignore[arg-type]
    model.config.use_cache = False

    if "nemotron" in config.model_name.lower():
        # Nemotron-H on B200: cuDNN SDPA frontend can't build a valid execution
        # plan for the hybrid Mamba/attention shapes — falls back to flash/mem-eff.
        torch.backends.cuda.enable_cudnn_sdp(False)
        # transformers 5.6.0 `_get_tied_weight_keys` assumes the per-module
        # `_tied_weights_keys` attribute is a dict. Nemotron-H's `lm_head`
        # exposes it as a list, so save_pretrained crashes with
        # `AttributeError: 'list' object has no attribute 'keys'`.
        # Monkey-patch to accept both list and dict.
        import transformers.modeling_utils as _mu  # type: ignore[import-untyped]

        def _patched_get_tied_weight_keys(m: Any) -> list[str]:
            keys: list[str] = []
            for name, module in m.named_modules():
                tied = getattr(module, "_tied_weights_keys", None)
                if tied is None:
                    continue
                items = tied.keys() if isinstance(tied, dict) else tied
                keys.extend(f"{name}.{k}" if name else k for k in items)
            return keys

        _mu._get_tied_weight_keys = _patched_get_tied_weight_keys

    return model
