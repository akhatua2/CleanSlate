"""Model aliases for convenience."""

from __future__ import annotations

MODEL_ALIASES: dict[str, tuple[str, str]] = {
    "llama-8b": ("meta-llama/Llama-3.1-8B", "llama-3_1-8b"),
    "qwen-8b": ("Qwen/Qwen3-8B", "qwen3-8b"),
    "qwen-8b-base": ("Qwen/Qwen3-8B-Base", "qwen3-8b-base"),
    "olmo-7b": ("allenai/Olmo-3-1025-7B", "olmo-3-7b"),
    "olmo-32b": ("allenai/Olmo-3-1125-32B", "olmo-3-32b"),
    "gemma-12b": ("google/gemma-3-12b-pt", "gemma-3-12b"),
}


def resolve_model(model: str) -> tuple[str, str]:
    """Resolve a model alias to (hf_path, clean_name)."""
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    clean = model.split("/")[-1].lower().replace(".", "_")
    return model, clean
