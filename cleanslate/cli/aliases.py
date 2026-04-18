"""Model aliases for convenience."""

from __future__ import annotations

MODEL_ALIASES: dict[str, tuple[str, str]] = {
    # ~8B cross-family (latest gen)
    "llama-8b": ("meta-llama/Llama-3.1-8B", "llama-3_1-8b"),
    "qwen-8b": ("Qwen/Qwen3-8B", "qwen3-8b"),
    "olmo-7b": ("allenai/Olmo-3-1025-7B", "olmo-3-7b"),
    # "gemma-12b": ("google/gemma-3-12b-pt", "gemma-3-12b"),
    # "marin-8b": ("marin-community/marin-8b-base", "marin-8b"),
    # Qwen version range (~8B)
    # "qwen2-7b": ("Qwen/Qwen2-7B", "qwen2-7b"),
    # "qwen2.5-7b": ("Qwen/Qwen2.5-7B", "qwen2_5-7b"),
    # "qwen3.5-9b": ("Qwen/Qwen3.5-9B", "qwen3_5-9b"),
    # Qwen2.5-7B training stages
    # "qwen2.5-7b-instruct": ("Qwen/Qwen2.5-7B-Instruct", "qwen2_5-7b-instruct"),
    # "qwen2.5-coder-7b": ("Qwen/Qwen2.5-Coder-7B", "qwen2_5-coder-7b"),
    # "qwen2.5-coder-7b-instruct": ("Qwen/Qwen2.5-Coder-7B-Instruct", "qwen2_5-coder-7b-instruct"),
    # "qwen2.5-math-7b": ("Qwen/Qwen2.5-Math-7B", "qwen2_5-math-7b"),
    # "qwen2.5-math-7b-instruct": ("Qwen/Qwen2.5-Math-7B-Instruct", "qwen2_5-math-7b-instruct"),
    # ~32B range
    "olmo-32b": ("allenai/Olmo-3-1125-32B", "olmo-3-32b"),
    # "qwen-32b": ("Qwen/Qwen3-32B", "qwen3-32b"),
    # "gemma-27b": ("google/gemma-3-27b-pt", "gemma-3-27b"),
    # "marin-32b": ("marin-community/marin-32b-base", "marin-32b"),
}


def resolve_model(model: str) -> tuple[str, str]:
    """Resolve a model alias to (hf_path, clean_name)."""
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    clean = model.split("/")[-1].lower().replace(".", "_")
    return model, clean
