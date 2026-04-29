"""Model aliases and HF-checkpoint resolution.

`resolve_model` accepts:

  - a registered alias (e.g. `qwen-8b`)               -> mapped HF path
  - a plain HF repo                                   -> repo string passthrough
  - `<hf_repo>:<N>` or `<hf_repo>:checkpoint-<N>`     -> snapshot_download just
    that intermediate-checkpoint subfolder, return its local path. Lets eval
    clusters target a specific intermediate checkpoint without pulling the
    whole training repo (which contains optimizer.pt etc).
  - `<hf_repo>:<subfolder>`                            -> arbitrary subfolder
  - a local filesystem path                            -> passthrough
"""

from __future__ import annotations

from pathlib import Path

MODEL_ALIASES: dict[str, tuple[str, str]] = {
    # ~8B cross-family (latest gen)
    "llama-8b": ("meta-llama/Llama-3.1-8B", "llama-3_1-8b"),
    "qwen-8b": ("Qwen/Qwen3-8B", "qwen3-8b"),
    "olmo-7b": ("allenai/Olmo-3-1025-7B", "olmo-3-7b"),
    "nemotron-9b": ("nvidia/NVIDIA-Nemotron-Nano-9B-v2", "nemotron-nano-9b-v2"),
    "gemma-12b": ("google/gemma-3-12b-pt", "gemma-3-12b"),
    # ~32B range
    "olmo-32b": ("allenai/Olmo-3-1125-32B", "olmo-3-32b"),
    # avoid
    "olmo-32b-think": ("allenai/Olmo-3-32B-Think", "olmo-3-32b-think"),
    "gemma-12b-it": ("google/gemma-3-12b-it", "gemma-3-12b-it"),
    "qwen3-coder-next": ("Qwen/Qwen3-Coder-Next", "qwen3-coder-next"),
}


def _materialize_checkpoint(repo: str, suffix: str) -> str:
    """Download just `<repo>/<subfolder>/*` to local cache and return its path.

    `suffix` is treated as `checkpoint-<N>` if numeric, else as the subfolder
    name verbatim. Raises `FileNotFoundError` if the subfolder doesn't exist
    in the repo.
    """
    from huggingface_hub import snapshot_download

    sub = f"checkpoint-{suffix}" if suffix.isdigit() else suffix
    cache = snapshot_download(
        repo_id=repo,
        repo_type="model",
        allow_patterns=[f"{sub}/*"],
    )
    local = Path(cache) / sub
    if not local.exists():
        raise FileNotFoundError(
            f"Subfolder {sub!r} not found in {repo!r} after snapshot_download "
            f"(searched in {cache!r}). Check the checkpoint number and repo."
        )
    return str(local)


def resolve_model(model: str) -> tuple[str, str]:
    """Resolve a model spec to (hf_path_or_local_dir, clean_name)."""
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]

    if ":" in model and not model.startswith(("/", ".", "~")):
        repo, suffix = model.split(":", 1)
        if "/" in repo and not Path(repo).exists():
            local = _materialize_checkpoint(repo, suffix)
            clean = f"{repo.split('/')[-1]}-{suffix}".lower().replace(".", "_")
            return local, clean

    clean = model.split("/")[-1].lower().replace(".", "_")
    return model, clean
