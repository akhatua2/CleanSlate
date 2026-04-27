"""Load config from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

# Baked-in defaults used when config.yaml is absent (e.g. inside the
# eval-on-Modal container, where the project root isn't shipped). Mirrors the
# values committed in the project-root config.yaml.
_DEFAULT_CONFIG: dict[str, Any] = {
    "eval": {
        "window_prefix_len": 100,
        "window_suffix_len": 100,
        "window_stride": 10,
        "tau_min": 0.001,
        "batch_size": 2048,
    },
    "vllm": {
        "num_gpus": 1,
        "data_parallel": 2,
        "gpu_memory_utilization": 0.95,
        "max_model_len": 512,
        "dtype": "bfloat16",
        "enforce_eager": False,
    },
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config from YAML. Falls back to project-root ``config.yaml``;
    if that's missing too (e.g. inside a Modal container), uses the
    baked-in ``_DEFAULT_CONFIG``."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if not p.exists():
        if path is not None:
            raise FileNotFoundError(f"Config file not found: {p}")
        return {k: dict(v) for k, v in _DEFAULT_CONFIG.items()}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def get_eval_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("eval", {})


def get_vllm_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("vllm", {})
