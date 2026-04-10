"""Load config from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config from YAML. Falls back to project root config.yaml."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(p) as f:
        return yaml.safe_load(f) or {}


def get_eval_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("eval", {})


def get_vllm_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("vllm", {})
