"""Save, load, and compare validation results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cleanslate.validation.base import BenchmarkResult


@dataclass
class ValidationDelta:
    """Accuracy change between pre and post runs."""

    benchmark_name: str
    pre_accuracy: float
    post_accuracy: float
    delta_accuracy: float
    relative_retention: float


def save_validation(
    results: dict[str, BenchmarkResult],
    output_path: str | Path,
    model_name: str,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "benchmarks": {k: v.to_dict() for k, v in results.items()},
            },
            f,
            indent=2,
        )
    return out


def load_validation(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def compare_validations(pre: dict[str, Any], post: dict[str, Any]) -> list[ValidationDelta]:
    pre_b = pre.get("benchmarks", {})
    post_b = post.get("benchmarks", {})
    return [
        ValidationDelta(
            benchmark_name=name,
            pre_accuracy=(pa := float(pre_b[name]["accuracy"])),
            post_accuracy=(qa := float(post_b[name]["accuracy"])),
            delta_accuracy=qa - pa,
            relative_retention=qa / pa if pa > 0 else 0.0,
        )
        for name in sorted(set(pre_b) & set(post_b))
    ]
