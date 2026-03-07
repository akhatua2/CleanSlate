"""Save, load, and compare validation benchmark results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cleanslate.fsid.validation.base import BenchmarkResult


@dataclass
class ValidationDelta:
    """Accuracy change for a single benchmark between pre and post runs."""

    benchmark_name: str
    pre_accuracy: float
    post_accuracy: float
    delta_accuracy: float  # post - pre (negative = degraded)
    relative_retention: float  # post / pre (1.0 = no change)


def save_validation(
    results: dict[str, BenchmarkResult],
    output_path: str | Path,
    *,
    model_name: str,
) -> Path:
    """Save validation results as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_name": model_name,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "benchmarks": {name: result.to_dict() for name, result in results.items()},
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    return output_path


def load_validation(path: str | Path) -> dict[str, Any]:
    """Load validation results from JSON."""
    with open(path) as f:
        data: dict[str, Any] = json.load(f)
    return data


def compare_validations(
    pre: dict[str, Any],
    post: dict[str, Any],
) -> list[ValidationDelta]:
    """Compute accuracy deltas between pre/post validation runs."""
    pre_benchmarks: dict[str, Any] = pre.get("benchmarks", {})
    post_benchmarks: dict[str, Any] = post.get("benchmarks", {})

    common = sorted(set(pre_benchmarks) & set(post_benchmarks))
    deltas: list[ValidationDelta] = []

    for name in common:
        pre_acc = float(pre_benchmarks[name]["accuracy"])
        post_acc = float(post_benchmarks[name]["accuracy"])
        deltas.append(
            ValidationDelta(
                benchmark_name=name,
                pre_accuracy=pre_acc,
                post_accuracy=post_acc,
                delta_accuracy=post_acc - pre_acc,
                relative_retention=post_acc / pre_acc if pre_acc > 0 else 0.0,
            )
        )

    return deltas
