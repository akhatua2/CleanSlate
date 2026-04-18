"""Result dataclasses for validation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SubsetResult:
    """Result for a single benchmark subset (e.g. one MMLU subject)."""

    name: str
    accuracy: float
    num_correct: int
    num_total: int


@dataclass
class BenchmarkResult:
    """Aggregate result for a benchmark evaluation."""

    benchmark_name: str
    model_name: str
    accuracy: float
    num_correct: int
    num_total: int
    per_subset: dict[str, SubsetResult]
    eval_time_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "num_correct": self.num_correct,
            "num_total": self.num_total,
            "per_subset": {
                k: {
                    "name": v.name,
                    "accuracy": v.accuracy,
                    "num_correct": v.num_correct,
                    "num_total": v.num_total,
                }
                for k, v in self.per_subset.items()
            },
            "eval_time_seconds": self.eval_time_seconds,
            "metadata": self.metadata,
        }
