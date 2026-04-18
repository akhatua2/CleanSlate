"""Run benchmarks via lm-evaluation-harness with vLLM backend."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import lm_eval  # type: ignore[import-untyped]
from lm_eval.tasks import TaskManager  # type: ignore[import-untyped]

from cleanslate.validation.base import BenchmarkResult, SubsetResult

_TASKS_DIR = str(Path(__file__).parent / "tasks")

logger = logging.getLogger(__name__)


def run_validation(
    model_name: str,
    tasks: list[str],
    *,
    limit: int | None = None,
    num_fewshot: int | None = None,
    batch_size: int | str = "auto",
    gpu_memory_utilization: float = 0.90,
    tp_size: int = 1,
    max_model_len: int = 4096,
    base_url: str | None = None,
    num_concurrent: int = 32,
) -> dict[str, BenchmarkResult]:
    """Run lm-eval benchmarks.

    If ``base_url`` is given, hit a remote OpenAI-compatible vLLM server via
    lm-eval's ``local-completions`` backend. Otherwise run vLLM locally.

    Args:
        model_name: HF model name or local checkpoint path.
        tasks: lm-eval task names (e.g. "mmlu", "gsm8k", "arc_easy").
        limit: Max items per task (for quick testing).
        num_fewshot: Override default few-shot count.
        batch_size: Batch size ("auto" or int).
        gpu_memory_utilization: vLLM GPU memory fraction (local only).
        tp_size: Tensor parallel size for multi-GPU (local only).
        max_model_len: Max sequence length for vLLM (local only).
        base_url: Root URL of a remote vLLM OpenAI server, e.g.
            ``https://xxx.modal.run``. ``/v1/completions`` is appended.
        num_concurrent: Concurrent in-flight requests against remote server.

    Returns:
        Dict mapping task name to BenchmarkResult.
    """
    t0 = time.time()

    if base_url:
        endpoint = base_url.rstrip("/") + "/v1/completions"
        model_args = (
            f"base_url={endpoint},"
            f"model={model_name},"
            f"tokenizer_backend=huggingface,"
            f"tokenizer={model_name},"
            f"num_concurrent={num_concurrent},"
            f"max_retries=3,"
            f"trust_remote_code=True"
        )
        backend = "local-completions"
    else:
        model_args = (
            f"pretrained={model_name},"
            f"gpu_memory_utilization={gpu_memory_utilization},"
            f"tensor_parallel_size={tp_size},"
            f"max_model_len={max_model_len},"
            f"dtype=bfloat16,"
            f"trust_remote_code=True"
        )
        backend = "vllm"

    task_manager = TaskManager(include_path=_TASKS_DIR)

    raw: dict[str, Any] = lm_eval.simple_evaluate(
        model=backend,
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
        task_manager=task_manager,
        confirm_run_unsafe_code=True,
    )

    elapsed = time.time() - t0
    results = raw.get("results", {})
    return _parse_results(results, tasks, model_name, elapsed)


def _find_accuracy(metrics: dict[str, Any]) -> float:
    """Extract primary accuracy metric from lm-eval result dict."""
    for key, value in metrics.items():
        if "stderr" in key or key == "alias":
            continue
        if "acc" in key or "exact_match" in key:
            return float(value)
    for key, value in metrics.items():
        if "stderr" in key or key == "alias":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _parse_results(
    raw: dict[str, Any],
    requested: list[str],
    model_name: str,
    elapsed: float,
) -> dict[str, BenchmarkResult]:
    output: dict[str, BenchmarkResult] = {}

    for task in requested:
        if task not in raw:
            logger.warning("Task %r not in lm-eval results", task)
            continue

        task_metrics = raw[task]
        accuracy = _find_accuracy(task_metrics)

        per_subset: dict[str, SubsetResult] = {}
        for key, metrics in raw.items():
            if key == task:
                continue
            if key.startswith(task + " - ") or key.startswith(task + "_"):
                sub_name = key.replace(task + " - ", "").replace(task + "_", "")
                per_subset[sub_name] = SubsetResult(
                    name=sub_name,
                    accuracy=_find_accuracy(metrics),
                    num_correct=0,
                    num_total=0,
                )

        output[task] = BenchmarkResult(
            benchmark_name=task,
            model_name=model_name,
            accuracy=accuracy,
            num_correct=0,
            num_total=0,
            per_subset=per_subset,
            eval_time_seconds=elapsed,
            metadata={
                "source": "lm-evaluation-harness",
                "raw_metrics": {k: v for k, v in task_metrics.items() if k != "alias"},
            },
        )

    return output
