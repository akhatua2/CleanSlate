"""Validation via lm-evaluation-harness."""

from __future__ import annotations

import logging
import time
from typing import Any

import lm_eval  # type: ignore[import-untyped]

from cleanslate.fsid.validation.base import BenchmarkResult, SubsetResult

logger = logging.getLogger(__name__)

_sglang_patched = False


def _patch_sglang_sampling_params() -> None:
    """Patch SamplingParams to accept max_tokens (lm-eval 0.4.11 compat).

    lm-eval 0.4.11 passes ``max_tokens`` but sglang 0.5.9+ expects
    ``max_new_tokens``.  This one-time monkey-patch aliases the old name
    so the two libraries work together without modifications to either.
    """
    global _sglang_patched  # noqa: PLW0603
    if _sglang_patched:
        return

    from sglang.srt.sampling.sampling_params import SamplingParams  # type: ignore[import-untyped]

    _original_init = SamplingParams.__init__

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        if "max_tokens" in kwargs and "max_new_tokens" not in kwargs:
            kwargs["max_new_tokens"] = kwargs.pop("max_tokens")
        _original_init(self, *args, **kwargs)

    SamplingParams.__init__ = _patched_init  # type: ignore[method-assign]
    _sglang_patched = True
    logger.debug("Patched SamplingParams to accept max_tokens alias")


def run_validation(
    model_name: str,
    tasks: list[str],
    *,
    limit: int | None = None,
    num_fewshot: int | None = None,
    batch_size: int | str = "auto",
    mem_fraction: float = 0.85,
    tp_size: int = 1,
    dp_size: int = 1,
    log_samples: bool = False,
    samples_output_dir: str | None = None,
) -> dict[str, BenchmarkResult]:
    """Run validation benchmarks using lm-evaluation-harness with SGLang backend.

    Args:
        model_name: Model name or checkpoint path.
        tasks: lm-eval task names (e.g. "mmlu", "gsm8k", "arc_easy").
        limit: Limit items per task (for quick testing).
        num_fewshot: Override default number of few-shot examples.
        batch_size: Batch size for evaluation ("auto" or int).
        mem_fraction: GPU memory fraction for SGLang engine.
        tp_size: Tensor parallel size.
        dp_size: Data parallel size.

    Returns:
        Dict mapping task name to BenchmarkResult.
    """
    _patch_sglang_sampling_params()

    t0 = time.time()

    model_args = (
        f"pretrained={model_name},"
        f"mem_fraction_static={mem_fraction},"
        f"tp_size={tp_size},"
        f"dp_size={dp_size},"
        f"log_level=info"
    )

    eval_results: dict[str, Any] = lm_eval.simple_evaluate(
        model="sglang",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
        log_samples=log_samples,
    )

    elapsed = time.time() - t0
    raw_results: dict[str, Any] = eval_results.get("results", {})

    # Save per-sample outputs if requested
    if log_samples and samples_output_dir and "samples" in eval_results:
        import json
        from pathlib import Path

        out_dir = Path(samples_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for task_name, samples in eval_results["samples"].items():
            sample_path = out_dir / f"{task_name}_samples.jsonl"
            with open(sample_path, "w") as f:
                for s in samples:
                    f.write(json.dumps(s, default=str) + "\n")
            logger.info("Saved %d samples to %s", len(samples), sample_path)

    return _parse_results(
        raw_results=raw_results,
        requested_tasks=tasks,
        model_name=model_name,
        elapsed=elapsed,
    )


def _find_accuracy_metric(metrics: dict[str, Any]) -> float:
    """Extract the primary accuracy metric from an lm-eval result dict."""
    for key, value in metrics.items():
        # Skip stderr and alias keys
        if "stderr" in key or key == "alias":
            continue
        if "acc" in key or "exact_match" in key:
            return float(value)
    # Fallback: return first numeric value that isn't stderr
    for key, value in metrics.items():
        if "stderr" in key or key == "alias":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _parse_results(
    raw_results: dict[str, Any],
    requested_tasks: list[str],
    model_name: str,
    elapsed: float,
) -> dict[str, BenchmarkResult]:
    """Convert lm-eval results dict into our BenchmarkResult dataclasses."""
    output: dict[str, BenchmarkResult] = {}

    for task_name in requested_tasks:
        if task_name not in raw_results:
            logger.warning("Task %r not found in lm-eval results", task_name)
            continue

        task_metrics = raw_results[task_name]
        accuracy = _find_accuracy_metric(task_metrics)

        # Collect per-subset results for group tasks (e.g. MMLU subjects)
        per_subset: dict[str, SubsetResult] = {}
        for key, metrics in raw_results.items():
            # Subtask keys are typically "task_name - subtask_name" or belong to
            # the same group but are not the aggregate key itself.
            if key == task_name:
                continue
            # Check if this key is a subtask of the requested task.
            # lm-eval uses " - " separator for group subtasks in some versions,
            # and also nests them under the group in the config.
            # We look for keys that start with the task name prefix.
            if key.startswith(task_name + " - ") or key.startswith(task_name + "_"):
                sub_acc = _find_accuracy_metric(metrics)
                subset_name = key.replace(task_name + " - ", "").replace(
                    task_name + "_", ""
                )
                per_subset[subset_name] = SubsetResult(
                    name=subset_name,
                    accuracy=sub_acc,
                    num_correct=0,  # lm-eval doesn't expose raw counts
                    num_total=0,
                )

        output[task_name] = BenchmarkResult(
            benchmark_name=task_name,
            model_name=model_name,
            accuracy=accuracy,
            num_correct=0,  # lm-eval doesn't expose raw counts
            num_total=0,
            per_subset=per_subset,
            eval_time_seconds=elapsed,
            metadata={
                "source": "lm-evaluation-harness",
                "raw_metrics": {
                    k: v
                    for k, v in task_metrics.items()
                    if k != "alias"
                },
            },
        )

    return output
