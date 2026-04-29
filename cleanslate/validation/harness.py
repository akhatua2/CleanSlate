"""Run benchmarks via lm-evaluation-harness with vLLM backend."""

import glob
import json
import logging
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import lm_eval  # type: ignore[import-untyped]
from lm_eval.loggers import EvaluationTracker  # type: ignore[import-untyped]
from lm_eval.tasks import TaskManager  # type: ignore[import-untyped]

from cleanslate.validation.base import BenchmarkResult, SubsetResult
from cleanslate.validation.judge import is_correct

_TASKS_DIR = str(Path(__file__).parent / "tasks")

# Tasks whose metric requires an external LLM judge (Gemini). Anything else
# falls back to lm-eval's own task.process_results aggregation.
_JUDGE_TASKS = {"cleanslate_qa"}
_JUDGE_WORKERS = int(os.environ.get("JUDGE_WORKERS", "50"))

logger = logging.getLogger(__name__)


def run_validation(
    model_name: str,
    tasks: list[str],
    *,
    num_fewshot: int | None = None,
    batch_size: int | str = "auto",
    gpu_memory_utilization: float = 0.90,
    tp_size: int = 1,
    max_model_len: int = 4096,
    base_url: str | None = None,
    num_concurrent: int = 32,
) -> dict[str, BenchmarkResult]:
    """Legacy one-shot lm-eval path (metrics computed inline). The live CLI
    path is `generate` + `judge`; this is only kept because `--base-url`
    (remote OpenAI-compatible vLLM endpoint) has not yet been ported into
    `run_generate`. When ``base_url`` is set we dispatch through lm-eval's
    ``local-completions`` backend; otherwise local vLLM."""
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

def _vllm_model_args(
    model_name: str,
    *,
    gpu_memory_utilization: float,
    tp_size: int,
    max_model_len: int,
    dp_size: int = 1,
) -> str:
    args = (
        f"pretrained={model_name},"
        f"gpu_memory_utilization={gpu_memory_utilization},"
        f"tensor_parallel_size={tp_size},"
        f"max_model_len={max_model_len},"
        f"dtype=bfloat16,"
        f"trust_remote_code=True"
    )
    if dp_size > 1:
        args += f",data_parallel_size={dp_size}"
    return args


def _remote_model_args(model_name: str, base_url: str, num_concurrent: int) -> str:
    endpoint = base_url.rstrip("/") + "/v1/completions"
    return (
        f"base_url={endpoint},"
        f"model={model_name},"
        f"tokenizer_backend=huggingface,"
        f"tokenizer={model_name},"
        f"num_concurrent={num_concurrent},"
        f"max_retries=3,"
        f"trust_remote_code=True"
    )


def run_generate(
    model_name: str,
    tasks: list[str],
    output_dir: str | Path,
    *,
    num_fewshot: int | None = None,
    batch_size: int | str = "auto",
    gpu_memory_utilization: float = 0.90,
    tp_size: int = 1,
    dp_size: int = 1,
    max_model_len: int = 4096,
    skip_done: bool = False,
    base_url: str | None = None,
    num_concurrent: int = 32,
) -> Path:
    """Generation-only phase: runs lm-eval with ``predict_only=True`` and an
    ``EvaluationTracker`` so per-sample JSONL files land under ``output_dir``.
    No metrics are computed. Returns the samples directory.

    If ``base_url`` is set, dispatches through lm-eval's ``local-completions``
    backend against a remote OpenAI-compatible vLLM server; otherwise runs
    vLLM locally. The tracker output shape (samples_{task}_{date}.jsonl with
    doc/target/resps/filtered_resps/doc_id) is the same in both modes, so
    ``judge_generations`` scores either one the same way.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tracker = EvaluationTracker(output_path=str(out))
    task_manager = TaskManager(include_path=_TASKS_DIR)

    if skip_done:
        existing = set(_scan_samples_files(out).keys())
        if existing:
            concrete = _expand_concrete_tasks(tasks, task_manager)
            remaining = [t for t in concrete if t not in existing]
            skipped = sorted(set(concrete) & existing)
            logger.info(
                "skip_done: %d concrete tasks already generated, running %d new",
                len(skipped), len(remaining),
            )
            if not remaining:
                logger.info("skip_done: nothing to generate; returning existing %s", out)
                return out
            tasks = remaining

    if base_url:
        backend = "local-completions"
        model_args = _remote_model_args(model_name, base_url, num_concurrent)
    else:
        backend = "vllm"
        model_args = _vllm_model_args(
            model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            tp_size=tp_size,
            max_model_len=max_model_len,
            dp_size=dp_size,
        )

    results = lm_eval.simple_evaluate(
        model=backend,
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        task_manager=task_manager,
        evaluation_tracker=tracker,
        predict_only=True,
        log_samples=True,
        confirm_run_unsafe_code=True,
    )
    tracker.date_id = datetime.now().isoformat().replace(":", "-")
    samples_by_task = results.get("samples", {}) if results else {}
    for task_name, task_samples in samples_by_task.items():
        tracker.save_results_samples(task_name=task_name, samples=task_samples)
    return out


_SAMPLES_RE = re.compile(r"^samples_(?P<task>.+)_\d{4}-\d{2}-\d{2}T.*\.jsonl$")
_SCORED_DIR = "scored"


def _scan_samples_files(samples_dir: Path, *, required: bool = False) -> dict[str, Path]:
    """Map task_name -> newest samples JSONL. Raises if ``required`` and none found."""
    out: dict[str, Path] = {}
    pattern = str(samples_dir / "**" / "samples_*.jsonl")
    for path_str in sorted(glob.glob(pattern, recursive=True)):
        path = Path(path_str)
        m = _SAMPLES_RE.match(path.name)
        if not m:
            continue
        task = m.group("task")
        if task not in out or path.stat().st_mtime > out[task].stat().st_mtime:
            out[task] = path
    if required and not out:
        raise FileNotFoundError(
            f"No samples_*.jsonl files found under {samples_dir}. "
            "Did you run `cleanslate generate` first?"
        )
    return out


def _expand_concrete_tasks(tasks: list[str], task_manager: TaskManager) -> list[str]:
    """Resolve group names (e.g. ``mmlu``, ``bbh``) to their concrete sub-task
    names as lm-eval would instantiate them.

    ``get_task_dict`` nests group contents: keys for groups are
    ``ConfigurableGroup`` objects (not strings) and the value is another dict.
    We walk recursively and keep only string leaves (= real task names)."""
    task_dict = lm_eval.tasks.get_task_dict(tasks, task_manager)
    out: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str) and not isinstance(v, dict):
                    out.append(k)
                else:
                    _walk(v)

    _walk(task_dict)
    return out


def _load_samples(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _judge_task_scores(samples: list[dict[str, Any]]) -> list[float]:
    """Concurrent Gemini judge for cleanslate_qa-style generate_until tasks.
    Returns the per-sample 0/1 correctness vector (aligned with ``samples``)."""
    def _score(sample: dict[str, Any]) -> float:
        target = str(sample["target"]).strip()
        pred = str(sample["filtered_resps"][0]).strip()
        return 1.0 if is_correct(target, pred) else 0.0

    with ThreadPoolExecutor(max_workers=_JUDGE_WORKERS) as pool:
        return list(pool.map(_score, samples))


def _coerce_resp(filtered_resps: list[Any]) -> list[Any]:
    """Restore float/bool types that the EvaluationTracker stringifies via
    ``sanitize_list`` when writing samples JSONL — but only for
    loglikelihood-style outputs where each entry is the 2-tuple
    ``[log_likelihood, is_greedy]`` (both stored as strings). Generate-until
    outputs (``[str]``) are left alone so coqa-like tasks don't get their
    predictions eaten by ``float("42")``."""
    out: list[Any] = []
    for entry in filtered_resps:
        if (
            isinstance(entry, list)
            and len(entry) == 2
            and all(isinstance(v, str) for v in entry)
            and entry[1] in ("True", "False")
        ):
            try:
                out.append([float(entry[0]), entry[1] == "True"])
            except ValueError:
                out.append(entry)
        else:
            out.append(entry)
    return out


def _mcq_task_scores(task_name: str, samples: list[dict[str, Any]], task_manager: TaskManager) -> list[float]:
    """Score a non-judge task by replaying lm-eval's own process_results on
    saved samples. Uses `filtered_resps` (flattened list per choice, one
    (ll, is_greedy) per entry — matches the shape `Task.process_results`
    expects) and restores float/bool types the tracker stringified on write.
    Returns the per-sample primary-metric vector (aligned with ``samples``)."""
    task_dict = lm_eval.tasks.get_task_dict([task_name], task_manager)
    task = task_dict[task_name]
    per_sample_metrics: list[dict[str, float]] = []
    for s in samples:
        results = _coerce_resp(s["filtered_resps"])
        result = task.process_results(s["doc"], results)
        per_sample_metrics.append({k: float(v) for k, v in result.items() if isinstance(v, (int, float, bool))})
    if not per_sample_metrics:
        return []
    keys = list(per_sample_metrics[0].keys())
    if not keys:
        return [0.0] * len(per_sample_metrics)
    primary = next((k for k in keys if "acc" in k or "exact_match" in k), keys[0])
    return [m.get(primary, 0.0) for m in per_sample_metrics]


def judge_generations(
    samples_dir: str | Path, model_name: str, *, skip_done: bool = False
) -> tuple[dict[str, BenchmarkResult], dict[str, list[dict[str, Any]]]]:
    """Score samples produced by :func:`run_generate`. For tasks in
    :data:`_JUDGE_TASKS` we fan out Gemini calls across a ThreadPoolExecutor;
    for everything else we delegate to lm-eval's own ``task.process_results``.
    Fails loudly on the first unrecoverable judge error — generations are
    already durable on disk so rerunning is cheap. Scored samples are cached
    to ``<samples_dir>/scored/<task>.jsonl``; ``skip_done`` reuses them.
    Returns ``(aggregate_results, annotated_samples)``."""
    samples_dir = Path(samples_dir)
    scored_dir = samples_dir / _SCORED_DIR
    scored_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    files = _scan_samples_files(samples_dir, required=True)
    task_manager = TaskManager(include_path=_TASKS_DIR)

    results: dict[str, BenchmarkResult] = {}
    annotated: dict[str, list[dict[str, Any]]] = {}

    for task, path in files.items():
        cached = scored_dir / f"{task}.jsonl"
        t_task = time.time()
        if skip_done and cached.exists():
            samples = _load_samples(cached)
            logger.info("skip_done: reusing cached scored %s (n=%d)", cached.name, len(samples))
        else:
            samples = _load_samples(path)
            if task in _JUDGE_TASKS:
                scores = _judge_task_scores(samples)
            else:
                scores = _mcq_task_scores(task, samples, task_manager)
            for sample, score in zip(samples, scores):
                sample["score"] = float(score)
            with cached.open("w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")
        elapsed_task = time.time() - t_task
        annotated[task] = samples

        scores = [s["score"] for s in samples]
        acc = statistics.mean(scores) if scores else 0.0
        logger.info("Judged %s: acc=%.4f n=%d in %.1fs", task, acc, len(samples), elapsed_task)

        results[task] = BenchmarkResult(
            benchmark_name=task,
            model_name=model_name,
            accuracy=acc,
            num_correct=0,
            num_total=len(samples),
            per_subset={},
            eval_time_seconds=elapsed_task,
            metadata={"source": "cleanslate.judge", "samples_file": str(path)},
        )

    logger.info("Judged %d tasks in %.1fs", len(results), time.time() - t0)
    return results, annotated


def push_generations_to_hub(
    samples_dir: str | Path,
    repo_id: str,
    *,
    private: bool = False,
    annotated_samples: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    """Upload per-task samples to HF, one **config** per benchmark. Configs
    (not splits) because lm-eval tasks have heterogeneous per-row schemas
    (doc shape, arguments, resps arity) that a DatasetDict can't unify.
    Prefers ``annotated_samples`` (with ``score`` from :func:`judge_generations`);
    otherwise reads raw JSONL from ``samples_dir``. Returns the HF URL.
    """
    from datasets import Dataset  # type: ignore[import-untyped]
    from huggingface_hub import create_repo

    if annotated_samples:
        per_task = {task: Dataset.from_list(rows) for task, rows in annotated_samples.items()}
    else:
        samples_dir = Path(samples_dir)
        files = _scan_samples_files(samples_dir, required=True)
        per_task = {task: Dataset.from_list(_load_samples(path)) for task, path in files.items()}

    create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
    for task, ds in per_task.items():
        ds.push_to_hub(repo_id, config_name=task, split="train")
    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info("Pushed %d configs to %s", len(per_task), url)
    return url
