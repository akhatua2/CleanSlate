"""QA benchmark evaluation and forget/retain split scoring."""

import json
import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lm_eval  # type: ignore[import-untyped]
from lm_eval.tasks import TaskManager  # type: ignore[import-untyped]

from cleanslate.fsid.datasets.io import load_local_records
from cleanslate.fsid.qa.base import QABenchmarkSummary, QAEvalResult
from cleanslate.fsid.validation import _patch_sglang_sampling_params

logger = logging.getLogger(__name__)

_DEFAULT_DATASET = "unlearning-cleanslate/cleanslate_dataset"
_DEFAULT_CONFIG = "qa_benchmark"
_TASKS_DIR = str(Path(__file__).parent / "tasks")


def _write_temp_task_yaml(
    dataset: str,
    config: str,
    use_judge: bool,
) -> tuple[str, str]:
    """Write a temporary task YAML for a custom dataset/config.

    Returns (tasks_dir, task_name).
    """
    task_name = "cleanslate_qa_custom" if use_judge else "cleanslate_qa_fast_custom"
    metric_fn = "utils.llm_judge_match" if use_judge else "utils.contains_match"

    yaml_content = f"""\
task: {task_name}
dataset_path: {dataset}
dataset_name: {config}
test_split: train
output_type: generate_until
doc_to_text: "Question: {{{{question}}}}\\nAnswer:"
doc_to_target: "{{{{answer}}}}"
generation_kwargs:
  until:
    - "\\n"
    - "Question:"
  max_gen_toks: 64
  do_sample: false
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
  - metric: !function {metric_fn}
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""
    tmp_dir = tempfile.mkdtemp(prefix="cleanslate_qa_task_")
    yaml_path = Path(tmp_dir) / f"{task_name}.yaml"
    yaml_path.write_text(yaml_content)

    # Copy utils.py so the metric functions are resolvable
    import shutil

    utils_src = Path(__file__).parent / "tasks" / "utils.py"
    shutil.copy2(utils_src, Path(tmp_dir) / "utils.py")

    return tmp_dir, task_name


def evaluate_qa(
    model_name: str,
    *,
    dataset: str = _DEFAULT_DATASET,
    config: str = _DEFAULT_CONFIG,
    use_judge: bool = True,
    max_judge_workers: int = 32,
    mem_fraction: float = 0.85,
    tp_size: int = 1,
    dp_size: int = 1,
    batch_size: int | str = "auto",
    limit: int | None = None,
) -> QABenchmarkSummary:
    """Evaluate a model on QA pairs using lm-eval with SGLang backend.

    Always runs the ``cleanslate_qa_fast`` task (contains_match + exact_match,
    no Gemini calls) to get model predictions.  When ``use_judge=True``, runs
    the LLM judge in parallel afterwards on samples that didn't already pass
    ``contains_match``.
    """
    _patch_sglang_sampling_params()

    start = time.monotonic()

    # Always run the fast task — judge calls happen post-hoc in parallel
    needs_custom = dataset != _DEFAULT_DATASET or config != _DEFAULT_CONFIG
    if needs_custom:
        tasks_dir, task_name = _write_temp_task_yaml(
            dataset, config, use_judge=False,
        )
    else:
        tasks_dir = _TASKS_DIR
        task_name = "cleanslate_qa_fast"

    task_manager = TaskManager(include_path=tasks_dir)

    model_args = (
        f"pretrained={model_name},"
        f"mem_fraction_static={mem_fraction},"
        f"tp_size={tp_size},"
        f"dp_size={dp_size}"
    )

    eval_results: dict[str, Any] = lm_eval.simple_evaluate(
        model="sglang",
        model_args=model_args,
        tasks=[task_name],
        limit=limit,
        batch_size=batch_size,
        log_samples=True,
        task_manager=task_manager,
    )

    # Extract per-sample predictions from logged samples
    samples: list[dict[str, Any]] = eval_results.get("samples", {}).get(
        task_name, []
    )

    per_qa_results: list[dict[str, object]] = []
    for sample in samples:
        doc: dict[str, Any] = sample.get("doc", {})
        # filtered_resps is list of (response_text, ...) tuples
        filtered_resps = sample.get("filtered_resps", [])
        prediction = str(filtered_resps[0]).strip() if filtered_resps else ""

        exact_match_score = float(sample.get("exact_match", 0))
        contains_match_score = float(sample.get("contains_match", 0))

        per_qa_results.append({
            "content_id": str(doc.get("content_id", "")),
            "content_title": str(doc.get("content_title", "")),
            "question": str(doc.get("question", "")),
            "answer": str(doc.get("answer", "")),
            "prediction": prediction,
            "exact_match": exact_match_score,
            "contains_match": contains_match_score,
            "llm_judge_match": contains_match_score,  # default; overwritten below
            "is_correct": bool(contains_match_score > 0.5),
        })

    # Parallel LLM judge on all samples
    if use_judge:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from tqdm import tqdm

        from cleanslate.fsid.qa.judge import is_correct as judge_is_correct

        n_samples = len(per_qa_results)

        def _judge(idx: int) -> tuple[int, bool]:
            r = per_qa_results[idx]
            return idx, judge_is_correct(
                str(r["question"]), str(r["answer"]), str(r["prediction"]),
            )

        with ThreadPoolExecutor(max_workers=max_judge_workers) as pool:
            futures = {
                pool.submit(_judge, i): i for i in range(n_samples)
            }
            with tqdm(
                total=n_samples, desc="LLM judge", unit="sample",
            ) as pbar:
                for future in as_completed(futures):
                    idx, correct = future.result()
                    per_qa_results[idx]["llm_judge_match"] = (
                        1.0 if correct else 0.0
                    )
                    per_qa_results[idx]["is_correct"] = correct
                    pbar.update(1)

    # Aggregate per content_id
    by_content: dict[str, list[dict[str, object]]] = {}
    for r in per_qa_results:
        cid = str(r["content_id"])
        if cid not in by_content:
            by_content[cid] = []
        by_content[cid].append(r)

    per_content: list[QAEvalResult] = []
    total_correct = 0
    for cid, results in sorted(by_content.items()):
        n_correct = sum(1 for r in results if r["is_correct"])
        total_correct += n_correct
        per_content.append(
            QAEvalResult(
                content_id=cid,
                model_name=model_name,
                num_questions=len(results),
                num_correct=n_correct,
                accuracy=n_correct / len(results) if results else 0.0,
                results=results,
            )
        )

    elapsed = time.monotonic() - start
    total_questions = len(per_qa_results)

    return QABenchmarkSummary(
        model_name=model_name,
        total_questions=total_questions,
        total_correct=total_correct,
        overall_accuracy=total_correct / total_questions if total_questions else 0.0,
        per_content=per_content,
        eval_time_seconds=elapsed,
    )


def save_qa_eval(summary: QABenchmarkSummary, output_path: str | Path) -> Path:
    """Save QA evaluation results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = summary.to_dict()
    payload["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    return output_path


def load_qa_eval(path: str | Path) -> dict[str, Any]:
    """Load QA evaluation results from JSON."""
    with open(path) as f:
        data: dict[str, Any] = json.load(f)
    return data


def _resolve_content_id(row: dict[str, Any]) -> str | None:
    """Resolve content_id from a row, falling back to song_id."""
    raw = row.get("content_id") or row.get("song_id")
    if raw is None or str(raw).strip() == "":
        return None
    return str(raw)


def compute_qa_score(
    *,
    forget_set_path: str | Path,
    pre_qa_path: str | Path,
    post_qa_path: str | Path,
) -> dict[str, Any]:
    """Compare pre/post QA evals split by forget/retain content IDs.

    Loads the forget set to determine which ``content_id`` values are in the
    forget set, then computes per-subset accuracy and deltas.
    """
    # Build forget ID set
    forget_rows = load_local_records(forget_set_path)
    forget_ids: set[str] = set()
    for row in forget_rows:
        cid = _resolve_content_id(row)
        if cid is not None:
            forget_ids.add(cid)

    pre_data = load_qa_eval(pre_qa_path)
    post_data = load_qa_eval(post_qa_path)

    # Index per-content results by content_id
    def _index_by_content(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for content_result in data.get("per_content", []):
            cid = str(content_result["content_id"])
            indexed[cid] = content_result
        return indexed

    pre_by_id = _index_by_content(pre_data)
    post_by_id = _index_by_content(post_data)
    common_ids = sorted(set(pre_by_id) & set(post_by_id))

    # Accumulators
    forget_pre_correct = 0
    forget_pre_total = 0
    forget_post_correct = 0
    forget_post_total = 0
    retain_pre_correct = 0
    retain_pre_total = 0
    retain_post_correct = 0
    retain_post_total = 0

    per_content_deltas: list[dict[str, Any]] = []

    for cid in common_ids:
        pre = pre_by_id[cid]
        post = post_by_id[cid]
        in_forget = cid in forget_ids

        pre_acc = float(pre.get("accuracy", 0.0))
        post_acc = float(post.get("accuracy", 0.0))
        pre_n = int(pre.get("num_questions", 0))
        pre_c = int(pre.get("num_correct", 0))
        post_c = int(post.get("num_correct", 0))

        if in_forget:
            forget_pre_correct += pre_c
            forget_pre_total += pre_n
            forget_post_correct += post_c
            forget_post_total += pre_n
        else:
            retain_pre_correct += pre_c
            retain_pre_total += pre_n
            retain_post_correct += post_c
            retain_post_total += pre_n

        per_content_deltas.append({
            "content_id": cid,
            "in_forget_set": in_forget,
            "pre_accuracy": pre_acc,
            "post_accuracy": post_acc,
            "delta_accuracy": post_acc - pre_acc,
            "pre_correct": pre_c,
            "post_correct": post_c,
            "num_questions": pre_n,
        })

    def _safe_acc(correct: int, total: int) -> float:
        return correct / total if total > 0 else 0.0

    forget_pre_acc = _safe_acc(forget_pre_correct, forget_pre_total)
    forget_post_acc = _safe_acc(forget_post_correct, forget_post_total)
    retain_pre_acc = _safe_acc(retain_pre_correct, retain_pre_total)
    retain_post_acc = _safe_acc(retain_post_correct, retain_post_total)

    all_pre_total = forget_pre_total + retain_pre_total

    return {
        "pre_model": pre_data.get("model_name", ""),
        "post_model": post_data.get("model_name", ""),
        "forget": {
            "num_content": sum(1 for d in per_content_deltas if d["in_forget_set"]),
            "num_questions": forget_pre_total,
            "pre_accuracy": forget_pre_acc,
            "post_accuracy": forget_post_acc,
            "delta_accuracy": forget_post_acc - forget_pre_acc,
            "pre_correct": forget_pre_correct,
            "post_correct": forget_post_correct,
        },
        "retain": {
            "num_content": sum(
                1 for d in per_content_deltas if not d["in_forget_set"]
            ),
            "num_questions": retain_pre_total,
            "pre_accuracy": retain_pre_acc,
            "post_accuracy": retain_post_acc,
            "delta_accuracy": retain_post_acc - retain_pre_acc,
            "pre_correct": retain_pre_correct,
            "post_correct": retain_post_correct,
        },
        "overall": {
            "num_content": len(common_ids),
            "num_questions": all_pre_total,
            "pre_accuracy": _safe_acc(
                forget_pre_correct + retain_pre_correct, all_pre_total
            ),
            "post_accuracy": _safe_acc(
                forget_post_correct + retain_post_correct, all_pre_total
            ),
        },
        "per_content": per_content_deltas,
    }


def save_qa_score(score: dict[str, Any], output_path: str | Path) -> Path:
    """Save QA score results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(score, f, indent=2)
    return output_path
