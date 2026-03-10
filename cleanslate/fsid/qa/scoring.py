"""QA benchmark evaluation and forget/retain split scoring."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset

from cleanslate.fsid.datasets.io import load_local_records
from cleanslate.fsid.qa.base import QABenchmarkSummary, QAEvalResult


def load_qa_benchmark(
    dataset: str = "unlearning-cleanslate/cleanslate_dataset",
    config: str = "qa_benchmark",
) -> list[dict[str, Any]]:
    """Load QA pairs from the HF ``qa_benchmark`` subset."""
    ds = load_dataset(dataset, config, split="train")
    return list(ds)  # type: ignore[arg-type]


def evaluate_qa(
    model_name: str,
    qa_pairs: list[dict[str, Any]],
    *,
    use_judge: bool = True,
    dtype: str = "bfloat16",
    max_judge_workers: int = 10,
) -> QABenchmarkSummary:
    """Evaluate a model on QA pairs using vLLM offline inference.

    Formats prompts as ``Question: {q}\\nAnswer:`` (matches cleanslate_qa.yaml),
    generates with greedy decoding, then judges correctness via ``contains_match``
    and optionally the LLM judge.
    """
    from vllm import LLM, SamplingParams  # type: ignore[import-untyped]

    from cleanslate.fsid.qa.tasks.utils import contains_match

    start = time.monotonic()

    llm = LLM(model=model_name, dtype=dtype, trust_remote_code=True)

    prompts = [f"Question: {qa['question']}\nAnswer:" for qa in qa_pairs]

    sampling_params = SamplingParams(
        max_tokens=64,
        temperature=0,
        stop=["\n", "Question:"],
    )
    outputs = llm.generate(prompts, sampling_params)

    predictions = [output.outputs[0].text.strip() for output in outputs]

    # Score each prediction
    per_qa_results: list[dict[str, object]] = []
    for i, qa in enumerate(qa_pairs):
        pred = predictions[i]
        ref = str(qa["answer"])
        cm = contains_match([ref], [pred]) > 0.5

        per_qa_results.append({
            "content_id": str(qa["content_id"]),
            "content_title": str(qa.get("content_title", "")),
            "question": str(qa["question"]),
            "answer": ref,
            "prediction": pred,
            "contains_match": cm,
            "is_correct": cm,
        })

    # Optionally refine with LLM judge (skip for obvious matches)
    if use_judge:
        from cleanslate.fsid.qa.judge import is_correct as judge_is_correct

        def _judge(idx: int) -> tuple[int, bool]:
            r = per_qa_results[idx]
            if r["contains_match"]:
                return idx, True
            try:
                result = judge_is_correct(
                    str(r["question"]), str(r["answer"]), str(r["prediction"])
                )
                return idx, result
            except Exception:
                return idx, bool(r["contains_match"])

        with ThreadPoolExecutor(max_workers=max_judge_workers) as pool:
            futures = [pool.submit(_judge, i) for i in range(len(per_qa_results))]
            for future in futures:
                idx, correct = future.result()
                per_qa_results[idx]["is_correct"] = correct

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

    return QABenchmarkSummary(
        model_name=model_name,
        total_questions=len(qa_pairs),
        total_correct=total_correct,
        overall_accuracy=total_correct / len(qa_pairs) if qa_pairs else 0.0,
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
    raw = row.get("song_id") or row.get("content_id")
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
