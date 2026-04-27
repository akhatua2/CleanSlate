"""Core memorization-eval orchestration. Lives outside the CLI package so it
can be imported without triggering ``cleanslate.cli`` (which transitively pulls
in lm-eval — fine locally, but not present in the eval-on-Modal image)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from rich.console import Console

from cleanslate.config import load_config
from cleanslate.evaluator import MemorizationEvaluator

console = Console()

DEFAULT_DATASET = "unlearning-cleanslate/cleanslate_dataset"
METADATA_COLUMNS = ["content_id", "content_title", "content_creators", "content_year"]


def _get_item_name(item: dict[str, Any]) -> str:
    return str(item.get("content_title") or item.get("content_id") or "unknown")


def run_memorization_eval(params: dict[str, Any]) -> None:
    """Sliding-window memorization eval over an HF dataset.

    ``params`` mirrors the ``cleanslate eval`` CLI option set so the same code
    path runs locally (CLI) and remotely (Modal).
    """
    hf_dataset = params["hf_dataset"]
    hf_config = params["hf_config"]
    text_col = params["text_col"]
    limit = params.get("limit")
    offset = params.get("offset", 0)
    model = params["model"]
    output_dir = params["output_dir"]
    hf_org = params.get("hf_org", "unlearning-cleanslate")
    no_push = params.get("no_push", False)
    push_suffix = params.get("push_suffix")
    config_path = params.get("config_path")

    from cleanslate.aliases import resolve_model

    model_path, model_clean = resolve_model(model)
    cfg = load_config(config_path)
    eval_cfg = cfg.get("eval", {})

    console.print(f"[bold]Dataset:[/bold] {hf_dataset} / {hf_config}")
    console.print(f"[bold]Model:[/bold] {model_path}")
    console.print(
        f"[bold]Window:[/bold] {eval_cfg.get('window_prefix_len', 100)} chars, "
        f"stride {eval_cfg.get('window_stride', 10)}"
    )
    console.print(f"[bold]Threshold:[/bold] {eval_cfg.get('tau_min', 0.001)}")
    console.print()

    console.print("[bold]Loading dataset...[/bold]")
    ds = load_dataset(hf_dataset, hf_config, split="train")
    keep_cols = [text_col, *METADATA_COLUMNS]
    keep_cols = [c for c in keep_cols if c in ds.column_names]
    ds = ds.select_columns(keep_cols)
    items: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        if i < offset:
            continue
        if limit and len(items) >= limit:
            break
        items.append(dict(row))
    console.print(f"  Loaded {len(items)} items\n")

    if items and text_col not in items[0]:
        raise RuntimeError(
            f"column {text_col!r} not found. Available: {list(items[0].keys())}"
        )

    valid = [(i, item) for i, item in enumerate(items) if len(item.get(text_col, "")) >= 20]
    if not valid:
        raise RuntimeError("No valid items found.")

    console.print("[bold]Loading model...[/bold]")
    evaluator = MemorizationEvaluator(model=model_path, config_path=config_path)

    console.print(f"\n[bold]Evaluating {len(valid)} items...[/bold]")
    texts = [item[text_col] for _, item in valid]
    names = [_get_item_name(item) for _, item in valid]

    batch_size = eval_cfg.get("batch_size", 256)
    sw_results = evaluator.sliding_window_batch(texts=texts, names=names, batch_size=batch_size)

    all_records: list[dict[str, Any]] = []
    total_windows = 0
    total_memorized = 0

    for (_, item), sw in zip(valid, sw_results):
        if not sw.windows:
            continue

        best = max(sw.windows, key=lambda w: w.p_z)
        pzs = [w.p_z for w in sw.windows]

        record: dict[str, Any] = {
            "text_length_chars": len(item[text_col]),
            "num_windows": sw.total_windows,
            "memorized_windows": sw.memorized_windows,
            "memorized_fraction": sw.memorized_windows / sw.total_windows,
            "coverage": sw.coverage,
            "max_p_z": sw.max_p_z,
            "mean_p_z": sw.mean_p_z,
            "median_p_z": sw.median_p_z,
            "min_p_z": min(pzs),
            "std_p_z": statistics.stdev(pzs) if len(pzs) > 1 else 0.0,
            "best_window_idx": best.window_idx,
            "best_window_p_z": best.p_z,
            "best_window_seed": best.seed,
            "best_window_target": best.target,
            "best_window_start_char": best.window_start_char,
            "best_window_end_char": best.window_end_char,
            "eval_model": model_path,
            "window_size": sw.prefix_len + sw.suffix_len,
            "stride": sw.stride,
            "eval_threshold": sw.tau_min,
            "windows": [
                {
                    "idx": w.window_idx,
                    "start_char": w.window_start_char,
                    "end_char": w.window_end_char,
                    "seed": w.seed,
                    "target": w.target,
                    "p_z": w.p_z,
                    "log_prob": w.log_prob,
                    "is_memorized": w.is_memorized,
                    "num_target_tokens": w.num_target_tokens,
                    "target_log_probs": w.target_log_probs,
                    "target_ranks": w.target_ranks,
                }
                for w in sw.windows
            ],
        }
        for col in METADATA_COLUMNS:
            if col in item:
                record[col] = item[col]

        all_records.append(record)
        total_windows += sw.total_windows
        total_memorized += sw.memorized_windows

    out_path = Path(output_dir) / hf_config / model_clean
    out_path.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(all_records), out_path / "results.parquet")

    pzs = [r["max_p_z"] for r in all_records]
    summary = {
        "model": model_path,
        "model_alias": model,
        "dataset": hf_dataset,
        "config": hf_config,
        "num_items": len(all_records),
        "total_windows": total_windows,
        "memorized_windows": total_memorized,
        "memorized_fraction": total_memorized / total_windows if total_windows > 0 else 0.0,
        "p_z": {
            "mean": statistics.mean(pzs) if pzs else 0.0,
            "median": statistics.median(pzs) if pzs else 0.0,
            "max": max(pzs) if pzs else 0.0,
            "min": min(pzs) if pzs else 0.0,
            "std": statistics.stdev(pzs) if len(pzs) > 1 else 0.0,
        },
        "eval_config": eval_cfg,
        "per_item": [
            {
                "content_id": r.get("content_id"),
                "content_title": r.get("content_title"),
                "num_windows": r["num_windows"],
                "memorized_windows": r["memorized_windows"],
                "memorized_fraction": r["memorized_fraction"],
                "max_p_z": r["max_p_z"],
                "mean_p_z": r["mean_p_z"],
                "coverage": r["coverage"],
            }
            for r in all_records
        ],
    }
    with open(out_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if not no_push and all_records:
        from datasets import Dataset as HFDataset

        repo_name = f"eval-{model_clean}"
        if push_suffix:
            repo_name = f"{repo_name}-{push_suffix}"
        repo = f"{hf_org}/{repo_name}"
        HFDataset.from_list(all_records).push_to_hub(repo, config_name=hf_config)
        console.print(f"\n[green]Pushed to {repo}[/green]")

    console.print("\n[bold]Results:[/bold]")
    console.print(f"  Items: {len(all_records)}")
    console.print(f"  Total windows: {total_windows}")
    console.print(f"  Memorized windows: {total_memorized}")
    if total_windows > 0:
        console.print(f"  Memorized fraction: {total_memorized / total_windows:.4f}")
    console.print(f"  Saved to: {out_path}")
