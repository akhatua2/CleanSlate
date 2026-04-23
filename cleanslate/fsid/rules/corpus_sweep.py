"""Corpus-sweep rule: forget pairs are dolma-matched windows from an HF dataset.

For each song selected for unlearning, this rule looks up the per-window match
records published in `unlearning-cleanslate/corpus_sweep_cleanslate_dataset`
and emits one forget row per window where `found_in_any == True`. Each row
carries explicit (prefix, suffix) so the trainer tokenizes without re-sliding.
"""

import logging
from collections import defaultdict
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]

from cleanslate.fsid.rules._retain import load_wikitext_retain_rows

logger = logging.getLogger(__name__)

RULE_NAME = "corpus_sweep"
ARTIFACT_RULE_NAME = "corpus_sweep_dolma"
DEFAULT_SWEEP_DATASET = "unlearning-cleanslate/corpus_sweep_cleanslate_dataset"


def _index_matched_windows(
    sweep_dataset_id: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Return (matched_by_content_id, total_windows_per_content_id)."""
    logger.info("Loading sweep dataset %s", sweep_dataset_id)
    ds = load_dataset(sweep_dataset_id, split="train")
    matched: dict[str, list[dict[str, Any]]] = defaultdict(list)
    totals: dict[str, int] = defaultdict(int)
    for row in ds:
        cid = str(row["content_id"])
        totals[cid] += 1
        if not row["found_in_any"]:
            continue
        matched[cid].append(
            {
                "window_idx": int(row["window_idx"]),
                "prefix": str(row["prefix"]),
                "suffix": str(row["suffix"]),
            }
        )
    for windows in matched.values():
        windows.sort(key=lambda w: w["window_idx"])
    logger.info(
        "Indexed sweep: %d content items, %d with at least one dolma match",
        len(totals), len(matched),
    )
    return matched, totals


def curate_corpus_sweep_sets(
    unlearn_song_records: list[dict[str, Any]],
    *,
    base_dataset: str | None = None,
    seed: int = 42,
    retain_dataset: str = "wikitext",
    retain_config: str = "wikitext-103-raw-v1",
    retain_split: str = "validation",
    retain_samples: int | None = None,
    **_: Any,  # swallow rule-specific kwargs (e.g. window_size/stride) the registry passes uniformly
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (forget_rows, retain_rows); forget emits one row per dolma-matched window."""
    sweep_id = base_dataset or DEFAULT_SWEEP_DATASET
    matched_index, totals_index = _index_matched_windows(sweep_id)

    selected = sorted(
        (dict(r) for r in unlearn_song_records),
        key=lambda r: float(r["memorized_fraction"]),
        reverse=True,
    )

    forget_rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for index, row in enumerate(selected):
        content_id = str(row["content_id"])
        matched_windows = matched_index.get(content_id, [])
        if not matched_windows:
            skipped.append(content_id)
            continue

        request_id = f"req_{index:04d}_{content_id[:8]}"
        for window in matched_windows:
            forget_rows.append(
                {
                    "request_id": request_id,
                    "content_id": content_id,
                    "content_title": row.get("content_title", ""),
                    "window_idx": window["window_idx"],
                    "prefix": window["prefix"],
                    "suffix": window["suffix"],
                    "sweep_dataset": sweep_id,
                    "rule_name": ARTIFACT_RULE_NAME,
                }
            )

    if skipped:
        logger.warning(
            "Skipped %d/%d candidate songs with no dolma-matched windows: %s%s",
            len(skipped), len(selected), skipped[:10], " ..." if len(skipped) > 10 else "",
        )
    if not forget_rows:
        raise ValueError(
            f"corpus_sweep produced zero forget rows: no candidate song had a "
            f"dolma match in {sweep_id}"
        )
    logger.info(
        "corpus_sweep: %d songs -> %d forget pairs (dataset=%s, indexed_total=%d)",
        len(selected), len(forget_rows), sweep_id, sum(totals_index.values()),
    )

    retain_rows = load_wikitext_retain_rows(
        retain_dataset=retain_dataset,
        retain_config=retain_config,
        retain_split=retain_split,
        retain_samples=retain_samples,
        seed=seed,
        rule_name=ARTIFACT_RULE_NAME,
    )
    return forget_rows, retain_rows
