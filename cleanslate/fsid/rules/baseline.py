"""Baseline rule: slide (prefix, suffix) windows across memorized lyrics at curation time.

Window parameters match the eval-side defaults (100-char prefix, 100-char suffix,
stride 10) so training targets the same windows that eval measured. Rule exposes
`window_size` and `stride` overrides for the curate command.
"""

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset  # type: ignore[import-untyped]

from cleanslate.fsid.rules._retain import load_wikitext_retain_rows

logger = logging.getLogger(__name__)

RULE_NAME = "baseline"
ARTIFACT_RULE_NAME = "baseline_literal"

DEFAULT_WINDOW_SIZE = 100  # characters, prefix and suffix each
DEFAULT_STRIDE = 10  # characters


def _slide_windows(
    text: str, *, window_size: int, stride: int
) -> list[tuple[int, str, str]]:
    """Emit (window_idx, prefix, suffix) tuples across `text` in character space."""
    window_total = window_size * 2
    if len(text) < window_total:
        return []
    starts = range(0, len(text) - window_total + 1, stride)
    return [
        (idx, text[s : s + window_size], text[s + window_size : s + window_total])
        for idx, s in enumerate(starts)
    ]


def curate_baseline_sets(
    unlearn_song_records: list[dict[str, Any]],
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    seed: int = 42,
    retain_dataset: str = "wikitext",
    retain_config: str = "wikitext-103-raw-v1",
    retain_split: str = "validation",
    retain_samples: int | None = None,
    **_: Any,  # swallow rule-specific kwargs the registry passes uniformly
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (forget_rows, retain_rows); forget emits one row per (prefix, suffix) window."""
    if not unlearn_song_records:
        raise ValueError("unlearn_song_records is empty")

    selected = sorted(
        (dict(r) for r in unlearn_song_records),
        key=lambda r: float(r.get("memorized_fraction", 0.0)),
        reverse=True,
    )

    forget_rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for index, row in enumerate(selected):
        content_id = str(row["content_id"])
        lyrics: str = row["lyrics"]
        windows = _slide_windows(lyrics, window_size=window_size, stride=stride)
        if not windows:
            skipped.append(content_id)
            continue
        request_id = f"req_{index:04d}_{content_id[:8]}"
        for window_idx, prefix, suffix in windows:
            forget_rows.append(
                {
                    "request_id": request_id,
                    "content_id": content_id,
                    "content_title": row.get("content_title", ""),
                    "window_idx": window_idx,
                    "prefix": prefix,
                    "suffix": suffix,
                    "memorized_fraction": row.get("memorized_fraction"),
                    "rule_name": ARTIFACT_RULE_NAME,
                }
            )

    if skipped:
        logger.warning(
            "Skipped %d/%d songs too short for window_size=%d: %s",
            len(skipped), len(selected), window_size, skipped[:10],
        )
    if not forget_rows:
        raise ValueError(
            f"baseline rule produced zero forget rows "
            f"(window_size={window_size} stride={stride})"
        )
    logger.info(
        "baseline: %d songs -> %d forget windows (window_size=%d, stride=%d)",
        len(selected), len(forget_rows), window_size, stride,
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


def save_curated_sets(
    *,
    output_dir: str | Path,
    forget_rows: list[dict[str, Any]],
    retain_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Path]:
    """Persist forget_set.parquet, retain_set.parquet, metadata.json to output_dir."""
    if not forget_rows:
        raise ValueError("forget_rows is empty — curation produced nothing to unlearn")
    if not retain_rows:
        raise ValueError("retain_rows is empty — retain sampling produced nothing")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    forget_path = out / "forget_set.parquet"
    retain_path = out / "retain_set.parquet"
    metadata_path = out / "metadata.json"

    Dataset.from_list(forget_rows).to_parquet(str(forget_path))
    Dataset.from_list(retain_rows).to_parquet(str(retain_path))
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return {"forget": forget_path, "retain": retain_path, "metadata": metadata_path}
