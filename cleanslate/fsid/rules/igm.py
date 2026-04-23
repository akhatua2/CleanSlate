"""Infini-gram-mini rule: forget windows centered on the matched n-gram inside the Dolma doc.

IGM's `unit_char_start/end` indexes into the reference lyric (the query text),
so it does not localize the match inside the retrieved corpus doc. The matched
n-gram itself lives in `metadata.span_text` (JSON string). For each pool
`content_id` we rank hits by score, locate `span_text` inside `retrieved_text`,
and emit one (prefix, suffix) window of `window_size` chars per side centered
on the midpoint of that occurrence. Forget rows stay attributed to the song's
`content_id` so eval scoring remains song-anchored.

Default source: `unlearning-cleanslate/infini_gram_mini_cleanslate_dolma3_10B`.
"""

import json
import logging
from collections import defaultdict
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]

from cleanslate.fsid.rules._retain import load_wikitext_retain_rows

logger = logging.getLogger(__name__)

RULE_NAME = "igm"
ARTIFACT_RULE_NAME = "igm_dolma"
DEFAULT_IGM_DATASET = "unlearning-cleanslate/infini_gram_mini_cleanslate_dolma3_10B"
DEFAULT_MAX_HITS_PER_SONG = 32


def _parse_span_text(raw_metadata: Any) -> str:
    if not raw_metadata:
        return ""
    try:
        obj = json.loads(str(raw_metadata))
    except (json.JSONDecodeError, TypeError):
        return ""
    value = obj.get("span_text", "") if isinstance(obj, dict) else ""
    return str(value) if value else ""


def _index_igm_hits(
    dataset_id: str,
    *,
    pool_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Group IGM hits by content_id (pool-restricted, sorted by score desc)."""
    logger.info("Loading IGM retrieval dataset %s", dataset_id)
    ds = load_dataset(dataset_id, split="train")
    hits_by_content: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ds:
        cid = str(row["content_id"])
        if cid not in pool_ids:
            continue
        span_text = _parse_span_text(row.get("metadata"))
        if not span_text:
            continue
        hits_by_content[cid].append(
            {
                "retrieved_text": str(row["retrieved_text"]),
                "span_text": span_text,
                "score": float(row.get("score", 0.0)),
            }
        )
    for hits in hits_by_content.values():
        hits.sort(key=lambda h: h["score"], reverse=True)
    logger.info("IGM: %d/%d pool songs have hits", len(hits_by_content), len(pool_ids))
    return hits_by_content


def curate_igm_sets(
    unlearn_song_records: list[dict[str, Any]],
    *,
    window_size: int = 100,
    base_dataset: str | None = None,
    max_hits_per_song: int = DEFAULT_MAX_HITS_PER_SONG,
    seed: int = 42,
    retain_dataset: str = "wikitext",
    retain_config: str = "wikitext-103-raw-v1",
    retain_split: str = "validation",
    retain_samples: int | None = None,
    **_: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (forget_rows, retain_rows); windows centered on n-gram matches inside Dolma docs."""
    if not unlearn_song_records:
        raise ValueError("unlearn_song_records is empty")
    if max_hits_per_song < 1:
        raise ValueError("max_hits_per_song must be >= 1")

    pool_ids = {str(r["content_id"]) for r in unlearn_song_records}
    id_to_meta = {str(r["content_id"]): r for r in unlearn_song_records}
    dataset_id = base_dataset or DEFAULT_IGM_DATASET
    hits_by_content = _index_igm_hits(dataset_id, pool_ids=pool_ids)

    ordered_ids = sorted(
        hits_by_content.keys(),
        key=lambda cid: float(id_to_meta[cid].get("memorized_fraction", 0.0)),
        reverse=True,
    )

    forget_rows: list[dict[str, Any]] = []
    skipped_no_window: list[str] = []
    for index, cid in enumerate(ordered_ids):
        pool_row = id_to_meta[cid]
        request_id = f"req_{index:04d}_{cid[:8]}"
        seen: set[tuple[str, str]] = set()
        emitted = 0
        for hit in hits_by_content[cid]:
            if emitted >= max_hits_per_song:
                break
            doc = hit["retrieved_text"]
            span_text = hit["span_text"]
            pos = doc.find(span_text)
            if pos < 0:
                continue
            mid = pos + len(span_text) // 2
            p_start = mid - window_size
            s_end = mid + window_size
            if p_start < 0 or s_end > len(doc):
                continue
            prefix, suffix = doc[p_start:mid], doc[mid:s_end]
            key = (prefix, suffix)
            if key in seen:
                continue
            seen.add(key)
            forget_rows.append(
                {
                    "request_id": request_id,
                    "content_id": cid,
                    "content_title": pool_row.get("content_title", ""),
                    "window_idx": emitted,
                    "prefix": prefix,
                    "suffix": suffix,
                    "memorized_fraction": pool_row.get("memorized_fraction"),
                    "rule_name": ARTIFACT_RULE_NAME,
                }
            )
            emitted += 1
        if emitted == 0:
            skipped_no_window.append(cid)

    songs_without_hits = len(pool_ids) - len(hits_by_content)
    if songs_without_hits:
        logger.warning(
            "Skipped %d/%d pool songs with no IGM hits in %s",
            songs_without_hits, len(pool_ids), dataset_id,
        )
    if skipped_no_window:
        logger.warning(
            "Skipped %d songs whose hits could not be centered within the Dolma doc (window_size=%d)",
            len(skipped_no_window), window_size,
        )
    if not forget_rows:
        raise ValueError(
            f"igm rule produced zero forget rows from {dataset_id} "
            f"(window_size={window_size}, cap={max_hits_per_song})"
        )
    logger.info(
        "igm: %d songs -> %d forget windows (dataset=%s, window_size=%d, cap=%d)",
        len(ordered_ids) - len(skipped_no_window),
        len(forget_rows), dataset_id, window_size, max_hits_per_song,
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
