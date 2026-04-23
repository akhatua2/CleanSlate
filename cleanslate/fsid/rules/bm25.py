"""BM25 rule: forget windows taken from the BM25-retrieved Dolma doc.

For each pool `content_id` we pull all its BM25 hits, rank by score, and slide
baseline-style (prefix, suffix) windows over `retrieved_text[unit_char_start:
unit_char_end]` — i.e. the scored span of the Dolma document itself, not the
reference lyrics. Each forget row is still attributed to the song's
`content_id` so eval scoring stays song-anchored.

Default source: `unlearning-cleanslate/verif-bm25-dolma-10B`.
"""

import logging
from collections import defaultdict
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]

from cleanslate.fsid.rules._retain import load_wikitext_retain_rows
from cleanslate.fsid.rules.baseline import DEFAULT_STRIDE, DEFAULT_WINDOW_SIZE, _slide_windows

logger = logging.getLogger(__name__)

RULE_NAME = "bm25"
ARTIFACT_RULE_NAME = "bm25_dolma"
DEFAULT_BM25_DATASET = "unlearning-cleanslate/verif-bm25-dolma-10B"
DEFAULT_MAX_HITS_PER_SONG = 32


def _index_bm25_hits(
    dataset_id: str,
    *,
    pool_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Group BM25 hits by content_id (pool-restricted, sorted by score desc)."""
    logger.info("Loading BM25 retrieval dataset %s", dataset_id)
    ds = load_dataset(dataset_id, split="train")
    hits_by_content: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ds:
        cid = str(row["content_id"])
        if cid not in pool_ids:
            continue
        hits_by_content[cid].append(
            {
                "retrieved_text": str(row["retrieved_text"]),
                "unit_char_start": int(row["unit_char_start"]),
                "unit_char_end": int(row["unit_char_end"]),
                "score": float(row.get("score", 0.0)),
            }
        )
    for hits in hits_by_content.values():
        hits.sort(key=lambda h: h["score"], reverse=True)
    logger.info("BM25: %d/%d pool songs have hits", len(hits_by_content), len(pool_ids))
    return hits_by_content


def curate_bm25_sets(
    unlearn_song_records: list[dict[str, Any]],
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    base_dataset: str | None = None,
    max_hits_per_song: int = DEFAULT_MAX_HITS_PER_SONG,
    seed: int = 42,
    retain_dataset: str = "wikitext",
    retain_config: str = "wikitext-103-raw-v1",
    retain_split: str = "validation",
    retain_samples: int | None = None,
    **_: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (forget_rows, retain_rows); baseline-style windows over BM25-retrieved Dolma spans."""
    if not unlearn_song_records:
        raise ValueError("unlearn_song_records is empty")
    if max_hits_per_song < 1:
        raise ValueError("max_hits_per_song must be >= 1")

    pool_ids = {str(r["content_id"]) for r in unlearn_song_records}
    id_to_meta = {str(r["content_id"]): r for r in unlearn_song_records}
    dataset_id = base_dataset or DEFAULT_BM25_DATASET
    hits_by_content = _index_bm25_hits(dataset_id, pool_ids=pool_ids)

    ordered_ids = sorted(
        hits_by_content.keys(),
        key=lambda cid: float(id_to_meta[cid].get("memorized_fraction", 0.0)),
        reverse=True,
    )

    forget_rows: list[dict[str, Any]] = []
    skipped_empty_span: list[str] = []
    for index, cid in enumerate(ordered_ids):
        pool_row = id_to_meta[cid]
        request_id = f"req_{index:04d}_{cid[:8]}"
        seen: set[tuple[str, str]] = set()
        emitted = 0
        for hit in hits_by_content[cid]:
            if emitted >= max_hits_per_song:
                break
            doc = hit["retrieved_text"]
            start = max(0, hit["unit_char_start"])
            end = min(len(doc), hit["unit_char_end"])
            if end <= start:
                continue
            span = doc[start:end]
            for window_idx, prefix, suffix in _slide_windows(
                span, window_size=window_size, stride=stride,
            ):
                if emitted >= max_hits_per_song:
                    break
                key = (prefix, suffix)
                if key in seen:
                    continue
                seen.add(key)
                forget_rows.append(
                    {
                        "request_id": request_id,
                        "content_id": cid,
                        "content_title": pool_row.get("content_title", ""),
                        "window_idx": window_idx,
                        "prefix": prefix,
                        "suffix": suffix,
                        "memorized_fraction": pool_row.get("memorized_fraction"),
                        "rule_name": ARTIFACT_RULE_NAME,
                    }
                )
                emitted += 1
        if emitted == 0:
            skipped_empty_span.append(cid)

    songs_without_hits = len(pool_ids) - len(hits_by_content)
    if songs_without_hits:
        logger.warning(
            "Skipped %d/%d pool songs with no BM25 hit in %s",
            songs_without_hits, len(pool_ids), dataset_id,
        )
    if skipped_empty_span:
        logger.warning(
            "Skipped %d songs whose hit spans produced no windows (window_size=%d)",
            len(skipped_empty_span), window_size,
        )
    if not forget_rows:
        raise ValueError(
            f"bm25 produced zero forget rows from {dataset_id} "
            f"(window_size={window_size} stride={stride})"
        )
    logger.info(
        "bm25: %d songs -> %d forget windows (dataset=%s, window_size=%d, stride=%d, cap=%d)",
        len(ordered_ids) - len(skipped_empty_span),
        len(forget_rows), dataset_id, window_size, stride, max_hits_per_song,
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
