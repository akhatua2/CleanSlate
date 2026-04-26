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

from cleanslate.fsid.rules._alloc import allocate_per_doc_quota
from cleanslate.fsid.rules._retain import load_wikitext_retain_rows
from cleanslate.fsid.rules.baseline import DEFAULT_STRIDE, DEFAULT_WINDOW_SIZE, _slide_windows

logger = logging.getLogger(__name__)

RULE_NAME = "bm25"
ARTIFACT_RULE_NAME = "bm25_dolma"
DEFAULT_BM25_DATASET = "unlearning-cleanslate/verif-bm25-dolma-10B"
DEFAULT_MAX_HITS_PER_SONG = 128
DEFAULT_MAX_WINDOWS_PER_HIT = 4


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
    max_windows_per_hit: int = DEFAULT_MAX_WINDOWS_PER_HIT,
    cross_hit_dedup: bool = True,
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
    if max_windows_per_hit < 1:
        raise ValueError("max_windows_per_hit must be >= 1")

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
    docs_used_per_song: list[int] = []
    for index, cid in enumerate(ordered_ids):
        pool_row = id_to_meta[cid]
        request_id = f"req_{index:04d}_{cid[:8]}"
        song_seen: set[tuple[str, str]] = set()

        hit_windows: list[list[tuple[int, str, str]]] = []
        for hit in hits_by_content[cid]:
            doc = hit["retrieved_text"]
            start = max(0, hit["unit_char_start"])
            end = min(len(doc), hit["unit_char_end"])
            if end <= start:
                continue
            ws = _slide_windows(doc[start:end], window_size=window_size, stride=stride)
            if ws:
                hit_windows.append(ws)
        if not hit_windows:
            skipped_empty_span.append(cid)
            continue

        def _emit(window_idx: int, prefix: str, suffix: str) -> bool:
            key = (prefix, suffix)
            if cross_hit_dedup and key in song_seen:
                return False
            song_seen.add(key)
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
            return True

        available = [len(ws) for ws in hit_windows]
        quotas = allocate_per_doc_quota(
            available, cap=max_hits_per_song, per_doc_floor=max_windows_per_hit,
        )
        docs_contributing = 0
        for ws, quota in zip(hit_windows, quotas, strict=True):
            if quota <= 0:
                continue
            if quota >= len(ws):
                take = ws
            else:
                step = len(ws) / quota
                take = [ws[int(j * step)] for j in range(quota)]
            emitted_from_doc = 0
            for w in take:
                if _emit(*w):
                    emitted_from_doc += 1
            if emitted_from_doc > 0:
                docs_contributing += 1
        docs_used_per_song.append(docs_contributing)

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
    avg_docs = sum(docs_used_per_song) / len(docs_used_per_song) if docs_used_per_song else 0.0
    logger.info(
        "bm25: %d songs -> %d forget windows "
        "(dataset=%s, window_size=%d, stride=%d, cap=%d, per_hit=%d, avg_docs/song=%.1f)",
        len(ordered_ids) - len(skipped_empty_span),
        len(forget_rows), dataset_id, window_size, stride,
        max_hits_per_song, max_windows_per_hit, avg_docs,
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
