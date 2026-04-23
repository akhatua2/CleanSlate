"""Infini-gram-mini rule: forget windows centered on the matched n-gram inside the Dolma doc.

IGM's `unit_char_start/end` indexes into the reference lyric (the query text),
so it does not localize the match inside the retrieved corpus doc. The matched
n-gram itself lives in `metadata.span_text` (JSON string). For each pool
`content_id` we group IGM hits by `doc_id` (one group per unique Dolma doc),
compute a per-doc quota via `allocate_per_doc_quota`, and emit one (prefix,
suffix) window of `window_size` chars per side centered on each chosen hit's
span. Forget rows stay attributed to the song's `content_id` so eval scoring
remains song-anchored.

Default source: `unlearning-cleanslate/infini_gram_mini_cleanslate_dolma3_10B`.
"""

import json
import logging
from collections import defaultdict
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]

from cleanslate.fsid.rules._alloc import allocate_per_doc_quota
from cleanslate.fsid.rules._retain import load_wikitext_retain_rows
from cleanslate.fsid.rules.baseline import DEFAULT_STRIDE

logger = logging.getLogger(__name__)

RULE_NAME = "igm"
ARTIFACT_RULE_NAME = "igm_dolma"
DEFAULT_IGM_DATASET = "unlearning-cleanslate/infini_gram_mini_cleanslate_dolma3_10B"
DEFAULT_MAX_HITS_PER_SONG = 128
DEFAULT_MAX_WINDOWS_PER_HIT = 4


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
                "doc_id": str(row.get("doc_id", "") or ""),
                "retrieved_text": str(row["retrieved_text"]),
                "span_text": span_text,
                "score": float(row.get("score", 0.0)),
            }
        )
    for hits in hits_by_content.values():
        hits.sort(key=lambda h: h["score"], reverse=True)
    logger.info("IGM: %d/%d pool songs have hits", len(hits_by_content), len(pool_ids))
    return hits_by_content


def _hit_windows(
    doc: str, span_text: str, *, window_size: int, stride: int,
) -> list[tuple[str, str]]:
    """All 2*window_size-char sliding windows whose frame contains the span midpoint.

    Windows are emitted at stride-spaced positions; every returned window has
    `span_text`'s midpoint inside `[start, start + 2*window_size)`, so the
    matched n-gram is always touched but can sit anywhere from the window's
    left edge to its right edge (off-centered by `k*stride` chars).
    """
    pos = doc.find(span_text)
    if pos < 0:
        return []
    full_win = 2 * window_size
    if full_win > len(doc):
        return []
    span_mid = pos + len(span_text) // 2
    min_start = max(0, span_mid - full_win + 1)
    max_start = min(len(doc) - full_win, span_mid)
    if max_start < min_start:
        return []
    return [
        (doc[s : s + window_size], doc[s + window_size : s + full_win])
        for s in range(min_start, max_start + 1, stride)
    ]


def curate_igm_sets(
    unlearn_song_records: list[dict[str, Any]],
    *,
    window_size: int = 100,
    stride: int = DEFAULT_STRIDE,
    base_dataset: str | None = None,
    max_hits_per_song: int = DEFAULT_MAX_HITS_PER_SONG,
    max_windows_per_hit: int = DEFAULT_MAX_WINDOWS_PER_HIT,
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
    if max_windows_per_hit < 1:
        raise ValueError("max_windows_per_hit must be >= 1")

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
    docs_used_per_song: list[int] = []
    for index, cid in enumerate(ordered_ids):
        pool_row = id_to_meta[cid]
        request_id = f"req_{index:04d}_{cid[:8]}"
        seen: set[tuple[str, str]] = set()

        # Group candidate windows by doc_id. For each hit, slide 2*window_size
        # frames across positions whose window contains the span midpoint; union
        # under the doc_id group while deduping within the group.
        doc_windows: dict[str, list[tuple[str, str]]] = {}
        doc_seen: dict[str, set[tuple[str, str]]] = {}
        doc_order: list[str] = []
        for hit in hits_by_content[cid]:
            ws = _hit_windows(
                hit["retrieved_text"], hit["span_text"],
                window_size=window_size, stride=stride,
            )
            if not ws:
                continue
            doc_id = hit["doc_id"] or f"__anonymous_{len(doc_order)}"
            if doc_id not in doc_windows:
                doc_order.append(doc_id)
                doc_windows[doc_id] = []
                doc_seen[doc_id] = set()
            bucket = doc_windows[doc_id]
            seen_local = doc_seen[doc_id]
            for w in ws:
                if w not in seen_local:
                    seen_local.add(w)
                    bucket.append(w)
        if not doc_order:
            skipped_no_window.append(cid)
            continue

        windows_per_doc = [doc_windows[d] for d in doc_order]
        available = [len(ws) for ws in windows_per_doc]
        quotas = allocate_per_doc_quota(
            available, cap=max_hits_per_song, per_doc_floor=max_windows_per_hit,
        )

        window_idx = 0
        docs_contributing = 0
        for ws, quota in zip(windows_per_doc, quotas, strict=True):
            if quota <= 0:
                continue
            if quota >= len(ws):
                take = ws
            else:
                step = len(ws) / quota
                take = [ws[int(j * step)] for j in range(quota)]
            emitted_from_doc = 0
            for prefix, suffix in take:
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
                window_idx += 1
                emitted_from_doc += 1
            if emitted_from_doc > 0:
                docs_contributing += 1
        docs_used_per_song.append(docs_contributing)

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
    avg_docs = sum(docs_used_per_song) / len(docs_used_per_song) if docs_used_per_song else 0.0
    logger.info(
        "igm: %d songs -> %d forget windows "
        "(dataset=%s, window_size=%d, stride=%d, cap=%d, per_doc=%d, avg_docs/song=%.1f)",
        len(ordered_ids) - len(skipped_no_window),
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
