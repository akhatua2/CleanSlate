"""FM-index exact n-gram retrieval via infini-gram-mini.

OLMoTrace-style: for each position, binary-search for longest verbatim
substring in the corpus, suppress non-maximal spans, compute coverage.
"""

import json
import os
import resource
import sys
from pathlib import Path

from curation.retrieval.schema import Cfg, Span, VENDOR_ROOT

ENGINE_DIR = VENDOR_ROOT / "engine"
INDEXING_DIR = VENDOR_ROOT / "src"


# -- Engine loading ----------------------------------------------------------

def load_engine(index_dir: Path, load_to_ram: bool = False):
    engine_str = str(ENGINE_DIR)
    if engine_str not in sys.path:
        sys.path.insert(0, engine_str)
    from src.engine import InfiniGramMiniEngine  # type: ignore[import-untyped]
    return InfiniGramMiniEngine(
        index_dirs=[str(index_dir)], load_to_ram=load_to_ram, get_metadata=True,
    )


# -- Index building ----------------------------------------------------------

def build_fm_index(cfg: Cfg) -> None:
    """Build FM-index from raw Dolma shards. Idempotent: skips if data.fm9 exists."""
    idx = cfg.index_dir
    if (idx / "data.fm9").exists():
        print(f"FM-index exists at {idx}, skipping.")
        return

    shards = cfg.raw_shards()
    if not shards:
        raise FileNotFoundError(f"No .jsonl.zst in {cfg.raw_data}")

    idx.mkdir(parents=True, exist_ok=True)
    scratch = cfg.index_dir.parent / "temp_index"
    scratch.mkdir(parents=True, exist_ok=True)

    print(f"Building FM-index: {len(shards)} shards → {idx}")

    indexing_str = str(INDEXING_DIR)
    if indexing_str not in sys.path:
        sys.path.insert(0, indexing_str)

    import argparse as _ap
    from indexing import build_sa_bwt, prepare  # type: ignore[import-untyped]

    args = _ap.Namespace(
        data_dir=str(cfg.raw_data),
        save_dir=str(idx),
        temp_dir=str(scratch),
        doc_sep=b"\xff",
        batch_size=65536,
        cpus=cfg.workers,
        mem=cfg.index_mem_gb,
        ulimit=1048576,
    )

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(args.ulimit, hard), hard))

    prepare(args)
    build_sa_bwt(args, mode="data")
    build_sa_bwt(args, mode="meta")

    prev_cwd = os.getcwd()
    os.chdir(str(INDEXING_DIR))
    binary = "./cpp_indexing_local" if (INDEXING_DIR / "cpp_indexing_local").exists() else "./cpp_indexing"
    ret = os.system(f"{binary} {idx} 2>&1")
    os.chdir(prev_cwd)
    if ret != 0:
        raise RuntimeError(f"{binary} failed with exit code {ret}")

    print(f"FM-index complete: {idx}")


# -- Span computation -------------------------------------------------------

def compute_verbatim_spans(text: str, engine, min_chars: int = 20) -> list[Span]:
    """Binary search for maximal verbatim spans of text in the corpus."""
    n = len(text)
    raw: list[Span] = []

    for i in range(n):
        if n - i < min_chars:
            break
        # Quick reject: does minimum-length substring exist?
        if engine.count(text[i : i + min_chars])["count"] == 0:
            continue

        best = min_chars
        lo, hi = min_chars + 1, n - i
        while lo <= hi:
            mid = (lo + hi) // 2
            if engine.count(text[i : i + mid])["count"] > 0:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        cnt = engine.count(text[i : i + best])["count"]
        raw.append(Span(start=i, end=i + best, count=cnt))

    return _suppress_non_maximal(raw)


def _suppress_non_maximal(spans: list[Span]) -> list[Span]:
    if not spans:
        return []
    spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
    result: list[Span] = []
    max_end = -1
    for s in spans:
        if s.end > max_end:
            result.append(s)
            max_end = s.end
    return result


def compute_coverage(spans: list[Span], text_len: int) -> float:
    """Fraction of text covered by spans. Interval merge, not char-set."""
    if not spans or text_len == 0:
        return 0.0
    intervals = sorted((s.start, s.end) for s in spans)
    merged: list[tuple[int, int]] = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return sum(e - s for s, e in merged) / text_len


# -- Document retrieval ------------------------------------------------------

def retrieve_docs(text: str, span: Span, engine, max_docs: int = 5) -> list[dict]:
    """Retrieve Dolma documents containing the span."""
    query = text[span.start : span.end]
    segments = engine.find(query).get("segment_by_shard", [])
    docs: list[dict] = []
    for shard_idx, seg in enumerate(segments):
        if not seg or len(seg) != 2:
            continue
        for rank in range(seg[0], min(seg[0] + max_docs - len(docs), seg[1])):
            r = engine.get_doc_by_rank(s=shard_idx, rank=rank, needle_len=len(query), max_ctx_len=300)
            if "error" not in r:
                meta = r.get("metadata", {})
                if isinstance(meta, str):
                    meta = json.loads(meta) if meta else {}
                docs.append({"doc_ix": r.get("doc_ix"), "text": r.get("text", ""),
                             "path": meta.get("path", "") if isinstance(meta, dict) else ""})
            if len(docs) >= max_docs:
                return docs
    return docs


# -- Search one item ---------------------------------------------------------

def search_one_item(
    content_id: str, title: str, text: str,
    engine, cfg: Cfg,
) -> list[dict]:
    """Search one query item. Returns list of hit dicts matching MATCH_SCHEMA."""
    spans = compute_verbatim_spans(text, engine, min_chars=cfg.min_span_chars)
    if not spans:
        return []

    coverage = compute_coverage(spans, len(text))
    ranked = sorted(spans, key=lambda s: s.end - s.start, reverse=True)[: cfg.top_k]
    hits: list[dict] = []

    for rank, span in enumerate(ranked, 1):
        span_text = text[span.start : span.end]
        docs = retrieve_docs(text, span, engine, max_docs=3)
        first = docs[0] if docs else {}

        hits.append({
            "content_id": content_id,
            "content_title": title,
            "reference_target": text,
            "retrieval_method": "infini_gram_mini",
            "corpus_name": f"dolma3_{cfg.variant}",
            "doc_id": str(first.get("doc_ix", "")),
            "doc_source": first.get("path", ""),
            "doc_category": first.get("path", "").split("/")[0] if "/" in first.get("path", "") else "",
            "unit_id": f"{content_id}_span_{span.start}_{span.end}",
            "unit_char_start": span.start,
            "unit_char_end": span.end,
            "rank": rank,
            "score": float(span.end - span.start),
            "retrieved_text": first.get("text", ""),
            "metadata": json.dumps({
                "span_text": span_text,
                "corpus_count": span.count,
                "coverage": round(coverage, 4),
                "total_spans": len(spans),
                "num_docs": len(docs),
            }),
        })

    return hits
