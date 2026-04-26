"""Shared infini-gram retrieval helpers for the dataset_search notebook + batch script.

Document retrieval delegates to `curation.retrieval.backend_infini_gram.retrieve_docs`
to avoid drift with the curation pipeline; this module just adds word-ngram
matching, per-N occurrence accounting, and parallel span->docs aggregation.
"""
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm

from curation.retrieval.backend_infini_gram import retrieve_docs as _curation_retrieve_docs
from curation.retrieval.schema import Span

WORD_RE = re.compile(r"\S+")


@dataclass
class WordNgramMatch:
    word_start: int
    word_end: int
    char_start: int
    char_end: int
    count: int


def word_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in WORD_RE.finditer(text)]


def compute_matched_word_ngrams(
    text: str,
    engine: any,
    min_n: int = 3,
    min_count: int = 0,
    workers: int = 1,
    progress: bool = False,
    progress_desc: str = "ngrams",
) -> list[WordNgramMatch]:
    """Find every contiguous word-ngram of length >= min_n with corpus count > min_count.

    `workers > 1` dispatches start indices `i` across a ThreadPoolExecutor.
    `engine.count` releases the GIL inside the C extension; the engine has an
    internal mutex so effective parallelism caps around 4-8 threads, but going
    above serial still gives a real speedup. Default 1 preserves byte-identical
    behavior for existing callers (notebook).
    """
    spans = word_spans(text)
    W = len(spans)
    if W < min_n:
        return []
    # Shared memo: dict assignments are GIL-protected and engine.count is a
    # deterministic function of the string, so concurrent threads racing to
    # populate the same key is correct (worst case: occasional duplicate work).
    memo: dict[str, int] = {}

    def get_count(q: str) -> int:
        c = memo.get(q)
        if c is None:
            c = engine.count(q)["count"]
            memo[q] = c
        return c

    def extend_from_i(i: int) -> list[WordNgramMatch]:
        out: list[WordNgramMatch] = []
        cs = spans[i][0]
        ce = spans[i + min_n - 1][1]
        cnt = get_count(text[cs:ce])
        if cnt <= min_count:
            return out
        out.append(WordNgramMatch(i, i + min_n, cs, ce, cnt))
        n = min_n + 1
        while i + n <= W:
            ce_n = spans[i + n - 1][1]
            cnt = get_count(text[cs:ce_n])
            if cnt <= min_count:
                break
            out.append(WordNgramMatch(i, i + n, cs, ce_n, cnt))
            n += 1
        return out

    starts = range(W - min_n + 1)
    if workers <= 1:
        it = tqdm(starts, desc=progress_desc) if progress else starts
        return [m for i in it for m in extend_from_i(i)]

    matches: list[WordNgramMatch] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results_it = pool.map(extend_from_i, starts)
        if progress:
            results_it = tqdm(results_it, total=len(starts), desc=progress_desc)
        for chunk in results_it:
            matches.extend(chunk)
    matches.sort(key=lambda m: (m.word_start, m.word_end))
    return matches


def retrieve_docs(query: str, engine: any, max_docs: int = 5) -> list[dict]:
    """Thin wrapper over curation.retrieval.backend_infini_gram.retrieve_docs.

    The canonical function expects a (text, Span) pair and slices `text[span.start:span.end]`
    as the query. We pass the raw query as both, with a 0..len(query) span.
    """
    return _curation_retrieve_docs(query, Span(0, len(query), 0), engine, max_docs=max_docs)


def per_word_occ_from_matches(text: str, matches: list[WordNgramMatch], n: int) -> np.ndarray:
    W = len(word_spans(text))
    occ = np.zeros(W, dtype=np.int64)
    for m in matches:
        if m.word_end - m.word_start != n:
            continue
        occ[m.word_start:m.word_end] += m.count
    return occ


def top_spans_by_length(matches: list[WordNgramMatch], k: int) -> list[WordNgramMatch]:
    return sorted(
        matches,
        key=lambda m: (m.word_end - m.word_start, m.count),
        reverse=True,
    )[:k]


def unique_docs_for_spans(
    text: str,
    spans: list[WordNgramMatch],
    engine: any,
    docs_per_span: int,
    workers: int = 8,
    progress: bool = True,
    progress_desc: str = "retrieve_unique_docs_spans",
    progress_position: int = 0,
    progress_leave: bool = True,
) -> list[dict]:
    """Fetch top `docs_per_span` docs per span in parallel; dedupe by doc_ix.

    Each entry: {doc_ix, path, doc_content, unique_source_spans: set[str]}.

    Distinct query strings are deduplicated before hitting the engine — many
    `WordNgramMatch` rows share the same query text (overlapping ngrams,
    repeated phrases). Retrieval is then parallelized over unique queries via
    a small ThreadPoolExecutor: the infini-gram engine has an internal C-level
    mutex, so very high thread counts (≥ ~32) just queue on the lock; a small
    pool (default 8) gives the best throughput.
    """
    span_texts = [text[m.char_start:m.char_end] for m in spans]
    unique_queries = list(set(span_texts))

    def fetch(q: str) -> tuple[str, list[dict]]:
        return q, retrieve_docs(q, engine, max_docs=docs_per_span)

    query_to_docs: dict[str, list[dict]] = {}
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            it = pool.map(fetch, unique_queries)
            if progress:
                it = tqdm(
                    it, total=len(unique_queries), desc=progress_desc,
                    position=progress_position, leave=progress_leave,
                )
            for q, docs in it:
                query_to_docs[q] = docs
    else:
        it = unique_queries
        if progress:
            it = tqdm(
                it, total=len(unique_queries), desc=progress_desc,
                position=progress_position, leave=progress_leave,
            )
        for q in it:
            query_to_docs[q] = retrieve_docs(q, engine, max_docs=docs_per_span)

    seen: dict[int, dict] = {}
    for q in span_texts:
        for doc in query_to_docs.get(q, []):
            ix = doc.get("doc_ix")
            if ix is None:
                continue
            if ix not in seen:
                seen[ix] = {
                    "doc_ix": ix,
                    "path": doc.get("path", ""),
                    "doc_content": doc.get("text", ""),
                    "unique_source_spans": set(),
                }
            seen[ix]["unique_source_spans"].add(q)
    return list(seen.values())


def plot_word_occ_strips(
    text: str,
    occ_counts_by_n: dict[int, np.ndarray],
    title: str = "",
):
    """One viridis log-scale strip per N (rows). x = word index, color = Σ
    occurrence count over length-N matches covering each word; white = 0.

    Implemented as a single `imshow` over the (n_N x n_words) matrix — fast
    even for >100k word references where the per-cell Rectangle approach
    stalls inside matplotlib's data-limit recomputation.
    """
    ns = sorted(occ_counts_by_n.keys())
    n_words = (
        len(word_spans(text)) if text else int(occ_counts_by_n[ns[0]].shape[0])
    )
    mat = np.stack([occ_counts_by_n[n].astype(float) for n in ns])
    masked = np.where(mat == 0, np.nan, mat)
    finite = mat[mat > 0]
    vmax = int(finite.max()) if finite.size else 2
    norm = LogNorm(vmin=1, vmax=max(vmax, 2))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(14, 0.55 * len(ns) + 1.5))
    im = ax.imshow(
        masked, aspect="auto", cmap=cmap, norm=norm,
        extent=(0, n_words, len(ns), 0), interpolation="nearest",
    )
    ax.set_yticks([i + 0.5 for i in range(len(ns))])
    ax.set_yticklabels([f"N={n}" for n in ns])
    ax.set_xlabel("word index in reference")
    ax.set_title(f"Per-word Σ occurrence count by N — {title}")
    fig.colorbar(
        im, ax=ax, label="Σ occurrence count (log; white = 0)",
        fraction=0.02, pad=0.01,
    )
    fig.tight_layout()
    return fig
