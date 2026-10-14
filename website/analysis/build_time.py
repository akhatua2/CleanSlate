"""Data for the 'time machine' figure: every song, by release year, with how widely its
text appears in the Dolmino mid-training mix (from unlearning-cleanslate/igm-retrievals).

  docs    documents sharing at least a short span (>= 3 words) with the song
  strong  documents sharing a whole line or more (>= 8 consecutive words)
For the most-matched songs released in 2025 we also keep the lyrics, the borrowed spans
located in them, and a few example documents, so the figure can show inward diffusion.

Writes public/data/time.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

HERE = Path(__file__).parent
CACHE = HERE / "cache"
OUT = HERE.parent / "public/data/time.json"
ROOT = "datasets/unlearning-cleanslate/igm-retrievals"
STRONG = 8
N_DETAIL = 8


def slug(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (t or "").lower()).strip("_")


def category(path: str) -> str:
    head = path.split("/")[0]
    m = re.match(r"common_crawl-high-quality_\d+_(.+)", head)
    if m:
        return "web: " + m.group(1).replace("_", " ")
    if "reddit" in head:
        return "Reddit"
    if "synth" in head or "reasoning" in head or "rcqa" in head:
        return "synthetic Q&A"
    if "flan" in head or "tulu" in head:
        return "instruction data"
    if head.startswith(("stack_edu", "cranecode", "stem-heavy")):
        return "code"
    return head


def main() -> None:
    ds = pd.read_parquet(CACHE / "doc_songs_v2.parquet")
    ds["slug"] = ds.song_dir.str.rsplit("__", n=1).str[0]
    per = ds.groupby("slug").agg(
        docs=("doc_ix", "nunique"),
        strong=("max_words", lambda x: int((x >= STRONG).sum())),
        dir=("song_dir", "first"),
    )
    w = pq.read_table(CACHE / "works.parquet").to_pandas()
    w = w[w.kind == "song"].copy()
    w["slug"] = w.title.map(slug)
    w = w.join(per, on="slug")
    w = w[w.docs.notna() & w.year.notna()]
    songs = [
        {"id": r.content_id, "t": r.title, "a": r.creators, "y": int(r.year), "d": int(r.docs), "s": int(r.strong)}
        for r in w.itertuples()
    ]

    # inward diffusion detail: the 2025 songs that already match the most documents
    fs = HfFileSystem()
    detail = {}
    late = w[w.year >= 2025].sort_values("docs", ascending=False).head(N_DETAIL)
    for r in late.itertuples():
        with fs.open(f"{ROOT}/{r.dir}/train.parquet", "rb") as f:
            t = pq.read_table(f, columns=["path", "unique_source_spans", "doc_content"]).to_pandas()
        text = r.text or ""
        lo = text.lower()
        counts: dict[str, int] = {}
        for spans in t.unique_source_spans:
            for s in set(spans):
                if len(s.split()) >= 4:
                    counts[s.strip()] = counts.get(s.strip(), 0) + 1
        # keep the longest non-overlapping borrowed phrases, located in the lyrics
        marks, taken = [], [False] * len(text)
        for s, c in sorted(counts.items(), key=lambda kv: (-len(kv[0]), -kv[1])):
            i = lo.find(s.lower())
            while i != -1:
                if not any(taken[i:i + len(s)]):
                    marks.append([i, i + len(s), c])
                    for k in range(i, i + len(s)):
                        taken[k] = True
                i = lo.find(s.lower(), i + 1)
        # where each located phrase was found: first document containing it (category + snippet)
        first_doc = {}
        for row in t.itertuples():
            for sp in set(row.unique_source_spans):
                sp = sp.strip()
                if sp in counts and sp not in first_doc:
                    body = row.doc_content
                    j = body.find(sp)
                    a2, b2 = max(0, j - 110), min(len(body), j + len(sp) + 110)
                    first_doc[sp] = {"src": category(row.path), "snippet": body[a2:b2], "at": [j - a2, j - a2 + len(sp)]}
        for m in marks:
            phrase = text[m[0]:m[1]]
            info = next((v for k, v in first_doc.items() if k.lower() == phrase.lower()), None)
            m.append(info)
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
        examples = []
        for phrase, c in top[:3]:
            row = t[t.unique_source_spans.map(lambda xs, p=phrase: p in xs)].head(1)
            if len(row):
                body = row.doc_content.iloc[0]
                j = body.find(phrase)
                a, b = max(0, j - 140), min(len(body), j + len(phrase) + 140)
                examples.append({"phrase": phrase, "count": c, "src": category(row.path.iloc[0]),
                                 "snippet": body[a:b], "at": [j - a, j - a + len(phrase)], "pre": a > 0, "post": b < len(body)})
        detail[r.content_id] = {"text": text, "marks": sorted(marks), "top": top, "examples": examples}
        print(f"{r.title}: {int(r.docs)} docs, {len(marks)} borrowed phrases located")

    # share of songs with a whole-line copy, by year
    by_year = (
        pd.DataFrame(songs).assign(has=lambda d: d.s > 0)
        .groupby("y").agg(n=("has", "size"), share=("has", "mean"), med=("d", "median")).reset_index()
    )
    OUT.write_text(json.dumps({
        "strong_words": STRONG,
        "songs": songs,
        "years": [{"y": int(r.y), "n": int(r.n), "share": round(float(r.share), 4), "med": float(r.med)} for r in by_year.itertuples()],
        "detail": detail,
    }, separators=(",", ":")))
    print(f"{len(songs)} songs, {OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
