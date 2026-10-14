"""Corpus footprint for the featured songs, from unlearning-cleanslate/igm-retrievals.

For every mid-training document that shares a span with the lyrics:
  source category (from its shard path), the matched spans (located in the lyrics),
  and a short snippet around the longest match.
Writes public/data/footprint/<content_id>.json
  {text, docs: [{cat, src, spans: [[a, b], ...], snippet, hl: [[a, b], ...], words}]}
where spans are character ranges in the lyrics and hl are ranges inside the snippet.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

HERE = Path(__file__).parent
DATA = HERE.parent / "public/data"
(DATA / "footprint").mkdir(exist_ok=True)
fs = HfFileSystem()
ROOT = "datasets/unlearning-cleanslate/igm-retrievals"
MIN_WORDS = 3  # keep short spans in the file; the figure filters by length (default 5)
SNIP = 220


def category(path: str) -> tuple[str, str]:
    head = path.split("/")[0]
    m = re.match(r"common_crawl-high-quality_\d+_(.+)", head)
    if m:
        return "Web pages", m.group(1).replace("_", " ")
    if "synth" in head:
        return "Synthetic Q&A", head
    if "flan" in head or "tulu" in head or "sft" in head:
        return "Instruction tuning", head
    if "software" in path or "code" in head or "stack" in head or "github" in head:
        return "Code & software", head
    if "math" in head:
        return "Math", head
    if "wiki" in head:
        return "Wikipedia", head
    return "Other", head


def find_all(text: str, sub: str) -> list[tuple[int, int]]:
    out, i = [], text.find(sub)
    while i != -1:
        out.append((i, i + len(sub)))
        i = text.find(sub, i + 1)
    return out


def main() -> None:
    ix = json.loads((DATA / "flame_index.json").read_text())
    dirs = fs.ls(ROOT, detail=False)
    unknown = Counter()
    for cid in ix["featured"]:
        rec = json.loads((DATA / "flame" / f"{cid}.json").read_text())
        text, title = rec["text"], rec["title"]
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        cands = [d for d in dirs if d.split("/")[-1].rsplit("__", 1)[0] == slug]
        if len(cands) != 1:
            print(f"SKIP {title}: {len(cands)} matching dirs {cands[:3]}")
            continue
        with fs.open(cands[0] + "/train.parquet", "rb") as f:
            df = pq.read_table(f).to_pandas()
        docs = []
        for r in df.itertuples():
            spans = []
            for s in r.unique_source_spans:
                if len(s.split()) >= MIN_WORDS:
                    spans += find_all(text, s)
            if not spans:
                continue
            longest = max(r.unique_source_spans, key=len)
            body = r.doc_content
            j = body.find(longest)
            a = max(0, j - SNIP) if j >= 0 else 0
            b = min(len(body), (j + len(longest) + SNIP) if j >= 0 else 2 * SNIP)
            snippet = body[a:b]
            hl = []
            for s in r.unique_source_spans:
                if len(s.split()) >= MIN_WORDS:
                    hl += find_all(snippet, s)
            cat, src = category(r.path)
            if cat == "Other":
                unknown[src] += 1
            docs.append({
                "cat": cat, "src": src, "spans": sorted(set(spans)),
                "words": max(len(s.split()) for s in r.unique_source_spans),
                "snippet": snippet, "pre": a > 0, "post": b < len(body), "hl": sorted(set(hl)),
            })
        out = {"id": cid, "title": title, "creators": rec["creators"], "year": rec["year"], "text": text, "docs": docs}
        (DATA / "footprint" / f"{cid}.json").write_text(json.dumps(out, separators=(",", ":")))
        cats = Counter(d["cat"] for d in docs if d["words"] >= 5)
        print(f"{title}: {len(df)} docs retrieved, {sum(cats.values())} with a >=5-word span  {dict(cats)}")
    if unknown:
        print("uncategorized sources:", unknown.most_common(15))


if __name__ == "__main__":
    main()
