"""Find real Dolmino mid-training documents that contain the paper's post-cutoff phrases,
by scanning every document already retrieved in unlearning-cleanslate/igm-retrievals.
Writes cache/phrase_docs.json: phrase -> up to 5 {src, snippet, at}.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from build_time import category

ROOT = "datasets/unlearning-cleanslate/igm-retrievals"
OUT = Path(__file__).parent / "cache/phrase_docs.json"
PHRASES = [
    "But it was good to see you",
    "the old life that we lived so fondly together",
    "I could go on and on",
    "the land, the sea, the sky",
    "the love of her life",
]
fs = HfFileSystem()
pats = {p: re.compile(re.escape(p), re.I) for p in PHRASES}


def scan(d: str) -> list[tuple[str, dict]]:
    for attempt in range(6):
        try:
            with fs.open(d + "/train.parquet", "rb") as f:
                t = pq.read_table(f, columns=["doc_ix", "path", "doc_content"]).to_pandas()
            break
        except Exception:
            time.sleep(2 ** attempt)
    else:
        return []
    out = []
    for r in t.itertuples():
        for p, rx in pats.items():
            m = rx.search(r.doc_content)
            if m:
                a, b = max(0, m.start() - 260), min(len(r.doc_content), m.end() + 260)
                out.append((p, {"ix": int(r.doc_ix), "src": category(r.path), "snippet": r.doc_content[a:b],
                                "at": [m.start() - a, m.end() - a], "pre": a > 0, "post": b < len(r.doc_content)}))
    return out


def main() -> None:
    dirs = [d for d in fs.ls(ROOT, detail=False) if "__" in d.split("/")[-1]]
    found: dict[str, dict] = {p: {} for p in PHRASES}
    with ThreadPoolExecutor(max_workers=12) as ex:
        for k, hits in enumerate(ex.map(scan, dirs)):
            for p, h in hits:
                found[p].setdefault(h["ix"], h)
            if k % 500 == 0:
                print(k, {p[:18]: len(v) for p, v in found.items()}, flush=True)
    OUT.write_text(json.dumps({p: list(v.values())[:40] for p, v in found.items()}))
    print({p: len(v) for p, v in found.items()})


if __name__ == "__main__":
    main()
