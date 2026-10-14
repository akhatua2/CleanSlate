"""Per-song coverage, the paper's Figure 2 measure: the share of a song's (non-space)
characters covered by spans of >= n words that also occur in the Dolmino mid-training mix
(unlearning-cleanslate/igm-retrievals, Infini-gram exact matches), for n = 5, 7, 10.

Writes cache/coverage.parquet (content_id, title, year, cov5, cov7, cov10) and adds
`coverage` (per-year median and quartiles of cov5) to public/data/time.json.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

HERE = Path(__file__).parent
CACHE = HERE / "cache"
ROOT = "datasets/unlearning-cleanslate/igm-retrievals"
fs = HfFileSystem()


def slug(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (t or "").lower()).strip("_")


def spans_for(d: str) -> tuple[str, set[str]] | None:
    for attempt in range(6):
        try:
            with fs.open(d + "/train.parquet", "rb") as f:
                t = pq.read_table(f, columns=["unique_source_spans"]).to_pandas()
            return d.split("/")[-1], {s for xs in t.unique_source_spans for s in xs}
        except Exception as e:  # network blips: back off and retry
            err = e
            time.sleep(2 ** attempt)
    print("FAIL", d, type(err).__name__, flush=True)
    return None


def coverage(text: str, spans: set[str], n: int) -> float:
    lo = text.lower()
    hit = np.zeros(len(text), dtype=bool)
    for s in spans:
        if len(s.split()) < n:
            continue
        sl = s.lower()
        i = lo.find(sl)
        while i != -1:
            hit[i:i + len(sl)] = True
            i = lo.find(sl, i + 1)
    keep = np.array([not c.isspace() for c in text])
    return float(hit[keep].mean()) if keep.any() else 0.0


def main() -> None:
    dirs = [d for d in fs.ls(ROOT, detail=False) if "__" in d.split("/")[-1]]
    print(len(dirs), "song dirs", flush=True)
    got = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        for k, r in enumerate(ex.map(spans_for, dirs)):
            if r:
                got[r[0]] = r[1]
            if k % 500 == 0:
                print(k, flush=True)
    if len(got) < 0.99 * len(dirs):
        raise SystemExit(f"only {len(got)} of {len(dirs)} read; not writing partial results")

    w = pq.read_table(CACHE / "works.parquet").to_pandas()
    w = w[w.kind == "song"].copy()
    by_slug = {}
    for d in got:
        by_slug.setdefault(d.rsplit("__", 1)[0], d)
    rows = []
    for r in w.itertuples():
        d = by_slug.get(slug(r.title))
        if d is None or not r.text or r.year != r.year:
            continue
        sp = got[d]
        rows.append({"content_id": r.content_id, "title": r.title, "year": int(r.year),
                     "cov5": coverage(r.text, sp, 5), "cov7": coverage(r.text, sp, 7), "cov10": coverage(r.text, sp, 10)})
    df = pd.DataFrame(rows)
    df.to_parquet(CACHE / "coverage.parquet")
    print(f"{len(df)} songs; mean cov5 {df.cov5.mean():.3f}, median {df.cov5.median():.3f}")

    g = df.groupby("year")
    yr = pd.DataFrame({"med": g.cov5.median(), "q1": g.cov5.quantile(0.25), "q3": g.cov5.quantile(0.75),
                       "mean": g.cov5.mean(), "n": g.size()}).reset_index()
    print(yr.iloc[::5].to_string())
    p = HERE.parent / "public/data/time.json"
    d = json.loads(p.read_text())
    d["coverage"] = [{"y": int(r.year), "med": round(r.med, 4), "q1": round(r.q1, 4), "q3": round(r.q3, 4),
                      "mean": round(r.mean, 4), "n": int(r.n)} for r in yr.itertuples()]
    d["coverage_mean"] = round(float(df.cov5.mean()), 4)
    p.write_text(json.dumps(d, separators=(",", ":")))


if __name__ == "__main__":
    main()
