"""Which songs share each retrieved document, across all songs in igm-retrievals.

Reads only (song_id, doc_ix) from every song's retrieval file, then for the featured songs
records, per document: full length, characters that are this song, and the other
benchmark songs whose text the same document also contains.

Writes cache/doc_songs.parquet (doc_ix, song_dir) and enriches public/data/footprint/*.json
docs with: ix, len, other (list of other song titles).
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

HERE = Path(__file__).parent
CACHE = HERE / "cache"
DATA = HERE.parent / "public/data"
ROOT = "datasets/unlearning-cleanslate/igm-retrievals"
fs = HfFileSystem()


def read_dir(d: str) -> pd.DataFrame | None:
    if not d.split("/")[-1].count("__"):
        return None  # README and other non-song entries
    for attempt in range(6):
        try:
            with fs.open(d + "/train.parquet", "rb") as f:
                t = pq.read_table(f, columns=["doc_ix", "unique_source_spans"]).to_pandas()
            # strength of the match: longest shared span in words. Short spans ("say goodbye")
            # match stock phrases, not copies of the song.
            t["max_words"] = t.unique_source_spans.map(lambda xs: max((len(x.split()) for x in xs), default=0))
            t["song_dir"] = d.split("/")[-1]
            return t[["doc_ix", "song_dir", "max_words"]]
        except Exception as e:  # network blips: back off and retry, report if it never works
            err = e
            time.sleep(2 ** attempt)
    print("FAIL", d, type(err).__name__, str(err)[:80], flush=True)
    return None


def main() -> None:
    out = CACHE / "doc_songs_v2.parquet"
    if not out.exists():
        dirs = fs.ls(ROOT, detail=False)
        print(len(dirs), "song dirs", flush=True)
        parts = []
        with ThreadPoolExecutor(max_workers=12) as ex:
            for k, t in enumerate(ex.map(read_dir, dirs)):
                if t is not None:
                    parts.append(t)
                if k % 500 == 0:
                    print(k, flush=True)
        n_ok = len({t.song_dir.iloc[0] for t in parts if len(t)})
        print(f"read {len(parts)} song files ({n_ok} non-empty)", flush=True)
        if len(parts) < 0.99 * sum(1 for d in dirs if d.split("/")[-1].count("__")):
            raise SystemExit("too many failed reads; not writing a partial cache")
        pd.concat(parts).to_parquet(out)
    ds = pd.read_parquet(out)
    STRONG = 8  # count another song only if the document shares >= 8 consecutive words of it
    strong = ds[ds.max_words >= STRONG]
    by_doc = strong.groupby("doc_ix").song_dir.apply(set).to_dict()
    print(f"strong (>= {STRONG} words) pairs: {len(strong):,} of {len(ds):,}")
    print(f"{len(ds):,} (doc, song) pairs over {ds.doc_ix.nunique():,} documents and {ds.song_dir.nunique():,} songs")

    pretty = lambda s: re.sub(r"_+", " ", s.rsplit("__", 1)[0]).strip().title()
    for f in (DATA / "footprint").glob("*.json"):
        rec = json.loads(f.read_text())
        slug = re.sub(r"[^a-z0-9]+", "_", rec["title"].lower()).strip("_")
        mine = [d for d in ds.song_dir.unique() if d.rsplit("__", 1)[0] == slug]
        if len(mine) != 1:
            print("skip", rec["title"], mine[:3])
            continue
        with fs.open(f"{ROOT}/{mine[0]}/train.parquet", "rb") as fh:
            src = pq.read_table(fh, columns=["doc_ix", "doc_content", "unique_source_spans"]).to_pandas()
        # footprint docs were written in the same row order, after the min-words filter
        kept = [r for r in src.itertuples() if any(len(s.split()) >= 3 for s in r.unique_source_spans)]
        if len(kept) != len(rec["docs"]):
            print("row mismatch", rec["title"], len(kept), len(rec["docs"]))
            continue
        for doc, r in zip(rec["docs"], kept):
            doc["ix"] = int(r.doc_ix)
            doc["len"] = len(r.doc_content)
            doc["other"] = sorted(pretty(s) for s in by_doc.get(r.doc_ix, set()) if s != mine[0])
        f.write_text(json.dumps(rec, separators=(",", ":")))
        shared = sum(bool(d["other"]) for d in rec["docs"])
        print(f"{rec['title']}: {shared}/{len(rec['docs'])} docs also carry other songs; "
              f"{sum(d['len'] for d in rec['docs']):,} chars total")


if __name__ == "__main__":
    main()
