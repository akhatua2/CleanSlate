"""Pull work metadata, reference texts and per-model forget/retain pools.

Output (cache/):
  works.parquet  content_id, title, creators, year, kind (song|book), public_domain, text
  pools.parquet  model, content_id, pool (forget|retain)
"""
from __future__ import annotations

import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

CACHE = Path(__file__).parent / "cache"
BOOKS_CSV = Path(__file__).parents[2] / "scripts/books-data/books_master_list.csv"
fs = HfFileSystem()
UC = "unlearning-cleanslate"

FSID = {
    "llama-8b": "llama-8b", "olmo-7b": "olmo-7b", "qwen3-8b": "qwen3-8b",
    "nemotron-9b": "nemotron-9b", "gemma-12b": "gemma-12b", "olmo-32b": "olmo-32b",
}


def read(path: str, columns: list[str]) -> pa.Table:
    with fs.open(path, "rb", block_size=2**20) as f:
        return pq.ParquetFile(f).read(columns=columns)


def main() -> None:
    t = read(f"datasets/{UC}/cleanslate_dataset/data/train-00000-of-00001.parquet",
             ["content_id", "content_title", "content_creators", "content_year", "reference_target"])
    books = {}
    with open(BOOKS_CSV) as f:
        for r in csv.DictReader(f):
            books[r["title"].strip().lower()] = r
    rows = []
    for r in t.to_pylist():
        b = books.get((r["content_title"] or "").strip().lower())
        year = r["content_year"]
        # Public domain in the US: published before 1930 and available on Gutenberg.
        pd = bool(b and b.get("gutenberg_available") == "yes" and year is not None and year < 1930)
        rows.append({
            "content_id": r["content_id"], "title": r["content_title"], "creators": r["content_creators"],
            "year": year, "kind": "book" if (b or len(r["reference_target"] or "") > 30000) else "song", "public_domain": pd,
            "text": r["reference_target"] or "",
        })
    pq.write_table(pa.Table.from_pylist(rows), CACHE / "works.parquet")
    print(f"works: {len(rows)} ({sum(r['kind'] == 'book' for r in rows)} books, "
          f"{sum(r['public_domain'] for r in rows)} public domain)")

    pools = []
    for m, short in FSID.items():
        for pool in ("forget_pool", "retain_pool"):
            files = sorted(fs.glob(f"datasets/{UC}/fsid-curated-{short}/{pool}/*.parquet"))
            if not files:
                print(f"MISSING {m} {pool}")
                continue
            ids = set()
            for p in files:
                ids |= set(read(p, ["content_id"]).column("content_id").to_pylist())
            pools += [{"model": m, "content_id": c, "pool": pool.split("_")[0]} for c in ids]
            print(f"{m} {pool}: {len(ids)}")
    pq.write_table(pa.Table.from_pylist(pools), CACHE / "pools.parquet")


if __name__ == "__main__":
    main()
