"""Convert plain-text book files into a parquet that cleanslate eval can consume.

Reads every .txt under data/post_cutoff_books/text/ , slices each to a
manageable excerpt (default: first 25K chars after the front-matter), and
writes a parquet whose schema matches unlearning-cleanslate/cleanslate_dataset:

    content_id        sha1 of "<title>|||<author>|||<year>"
    content_title     filled from the SLUG (override via metadata.json)
    content_creators  filled from metadata.json
    content_year      int parsed from filename prefix
    reference_target  the trimmed plain-text body

Optionally pushes to HF: --push --hf-org unlearning-cleanslate --suffix v1

Run:
    uv run python scripts/books-data/build_books_dataset.py \
        --in-dir data/post_cutoff_books/text \
        --output data/post_cutoff_books/lyrics.parquet \
        --push  # optional
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_filename(stem: str) -> tuple[int, str]:
    """`2025-onyx-storm` -> (2025, 'onyx storm')."""
    m = re.match(r"^(\d{4})[-_](.+)$", stem)
    if not m:
        raise ValueError(f"can't parse year from {stem!r}; rename like 2025-<slug>.txt")
    year = int(m.group(1))
    title_slug = m.group(2).replace("-", " ").replace("_", " ").strip()
    return year, title_slug


def trim(text: str, max_chars: int) -> str:
    text = text.strip()
    # Cut common front-matter headers
    text = re.sub(r"^(Copyright|Published|First Edition|ISBN|All rights reserved).*?\n\n", "",
                  text, flags=re.IGNORECASE | re.DOTALL)
    return text[:max_chars]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="data/post_cutoff_books/text")
    p.add_argument("--output", default="data/post_cutoff_books/lyrics.parquet")
    p.add_argument("--max-chars", type=int, default=25000)
    p.add_argument("--metadata", default="data/post_cutoff_books/metadata.json",
                   help="Optional JSON: { '<slug>': {'title': ..., 'creators': ..., 'year': ...} }")
    p.add_argument("--push", action="store_true")
    p.add_argument("--hf-repo", default="unlearning-cleanslate/cleanslate_dataset_post_books_2025_2026")
    args = p.parse_args()

    metadata: dict[str, dict] = {}
    meta_path = Path(args.metadata)
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())

    rows: list[dict] = []
    for txt in sorted(Path(args.in_dir).glob("*.txt")):
        slug = txt.stem
        try:
            year, title_default = parse_filename(slug)
        except ValueError as e:
            print(f"  [skip] {txt.name}: {e}")
            continue
        meta = metadata.get(slug, {})
        title = meta.get("title", title_default.title())
        creators = meta.get("creators", "Unknown")
        year = int(meta.get("year", year))
        body = trim(txt.read_text(encoding="utf-8", errors="replace"), args.max_chars)
        if len(body) < 1000:
            print(f"  [skip] {txt.name}: only {len(body)} chars after trim")
            continue
        cid = hashlib.sha1(f"{title}|||{creators}|||{year}".encode()).hexdigest()
        rows.append({
            "content_id": cid,
            "content_title": title,
            "content_creators": creators,
            "content_year": year,
            "reference_target": body,
        })
        print(f"  [keep] {title} -- {creators} ({year})  {len(body):,} chars")

    if not rows:
        print("No usable rows. Drop .txt files into the in-dir first.")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out_path)
    print(f"\nWrote {len(rows)} rows -> {out_path}")

    if args.push:
        import os
        from datasets import Dataset
        ds = Dataset.from_list(rows)
        ds.push_to_hub(args.hf_repo, config_name="default", token=os.environ["HF_TOKEN"])
        print(f"Pushed to {args.hf_repo}")


if __name__ == "__main__":
    main()
