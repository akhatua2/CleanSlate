"""
Fetch book texts from Project Gutenberg for books in the master list CSV.

Downloads full text, strips Gutenberg header/footer boilerplate,
caches locally, and saves enriched data to parquet.

Usage:
    python fetch_gutenberg.py                          # fetch all with gutenberg_id
    python fetch_gutenberg.py --max 5                  # fetch first 5 only
    python fetch_gutenberg.py --dry-run                # just check availability
"""

import argparse
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm

GUTENBERG_TEXT_URL = "https://www.gutenberg.org/ebooks/{id}.txt.utf-8"
GUTENBERG_MIRROR_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
MAX_WORKERS = 5  # Be polite to Gutenberg
HEADERS = {
    "User-Agent": "CleanSlate/1.0 (academic research; https://github.com/akhatua2/cleanslate)"
}

SCRIPT_DIR = Path(__file__).parent
CSV_PATH = SCRIPT_DIR / "books_master_list.csv"
CACHE_DIR = SCRIPT_DIR / "cache"
TEXT_CACHE_DIR = CACHE_DIR / "gutenberg_texts"
CACHE_META_PATH = CACHE_DIR / "gutenberg_meta.json"


def load_cache_meta() -> dict[str, dict]:
    """Load metadata cache (status, word count, etc.)."""
    if CACHE_META_PATH.exists():
        with open(CACHE_META_PATH) as f:
            return json.load(f)
    return {}


def save_cache_meta(cache: dict[str, dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_META_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def get_text_cache_path(gutenberg_id: str) -> Path:
    return TEXT_CACHE_DIR / f"pg{gutenberg_id}.txt"


def parse_gutenberg_metadata(raw_text: str) -> dict[str, str]:
    """Extract metadata fields from the Gutenberg header before the *** START marker."""
    # Only look at text before the START marker
    start_match = re.search(r"\*\*\*\s*START OF", raw_text, re.IGNORECASE)
    header = raw_text[:start_match.start()] if start_match else raw_text[:3000]

    fields = {}
    for field in ["Title", "Author", "Translator", "Release date", "Language", "Credits"]:
        # Match "Field: value" possibly spanning multiple lines (until next field or blank line)
        pattern = rf"^{field}:\s*(.+?)(?=\n\s*\n|\n[A-Z][\w\s]*:|\*\*\*)"
        match = re.search(pattern, header, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        if match:
            val = match.group(1).strip()
            # Collapse internal whitespace
            val = re.sub(r"\s+", " ", val)
            fields[field.lower().replace(" ", "_")] = val

    return fields


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header and footer boilerplate."""
    # Find start of actual text (after the header)
    start_markers = [
        r"\*\*\*\s*START OF TH(?:IS|E) PROJECT GUTENBERG EBOOK",
        r"\*\*\*\s*START OF THE PROJECT GUTENBERG",
        r"\*\*\* START OF THIS PROJECT GUTENBERG",
    ]
    end_markers = [
        r"\*\*\*\s*END OF TH(?:IS|E) PROJECT GUTENBERG EBOOK",
        r"\*\*\*\s*END OF THE PROJECT GUTENBERG",
        r"\*\*\* END OF THIS PROJECT GUTENBERG",
    ]

    # Strip header
    for pattern in start_markers:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Skip past the marker line
            rest = text[match.end():]
            # Skip to the next newline (end of marker line)
            newline_pos = rest.find("\n")
            if newline_pos != -1:
                text = rest[newline_pos + 1:]
            break

    # Strip footer
    for pattern in end_markers:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]
            break

    return text.strip()


def fetch_gutenberg_text(gutenberg_id: str, title: str) -> tuple[str, str, str | None, dict[str, str], int]:
    """
    Fetch text from Project Gutenberg.
    Returns (gutenberg_id, title, text_or_none, metadata_dict, status_code).
    """
    # Check local cache first
    cache_path = get_text_cache_path(gutenberg_id)
    if cache_path.exists():
        raw = cache_path.read_text(encoding="utf-8", errors="replace")
        if raw.strip():
            cleaned = strip_gutenberg_boilerplate(raw)
            metadata = parse_gutenberg_metadata(raw)
            return (gutenberg_id, title, cleaned, metadata, 200)

    # Try primary URL
    for url_template in [GUTENBERG_TEXT_URL, GUTENBERG_MIRROR_URL]:
        url = url_template.format(id=gutenberg_id)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200 and len(resp.text) > 500:
                # Cache raw text
                TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(resp.text, encoding="utf-8")
                cleaned = strip_gutenberg_boilerplate(resp.text)
                metadata = parse_gutenberg_metadata(resp.text)
                return (gutenberg_id, title, cleaned, metadata, 200)
        except requests.RequestException:
            continue

    return (gutenberg_id, title, None, {}, 404)


def read_csv() -> list[dict]:
    """Read master list CSV."""
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def write_csv(rows: list[dict]) -> None:
    """Write updated CSV back."""
    if not rows:
        return
    fieldnames = rows[0].keys()
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Fetch book texts from Project Gutenberg")
    parser.add_argument("--max", type=int, default=0, help="Max books to fetch (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Just check, don't download")
    parser.add_argument("--output", default=str(SCRIPT_DIR / "books_gutenberg.parquet"),
                        help="Output parquet path")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel workers")
    args = parser.parse_args()

    rows = read_csv()
    print(f"Loaded {len(rows)} books from CSV")

    # Filter to books with gutenberg_id
    gutenberg_rows = [r for r in rows if r.get("gutenberg_id", "").strip()]
    print(f"Books with Gutenberg IDs: {len(gutenberg_rows)}")

    if args.max > 0:
        gutenberg_rows = gutenberg_rows[:args.max]
        print(f"  (limited to {args.max})")

    if args.dry_run:
        print("\n[DRY RUN] Would fetch these books:")
        for r in gutenberg_rows:
            print(f"  PG#{r['gutenberg_id']}: {r['title']} by {r['author']}")
        return

    # Load cache metadata
    cache_meta = load_cache_meta()

    # Fetch texts in parallel
    results = {}  # gutenberg_id -> (text, metadata)
    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_gutenberg_text, r["gutenberg_id"].strip(), r["title"]): r
            for r in gutenberg_rows
        }

        pbar = tqdm(as_completed(futures), total=len(futures), desc="Fetching from Gutenberg")
        for future in pbar:
            row = futures[future]
            gid, title, text, metadata, status = future.result()

            if text and len(text) > 100:
                word_count = len(text.split())
                results[gid] = (text, metadata)
                cache_meta[gid] = {
                    "title": title,
                    "word_count": word_count,
                    "status": "ok",
                    "metadata": metadata,
                }
                success += 1
                pbar.set_postfix(ok=success, fail=failed)
            else:
                cache_meta[gid] = {"title": title, "status": "failed"}
                failed += 1
                tqdm.write(f"  FAILED: PG#{gid} - {title}")
                pbar.set_postfix(ok=success, fail=failed)

    save_cache_meta(cache_meta)

    # Update CSV with availability status
    for row in rows:
        gid = row.get("gutenberg_id", "").strip()
        if gid and gid in cache_meta:
            row["gutenberg_available"] = "yes" if cache_meta[gid]["status"] == "ok" else "no"

    write_csv(rows)
    print(f"\nUpdated CSV with availability status")

    # Build parquet with successful downloads
    parquet_rows = []
    for row in rows:
        gid = row.get("gutenberg_id", "").strip()
        if gid and gid in results:
            text, metadata = results[gid]
            parquet_rows.append({
                "title": row["title"],
                "author": row["author"],
                "year": int(row["year"]) if row["year"].lstrip("-").isdigit() else 0,
                "genre": row["genre"],
                "gutenberg_id": int(gid),
                "translator": metadata.get("translator", ""),
                "language": metadata.get("language", ""),
                "credits": metadata.get("credits", ""),
                "release_date": metadata.get("release_date", ""),
                "text": text,
                "word_count": len(text.split()),
                "source": "gutenberg",
            })

    if parquet_rows:
        df = pl.DataFrame(parquet_rows)
        df.write_parquet(args.output, compression="zstd")
        print(f"\nSaved {len(df)} books to {args.output}")
        print(f"  Total words: {df['word_count'].sum():,}")
        print(f"  Avg words/book: {df['word_count'].mean():,.0f}")
        print(f"  Min: {df['word_count'].min():,} ({df.filter(pl.col('word_count') == df['word_count'].min())['title'][0]})")
        print(f"  Max: {df['word_count'].max():,} ({df.filter(pl.col('word_count') == df['word_count'].max())['title'][0]})")
    else:
        print("No texts fetched!")

    print(f"\nSummary: {success} succeeded, {failed} failed out of {len(gutenberg_rows)} attempted")


if __name__ == "__main__":
    main()
