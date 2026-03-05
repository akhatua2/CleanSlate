"""
Check and fetch book texts from the Books3 dataset (Geralt-Targaryen/books3) on HuggingFace.

Books3 is a large collection of books stored as individual .epub files.
This script searches for our target books by title/author matching.

Usage:
    python fetch_books3.py                    # check all books without gutenberg text
    python fetch_books3.py --dry-run          # just list what would be searched
    python fetch_books3.py --check-only       # only check availability, don't download
"""

import argparse
import csv
import json
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
CSV_PATH = SCRIPT_DIR / "books_master_list.csv"
CACHE_DIR = SCRIPT_DIR / "cache"
BOOKS3_CACHE_PATH = CACHE_DIR / "books3_index.json"
BOOKS3_TEXTS_DIR = CACHE_DIR / "books3_texts"


def read_csv() -> list[dict]:
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def write_csv(rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = rows[0].keys()
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def normalize(s: str) -> str:
    """Normalize a string for fuzzy matching."""
    s = s.lower().strip()
    # Remove common prefixes/suffixes
    s = re.sub(r"^(the|a|an)\s+", "", s)
    # Remove punctuation
    s = re.sub(r"[^\w\s]", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def title_matches(book_title: str, dataset_title: str) -> bool:
    """Check if titles match with fuzzy matching."""
    n1 = normalize(book_title)
    n2 = normalize(dataset_title)

    # Exact match
    if n1 == n2:
        return True

    # One contains the other
    if n1 in n2 or n2 in n1:
        return True

    return False


def author_matches(book_author: str, dataset_author: str) -> bool:
    """Check if authors match (last name matching)."""
    # Extract last names
    book_parts = book_author.lower().split()
    dataset_parts = dataset_author.lower().split()

    if not book_parts or not dataset_parts:
        return False

    # Check if last name matches
    book_last = re.sub(r"[^\w]", "", book_parts[-1])
    dataset_last = re.sub(r"[^\w]", "", dataset_parts[-1])

    if book_last == dataset_last:
        return True

    # Check if any name part appears
    for part in book_parts:
        clean = re.sub(r"[^\w]", "", part)
        if len(clean) > 2 and clean in dataset_author.lower():
            return True

    return False


def load_books3_index() -> list[dict]:
    """Load or build the Books3 dataset index (titles only, not full text)."""
    if BOOKS3_CACHE_PATH.exists():
        print("Loading cached Books3 index...")
        with open(BOOKS3_CACHE_PATH) as f:
            return json.load(f)

    print("Loading Books3 dataset index from HuggingFace (this may take a while)...")
    print("Dataset: Geralt-Targaryen/books3")

    # Stream the dataset to build an index without loading everything into memory
    ds = load_dataset("Geralt-Targaryen/books3", split="train", streaming=True)

    index = []
    for i, item in enumerate(tqdm(ds, desc="Indexing Books3")):
        entry = {
            "idx": i,
            "title": item.get("title", ""),
            "author": item.get("author", ""),
        }
        index.append(entry)

        # Save progress periodically
        if (i + 1) % 10000 == 0:
            print(f"  Indexed {i + 1} books so far...")

    # Cache the index
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(BOOKS3_CACHE_PATH, "w") as f:
        json.dump(index, f, indent=2)

    print(f"Indexed {len(index)} books total")
    return index


def search_books3(target_books: list[dict], index: list[dict]) -> dict[str, list[dict]]:
    """Search Books3 index for target books. Returns {csv_title: [matching_entries]}."""
    matches = {}

    for book in tqdm(target_books, desc="Searching Books3"):
        title = book["title"]
        author = book["author"]
        book_matches = []

        for entry in index:
            b3_title = entry.get("title", "")
            b3_author = entry.get("author", "")

            if title_matches(title, b3_title):
                # If we also have author info, check that too
                if b3_author and author:
                    if author_matches(author, b3_author):
                        book_matches.append(entry)
                elif not b3_author:
                    # No author in Books3, accept title match
                    book_matches.append(entry)

        matches[title] = book_matches

    return matches


def fetch_books3_text(index_position: int) -> str | None:
    """Fetch the full text of a book from Books3 by its index position."""
    try:
        ds = load_dataset("Geralt-Targaryen/books3", split="train", streaming=True)
        for i, item in enumerate(ds):
            if i == index_position:
                return item.get("text", "")
        return None
    except Exception as e:
        print(f"Error fetching index {index_position}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Search/fetch books from Books3 dataset")
    parser.add_argument("--dry-run", action="store_true", help="Just list target books")
    parser.add_argument("--check-only", action="store_true", help="Only check availability")
    parser.add_argument("--fetch", action="store_true", help="Actually download matched texts")
    args = parser.parse_args()

    rows = read_csv()
    print(f"Loaded {len(rows)} books from CSV")

    # Find books that need Books3 (no gutenberg_id or gutenberg failed)
    need_books3 = [
        r for r in rows
        if not r.get("gutenberg_id", "").strip()
        or r.get("gutenberg_available", "") == "no"
    ]
    print(f"Books needing alternative source: {len(need_books3)}")

    if args.dry_run:
        print("\nWould search Books3 for:")
        for r in need_books3:
            print(f"  {r['title']} by {r['author']} ({r['year']})")
        return

    # Load Books3 index
    index = load_books3_index()
    print(f"Books3 index: {len(index)} entries")

    # Search for matches
    matches = search_books3(need_books3, index)

    # Report results
    found = 0
    not_found = 0
    print("\n=== Search Results ===")
    for title, book_matches in matches.items():
        if book_matches:
            found += 1
            print(f"  FOUND: {title}")
            for m in book_matches[:3]:  # Show top 3 matches
                print(f"    -> '{m['title']}' by {m.get('author', 'unknown')} (idx={m['idx']})")
        else:
            not_found += 1
            print(f"  MISS:  {title}")

    print(f"\nFound: {found}/{len(matches)} ({100*found/max(len(matches),1):.1f}%)")
    print(f"Missing: {not_found}/{len(matches)}")

    # Update CSV
    for row in rows:
        title = row["title"]
        if title in matches:
            if matches[title]:
                row["books3_available"] = "yes"
                # Store the Books3 index for later fetching
                row["other_source"] = f"books3_idx={matches[title][0]['idx']}"
            else:
                row["books3_available"] = "no"

    write_csv(rows)
    print("Updated CSV with Books3 availability")

    if args.fetch and found > 0:
        print("\nFetching full texts from Books3...")
        BOOKS3_TEXTS_DIR.mkdir(parents=True, exist_ok=True)

        for title, book_matches in tqdm(matches.items(), desc="Downloading"):
            if not book_matches:
                continue
            idx = book_matches[0]["idx"]
            cache_path = BOOKS3_TEXTS_DIR / f"books3_{idx}.txt"

            if cache_path.exists():
                continue

            text = fetch_books3_text(idx)
            if text:
                cache_path.write_text(text, encoding="utf-8")
                tqdm.write(f"  Saved: {title} ({len(text.split())} words)")


if __name__ == "__main__":
    main()
