"""
Create books dataset from Gutenberg parquet, matching songs_5000 schema.

Schema:
    book_id (str): SHA1 hash of title
    book_title (str): Book title
    book_authors (list[str]): Author name(s)
    book_year (int): Publication year
    reference_target (str): Full book text
    prompt (str): Blank string

Usage:
    python scripts/create_books_dataset.py --dry-run
    python scripts/create_books_dataset.py --output-repo unlearning-cleanslate/books
"""

import argparse
import hashlib
import logging

import polars as pl
from datasets import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SOURCE = "scripts/books-data/books_gutenberg.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create books dataset for HuggingFace")
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Source parquet file from fetch_gutenberg.py",
    )
    parser.add_argument(
        "--output-repo",
        default="unlearning-cleanslate/books",
        help="Output HuggingFace repo",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't push to HuggingFace, just show what would be done",
    )
    args = parser.parse_args()

    # Load source parquet
    logger.info(f"Loading source: {args.source}")
    df = pl.read_parquet(args.source)
    logger.info(f"Loaded {len(df)} books")
    logger.info(f"Columns: {df.columns}")

    # Build output rows
    output_rows = []
    seen_titles = set()

    for row in df.iter_rows(named=True):
        title = row["title"]
        title_key = title.lower().strip()

        if title_key in seen_titles:
            logger.warning(f"Duplicate title skipped: {title}")
            continue
        seen_titles.add(title_key)

        # Generate book_id as SHA1 of title
        book_id = hashlib.sha1(title.encode("utf-8")).hexdigest()

        # Parse author(s) — split on " and ", "/", "&"
        author_str = row["author"]
        authors = [a.strip() for a in author_str.replace(" and ", "/").replace("&", "/").split("/")]
        authors = [a for a in authors if a]

        output_rows.append({
            "book_id": book_id,
            "book_title": title,
            "book_authors": authors,
            "book_year": row["year"],
            "genre": row.get("genre", ""),
            "translator": row.get("translator", ""),
            "language": row.get("language", ""),
            "gutenberg_id": row.get("gutenberg_id", 0),
            "credits": row.get("credits", ""),
            "release_date": row.get("release_date", ""),
            "word_count": row.get("word_count", len(row["text"].split())),
            "reference_target": row["text"],
            "prompt": "",
        })

    output_ds = Dataset.from_list(output_rows)

    # Summary
    word_counts = [len(r["reference_target"].split()) for r in output_rows]
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Source: {args.source}")
    logger.info(f"Total books: {len(output_ds)}")
    logger.info(f"Total words: {sum(word_counts):,}")
    logger.info(f"Avg words/book: {sum(word_counts)/len(word_counts):,.0f}")
    logger.info(f"Min words: {min(word_counts):,}")
    logger.info(f"Max words: {max(word_counts):,}")
    logger.info(f"Output repo: {args.output_repo}")
    logger.info("")

    # Show first few
    logger.info("Sample books:")
    for r in output_rows[:10]:
        wc = len(r["reference_target"].split())
        logger.info(f"  {r['book_title']} by {', '.join(r['book_authors'])} ({r['book_year']}) - {wc:,} words")

    if args.dry_run:
        logger.info("")
        logger.info("DRY RUN - not pushing to HuggingFace")
        logger.info(f"Would push {len(output_ds)} books to {args.output_repo}")
    else:
        logger.info(f"Pushing to HuggingFace: {args.output_repo}")
        output_ds.push_to_hub(
            args.output_repo,
            split="train",
            private=False,
        )
        logger.info(f"Pushed to https://huggingface.co/datasets/{args.output_repo}")


if __name__ == "__main__":
    main()
