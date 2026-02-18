"""Create dataset with all songs from direct__none config."""

import argparse
import logging

from datasets import Dataset, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create songs dataset from direct__none")
    parser.add_argument(
        "--source-dataset",
        default="unlearning-cleanslate/generated_songs",
        help="Source HuggingFace dataset",
    )
    parser.add_argument(
        "--source-config",
        default="direct__none",
        help="Source config/split to use",
    )
    parser.add_argument(
        "--output-repo",
        default="unlearning-cleanslate/songs_5000",
        help="Output HuggingFace repo",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't push to HuggingFace, just show what would be done",
    )

    args = parser.parse_args()

    # Load source dataset
    logger.info(f"Loading dataset: {args.source_dataset} / {args.source_config}")
    ds = load_dataset(args.source_dataset, args.source_config, split="train")
    logger.info(f"Loaded {len(ds)} rows")

    # Show sample columns
    if len(ds) > 0:
        logger.info(f"Columns: {ds.column_names}")

    # Deduplicate by song_title (keep first occurrence)
    output_rows = []
    seen_titles = set()
    for row in ds:
        title_key = row.get("song_title", "").lower().strip()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)

        output_row = {
            "song_id": row.get("song_id", ""),
            "song_title": row.get("song_title", ""),
            "song_artists": row.get("song_artists", []),
            "song_year": row.get("song_year"),
            "reference_target": row.get("reference_target", ""),
            "prompt": "",  # Blank prompt
        }
        output_rows.append(output_row)

    output_ds = Dataset.from_list(output_rows)
    logger.info(f"Created output dataset with {len(output_ds)} unique songs")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Source: {args.source_dataset} / {args.source_config}")
    logger.info(f"Source rows: {len(ds)}")
    logger.info(f"Unique songs: {len(output_ds)}")
    logger.info(f"Output repo: {args.output_repo}")
    logger.info("")

    if args.dry_run:
        logger.info("DRY RUN - not pushing to HuggingFace")
        logger.info(f"Would push {len(output_ds)} songs to {args.output_repo}")
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
