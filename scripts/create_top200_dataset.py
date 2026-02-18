"""Create filtered dataset with only top 200 songs for evaluation."""

import argparse
import logging
import re
from pathlib import Path

from datasets import Dataset, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_song_line(line: str) -> tuple[str, str, int | None]:
    """Parse a song line like 'Royals - Lorde (2013)' into (title, artist, year)."""
    # Match pattern: "Title - Artist (Year)"
    match = re.match(r"^(.+?)\s*-\s*(.+?)\s*\((\d{4})\)\s*$", line.strip())
    if match:
        return match.group(1).strip(), match.group(2).strip(), int(match.group(3))
    # Fallback: try without year
    match = re.match(r"^(.+?)\s*-\s*(.+)$", line.strip())
    if match:
        return match.group(1).strip(), match.group(2).strip(), None
    return line.strip(), "", None


def load_top_songs(songs_file: str) -> list[dict]:
    """Load top songs from file."""
    songs = []
    with open(songs_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            title, artist, year = parse_song_line(line)
            songs.append({
                "title": title.lower(),
                "artist": artist.lower(),
                "year": year,
                "original_line": line,
            })
    logger.info(f"Loaded {len(songs)} songs from {songs_file}")
    return songs


def normalize_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching."""
    # Lowercase, remove special chars, normalize whitespace
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def song_matches(row: dict, top_songs: list[dict]) -> bool:
    """Check if a row matches any of the top songs."""
    row_title = normalize_for_matching(row.get("song_title", ""))
    row_artists = row.get("song_artists", [])
    if isinstance(row_artists, str):
        row_artists = [row_artists]
    row_artists_norm = [normalize_for_matching(a) for a in row_artists]

    for song in top_songs:
        song_title = normalize_for_matching(song["title"])
        song_artist = normalize_for_matching(song["artist"])

        # Check title match (allow partial)
        if song_title in row_title or row_title in song_title:
            # Check artist match
            for row_artist in row_artists_norm:
                if song_artist in row_artist or row_artist in song_artist:
                    return True
                # Check if any part of the artist matches
                for artist_part in song_artist.split(","):
                    artist_part = artist_part.strip()
                    if artist_part and (artist_part in row_artist or row_artist in artist_part):
                        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Create filtered top 200 songs dataset")
    parser.add_argument(
        "--songs-file",
        default="/dfs/project/kgrlm/akhatua/CleanSlate/songs.txt",
        help="Path to songs.txt file with top 200 songs",
    )
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
        default="unlearning-cleanslate/top200_songs",
        help="Output HuggingFace repo",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't push to HuggingFace, just show what would be done",
    )

    args = parser.parse_args()

    # Load top songs
    top_songs = load_top_songs(args.songs_file)

    # Load source dataset
    logger.info(f"Loading dataset: {args.source_dataset} / {args.source_config}")
    ds = load_dataset(args.source_dataset, args.source_config, split="train")
    logger.info(f"Loaded {len(ds)} rows")

    # Show sample columns
    if len(ds) > 0:
        logger.info(f"Columns: {ds.column_names}")
        sample = ds[0]
        logger.info(f"Sample row keys: {list(sample.keys())}")
        if "song_title" in sample:
            logger.info(f"Sample song_title: {sample['song_title']}")
        if "song_artists" in sample:
            logger.info(f"Sample song_artists: {sample['song_artists']}")

    # Filter to top 200 songs
    logger.info("Filtering to top 200 songs...")

    def filter_fn(row: dict) -> bool:
        return song_matches(row, top_songs)

    filtered_ds = ds.filter(filter_fn, num_proc=8, desc="Filtering to top 200 songs")
    logger.info(f"Filtered to {len(filtered_ds)} rows")

    # Show which songs were found
    found_titles = set()
    for row in filtered_ds:
        found_titles.add(row.get("song_title", "").lower())

    logger.info(f"Found {len(found_titles)} unique songs")

    # Show missing songs
    missing = []
    for song in top_songs:
        song_title_norm = normalize_for_matching(song["title"])
        found = False
        for title in found_titles:
            if song_title_norm in normalize_for_matching(title) or normalize_for_matching(title) in song_title_norm:
                found = True
                break
        if not found:
            missing.append(song["original_line"])

    if missing:
        logger.warning(f"Missing {len(missing)} songs:")
        for m in missing[:10]:
            logger.warning(f"  - {m}")
        if len(missing) > 10:
            logger.warning(f"  ... and {len(missing) - 10} more")

    # Create output dataset with simplified structure
    # Keep only essential columns, set prompt to blank
    # Deduplicate by song_title (keep first occurrence)
    output_rows = []
    seen_titles = set()
    for row in filtered_ds:
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
            "prompt": "",  # Blank prompt as requested
        }
        output_rows.append(output_row)

    # Limit to exactly 200 songs (matching songs.txt)
    output_rows = output_rows[:200]

    output_ds = Dataset.from_list(output_rows)
    logger.info(f"Created output dataset with {len(output_ds)} rows (deduped, max 200)")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Source: {args.source_dataset} / {args.source_config}")
    logger.info(f"Top songs file: {args.songs_file}")
    logger.info(f"Source rows: {len(ds)}")
    logger.info(f"Filtered rows: {len(output_ds)}")
    logger.info(f"Unique songs found: {len(found_titles)}")
    logger.info(f"Missing songs: {len(missing)}")
    logger.info(f"Output repo: {args.output_repo}")
    logger.info("")

    if args.dry_run:
        logger.info("DRY RUN - not pushing to HuggingFace")
        # Save locally for inspection
        local_path = Path("results/top200_preview.parquet")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        output_ds.to_parquet(str(local_path))
        logger.info(f"Saved preview to {local_path}")
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
