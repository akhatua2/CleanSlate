"""
Add lyrics to Billboard songs parquet file using Genius API.

Usage:
    python add_lyrics.py input.parquet output.parquet
"""

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import lyricsgenius
import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


def get_cache_path(cache_dir: Path) -> Path:
    """Get path to lyrics cache file."""
    return cache_dir / "lyrics_cache.json"


def load_cache(cache_dir: Path) -> dict[str, str | None]:
    """Load lyrics cache from disk."""
    cache_path = get_cache_path(cache_dir)
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict[str, str | None], cache_dir: Path) -> None:
    """Save lyrics cache to disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = get_cache_path(cache_dir)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def make_cache_key(title: str, artist: str) -> str:
    """Create a unique cache key for a song."""
    return f"{title.lower().strip()}|||{artist.lower().strip()}"


def fetch_lyrics(
    genius: lyricsgenius.Genius,
    title: str,
    artist: str,
    cache: dict[str, str | None],
    cache_lock: threading.Lock,
) -> str | None:
    """
    Fetch lyrics for a song, using cache if available.
    Returns lyrics string or None if not found.
    Thread-safe with cache_lock.
    """
    cache_key = make_cache_key(title, artist)
    
    # Check cache (thread-safe read)
    with cache_lock:
        if cache_key in cache:
            return cache[cache_key]
    
    try:
        song = genius.search_song(title, artist)
        if song and song.lyrics:
            lyrics = song.lyrics
            with cache_lock:
                cache[cache_key] = lyrics
            return lyrics
        else:
            with cache_lock:
                cache[cache_key] = None
            return None
    except Exception as e:
        print(f"Error fetching '{title}' by '{artist}': {e}")
        with cache_lock:
            cache[cache_key] = None
        return None


def fetch_lyrics_task(
    task: tuple[int, str, str],
    genius: lyricsgenius.Genius,
    cache: dict[str, str | None],
    cache_lock: threading.Lock,
) -> tuple[int, str | None]:
    """
    Worker task for ThreadPoolExecutor.
    Returns (index, lyrics) tuple.
    """
    idx, title, artist = task
    lyrics = fetch_lyrics(genius, title, artist, cache, cache_lock)
    return (idx, lyrics)


def add_lyrics_to_parquet(
    input_path: str,
    output_path: str,
    cache_dir: str = "data/cache",
    save_interval: int = 50,
    max_workers: int = 3,
) -> None:
    """
    Read parquet file, add lyrics column, and save to new file.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        cache_dir: Directory to store lyrics cache
        save_interval: How often to save cache (every N songs)
        max_workers: Number of concurrent API requests
    """
    # Get API token from environment
    token = os.environ.get("GENIUS_CLIENT_ACCESS_TOKEN")
    if not token:
        raise ValueError("GENIUS_CLIENT_ACCESS_TOKEN not found in environment")
    
    # Initialize Genius client
    genius = lyricsgenius.Genius(token, verbose=False, remove_section_headers=True)
    genius.timeout = 15
    genius.retries = 3
    
    # Load existing cache
    cache_path = Path(cache_dir)
    cache = load_cache(cache_path)
    cache_lock = threading.Lock()
    print(f"Loaded {len(cache)} cached lyrics")
    
    # Read input parquet
    df = pl.read_parquet(input_path)
    print(f"Loaded {len(df)} songs from {input_path}")
    
    # Check if lyrics column already exists
    if "lyrics" in df.columns:
        existing_lyrics = df["lyrics"].to_list()
        print("Found existing lyrics column, will skip songs with lyrics")
    else:
        existing_lyrics = [None] * len(df)
    
    # Prepare results array and tasks
    lyrics_results: dict[int, str | None] = {}
    tasks_to_fetch: list[tuple[int, str, str]] = []
    
    # Separate cached/existing from tasks to fetch
    for i, row in enumerate(df.iter_rows(named=True)):
        title = row["title"]
        artists = row["artists"]
        primary_artist = artists[0] if artists else ""
        
        # Skip if lyrics already exist in dataframe
        if existing_lyrics[i] is not None:
            lyrics_results[i] = existing_lyrics[i]
            continue
        
        # Check if in cache
        cache_key = make_cache_key(title, primary_artist)
        if cache_key in cache:
            lyrics_results[i] = cache[cache_key]
            continue
        
        # Need to fetch
        tasks_to_fetch.append((i, title, primary_artist))
    
    print(f"Already have: {len(lyrics_results)} | Need to fetch: {len(tasks_to_fetch)}")
    
    # Fetch lyrics using thread pool
    if tasks_to_fetch:
        completed = 0
        found_count = sum(1 for v in lyrics_results.values() if v)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fetch_lyrics_task, task, genius, cache, cache_lock): task
                for task in tasks_to_fetch
            }
            
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Fetching lyrics")
            for future in pbar:
                idx, lyrics = future.result()
                lyrics_results[idx] = lyrics
                completed += 1
                if lyrics:
                    found_count += 1
                
                pbar.set_postfix(found=found_count, cached=len(cache))
                
                # Periodically save cache
                if completed % save_interval == 0:
                    with cache_lock:
                        save_cache(cache, cache_path)
    
    # Save final cache
    save_cache(cache, cache_path)
    
    # Build lyrics list in correct order
    lyrics_list = [lyrics_results[i] for i in range(len(df))]
    
    # Add lyrics column to dataframe
    df = df.with_columns([pl.Series("lyrics", lyrics_list)])
    
    # Save output
    df.write_parquet(output_path, compression="zstd")
    
    # Print stats
    found = df.filter(pl.col("lyrics").is_not_null()).height
    print("\nComplete!")
    print(f"Songs with lyrics: {found}/{len(df)} ({100*found/len(df):.1f}%)")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Add lyrics to Billboard songs parquet")
    parser.add_argument("input", help="Input parquet file path")
    parser.add_argument("output", help="Output parquet file path")
    parser.add_argument("--cache-dir", default="data/cache", help="Cache directory")
    parser.add_argument("--save-interval", type=int, default=50, help="Cache save interval")
    parser.add_argument("--workers", type=int, default=2, help="Number of concurrent requests")
    args = parser.parse_args()
    
    add_lyrics_to_parquet(
        input_path=args.input,
        output_path=args.output,
        cache_dir=args.cache_dir,
        save_interval=args.save_interval,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()