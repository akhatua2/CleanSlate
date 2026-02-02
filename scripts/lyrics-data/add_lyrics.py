"""
Add lyrics to Billboard songs parquet file using LRCLIB API.

API docs: https://lrclib.net/docs

Usage:
    python add_lyrics.py input.parquet output.parquet
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm

LRCLIB_API_BASE = "https://lrclib.net/api"
MAX_WORKERS = 15  # Number of parallel requests
HEADERS = {
    "User-Agent": "BillboardLyrics/1.0 (https://github.com/example/cleanslate)"
}


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


def fetch_lyrics_from_lrclib(
    title: str,
    artist: str,
) -> tuple[str, str, str | None]:
    """
    Fetch lyrics from LRCLIB API.
    Returns (title, artist, lyrics) tuple.
    
    API endpoint: https://lrclib.net/api/get?artist_name=X&track_name=Y
    """
    try:
        url = f"{LRCLIB_API_BASE}/get"
        params = {
            "artist_name": artist,
            "track_name": title,
        }
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        
        if response.status_code == 404:
            # Song not found
            return (title, artist, None)
        
        response.raise_for_status()
        data = response.json()
        
        # Prefer plainLyrics
        lyrics = data.get("plainLyrics")
        return (title, artist, lyrics)
    except requests.RequestException as e:
        tqdm.write(f"Error fetching '{title}' by '{artist}': {e}")
        return (title, artist, None)


def add_lyrics_to_parquet(
    input_path: str,
    output_path: str,
    cache_dir: str = "data/cache",
    save_interval: int = 50,
) -> None:
    """
    Read parquet file, add lyrics column, and save to new file.
    
    Uses LRCLIB API (https://lrclib.net/docs) - no authentication required.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        cache_dir: Directory to store lyrics cache
        save_interval: How often to save cache (every N songs)
    """
    # Load existing cache
    cache_path = Path(cache_dir)
    cache = load_cache(cache_path)
    print(f"Loaded {len(cache)} cached lyrics")
    
    # Read input parquet
    df = pl.read_parquet(input_path)
    print(f"Loaded {len(df)} songs from {input_path}")
    
    # Check if lyrics column already exists
    if "lyrics" in df.columns:
        # Only process songs without lyrics
        existing_lyrics = df["lyrics"].to_list()
        print("Found existing lyrics column, will skip songs with lyrics")
    else:
        existing_lyrics = [None] * len(df)
    
    # Prepare list of songs to fetch (skip cached/existing)
    songs_to_fetch = []  # List of (index, title, artist)
    lyrics_list = [None] * len(df)
    found_count = 0
    
    for i, row in enumerate(df.iter_rows(named=True)):
        title = row["title"]
        artists = row["artists"]
        primary_artist = artists[0] if artists else ""
        cache_key = make_cache_key(title, primary_artist)
        
        # Check if lyrics already exist in dataframe
        if existing_lyrics[i] is not None:
            lyrics_list[i] = existing_lyrics[i]
            if existing_lyrics[i]:
                found_count += 1
            continue
        
        # Check cache
        if cache.get(cache_key):
            lyrics_list[i] = cache[cache_key]
            found_count += 1
            continue
        
        # Need to fetch this one
        songs_to_fetch.append((i, title, primary_artist))
    
    print(f"Songs to fetch: {len(songs_to_fetch)} (skipped {len(df) - len(songs_to_fetch)} cached/existing)")
    
    # Fetch lyrics in parallel using ThreadPoolExecutor
    songs_fetched = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all fetch tasks
        future_to_song = {
            executor.submit(fetch_lyrics_from_lrclib, title, artist): (idx, title, artist)
            for idx, title, artist in songs_to_fetch
        }
        
        # Process results as they complete
        pbar = tqdm(as_completed(future_to_song), total=len(songs_to_fetch), desc="Fetching lyrics")
        for future in pbar:
            idx, title, artist = future_to_song[future]
            try:
                _, _, lyrics = future.result()
                lyrics_list[idx] = lyrics
                
                # Update cache
                cache_key = make_cache_key(title, artist)
                cache[cache_key] = lyrics
                
                if lyrics:
                    found_count += 1
                
                songs_fetched += 1
                pbar.set_postfix(found=found_count, cached=len(cache))
                
                # Periodically save cache
                if songs_fetched % save_interval == 0:
                    save_cache(cache, cache_path)
            except Exception as e:
                tqdm.write(f"Error processing '{title}' by '{artist}': {e}")
    
    # Save final cache
    save_cache(cache, cache_path)
    
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
    parser.add_argument("--save-interval", type=int, default=10, help="Cache save interval")
    args = parser.parse_args()
    
    add_lyrics_to_parquet(
        input_path=args.input,
        output_path=args.output,
        cache_dir=args.cache_dir,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()