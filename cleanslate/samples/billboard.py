"""Billboard Hot 100 songs data loader.

Loads songs from parquet file with columns: title, artists, year, wikipedia_link, lyrics
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

# Default path to the billboard parquet file
DEFAULT_PARQUET_PATH = Path(__file__).parent.parent.parent / "data" / "billboard_songs_1970_2025_lyrics.parquet"


def load_billboard_songs(
    parquet_path: str | Path | None = None,
    year_range: tuple[int, int] | None = None,
    limit: int | None = None,
    require_lyrics: bool = True,
    shuffle: bool = False,
    seed: int = 42,
) -> list[dict]:
    """
    Load Billboard songs from parquet file as sample dicts.
    
    Args:
        parquet_path: Path to parquet file. Defaults to data/billboard_songs_1970_2025_lyrics.parquet
        year_range: Optional (start_year, end_year) inclusive filter
        limit: Maximum number of songs to return
        require_lyrics: If True, filter out songs with null/empty lyrics
        shuffle: If True, shuffle the songs (with seed for reproducibility)
        seed: Random seed for shuffling
        
    Returns:
        List of dicts with keys: name, title, artists, year, lyrics, wikipedia_link
        
    Example:
        >>> songs = load_billboard_songs(year_range=(2020, 2025), limit=100)
        >>> print(songs[0]["title"], songs[0]["artists"])
    """
    try:
        import polars as pl
    except ImportError:
        raise ImportError("polars is required to load billboard data: pip install polars")
    
    path = Path(parquet_path) if parquet_path else DEFAULT_PARQUET_PATH
    
    if not path.exists():
        raise FileNotFoundError(
            f"Billboard parquet file not found at {path}. "
            "Expected columns: title, artists, year, wikipedia_link, lyrics"
        )
    
    df = pl.read_parquet(path)
    
    # Filter by year range
    if year_range:
        start_year, end_year = year_range
        df = df.filter(
            (pl.col("year") >= start_year) & (pl.col("year") <= end_year)
        )
    
    # Filter out songs without lyrics
    if require_lyrics:
        df = df.filter(
            pl.col("lyrics").is_not_null() & (pl.col("lyrics").str.len_chars() > 0)
        )
    
    # Shuffle if requested
    if shuffle:
        df = df.sample(fraction=1.0, shuffle=True, seed=seed)
    
    # Limit
    if limit:
        df = df.head(limit)
    
    # Convert to list of dicts
    samples = []
    for row in df.iter_rows(named=True):
        artists = row["artists"] if row["artists"] else []
        artists_str = ", ".join(artists) if artists else "Unknown"
        
        samples.append({
            "name": f"{row['title']} - {artists_str} ({row['year']})",
            "title": row["title"],
            "artists": artists,
            "year": row["year"],
            "lyrics": row["lyrics"] or "",
            "wikipedia_link": row.get("wikipedia_link"),
        })
    
    return samples


def get_lyrics_prefix(lyrics: str, mode: Literal["none", "first_line", "first_verse", "half"] = "none") -> str:
    """
    Extract a prefix of the lyrics to use as a seed hint.
    
    Args:
        lyrics: Full lyrics text
        mode: How much of the lyrics to include
            - "none": Empty string (no hint)
            - "first_line": First line only
            - "first_verse": First verse (up to first double newline or ~4 lines)
            - "half": First half of the lyrics
            
    Returns:
        Lyrics prefix string
    """
    if not lyrics or mode == "none":
        return ""
    
    lines = lyrics.strip().split("\n")
    
    if mode == "first_line":
        return lines[0] if lines else ""
    
    elif mode == "first_verse":
        # Find first blank line (verse boundary) or take first 4 lines
        verse_lines = []
        for line in lines[:8]:  # Max 8 lines to search
            if not line.strip():
                break
            verse_lines.append(line)
            if len(verse_lines) >= 4:
                break
        return "\n".join(verse_lines)
    
    elif mode == "half":
        half_point = len(lyrics) // 2
        # Try to break at a line boundary
        newline_pos = lyrics.find("\n", half_point)
        if newline_pos != -1 and newline_pos < half_point + 100:
            return lyrics[:newline_pos]
        return lyrics[:half_point]
    
    return ""


# Convenience functions for common use cases
def load_recent_songs(limit: int = 100) -> list[dict]:
    """Load recent songs (2020-2025)."""
    return load_billboard_songs(year_range=(2020, 2025), limit=limit, shuffle=True)


def load_classic_songs(limit: int = 100) -> list[dict]:
    """Load classic songs (1970-1989)."""
    return load_billboard_songs(year_range=(1970, 1989), limit=limit, shuffle=True)


def load_songs_by_decade(decade: int, limit: int = 100) -> list[dict]:
    """Load songs from a specific decade (e.g., 1990 for 1990s)."""
    return load_billboard_songs(
        year_range=(decade, decade + 9), 
        limit=limit, 
        shuffle=True
    )
