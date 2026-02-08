"""Sample texts for testing memorization."""

from .shakespeare import SHAKESPEARE_SAMPLES
from .books import BOOK_SAMPLES
from .songs import SONG_SAMPLES
from .billboard import (
    load_billboard_songs,
    get_lyrics_prefix,
    load_recent_songs,
    load_classic_songs,
    load_songs_by_decade,
)

__all__ = [
    "SHAKESPEARE_SAMPLES", 
    "BOOK_SAMPLES", 
    "SONG_SAMPLES",
    # Billboard data loader
    "load_billboard_songs",
    "get_lyrics_prefix",
    "load_recent_songs",
    "load_classic_songs",
    "load_songs_by_decade",
]
