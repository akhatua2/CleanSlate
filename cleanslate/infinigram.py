"""Infini-gram API client for querying n-gram counts in training corpora.

The infini-gram API provides fast n-gram lookup across large training datasets.
This is useful for checking how often memorized windows appear in training data.

Documentation: https://infini-gram.readthedocs.io/en/latest/api.html
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)

# Available corpus indices (from API docs)
# See: https://infini-gram.readthedocs.io/en/latest/api.html
CORPUS_INDICES = {
    # Large pretraining corpora
    "rpj": "v4_rpj_llama_s4",              # RedPajama (1.4T tokens)
    "pile": "v4_piletrain_llama",          # The Pile train (383B tokens)
    "dolma": "v4_dolma-v1_7_llama",        # Dolma v1.7 (2.6T tokens)
    "dclm": "v4_dclm-baseline_llama",      # DCLM baseline (4.3T tokens)
    "c4": "v4_c4train_llama",              # C4 train (198B tokens)
    # OLMo training data
    "olmomix": "v4_olmo-mix-1124_llama",   # OLMo-mix-1124 (4.6T tokens)
    # Smaller/validation sets
    "dolma-sample": "v4_dolma-v1_6-sample_llama",  # Dolma sample (9B tokens)
    "pile-val": "v4_pileval_llama",        # Pile validation (394M tokens)
}

# Default corpora to query (major pretraining sources)
DEFAULT_CORPORA = ["rpj", "pile", "dolma", "dclm", "c4"]

API_URL = "https://api.infini-gram.io/"


@dataclass
class CorpusCount:
    """Count result for a single corpus."""

    corpus: str
    index: str
    count: int
    approx_count: int | None = None
    error: str | None = None


@dataclass
class WindowFrequency:
    """Frequency results for a window across multiple corpora."""

    window_text: str
    window_idx: int
    counts: dict[str, CorpusCount]
    total_count: int  # Sum across all corpora

    @property
    def found_in_any(self) -> bool:
        return self.total_count > 0

    @property
    def top_corpus(self) -> str | None:
        """Return the corpus with highest count."""
        if not self.counts:
            return None
        return max(self.counts.items(), key=lambda x: x[1].count)[0]


class RateLimitError(Exception):
    """Raised when API returns 429 rate limit error."""
    pass


class TransientError(Exception):
    """Raised for transient errors that should be retried."""
    pass


@retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=2, min=5, max=120),
    retry=retry_if_exception_type((RateLimitError, TransientError, requests.exceptions.Timeout)),
    reraise=True,
)
def _query_with_retry(payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    """Internal query function with tenacity retry logic."""
    response = requests.post(
        API_URL,
        json=payload,
        timeout=timeout,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code == 429:
        raise RateLimitError("Rate limited by infini-gram API")
    if response.status_code in (500, 502, 503, 504):
        raise TransientError(f"Server error: {response.status_code}")
    if response.status_code == 403:
        # IP might be temporarily blocked - treat as rate limit
        raise RateLimitError("Forbidden - possible IP block")

    response.raise_for_status()
    return response.json()


def query_count(
    text: str,
    index: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Query infini-gram API for n-gram count.

    Args:
        text: The text to search for
        index: The corpus index to query (e.g., "v4_rpj_llama_s4")
        timeout: Request timeout in seconds

    Returns:
        API response dict with 'count' field
    """
    payload = {
        "index": index,
        "query_type": "count",
        "query": text,
    }

    try:
        return _query_with_retry(payload, timeout)
    except Exception as e:
        logger.warning(f"Query failed after retries: {e}")
        return {"error": str(e), "count": 0}


def query_window(
    text: str,
    corpora: list[str] | None = None,
    window_idx: int = 0,
) -> WindowFrequency:
    """Query a text window across multiple corpora.

    Args:
        text: The window text to search for
        corpora: List of corpus short names (default: DEFAULT_CORPORA)
        window_idx: Window index for tracking

    Returns:
        WindowFrequency with counts per corpus
    """
    if corpora is None:
        corpora = DEFAULT_CORPORA

    counts: dict[str, CorpusCount] = {}
    total = 0

    for corpus in corpora:
        if corpus not in CORPUS_INDICES:
            counts[corpus] = CorpusCount(
                corpus=corpus,
                index="unknown",
                count=0,
                error=f"Unknown corpus: {corpus}",
            )
            continue

        index = CORPUS_INDICES[corpus]
        result = query_count(text, index)

        if "error" in result:
            counts[corpus] = CorpusCount(
                corpus=corpus,
                index=index,
                count=0,
                error=str(result["error"]),
            )
        else:
            count = result.get("count", 0)
            counts[corpus] = CorpusCount(
                corpus=corpus,
                index=index,
                count=count,
                approx_count=result.get("approx_count"),
            )
            total += count

    return WindowFrequency(
        window_text=text,
        window_idx=window_idx,
        counts=counts,
        total_count=total,
    )


def batch_query_windows(
    windows: list[tuple[int, str]],  # (window_idx, text)
    corpora: list[str] | None = None,
    rate_limit: float = 0.1,  # seconds between requests
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[WindowFrequency]:
    """Query multiple windows across corpora with rate limiting.

    Args:
        windows: List of (window_idx, text) tuples
        corpora: List of corpus short names
        rate_limit: Delay between API calls (be nice to the API)
        progress_callback: Optional callback(current, total) for progress

    Returns:
        List of WindowFrequency results
    """
    results = []

    for i, (idx, text) in enumerate(windows):
        if progress_callback:
            progress_callback(i, len(windows))

        result = query_window(text, corpora=corpora, window_idx=idx)
        results.append(result)

        # Rate limiting
        if i < len(windows) - 1 and rate_limit > 0:
            time.sleep(rate_limit)

    return results


def format_count(count: int) -> str:
    """Format count with K/M suffixes."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    else:
        return str(count)
