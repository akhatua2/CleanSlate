"""Data structures for generation-only mode (no metrics computed)."""

from dataclasses import dataclass, asdict
from typing import Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenData:
    """Per-token generation data."""

    token_id: int
    token_str: str
    logprob: float
    # Top-k alternatives for this position
    top_tokens: list[int]  # Token IDs of top K alternatives
    top_logprobs: list[float]  # Their log probabilities


@dataclass
class GenerationOutput:
    """
    Complete generation output with all data for later analysis.

    This is designed to be saved to Parquet/HF datasets format.
    """

    # Identification
    sample_id: str  # Unique ID for this sample
    sample_name: str  # Human-readable name
    generation_index: int  # Which of the N generations (0, 1, 2, ...)

    # Input
    seed: str  # The prompt/seed text
    original_target: Optional[str]  # Original target if doing memorization tests

    # Model info
    model_name: str
    backend: str
    device: str

    # Generation parameters
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int

    # Output
    generated_text: str
    num_tokens: int
    finish_reason: str  # "stop", "length", "eos", etc.

    # Token-level data
    tokens: list[int]  # Token IDs
    token_strings: list[str]  # Decoded token strings
    token_logprobs: list[float]  # Log probability of each token

    # Top-K alternatives per token (for analysis)
    # Format: list of dicts {token_id: int, logprob: float} for each position
    top_logprobs_data: Optional[list[list[dict]]] = None

    # Aggregate statistics (optional, can compute later)
    cumulative_logprob: Optional[float] = None
    mean_logprob: Optional[float] = None

    # Metadata
    timestamp: str
    generation_time_ms: float  # How long generation took

    def to_dict(self) -> dict:
        """Convert to dictionary for saving."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class GenerationBatch:
    """
    Container for multiple generations from the same seed.

    This groups N generations from the same seed for easier analysis.
    """

    sample_id: str
    sample_name: str
    seed: str
    original_target: Optional[str]

    model_name: str
    num_generations: int

    generations: list[GenerationOutput]

    def to_flat_records(self) -> list[dict]:
        """
        Flatten to list of records for Parquet.
        Each generation becomes one row.
        """
        return [gen.to_dict() for gen in self.generations]


def save_generations_parquet(
    generations: list[GenerationOutput],
    output_path: str,
):
    """
    Save generations to Parquet file.

    Args:
        generations: List of GenerationOutput objects
        output_path: Path to save .parquet file
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow required for Parquet export. Install with: pip install pyarrow"
        )

    # Convert to list of dicts
    records = [gen.to_dict() for gen in generations]

    # Create PyArrow table
    table = pa.Table.from_pylist(records)

    # Write to Parquet
    pq.write_table(table, output_path)
    logger.info("Saved %d generations to %s", len(generations), output_path)


def save_generations_hf_dataset(
    generations: list[GenerationOutput],
    output_path: str,
):
    """
    Save generations as HuggingFace dataset.

    Args:
        generations: List of GenerationOutput objects
        output_path: Path to save dataset directory
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "datasets required for HF export. Install with: pip install datasets"
        )

    # Convert to list of dicts
    records = [gen.to_dict() for gen in generations]

    # Create dataset
    dataset = Dataset.from_list(records)

    # Save to disk
    dataset.save_to_disk(output_path)
    logger.info("Saved %d generations to %s", len(generations), output_path)

    # Also save as Parquet for easy access
    parquet_path = f"{output_path}.parquet"
    dataset.to_parquet(parquet_path)
    logger.info("Also saved as Parquet: %s", parquet_path)


def load_generations_parquet(path: str) -> list[GenerationOutput]:
    """Load generations from Parquet file."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow required. Install with: pip install pyarrow")

    table = pq.read_table(path)
    records = table.to_pylist()

    return [GenerationOutput(**record) for record in records]


def load_generations_hf_dataset(path: str):
    """Load generations from HuggingFace dataset."""
    try:
        from datasets import load_from_disk
    except ImportError:
        raise ImportError("datasets required. Install with: pip install datasets")

    return load_from_disk(path)
