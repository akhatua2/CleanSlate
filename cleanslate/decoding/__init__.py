"""Decoding strategies for text generation.

This module provides pluggable decoding strategies:
- GreedyDecoding: Always pick the highest probability token
- BeamSearchDecoding: Maintain top-k hypotheses at each step

Usage:
    from cleanslate.decoding import GreedyDecoding, BeamSearchDecoding, create_strategy

    # Direct instantiation
    strategy = GreedyDecoding()
    strategy = BeamSearchDecoding(beam_width=5, length_penalty=1.0)

    # Factory function
    strategy = create_strategy("greedy")
    strategy = create_strategy("beam", beam_width=5)
"""

from cleanslate.decoding.base import (
    DecodingOutput,
    DecodingStrategy,
    GenerationSequence,
)
from cleanslate.decoding.greedy import GreedyDecoding
from cleanslate.decoding.beam import BeamSearchDecoding

__all__ = [
    # Base classes
    "DecodingStrategy",
    "DecodingOutput",
    "GenerationSequence",
    # Strategies
    "GreedyDecoding",
    "BeamSearchDecoding",
    # Factory
    "create_strategy",
    "STRATEGIES",
]

# Registry of available strategies
STRATEGIES: dict[str, type[DecodingStrategy]] = {
    "greedy": GreedyDecoding,
    "beam": BeamSearchDecoding,
}


def create_strategy(name: str, **kwargs) -> DecodingStrategy:
    """Create a decoding strategy by name.

    Args:
        name: Strategy name ("greedy", "beam")
        **kwargs: Strategy-specific parameters
            - beam: beam_width (int), length_penalty (float)

    Returns:
        Instantiated DecodingStrategy

    Raises:
        ValueError: If strategy name is unknown

    Example:
        >>> strategy = create_strategy("greedy")
        >>> strategy = create_strategy("beam", beam_width=10, length_penalty=0.8)
    """
    if name not in STRATEGIES:
        available = ", ".join(STRATEGIES.keys())
        raise ValueError(f"Unknown strategy: {name}. Available: {available}")

    strategy_cls = STRATEGIES[name]
    return strategy_cls(**kwargs)
