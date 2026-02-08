"""Abstract base class for inference backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class GenerationResult:
    """Result of a text generation with captured logits.

    Designed for HuggingFace parquet-compatible output.
    Each instance represents one generation for one seed.
    """

    # Generation output
    generated_text: str
    generated_token_ids: list[int]
    generated_tokens: list[str]

    # Token-level probability data
    token_log_probs: list[float]  # Log prob of each generated token
    top_k_tokens: list[list[int]] = field(
        default_factory=list
    )  # Top-K token IDs per position
    top_k_probs: list[list[float]] = field(
        default_factory=list
    )  # Top-K log probs per position

    # Generation parameters (for reproducibility)
    max_new_tokens: int = 0
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    do_sample: bool = True

    # Timing
    generation_time_ms: float = 0.0


@dataclass
class ProbabilityResult:
    """Result of a probability computation.

    Decoding options (passed via **kwargs) affect how probabilities are computed:

    - temperature: Scales logits before softmax. Higher = more uniform distribution,
                   lower = more peaked. Default 1.0 = no scaling.
    - top_k: Only consider top-k tokens at each position. If target token is outside
             top-k, its probability is 0 (log_prob = -inf). None = consider all tokens.
    - top_p: (future) Nucleus sampling threshold.
    - etc.
    """

    log_prob: float  # Total log probability: sum of log P(token_i | context)
    perplexity: float  # exp(-log_prob / num_tokens)
    num_tokens: int  # Number of target tokens
    token_log_probs: list[float]  # Per-token log probabilities
    token_ranks: list[int] | None = None  # Rank of each target token (1 = top choice)
    target_tokens: list[str] | None = None  # Decoded target tokens
    predicted_tokens: list[str] | None = None  # Model's top-1 predictions
    decoding_options: dict | None = None  # Options used (temperature, top_k, etc.)

    @property
    def avg_log_prob(self) -> float:
        """Average log probability per token."""
        if self.num_tokens == 0:
            return 0.0
        return self.log_prob / self.num_tokens

    @property
    def rank_1_fraction(self) -> float:
        """Fraction of tokens where model's top choice matched target."""
        if self.token_ranks is None or len(self.token_ranks) == 0:
            return 0.0
        return sum(1 for r in self.token_ranks if r == 1) / len(self.token_ranks)

    @property
    def avg_rank(self) -> float:
        """Average rank of target tokens (lower = more memorized)."""
        if self.token_ranks is None or len(self.token_ranks) == 0:
            return float("inf")
        return sum(self.token_ranks) / len(self.token_ranks)


class InferenceBackend(ABC):
    """
    Abstract base class for inference backends.

    Backends compute P(target | seed) by:
    1. Tokenizing seed + target
    2. Running forward pass to get logits
    3. Extracting log probabilities for target tokens
    4. Summing log probs in log-space
    """

    @property
    @abstractmethod
    def device(self) -> str:
        """Return the device this backend is using (cpu, cuda, mps, etc)."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/path."""
        pass

    @abstractmethod
    def compute(
        self,
        seed: str,
        target: str,
        **decoding_options,
    ) -> ProbabilityResult:
        """
        Compute P(target | seed) under a given decoding scheme.

        Args:
            seed: The prefix/prompt text
            target: The continuation text whose probability we want
            **decoding_options: Decoding parameters (backend-specific):
                - temperature (float): Logit scaling. Higher = flatter. Default 1.0.
                - top_k (int): Only consider top-k tokens. Default None (all).
                - top_p (float): Nucleus sampling threshold. Default None (disabled).
                - repetition_penalty (float): Penalize repeated tokens. Default 1.0.

        Returns:
            ProbabilityResult with log probability, perplexity, etc.
        """
        pass

    def compute_batch(
        self,
        pairs: list[tuple[str, str]],
        **decoding_options,
    ) -> list[ProbabilityResult]:
        """
        Compute probabilities for multiple (seed, target) pairs.

        Default implementation just loops. Backends can override for efficiency.
        """
        return [
            self.compute(seed, target, **decoding_options) for seed, target in pairs
        ]

    @abstractmethod
    def generate(
        self,
        seed: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
        capture_top_k: int = 50,
    ) -> GenerationResult:
        """
        Generate text from seed and capture token-level probabilities.

        Args:
            seed: The prefix/prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k filtering for sampling
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample (vs greedy decode)
            capture_top_k: Number of top tokens/probs to capture per position

        Returns:
            GenerationResult with generated text and probability data
        """
        pass

    def generate_batch(
        self,
        seeds: list[str],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
        capture_top_k: int = 50,
    ) -> list[GenerationResult]:
        """
        Generate text from multiple seeds in a batch.

        Default implementation loops. Backends can override for efficiency.

        Args:
            seeds: List of prefix/prompt texts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering for sampling
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample (vs greedy decode)
            capture_top_k: Number of top tokens/probs to capture per position

        Returns:
            List of GenerationResult, one per seed
        """
        return [
            self.generate(
                seed,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                do_sample,
                capture_top_k,
            )
            for seed in seeds
        ]
