"""Unified memorization evaluation: compute + generate + metrics.

This module provides a single interface for evaluating memorization that combines:
1. Probability computation (p_z) via teacher forcing
2. Text generation and comparison (BLEU, ROUGE, exact match)
3. Sliding window extraction scans

Based on the methodology from:
- Hayes et al. "Measuring memorization in language models"
- Cooper et al. "Extracting memorized pieces of copyrighted books"
"""

import logging
import math
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal
import json

from cleanslate.backends import create_backend
from cleanslate.backends.base import InferenceBackend, ProbabilityResult, GenerationResult
from cleanslate.decoding import create_strategy, DecodingOutput
from cleanslate.metrics.probability import compute_extraction_probability, ExtractionResult
from cleanslate.metrics.bleu import BLEU1, BLEU2
from cleanslate.metrics.rouge import ROUGE1, ROUGEL

logger = logging.getLogger(__name__)

BackendType = Literal["huggingface", "sglang"]


@dataclass
class MemorizationConfig:
    """Configuration for memorization evaluation."""

    # Compute settings (for p_z calculation)
    compute_top_k: int = 40  # Paper uses k=40 for top-k decoding
    tau_min: float = 0.001  # 0.1% threshold for "memorized"

    # Generation settings
    do_generate: bool = True  # Whether to also generate and compare
    num_generations: int = 1  # Generations per seed
    gen_temperature: float = 0.0  # 0 = greedy/deterministic
    gen_top_k: int | None = 40
    gen_top_p: float | None = None
    gen_min_p: float | None = None
    gen_max_new_tokens: int = 50
    gen_do_sample: bool = False
    gen_repetition_penalty: float | None = None
    capture_top_k: int = 50  # Top-k alternatives to save per token

    # Decoding strategy settings
    decode_strategy: str = "greedy"  # "greedy" or "beam"
    beam_width: int = 5  # Number of beams for beam search
    length_penalty: float = 1.0  # Length penalty for beam search

    # Reproduction detection
    reproduction_threshold: float = 0.7  # ROUGE-L threshold for "reproduced"

    # Sliding window settings
    window_prefix_len: int = 50  # Tokens
    window_suffix_len: int = 50  # Tokens
    window_stride: int = 25  # Tokens

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MemorizationResult:
    """Complete memorization evaluation for a single (seed, target) pair."""

    # Identifiers
    sample_id: str
    sample_name: str
    generation_idx: int = 0
    window_idx: int | None = None  # For sliding window

    # Input
    seed: str = ""
    target: str = ""

    # Position (for sliding window)
    window_start_char: int | None = None
    window_end_char: int | None = None
    window_start_token: int | None = None

    # === Probability metrics (from compute) ===
    p_z: float = 0.0  # Extraction probability
    log_prob: float = 0.0  # Sum of token log probs
    perplexity: float = 0.0
    is_memorized: bool = False  # p_z >= tau_min
    prompts_to_extract: float = float("inf")  # 1/p_z
    rank_1_fraction: float = 0.0  # % tokens that were top-1
    avg_rank: float = 0.0
    num_target_tokens: int = 0

    # Per-token probability data
    target_token_ids: list[int] = field(default_factory=list)
    target_log_probs: list[float] = field(default_factory=list)
    target_ranks: list[int] = field(default_factory=list)

    # === Generation output ===
    generated_text: str | None = None
    generated_token_ids: list[int] = field(default_factory=list)
    generated_tokens: list[str] = field(default_factory=list)
    gen_log_probs: list[float] = field(default_factory=list)
    gen_top_k_ids: list[list[int]] = field(default_factory=list)
    gen_top_k_probs: list[list[float]] = field(default_factory=list)
    num_generated_tokens: int = 0

    # === Comparison metrics ===
    exact_match: bool | None = None
    bleu_1: float | None = None
    bleu_2: float | None = None
    rouge_1: float | None = None
    rouge_l: float | None = None

    # === Reproduction detection (beam search) ===
    is_reproduced: bool = False  # Any beam has ROUGE-L > threshold
    best_beam_rouge_l: float | None = None  # Max ROUGE-L across all beams
    best_beam_idx: int | None = None  # Which beam had best ROUGE-L
    num_beams: int = 1  # Number of beams generated
    all_beam_texts: list[str] = field(default_factory=list)  # All beam outputs
    all_beam_rouge_l: list[float] = field(default_factory=list)  # ROUGE-L per beam

    # === Timing ===
    compute_time_ms: float = 0.0
    generate_time_ms: float = 0.0
    row_timestamp: str = ""

    # === Optional metadata ===
    song_id: str | None = None
    song_title: str | None = None
    song_artists: list[str] = field(default_factory=list)
    song_year: int | None = None
    prompt_template: str | None = None
    seed_mode: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WindowInfo:
    """Info about a sliding window position."""

    prefix: str
    suffix: str
    start_char: int
    end_char: int
    start_token: int
    prefix_token_ids: list[int]
    suffix_token_ids: list[int]


@dataclass
class SlidingWindowResult:
    """Full sliding window scan over a text."""

    text_name: str
    model_name: str

    # Config used
    prefix_len: int
    suffix_len: int
    stride: int
    tau_min: float

    # Results
    windows: list[MemorizationResult]

    # Aggregates
    total_windows: int = 0
    memorized_windows: int = 0
    coverage: float = 0.0  # % of characters extractable
    max_p_z: float = 0.0
    mean_p_z: float = 0.0
    median_p_z: float = 0.0

    # For heatmap
    char_to_max_pz: list[float] = field(default_factory=list)

    # Timing
    total_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "text_name": self.text_name,
            "model_name": self.model_name,
            "config": {
                "prefix_len": self.prefix_len,
                "suffix_len": self.suffix_len,
                "stride": self.stride,
                "tau_min": self.tau_min,
            },
            "summary": {
                "total_windows": self.total_windows,
                "memorized_windows": self.memorized_windows,
                "coverage": self.coverage,
                "max_p_z": self.max_p_z,
                "mean_p_z": self.mean_p_z,
                "median_p_z": self.median_p_z,
            },
            "timing": {
                "total_time_ms": self.total_time_ms,
            },
            "windows": [w.to_dict() for w in self.windows],
        }


@dataclass
class ExperimentSummary:
    """Summary statistics for an experiment."""

    experiment_id: str
    timestamp: str
    model_name: str
    dataset_name: str
    num_samples: int

    # Memorization stats
    memorized_count: int
    memorized_fraction: float
    exact_match_count: int
    exact_match_fraction: float

    # p_z statistics
    p_z_mean: float
    p_z_std: float
    p_z_min: float
    p_z_max: float
    p_z_median: float

    # Perplexity statistics
    ppl_mean: float
    ppl_std: float

    # Similarity statistics
    bleu_1_mean: float | None
    rouge_l_mean: float | None

    # Timing
    total_seconds: float
    samples_per_second: float

    def to_dict(self) -> dict:
        return asdict(self)


class MemorizationEvaluator:
    """Unified memorization evaluation: compute + generate + metrics."""

    def __init__(
        self,
        model: str,
        backend: BackendType | InferenceBackend = "sglang",
        config: MemorizationConfig | None = None,
        device: str | None = None,
        **backend_kwargs,
    ):
        """
        Initialize the evaluator.

        Args:
            model: Model name/path
            backend: Backend type or instance
            config: Evaluation configuration
            device: Device override
            **backend_kwargs: Additional backend arguments
        """
        self.model_name = model
        self.config = config or MemorizationConfig()
        self._device = device
        self._backend_kwargs = backend_kwargs

        if isinstance(backend, InferenceBackend):
            self._backend = backend
            self._backend_type = "custom"
        else:
            self._backend_type = backend
            self._backend: InferenceBackend | None = None

        # Metrics (initialized once)
        self._bleu1 = BLEU1()
        self._bleu2 = BLEU2()
        self._rouge1 = ROUGE1()
        self._rougel = ROUGEL()

    def _get_backend(self) -> InferenceBackend:
        """Lazy-load the backend."""
        if self._backend is None:
            self._backend = create_backend(
                self._backend_type,
                self.model_name,
                device=self._device,
                **self._backend_kwargs,
            )
        return self._backend

    @property
    def tokenizer(self):
        """Access the backend's tokenizer."""
        return self._get_backend().tokenizer

    def evaluate_single(
        self,
        seed: str,
        target: str,
        sample_id: str = "sample",
        sample_name: str = "sample",
        generation_idx: int = 0,
        metadata: dict | None = None,
    ) -> MemorizationResult:
        """
        Full evaluation of a single (seed, target) pair.

        1. Compute p_z via teacher forcing
        2. Optionally generate and compare
        3. Return unified result
        """
        backend = self._get_backend()
        config = self.config
        metadata = metadata or {}

        # Step 1: Compute P(target | seed)
        compute_start = time.time()
        prob_result = backend.compute(seed, target)
        compute_time_ms = (time.time() - compute_start) * 1000

        extraction = compute_extraction_probability(prob_result.token_log_probs)

        # Step 2: Generate (if enabled)
        generated_text = None
        generated_token_ids: list[int] = []
        generated_tokens: list[str] = []
        gen_log_probs: list[float] = []
        gen_top_k_ids: list[list[int]] = []
        gen_top_k_probs: list[list[float]] = []
        generate_time_ms = 0.0
        exact_match = None
        bleu_1 = bleu_2 = rouge_1 = rouge_l = None

        # Reproduction detection fields
        is_reproduced = False
        best_beam_rouge_l: float | None = None
        best_beam_idx: int | None = None
        num_beams = 1
        all_beam_texts: list[str] = []
        all_beam_rouge_l: list[float] = []

        if config.do_generate:
            gen_start = time.time()

            # Create decoding strategy
            if config.decode_strategy == "beam":
                strategy = create_strategy(
                    "beam",
                    beam_width=config.beam_width,
                    length_penalty=config.length_penalty,
                )
            else:
                strategy = create_strategy("greedy")

            # Generate using strategy
            decode_output = backend.generate(
                seed,
                strategy=strategy,
                max_new_tokens=config.gen_max_new_tokens,
                capture_top_k=config.capture_top_k,
            )
            generate_time_ms = (time.time() - gen_start) * 1000

            # Get best sequence (top beam or greedy result)
            best_seq = decode_output.sequences[0]
            generated_text = best_seq.generated_text
            generated_token_ids = best_seq.generated_token_ids
            generated_tokens = best_seq.generated_tokens
            gen_log_probs = best_seq.token_log_probs
            gen_top_k_ids = best_seq.top_k_tokens
            gen_top_k_probs = best_seq.top_k_probs
            num_beams = len(decode_output.sequences)

            # Step 3: Compare all beams and find best ROUGE-L
            for beam_idx, seq in enumerate(decode_output.sequences):
                beam_text = seq.generated_text
                all_beam_texts.append(beam_text)
                _, _, _, beam_rouge_l = self._compute_similarity(beam_text, target)
                all_beam_rouge_l.append(beam_rouge_l if beam_rouge_l is not None else 0.0)

            # Find best beam by ROUGE-L
            if all_beam_rouge_l:
                best_beam_idx = max(range(len(all_beam_rouge_l)), key=lambda i: all_beam_rouge_l[i])
                best_beam_rouge_l = all_beam_rouge_l[best_beam_idx]
                is_reproduced = best_beam_rouge_l >= config.reproduction_threshold

            # Metrics for top beam
            exact_match = self._check_exact_match(generated_text, target)
            bleu_1, bleu_2, rouge_1, rouge_l = self._compute_similarity(
                generated_text, target
            )

        return MemorizationResult(
            sample_id=sample_id,
            sample_name=sample_name,
            generation_idx=generation_idx,
            seed=seed,
            target=target,
            # Probability metrics
            p_z=extraction.prob,
            log_prob=extraction.log_prob,
            perplexity=extraction.perplexity,
            is_memorized=(extraction.prob >= config.tau_min),
            prompts_to_extract=(
                1 / extraction.prob if extraction.prob > 0 else float("inf")
            ),
            num_target_tokens=extraction.num_tokens,
            target_log_probs=prob_result.token_log_probs,
            target_ranks=prob_result.token_ranks or [],
            rank_1_fraction=prob_result.rank_1_fraction,
            avg_rank=prob_result.avg_rank,
            # Generation output
            generated_text=generated_text,
            generated_token_ids=generated_token_ids,
            generated_tokens=generated_tokens,
            gen_log_probs=gen_log_probs,
            gen_top_k_ids=gen_top_k_ids,
            gen_top_k_probs=gen_top_k_probs,
            num_generated_tokens=len(generated_token_ids),
            # Comparison metrics
            exact_match=exact_match,
            bleu_1=bleu_1,
            bleu_2=bleu_2,
            rouge_1=rouge_1,
            rouge_l=rouge_l,
            # Reproduction detection
            is_reproduced=is_reproduced,
            best_beam_rouge_l=best_beam_rouge_l,
            best_beam_idx=best_beam_idx,
            num_beams=num_beams,
            all_beam_texts=all_beam_texts,
            all_beam_rouge_l=all_beam_rouge_l,
            # Timing
            compute_time_ms=compute_time_ms,
            generate_time_ms=generate_time_ms,
            row_timestamp=datetime.now().isoformat(),
            # Metadata
            song_id=metadata.get("song_id"),
            song_title=metadata.get("song_title"),
            song_artists=metadata.get("song_artists", []),
            song_year=metadata.get("song_year"),
            prompt_template=metadata.get("prompt_template"),
            seed_mode=metadata.get("seed_mode"),
        )

    def evaluate_batch(
        self,
        samples: list[dict],
        show_progress: bool = True,
    ) -> list[MemorizationResult]:
        """
        Batch evaluation with efficient batching.

        Args:
            samples: List of dicts with keys: seed, target, name (optional), etc.
            show_progress: Whether to show progress bar

        Returns:
            List of MemorizationResult
        """
        from tqdm import tqdm

        backend = self._get_backend()
        config = self.config
        results: list[MemorizationResult] = []

        # Extract pairs
        pairs = [(s["seed"], s["target"]) for s in samples]

        # Batch compute
        logger.info("Computing p_z for %d samples...", len(pairs))
        compute_start = time.time()
        prob_results = backend.compute_batch(pairs)
        compute_time_total = (time.time() - compute_start) * 1000
        compute_time_per = compute_time_total / len(pairs) if pairs else 0

        # Batch generate (if enabled)
        decode_outputs: list[DecodingOutput] | None = None
        generate_time_per = 0.0

        if config.do_generate:
            logger.info("Generating for %d samples...", len(pairs))
            seeds = [s["seed"] for s in samples]

            # Create decoding strategy
            if config.decode_strategy == "beam":
                strategy = create_strategy(
                    "beam",
                    beam_width=config.beam_width,
                    length_penalty=config.length_penalty,
                )
            else:
                strategy = create_strategy("greedy")

            gen_start = time.time()
            decode_outputs = backend.generate_batch(
                seeds,
                strategy=strategy,
                max_new_tokens=config.gen_max_new_tokens,
                capture_top_k=config.capture_top_k,
            )
            generate_time_total = (time.time() - gen_start) * 1000
            generate_time_per = generate_time_total / len(pairs) if pairs else 0

        # Combine results
        iterator = enumerate(zip(samples, prob_results))
        if show_progress:
            iterator = tqdm(list(iterator), desc="Processing results")

        for i, (sample, prob_result) in iterator:
            seed = sample["seed"]
            target = sample["target"]
            sample_name = sample.get("name", sample.get("sample_name", f"sample_{i}"))
            sample_id = sample.get("sample_id", sample_name)

            extraction = compute_extraction_probability(prob_result.token_log_probs)

            # Generation data
            generated_text = None
            generated_token_ids: list[int] = []
            generated_tokens: list[str] = []
            gen_log_probs: list[float] = []
            gen_top_k_ids: list[list[int]] = []
            gen_top_k_probs: list[list[float]] = []
            exact_match = None
            bleu_1 = bleu_2 = rouge_1 = rouge_l = None

            # Reproduction detection fields
            is_reproduced = False
            best_beam_rouge_l: float | None = None
            best_beam_idx: int | None = None
            num_beams = 1
            all_beam_texts: list[str] = []
            all_beam_rouge_l: list[float] = []

            if decode_outputs:
                decode_output = decode_outputs[i]
                best_seq = decode_output.sequences[0]
                generated_text = best_seq.generated_text
                generated_token_ids = best_seq.generated_token_ids
                generated_tokens = best_seq.generated_tokens
                gen_log_probs = best_seq.token_log_probs
                gen_top_k_ids = best_seq.top_k_tokens
                gen_top_k_probs = best_seq.top_k_probs
                num_beams = len(decode_output.sequences)

                # Compare all beams and find best ROUGE-L
                for beam_idx, seq in enumerate(decode_output.sequences):
                    beam_text = seq.generated_text
                    all_beam_texts.append(beam_text)
                    _, _, _, beam_rouge_l = self._compute_similarity(beam_text, target)
                    all_beam_rouge_l.append(beam_rouge_l if beam_rouge_l is not None else 0.0)

                # Find best beam by ROUGE-L
                if all_beam_rouge_l:
                    best_beam_idx = max(range(len(all_beam_rouge_l)), key=lambda j: all_beam_rouge_l[j])
                    best_beam_rouge_l = all_beam_rouge_l[best_beam_idx]
                    is_reproduced = best_beam_rouge_l >= config.reproduction_threshold

                exact_match = self._check_exact_match(generated_text, target)
                bleu_1, bleu_2, rouge_1, rouge_l = self._compute_similarity(
                    generated_text, target
                )

            results.append(
                MemorizationResult(
                    sample_id=sample_id,
                    sample_name=sample_name,
                    generation_idx=0,
                    seed=seed,
                    target=target,
                    # Probability metrics
                    p_z=extraction.prob,
                    log_prob=extraction.log_prob,
                    perplexity=extraction.perplexity,
                    is_memorized=(extraction.prob >= config.tau_min),
                    prompts_to_extract=(
                        1 / extraction.prob if extraction.prob > 0 else float("inf")
                    ),
                    num_target_tokens=extraction.num_tokens,
                    target_log_probs=prob_result.token_log_probs,
                    target_ranks=prob_result.token_ranks or [],
                    rank_1_fraction=prob_result.rank_1_fraction,
                    avg_rank=prob_result.avg_rank,
                    # Generation output
                    generated_text=generated_text,
                    generated_token_ids=generated_token_ids,
                    generated_tokens=generated_tokens,
                    gen_log_probs=gen_log_probs,
                    gen_top_k_ids=gen_top_k_ids,
                    gen_top_k_probs=gen_top_k_probs,
                    num_generated_tokens=len(generated_token_ids),
                    # Comparison metrics
                    exact_match=exact_match,
                    bleu_1=bleu_1,
                    bleu_2=bleu_2,
                    rouge_1=rouge_1,
                    rouge_l=rouge_l,
                    # Reproduction detection
                    is_reproduced=is_reproduced,
                    best_beam_rouge_l=best_beam_rouge_l,
                    best_beam_idx=best_beam_idx,
                    num_beams=num_beams,
                    all_beam_texts=all_beam_texts,
                    all_beam_rouge_l=all_beam_rouge_l,
                    # Timing
                    compute_time_ms=compute_time_per,
                    generate_time_ms=generate_time_per,
                    row_timestamp=datetime.now().isoformat(),
                    # Metadata
                    song_id=sample.get("song_id"),
                    song_title=sample.get("song_title"),
                    song_artists=sample.get("song_artists", []),
                    song_year=sample.get("song_year"),
                    prompt_template=sample.get("prompt_template"),
                    seed_mode=sample.get("seed_mode"),
                )
            )

        return results

    def sliding_window(
        self,
        text: str,
        name: str = "text",
        show_progress: bool = True,
    ) -> SlidingWindowResult:
        """
        Sliding window extraction scan over full text.

        Args:
            text: Full text to scan
            name: Name for identification
            show_progress: Whether to show progress bar

        Returns:
            SlidingWindowResult with per-window p_z and coverage
        """
        from tqdm import tqdm

        backend = self._get_backend()
        config = self.config
        start_time = time.time()

        # Generate windows
        windows = self._generate_windows(text)
        logger.info("Generated %d windows for '%s'", len(windows), name)

        if not windows:
            return SlidingWindowResult(
                text_name=name,
                model_name=self.model_name,
                prefix_len=config.window_prefix_len,
                suffix_len=config.window_suffix_len,
                stride=config.window_stride,
                tau_min=config.tau_min,
                windows=[],
                total_windows=0,
            )

        # Batch compute all windows
        pairs = [(w.prefix, w.suffix) for w in windows]
        logger.info("Computing p_z for %d windows...", len(pairs))
        prob_results = backend.compute_batch(pairs)

        # Build window results
        window_results: list[MemorizationResult] = []
        iterator = enumerate(zip(windows, prob_results))
        if show_progress:
            iterator = tqdm(list(iterator), desc="Processing windows")

        for i, (window, prob_result) in iterator:
            extraction = compute_extraction_probability(prob_result.token_log_probs)

            window_results.append(
                MemorizationResult(
                    sample_id=f"{name}_window_{i}",
                    sample_name=name,
                    generation_idx=0,
                    window_idx=i,
                    seed=window.prefix,
                    target=window.suffix,
                    window_start_char=window.start_char,
                    window_end_char=window.end_char,
                    window_start_token=window.start_token,
                    # Probability metrics
                    p_z=extraction.prob,
                    log_prob=extraction.log_prob,
                    perplexity=extraction.perplexity,
                    is_memorized=(extraction.prob >= config.tau_min),
                    prompts_to_extract=(
                        1 / extraction.prob if extraction.prob > 0 else float("inf")
                    ),
                    num_target_tokens=extraction.num_tokens,
                    target_log_probs=prob_result.token_log_probs,
                    target_ranks=prob_result.token_ranks or [],
                    rank_1_fraction=prob_result.rank_1_fraction,
                    avg_rank=prob_result.avg_rank,
                    # Timing
                    row_timestamp=datetime.now().isoformat(),
                )
            )

        # Compute aggregates
        p_z_values = [w.p_z for w in window_results]
        memorized_windows = sum(1 for w in window_results if w.is_memorized)
        coverage = self._compute_coverage(window_results, len(text))
        char_to_max_pz = self._compute_heatmap(window_results, len(text))

        total_time_ms = (time.time() - start_time) * 1000

        return SlidingWindowResult(
            text_name=name,
            model_name=self.model_name,
            prefix_len=config.window_prefix_len,
            suffix_len=config.window_suffix_len,
            stride=config.window_stride,
            tau_min=config.tau_min,
            windows=window_results,
            total_windows=len(window_results),
            memorized_windows=memorized_windows,
            coverage=coverage,
            max_p_z=max(p_z_values) if p_z_values else 0.0,
            mean_p_z=sum(p_z_values) / len(p_z_values) if p_z_values else 0.0,
            median_p_z=sorted(p_z_values)[len(p_z_values) // 2] if p_z_values else 0.0,
            char_to_max_pz=char_to_max_pz,
            total_time_ms=total_time_ms,
        )

    def _generate_windows(self, text: str) -> list[WindowInfo]:
        """Generate sliding windows over text."""
        config = self.config
        tokenizer = self.tokenizer

        # Tokenize entire text
        tokens = tokenizer.encode(text, add_special_tokens=False)
        total_tokens = len(tokens)

        window_size = config.window_prefix_len + config.window_suffix_len
        if total_tokens < window_size:
            logger.warning(
                "Text too short (%d tokens) for window size (%d). Skipping.",
                total_tokens,
                window_size,
            )
            return []

        windows: list[WindowInfo] = []
        char_offset = 0

        for start_token in range(
            0, total_tokens - window_size + 1, config.window_stride
        ):
            prefix_end = start_token + config.window_prefix_len
            suffix_end = prefix_end + config.window_suffix_len

            prefix_ids = tokens[start_token:prefix_end]
            suffix_ids = tokens[prefix_end:suffix_end]

            prefix_text = tokenizer.decode(prefix_ids)
            suffix_text = tokenizer.decode(suffix_ids)

            # Calculate character positions
            prefix_before = tokenizer.decode(tokens[:start_token])
            start_char = len(prefix_before)
            end_char = start_char + len(prefix_text) + len(suffix_text)

            windows.append(
                WindowInfo(
                    prefix=prefix_text,
                    suffix=suffix_text,
                    start_char=start_char,
                    end_char=end_char,
                    start_token=start_token,
                    prefix_token_ids=prefix_ids,
                    suffix_token_ids=suffix_ids,
                )
            )

        return windows

    def _compute_coverage(
        self, windows: list[MemorizationResult], text_len: int
    ) -> float:
        """Compute extraction coverage (% of chars in memorized windows)."""
        if not windows or text_len == 0:
            return 0.0

        # Mark characters in memorized windows
        char_memorized = [False] * text_len

        for w in windows:
            if w.is_memorized and w.window_start_char is not None:
                start = w.window_start_char
                end = min(w.window_end_char or text_len, text_len)
                for i in range(start, end):
                    char_memorized[i] = True

        return sum(char_memorized) / text_len

    def _compute_heatmap(
        self, windows: list[MemorizationResult], text_len: int
    ) -> list[float]:
        """Compute max p_z at each character position."""
        if not windows or text_len == 0:
            return []

        char_max_pz = [0.0] * text_len

        for w in windows:
            if w.window_start_char is not None:
                start = w.window_start_char
                end = min(w.window_end_char or text_len, text_len)
                for i in range(start, end):
                    char_max_pz[i] = max(char_max_pz[i], w.p_z)

        return char_max_pz

    def _check_exact_match(self, generated: str, target: str) -> bool:
        """Check if generated text exactly matches target."""
        if not generated or not target:
            return False
        # Normalize whitespace
        gen_norm = " ".join(generated.split())
        tgt_norm = " ".join(target.split())
        return gen_norm == tgt_norm

    def _compute_similarity(
        self, generated: str, target: str
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """Compute BLEU and ROUGE scores."""
        if not generated or not target:
            return None, None, None, None

        # Tokenize (word-level)
        gen_tokens = generated.split()
        tgt_tokens = target.split()

        if not gen_tokens or not tgt_tokens:
            return None, None, None, None

        try:
            bleu_1 = self._bleu1.compute(gen_tokens, tgt_tokens).score
            bleu_2 = self._bleu2.compute(gen_tokens, tgt_tokens).score
            rouge_1 = self._rouge1.compute(gen_tokens, tgt_tokens).score
            rouge_l = self._rougel.compute(gen_tokens, tgt_tokens).score
            return bleu_1, bleu_2, rouge_1, rouge_l
        except Exception as e:
            logger.warning("Error computing similarity: %s", e)
            return None, None, None, None

    def compute_summary(
        self,
        results: list[MemorizationResult],
        dataset_name: str = "dataset",
        experiment_id: str | None = None,
    ) -> ExperimentSummary:
        """Compute summary statistics for a set of results."""
        import statistics

        if not results:
            raise ValueError("No results to summarize")

        experiment_id = experiment_id or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        p_z_values = [r.p_z for r in results]
        ppl_values = [r.perplexity for r in results if r.perplexity > 0]

        memorized_count = sum(1 for r in results if r.is_memorized)
        exact_match_count = sum(1 for r in results if r.exact_match is True)

        bleu_values = [r.bleu_1 for r in results if r.bleu_1 is not None]
        rouge_values = [r.rouge_l for r in results if r.rouge_l is not None]

        # Timing
        total_compute = sum(r.compute_time_ms for r in results)
        total_generate = sum(r.generate_time_ms for r in results)
        total_seconds = (total_compute + total_generate) / 1000

        return ExperimentSummary(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            model_name=self.model_name,
            dataset_name=dataset_name,
            num_samples=len(results),
            memorized_count=memorized_count,
            memorized_fraction=memorized_count / len(results),
            exact_match_count=exact_match_count,
            exact_match_fraction=exact_match_count / len(results),
            p_z_mean=statistics.mean(p_z_values),
            p_z_std=statistics.stdev(p_z_values) if len(p_z_values) > 1 else 0.0,
            p_z_min=min(p_z_values),
            p_z_max=max(p_z_values),
            p_z_median=statistics.median(p_z_values),
            ppl_mean=statistics.mean(ppl_values) if ppl_values else 0.0,
            ppl_std=statistics.stdev(ppl_values) if len(ppl_values) > 1 else 0.0,
            bleu_1_mean=statistics.mean(bleu_values) if bleu_values else None,
            rouge_l_mean=statistics.mean(rouge_values) if rouge_values else None,
            total_seconds=total_seconds,
            samples_per_second=len(results) / total_seconds if total_seconds > 0 else 0,
        )


def save_results(
    results: list[MemorizationResult],
    output_dir: str | Path,
    config: MemorizationConfig,
    model_name: str,
    dataset_name: str,
    summary: ExperimentSummary | None = None,
) -> None:
    """
    Save results to output directory.

    Creates:
      - config.json: Experiment configuration
      - results.parquet: Per-sample results
      - summary.json: Aggregate statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_data = {
        "experiment_id": summary.experiment_id if summary else "unknown",
        "timestamp": datetime.now().isoformat(),
        "model": {
            "name": model_name,
        },
        "compute": {
            "top_k": config.compute_top_k,
            "tau_min": config.tau_min,
        },
        "generation": {
            "enabled": config.do_generate,
            "temperature": config.gen_temperature,
            "top_k": config.gen_top_k,
            "top_p": config.gen_top_p,
            "min_p": config.gen_min_p,
            "max_new_tokens": config.gen_max_new_tokens,
            "do_sample": config.gen_do_sample,
            "repetition_penalty": config.gen_repetition_penalty,
            "num_generations": config.num_generations,
            "capture_top_k": config.capture_top_k,
        },
        "sliding_window": {
            "prefix_len": config.window_prefix_len,
            "suffix_len": config.window_suffix_len,
            "stride": config.window_stride,
        },
        "dataset": {
            "name": dataset_name,
            "num_samples": len(results),
        },
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    logger.info("Config saved to %s", config_path)

    # Save results as parquet
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        records = [r.to_dict() for r in results]
        table = pa.Table.from_pylist(records)
        parquet_path = output_dir / "results.parquet"
        pq.write_table(table, parquet_path)
        logger.info("Results saved to %s", parquet_path)
    except ImportError:
        # Fallback to JSON
        json_path = output_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        logger.info("Results saved to %s (parquet unavailable)", json_path)

    # Save summary
    if summary:
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        logger.info("Summary saved to %s", summary_path)
