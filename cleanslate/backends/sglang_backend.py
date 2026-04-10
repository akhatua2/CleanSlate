"""SGLang backend - GPU-optimized inference with continuous batching.

SGLang's Engine runs the model in a separate scheduler process and uses
continuous batching to maximize GPU utilization. This makes it much faster
than HuggingFace for large-scale generation workloads.

Key advantages over HuggingFace backend:
  - Continuous batching: many prompts scheduled concurrently
  - PagedAttention KV cache: efficient memory management
  - Native logprob support: no manual softmax needed
  - Async scheduler: overlaps compute and I/O

Requires: pip install cleanslate[sglang]  (or uv sync --extra sglang)
Important: Scripts using this backend MUST use `if __name__ == '__main__':` guard.
"""

import asyncio
import json
import logging
import math
import os
import time
from typing import Any

# Suppress all SGLang/torch verbose output
os.environ["SGLANG_LOG_LEVEL"] = "error"
os.environ["SGLANG_LOGGING_LEVEL"] = "error"
os.environ["GLOO_LOG_LEVEL"] = "error"
os.environ["NCCL_DEBUG"] = "ERROR"
os.environ["TQDM_DISABLE"] = "1"  # Disable progress bars in subprocesses

import warnings
warnings.filterwarnings("ignore", message=".*Gloo.*")
warnings.filterwarnings("ignore", message=".*safetensors.*")

for name in ["sglang", "sglang.srt", "sglang.srt.server_args", "sglang.srt.entrypoints",
             "sglang.srt.managers", "torch.distributed", "filelock"]:
    logging.getLogger(name).setLevel(logging.ERROR)

try:
    from sglang.srt.entrypoints.engine import Engine
except ImportError:
    raise ImportError(
        "SGLang not installed. Install with: pip install cleanslate[sglang] "
        "or uv sync --extra sglang"
    )

from cleanslate.backends.base import (
    InferenceBackend,
    ProbabilityResult,
    SongGenerationOutput,
)
from cleanslate.chat_format import format_chat as _format_chat
from cleanslate.decoding import DecodingOutput, DecodingStrategy, GreedyDecoding

logger = logging.getLogger(__name__)


class SGLangBackend(InferenceBackend):
    """SGLang backend for GPU-optimized inference with continuous batching.

    Uses SGLang's offline Engine API - no HTTP server needed.
    Supports native batch generation for full GPU utilization.
    """

    _json_output: str | None = None

    def __init__(
        self, model_name: str, json_decoding: bool = False, **engine_kwargs: Any
    ) -> None:
        self._model_name = model_name
        if json_decoding:
            self._json_output = json.dumps(SongGenerationOutput.model_json_schema())
        defaults: dict[str, Any] = {
            "mem_fraction_static": 0.92,
            "disable_cuda_graph": False,
            "cuda_graph_max_bs": 128,
            "chunked_prefill_size": 8192,
            "log_level": "error",
            "show_time_cost": False,
            "disable_flashinfer_autotune": True,
        }
        defaults.update(engine_kwargs)

        self.engine = Engine(model_path=model_name, **defaults)
        self.tokenizer = self.engine.tokenizer_manager.tokenizer

    @property
    def device(self) -> str:
        return "cuda"

    @property
    def model_name(self) -> str:
        return self._model_name

    # ------------------------------------------------------------------
    # Probability computation
    # ------------------------------------------------------------------

    def compute(
        self, seed: str, target: str, **decoding_options: Any
    ) -> ProbabilityResult:
        """Compute P(target | seed) using SGLang's logprob API."""
        return self._compute_single(seed, target)

    def compute_batch(
        self,
        pairs: list[tuple[str, str]],
        max_batch_size: int = 64,
        **decoding_options: Any,
    ) -> list[ProbabilityResult]:
        """Compute probabilities for many (seed, target) pairs in one batch.

        SGLang's continuous batching scheduler runs all prompts concurrently,
        giving much higher GPU utilization than sequential calls.

        Args:
            pairs: List of (seed, target) tuples
            max_batch_size: Maximum prompts per engine.generate() call.
                            Reduce if OOM during logits computation.
        """
        if not pairs:
            return []

        prompts, seed_lens, logprob_starts = self._prepare_logprob_inputs(pairs)

        # Process in chunks to avoid OOM on large batches
        out: list[ProbabilityResult] = []
        for batch_start in range(0, len(prompts), max_batch_size):
            batch_end = min(batch_start + max_batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_seed_lens = seed_lens[batch_start:batch_end]
            batch_logprob_starts = logprob_starts[batch_start:batch_end]

            results_raw = self.engine.generate(
                prompt=batch_prompts,
                sampling_params=[{"max_new_tokens": 0, "temperature": 0}] * len(batch_prompts),
                return_logprob=[True] * len(batch_prompts),
                logprob_start_len=batch_logprob_starts,
                top_logprobs_num=[1] * len(batch_prompts),
            )

            for i, raw in enumerate(results_raw):
                out.append(self._parse_logprobs(raw, batch_seed_lens[i], batch_logprob_starts[i]))

        return out

    async def compute_batch_async(
        self,
        pairs: list[tuple[str, str]],
        max_batch_size: int = 64,
        **decoding_options: Any,
    ) -> list[ProbabilityResult]:
        """Async variant of :meth:`compute_batch`.

        Uses ``engine.async_generate`` so the caller can submit the next
        chunk while the GPU is still processing the previous one. SGLang's
        continuous batching scheduler keeps all in-flight requests packed
        on-device, so this is the right primitive to keep the GPU saturated.
        """
        if not pairs:
            return []

        prompts, seed_lens, logprob_starts = self._prepare_logprob_inputs(pairs)

        out: list[ProbabilityResult] = [None] * len(prompts)  # type: ignore[list-item]

        async def _run_chunk(start: int, end: int) -> None:
            batch_prompts = prompts[start:end]
            batch_seed_lens = seed_lens[start:end]
            batch_logprob_starts = logprob_starts[start:end]
            results_raw = await self.engine.async_generate(
                prompt=batch_prompts,
                sampling_params=[{"max_new_tokens": 0, "temperature": 0}] * len(batch_prompts),
                return_logprob=[True] * len(batch_prompts),
                logprob_start_len=batch_logprob_starts,
                top_logprobs_num=[1] * len(batch_prompts),
            )
            # async_generate returns a list when given a list of prompts
            if isinstance(results_raw, dict):
                results_raw = [results_raw]
            for i, raw in enumerate(results_raw):
                out[start + i] = self._parse_logprobs(
                    raw, batch_seed_lens[i], batch_logprob_starts[i]
                )

        tasks = [
            _run_chunk(s, min(s + max_batch_size, len(prompts)))
            for s in range(0, len(prompts), max_batch_size)
        ]
        await asyncio.gather(*tasks)
        return out

    def _prepare_logprob_inputs(
        self, pairs: list[tuple[str, str]]
    ) -> tuple[list[str], list[int], list[int]]:
        """Tokenize seeds and build the (prompts, seed_lens, logprob_starts)
        tuple shared by the sync and async batch paths."""
        prompts: list[str] = []
        seed_lens: list[int] = []
        for seed, target in pairs:
            seed_ids = self.tokenizer.encode(seed, add_special_tokens=True)
            prompts.append(seed + target)
            seed_lens.append(len(seed_ids))
        # logprob_start_len one position early so the None falls on last seed token
        logprob_starts = [max(0, sl - 1) for sl in seed_lens]
        return prompts, seed_lens, logprob_starts

    # ------------------------------------------------------------------
    # Generation (strategy-based)
    # ------------------------------------------------------------------

    def format_chat(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        """
        Format chat messages using the model's chat template if available.
        """
        return _format_chat(
            self.tokenizer,
            messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    def generate(
        self,
        seed: str,
        strategy: DecodingStrategy | None = None,
        max_new_tokens: int = 128,
        capture_top_k: int = 50,
    ) -> DecodingOutput:
        """Generate text from a single seed using a decoding strategy.

        Args:
            seed: Prompt string
            strategy: Decoding strategy (default: GreedyDecoding)
            max_new_tokens: Maximum tokens to generate
            capture_top_k: Number of top tokens to capture per position

        Returns:
            DecodingOutput with all sequences from the strategy
        """
        results = self.generate_batch(
            seeds=[seed],
            strategy=strategy,
            max_new_tokens=max_new_tokens,
            capture_top_k=capture_top_k,
        )
        return results[0]

    def generate_batch(
        self,
        seeds: list[str],
        strategy: DecodingStrategy | None = None,
        max_new_tokens: int = 128,
        capture_top_k: int = 50,
    ) -> list[DecodingOutput]:
        """Generate text from many seeds using a decoding strategy.

        Args:
            seeds: List of prompt strings
            strategy: Decoding strategy (default: GreedyDecoding)
            max_new_tokens: Maximum tokens to generate per seed
            capture_top_k: Number of top tokens to capture per position

        Returns:
            List of DecodingOutput, one per seed
        """
        if strategy is None:
            strategy = GreedyDecoding()

        return strategy.decode(
            engine=self.engine,
            tokenizer=self.tokenizer,
            seeds=seeds,
            max_new_tokens=max_new_tokens,
            capture_top_k=capture_top_k,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_logprobs(
        self, raw: dict, seed_len: int, logprob_start: int
    ) -> ProbabilityResult:
        """Parse SGLang generate response into a ProbabilityResult."""
        meta = raw["meta_info"]
        input_logprobs = meta.get("input_token_logprobs", [])
        input_top_logprobs = meta.get("input_top_logprobs", [])

        # Skip entries before the target region
        offset = seed_len - logprob_start
        target_logprobs = input_logprobs[offset:]
        target_top = input_top_logprobs[offset:] if input_top_logprobs else []

        token_log_probs: list[float] = []
        token_ranks: list[int] = []

        for i, entry in enumerate(target_logprobs):
            lp = entry[0]
            token_id = entry[1]
            if lp is None:
                continue

            token_log_probs.append(float(lp))

            # Rank from top-1: if the actual token IS the top prediction, rank=1
            if i < len(target_top) and target_top[i]:
                top1_id = target_top[i][0][1]
                token_ranks.append(1 if token_id == top1_id else 2)
            else:
                token_ranks.append(1 if lp > -0.01 else 2)

        n = len(token_log_probs)
        total = sum(token_log_probs)
        ppl = math.exp(-total / n) if n > 0 else 1.0

        return ProbabilityResult(
            log_prob=total,
            perplexity=ppl,
            num_tokens=n,
            token_log_probs=token_log_probs,
            token_ranks=token_ranks,
        )

    def _compute_single(self, seed: str, target: str) -> ProbabilityResult:
        """Compute a single (seed, target) pair."""
        seed_ids = self.tokenizer.encode(seed, add_special_tokens=True)
        seed_len = len(seed_ids)
        full_text = seed + target
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=True)

        if len(full_ids) <= seed_len:
            return ProbabilityResult(
                log_prob=0.0, perplexity=1.0, num_tokens=0, token_log_probs=[]
            )

        logprob_start = max(0, seed_len - 1)
        raw = self.engine.generate(
            prompt=full_text,
            sampling_params={"max_new_tokens": 0, "temperature": 0},
            return_logprob=True,
            logprob_start_len=logprob_start,
            top_logprobs_num=1,
        )
        return self._parse_logprobs(raw, seed_len, logprob_start)


    def shutdown(self) -> None:
        """Shut down the SGLang engine and free GPU memory."""
        if hasattr(self, "engine"):
            self.engine.shutdown()

    def __del__(self) -> None:
        self.shutdown()
