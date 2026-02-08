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

import logging
import math
import time
from typing import Any

try:
    from sglang.srt.entrypoints.engine import Engine
except ImportError:
    raise ImportError(
        "SGLang not installed. Install with: pip install cleanslate[sglang] "
        "or uv sync --extra sglang"
    )

from cleanslate.backends.base import GenerationResult, InferenceBackend, ProbabilityResult

logger = logging.getLogger(__name__)


class SGLangBackend(InferenceBackend):
    """SGLang backend for GPU-optimized inference with continuous batching.

    Uses SGLang's offline Engine API - no HTTP server needed.
    Supports native batch generation for full GPU utilization.
    """

    def __init__(self, model_name: str, **engine_kwargs: Any) -> None:
        self._model_name = model_name

        defaults: dict[str, Any] = {
            "trust_remote_code": True,
            "mem_fraction_static": 0.85,
            "disable_cuda_graph": True,
        }
        defaults.update(engine_kwargs)

        logger.info("Loading %s with SGLang Engine...", model_name)
        self.engine = Engine(model_path=model_name, **defaults)
        self.tokenizer = self.engine.tokenizer_manager.tokenizer
        logger.info("Model loaded with SGLang on CUDA.")

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
        **decoding_options: Any,
    ) -> list[ProbabilityResult]:
        """Compute probabilities for many (seed, target) pairs in one batch.

        SGLang's continuous batching scheduler runs all prompts concurrently,
        giving much higher GPU utilization than sequential calls.
        """
        if not pairs:
            return []

        # Build batch: full texts + per-prompt logprob_start_len
        prompts: list[str] = []
        seed_lens: list[int] = []
        for seed, target in pairs:
            seed_ids = self.tokenizer.encode(seed, add_special_tokens=True)
            prompts.append(seed + target)
            seed_lens.append(len(seed_ids))

        # logprob_start_len one position early so the None falls on last seed token
        logprob_starts = [max(0, sl - 1) for sl in seed_lens]

        results_raw = self.engine.generate(
            prompt=prompts,
            sampling_params=[{"max_new_tokens": 0, "temperature": 0}] * len(prompts),
            return_logprob=[True] * len(prompts),
            logprob_start_len=logprob_starts,
            top_logprobs_num=[1] * len(prompts),
        )

        out: list[ProbabilityResult] = []
        for i, raw in enumerate(results_raw):
            out.append(self._parse_logprobs(raw, seed_lens[i], logprob_starts[i]))
        return out

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

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
        """Generate text from a single seed with full logprob capture."""
        results = self.generate_batch(
            seeds=[seed],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            capture_top_k=capture_top_k,
        )
        return results[0]

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
        """Generate text from many seeds in a single batched call.

        All prompts are submitted together and SGLang's continuous batching
        scheduler handles them concurrently for maximum GPU utilization.
        """
        if not seeds:
            return []

        start_time = time.time()

        sampling_params: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if do_sample else 0.0,
        }
        if top_k is not None:
            sampling_params["top_k"] = top_k
        if top_p is not None:
            sampling_params["top_p"] = top_p

        # Single batched call - SGLang schedules all prompts concurrently
        raw_results = self.engine.generate(
            prompt=seeds,
            sampling_params=[sampling_params] * len(seeds),
            return_logprob=[True] * len(seeds),
            logprob_start_len=[0] * len(seeds),
            top_logprobs_num=[capture_top_k] * len(seeds),
        )

        batch_time_ms = (time.time() - start_time) * 1000
        per_item_ms = batch_time_ms / len(seeds)

        out: list[GenerationResult] = []
        for raw in raw_results:
            out.append(
                self._parse_generation(
                    raw,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    generation_time_ms=per_item_ms,
                )
            )
        return out

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

    def _parse_generation(
        self,
        raw: dict,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        do_sample: bool,
        generation_time_ms: float,
    ) -> GenerationResult:
        """Parse a single SGLang generate response into GenerationResult."""
        meta = raw["meta_info"]
        generated_text: str = raw.get("text", "")
        output_ids: list[int] = raw.get("output_ids", [])

        output_logprobs = meta.get("output_token_logprobs", [])
        output_top = meta.get("output_top_logprobs", [])

        token_log_probs: list[float] = []
        generated_token_ids: list[int] = []
        top_k_tokens_list: list[list[int]] = []
        top_k_probs_list: list[list[float]] = []

        for i, entry in enumerate(output_logprobs):
            token_log_probs.append(float(entry[0]))
            generated_token_ids.append(int(entry[1]))

            if i < len(output_top) and output_top[i]:
                top_k_tokens_list.append([int(t[1]) for t in output_top[i]])
                top_k_probs_list.append([float(t[0]) for t in output_top[i]])
            else:
                top_k_tokens_list.append([])
                top_k_probs_list.append([])

        if not generated_token_ids and output_ids:
            generated_token_ids = list(output_ids)

        generated_tokens = [self.tokenizer.decode([tid]) for tid in generated_token_ids]

        return GenerationResult(
            generated_text=generated_text,
            generated_token_ids=generated_token_ids,
            generated_tokens=generated_tokens,
            token_log_probs=token_log_probs,
            top_k_tokens=top_k_tokens_list,
            top_k_probs=top_k_probs_list,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            generation_time_ms=generation_time_ms,
        )

    def shutdown(self) -> None:
        """Shut down the SGLang engine and free GPU memory."""
        if hasattr(self, "engine"):
            self.engine.shutdown()

    def __del__(self) -> None:
        self.shutdown()
