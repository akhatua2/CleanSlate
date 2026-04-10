"""vLLM backend using prompt_logprobs for teacher-forcing P(target|seed).

Requires vLLM >= 0.19.0. Uses the official LLM.generate() API with
SamplingParams(prompt_logprobs=1, max_tokens=1) to extract per-token
log probabilities for the prompt.
"""

from __future__ import annotations

import logging
from typing import Any

from vllm import LLM, SamplingParams

from cleanslate.backends.base import InferenceBackend, ProbabilityResult

logger = logging.getLogger(__name__)


class VLLMBackend(InferenceBackend):
    """vLLM-backed inference for p_z computation."""

    def __init__(self, model_name: str, vllm_config: dict[str, Any] | None = None) -> None:
        self._model_name = model_name
        cfg = vllm_config or {}

        logger.info("Loading %s with vLLM...", model_name)
        self._llm = LLM(
            model=model_name,
            trust_remote_code=cfg.get("trust_remote_code", True),
            dtype=cfg.get("dtype", "bfloat16"),
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.90),
            max_model_len=cfg.get("max_model_len", 4096),
            tensor_parallel_size=cfg.get("num_gpus", 1),
            enforce_eager=cfg.get("enforce_eager", True),
            disable_log_stats=cfg.get("disable_log_stats", True),
        )
        self._tokenizer = self._llm.get_tokenizer()
        logger.info("Model loaded.")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def compute(self, seed: str, target: str) -> ProbabilityResult:
        return self.compute_batch([(seed, target)])[0]

    def compute_batch(self, pairs: list[tuple[str, str]]) -> list[ProbabilityResult]:
        if not pairs:
            return []

        prompts: list[str] = []
        seed_lens: list[int] = []
        for seed, target in pairs:
            seed_ids = self._tokenizer.encode(seed, add_special_tokens=True)
            prompts.append(seed + target)
            seed_lens.append(len(seed_ids))

        params = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0)
        outputs = self._llm.generate(prompts, params)

        return [self._parse_output(output, seed_lens[i]) for i, output in enumerate(outputs)]

    def _parse_output(self, output: Any, seed_len: int) -> ProbabilityResult:
        prompt_logprobs = output.prompt_logprobs
        prompt_token_ids = output.prompt_token_ids

        if prompt_logprobs is None or len(prompt_token_ids) <= seed_len:
            return ProbabilityResult(log_prob=0.0, num_tokens=0, token_log_probs=[])

        token_log_probs: list[float] = []
        token_ranks: list[int] = []

        for pos in range(seed_len, len(prompt_token_ids)):
            token_id = prompt_token_ids[pos]
            logprob_dict = prompt_logprobs[pos]
            if logprob_dict is None:
                continue

            if token_id in logprob_dict:
                entry = logprob_dict[token_id]
                token_log_probs.append(float(entry.logprob))
                token_ranks.append(int(entry.rank) if entry.rank is not None else 0)
            else:
                token_log_probs.append(float("-inf"))
                token_ranks.append(0)

        total = sum(lp for lp in token_log_probs if lp != float("-inf"))
        return ProbabilityResult(
            log_prob=total,
            num_tokens=len(token_log_probs),
            token_log_probs=token_log_probs,
            token_ranks=token_ranks,
        )
