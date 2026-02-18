"""Greedy decoding strategy.

Always picks the highest probability token at each step.
Equivalent to temperature=0 sampling.
"""

import time
from dataclasses import dataclass
from typing import Any

from cleanslate.decoding.base import DecodingOutput, DecodingStrategy, GenerationSequence


@dataclass
class GreedyDecoding(DecodingStrategy):
    """Greedy decoding - always pick the top-1 token.

    This is deterministic and fast, producing a single sequence per seed.
    """

    @property
    def name(self) -> str:
        return "greedy"

    def decode(
        self,
        engine: Any,
        tokenizer: Any,
        seeds: list[str],
        max_new_tokens: int,
        capture_top_k: int = 50,
    ) -> list[DecodingOutput]:
        """Decode using greedy selection (temperature=0).

        Args:
            engine: SGLang Engine instance
            tokenizer: Tokenizer for encoding/decoding
            seeds: List of prompt strings
            max_new_tokens: Maximum tokens to generate per seed
            capture_top_k: Number of top tokens to capture per position

        Returns:
            List of DecodingOutput, one per seed (each with single sequence)
        """
        if not seeds:
            return []

        start_time = time.time()

        # Single batched call with temperature=0 (greedy)
        sampling_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
        }

        raw_results = engine.generate(
            prompt=seeds,
            sampling_params=[sampling_params] * len(seeds),
            return_logprob=[True] * len(seeds),
            top_logprobs_num=[capture_top_k] * len(seeds),
        )

        batch_time_ms = (time.time() - start_time) * 1000
        per_item_ms = batch_time_ms / len(seeds)

        outputs: list[DecodingOutput] = []

        for raw in raw_results:
            sequence = self._parse_result(raw, tokenizer, per_item_ms)
            outputs.append(
                DecodingOutput(
                    sequences=[sequence],
                    strategy_name=self.name,
                    strategy_params=self.to_dict(),
                )
            )

        return outputs

    def _parse_result(
        self,
        raw: dict,
        tokenizer: Any,
        generation_time_ms: float,
    ) -> GenerationSequence:
        """Parse SGLang generate response into GenerationSequence."""
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

        generated_tokens = tokenizer.convert_ids_to_tokens(generated_token_ids)
        total_log_prob = sum(token_log_probs)

        return GenerationSequence(
            generated_text=generated_text,
            generated_token_ids=generated_token_ids,
            generated_tokens=generated_tokens,
            token_log_probs=token_log_probs,
            top_k_tokens=top_k_tokens_list,
            top_k_probs=top_k_probs_list,
            total_log_prob=total_log_prob,
            score=total_log_prob,  # For greedy, score = total log prob
            generation_time_ms=generation_time_ms,
        )

    def to_dict(self) -> dict:
        return {"name": self.name}
