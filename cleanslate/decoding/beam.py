"""Beam search decoding strategy.

Maintains top-k hypotheses at each step, returning all beams for analysis.
Useful for memorization evaluation where the target may not be the top beam.

Supports batching across multiple seeds for efficiency.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from cleanslate.decoding.base import DecodingOutput, DecodingStrategy, GenerationSequence


@dataclass
class BeamHypothesis:
    """A single beam hypothesis during search."""

    token_ids: list[int]  # Full sequence including prompt
    cum_log_prob: float
    finished: bool = False

    # Token-level data for the generated portion
    generated_log_probs: list[float] = field(default_factory=list)
    generated_top_k_tokens: list[list[int]] = field(default_factory=list)
    generated_top_k_probs: list[list[float]] = field(default_factory=list)

    def score(self, length_penalty: float, prompt_len: int) -> float:
        """Compute score with length penalty.

        Score = cum_log_prob / (generated_length ^ length_penalty)
        """
        gen_len = len(self.token_ids) - prompt_len
        if gen_len <= 0:
            return self.cum_log_prob
        return self.cum_log_prob / (gen_len**length_penalty)


@dataclass
class SeedState:
    """Tracks beam search state for a single seed."""

    seed_idx: int
    prompt_len: int
    eos_token_id: int
    active_beams: list[BeamHypothesis]
    finished_beams: list[BeamHypothesis] = field(default_factory=list)
    is_done: bool = False


@dataclass
class BeamSearchDecoding(DecodingStrategy):
    """Beam search - maintain top-k hypotheses at each step.

    Returns ALL beam_width hypotheses for each seed, ranked by score.
    This is useful for memorization analysis where we want to check
    if the target appears in any beam, not just the top one.

    Supports batching across seeds for efficiency - at each token step,
    all seeds' beams are processed in a single SGLang call (chunked by
    max_batch_prompts to avoid OOM).

    Args:
        beam_width: Number of beams to maintain (default: 5)
        length_penalty: Exponent for length normalization (default: 1.0)
            - 1.0: Linear penalty (longer = lower score)
            - 0.0: No penalty (raw log prob)
            - >1.0: Stronger penalty for length
        max_batch_prompts: Maximum prompts per SGLang call (default: 256)
            Controls memory usage. Reduce if OOM.
    """

    beam_width: int = 5
    length_penalty: float = 1.0
    max_batch_prompts: int = 256

    @property
    def name(self) -> str:
        return "beam"

    def decode(
        self,
        engine: Any,
        tokenizer: Any,
        seeds: list[str],
        max_new_tokens: int,
        capture_top_k: int = 50,
    ) -> list[DecodingOutput]:
        """Decode using beam search with batching across seeds.

        Args:
            engine: SGLang Engine instance
            tokenizer: Tokenizer for encoding/decoding
            seeds: List of prompt strings
            max_new_tokens: Maximum tokens to generate per seed
            capture_top_k: Number of top tokens to capture per position

        Returns:
            List of DecodingOutput, one per seed (each with beam_width sequences)
        """
        if not seeds:
            return []

        # Fast path: use SGLang's native beam-search implementation when
        # available and configured on the engine.
        native_outputs = self._decode_with_native_beam(
            engine=engine,
            tokenizer=tokenizer,
            seeds=seeds,
            max_new_tokens=max_new_tokens,
            capture_top_k=capture_top_k,
        )
        if native_outputs is not None:
            return native_outputs

        start_time = time.time()
        eos_token_id = tokenizer.eos_token_id

        # Initialize state for each seed
        seed_states: list[SeedState] = []
        for idx, seed in enumerate(seeds):
            seed_ids = tokenizer.encode(seed, add_special_tokens=True)
            initial_beam = BeamHypothesis(token_ids=list(seed_ids), cum_log_prob=0.0)
            seed_states.append(
                SeedState(
                    seed_idx=idx,
                    prompt_len=len(seed_ids),
                    eos_token_id=eos_token_id,
                    active_beams=[initial_beam],
                )
            )

        top_k_request = max(self.beam_width * 2, capture_top_k)

        # Main loop: process all seeds in parallel at each step
        for step in range(max_new_tokens):
            # Collect all active beams across all seeds
            all_prompts: list[str] = []
            prompt_to_source: list[tuple[int, int]] = []  # (seed_idx, beam_idx)

            for state in seed_states:
                if state.is_done:
                    continue
                for beam_idx, beam in enumerate(state.active_beams):
                    prompt_text = tokenizer.decode(beam.token_ids)
                    all_prompts.append(prompt_text)
                    prompt_to_source.append((state.seed_idx, beam_idx))

            if not all_prompts:
                break  # All seeds done

            # Process in chunks to avoid OOM
            all_results: list[dict] = []
            for chunk_start in range(0, len(all_prompts), self.max_batch_prompts):
                chunk_end = min(chunk_start + self.max_batch_prompts, len(all_prompts))
                chunk_prompts = all_prompts[chunk_start:chunk_end]
                chunk_size = len(chunk_prompts)

                chunk_results = engine.generate(
                    prompt=chunk_prompts,
                    sampling_params=[{"max_new_tokens": 1, "temperature": 0.0}]
                    * chunk_size,
                    return_logprob=[True] * chunk_size,
                    top_logprobs_num=[top_k_request] * chunk_size,
                )
                all_results.extend(chunk_results)

            # Group results by seed and expand beams
            seed_candidates: dict[int, list[BeamHypothesis]] = {
                s.seed_idx: [] for s in seed_states if not s.is_done
            }

            for (seed_idx, beam_idx), result in zip(prompt_to_source, all_results):
                state = seed_states[seed_idx]
                beam = state.active_beams[beam_idx]
                meta = result["meta_info"]
                output_top = meta.get("output_top_logprobs", [])

                if not output_top or not output_top[0]:
                    # No valid output, mark beam as finished
                    beam.finished = True
                    state.finished_beams.append(beam)
                    continue

                # Expand beam with top-k candidates
                # SGLang format: each entry is (log_prob, token_id, ...) - may have extra fields
                top_tokens = output_top[0]
                for entry in top_tokens[: self.beam_width]:
                    log_prob = float(entry[0])
                    token_id = int(entry[1])
                    new_beam = BeamHypothesis(
                        token_ids=beam.token_ids + [token_id],
                        cum_log_prob=beam.cum_log_prob + log_prob,
                        finished=(token_id == state.eos_token_id),
                        generated_log_probs=beam.generated_log_probs + [log_prob],
                        generated_top_k_tokens=beam.generated_top_k_tokens
                        + [[int(t[1]) for t in top_tokens[:capture_top_k]]],
                        generated_top_k_probs=beam.generated_top_k_probs
                        + [[float(t[0]) for t in top_tokens[:capture_top_k]]],
                    )
                    seed_candidates[seed_idx].append(new_beam)

            # Prune beams for each seed
            for state in seed_states:
                if state.is_done:
                    continue

                candidates = seed_candidates[state.seed_idx]
                if not candidates:
                    state.is_done = True
                    continue

                # Sort by score
                candidates.sort(
                    key=lambda h: h.score(self.length_penalty, state.prompt_len),
                    reverse=True,
                )

                # Separate finished and active
                new_active = []
                for c in candidates:
                    if c.finished:
                        state.finished_beams.append(c)
                    elif len(new_active) < self.beam_width:
                        new_active.append(c)

                state.active_beams = new_active

                # Check if done
                if not new_active and len(state.finished_beams) >= self.beam_width:
                    state.is_done = True

        total_time_ms = (time.time() - start_time) * 1000
        per_seed_time = total_time_ms / len(seeds)

        # Convert states to outputs
        outputs: list[DecodingOutput] = []
        for state in seed_states:
            output = self._state_to_output(state, tokenizer, per_seed_time)
            outputs.append(output)

        return outputs

    def _decode_with_native_beam(
        self,
        engine: Any,
        tokenizer: Any,
        seeds: list[str],
        max_new_tokens: int,
        capture_top_k: int,
    ) -> list[DecodingOutput] | None:
        """Decode with SGLang's native beam search if the engine supports it.

        Returns None when native beam search is unavailable or misconfigured,
        so callers can transparently fall back to the manual implementation.
        """
        configured_beam_width = self._engine_beam_width(engine)
        if configured_beam_width <= 0:
            return None
        if configured_beam_width < self.beam_width:
            return None

        start_time = time.time()
        top_k_request = max(self.beam_width * 2, capture_top_k)

        raw_results = engine.generate(
            prompt=seeds,
            sampling_params=[{"max_new_tokens": max_new_tokens, "temperature": 0.0}]
            * len(seeds),
            return_logprob=[True] * len(seeds),
            top_logprobs_num=[top_k_request] * len(seeds),
        )
        if not isinstance(raw_results, list) or len(raw_results) != len(seeds):
            return None

        # If runtime does not provide native beam outputs, use fallback path.
        if any(
            not raw.get("meta_info", {}).get("beam_search_outputs")
            for raw in raw_results
        ):
            return None

        total_time_ms = (time.time() - start_time) * 1000
        per_seed_time = total_time_ms / len(seeds)

        outputs: list[DecodingOutput] = []
        for seed, raw in zip(seeds, raw_results):
            seed_ids = tokenizer.encode(seed, add_special_tokens=True)
            prompt_len = len(seed_ids)
            beam_output = raw["meta_info"]["beam_search_outputs"]

            sequences: list[GenerationSequence] = []
            for beam in beam_output.sequences[: self.beam_width]:
                beam_token_ids = [int(t) for t in getattr(beam, "tokens", [])]
                generated_ids = beam_token_ids[prompt_len:]
                generated_text = getattr(beam, "text", None) or tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
                total_log_prob = float(getattr(beam, "cum_logprob", 0.0))

                gen_len = len(generated_ids)
                if gen_len <= 0:
                    score = total_log_prob
                else:
                    score = total_log_prob / (gen_len**self.length_penalty)

                sequences.append(
                    GenerationSequence(
                        generated_text=generated_text,
                        generated_token_ids=generated_ids,
                        generated_tokens=generated_tokens,
                        token_log_probs=[],
                        top_k_tokens=[],
                        top_k_probs=[],
                        total_log_prob=total_log_prob,
                        score=score,
                        generation_time_ms=per_seed_time / max(self.beam_width, 1),
                    )
                )

            if not sequences:
                return None

            outputs.append(
                DecodingOutput(
                    sequences=sequences,
                    strategy_name=self.name,
                    strategy_params=self.to_dict() | {"native_beam_search": True},
                )
            )

        return outputs

    def _engine_beam_width(self, engine: Any) -> int:
        """Best-effort extraction of configured beam width from SGLang engine."""
        # Current SGLang versions expose this under engine.server_args.beam_width.
        server_args = getattr(engine, "server_args", None)
        if server_args is not None:
            width = getattr(server_args, "beam_width", 0)
            try:
                return int(width)
            except (TypeError, ValueError):
                return 0

        return 0

    def _state_to_output(
        self,
        state: SeedState,
        tokenizer: Any,
        generation_time_ms: float,
    ) -> DecodingOutput:
        """Convert a SeedState to DecodingOutput."""
        # Combine finished and remaining active beams
        all_beams = state.finished_beams + state.active_beams
        all_beams.sort(
            key=lambda h: h.score(self.length_penalty, state.prompt_len),
            reverse=True,
        )

        # Take top beam_width
        final_beams = all_beams[: self.beam_width]

        # Convert to GenerationSequence objects
        sequences = []
        for beam in final_beams:
            generated_ids = beam.token_ids[state.prompt_len :]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)

            sequences.append(
                GenerationSequence(
                    generated_text=generated_text,
                    generated_token_ids=generated_ids,
                    generated_tokens=generated_tokens,
                    token_log_probs=beam.generated_log_probs,
                    top_k_tokens=beam.generated_top_k_tokens,
                    top_k_probs=beam.generated_top_k_probs,
                    total_log_prob=beam.cum_log_prob,
                    score=beam.score(self.length_penalty, state.prompt_len),
                    generation_time_ms=generation_time_ms / max(len(final_beams), 1),
                )
            )

        # Handle case where no beams were generated
        if not sequences:
            sequences.append(
                GenerationSequence(
                    generated_text="",
                    generated_token_ids=[],
                    generated_tokens=[],
                    token_log_probs=[],
                    total_log_prob=0.0,
                    score=0.0,
                    generation_time_ms=generation_time_ms,
                )
            )

        return DecodingOutput(
            sequences=sequences,
            strategy_name=self.name,
            strategy_params=self.to_dict(),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "beam_width": self.beam_width,
            "length_penalty": self.length_penalty,
            "max_batch_prompts": self.max_batch_prompts,
        }
