#!/usr/bin/env python3
"""Unit tests for native SGLang beam-search integration path."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from cleanslate.decoding.beam import BeamSearchDecoding


class FakeTokenizer:
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        body = [ord(ch) - 96 for ch in text]
        return ([101] + body) if add_special_tokens else body

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        out = []
        for tid in token_ids:
            if skip_special_tokens and tid in {0, 101}:
                continue
            if tid == 101:
                continue
            if 1 <= tid <= 26:
                out.append(chr(tid + 96))
        return "".join(out)

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [str(x) for x in token_ids]


class FakeBeam:
    def __init__(self, tokens: list[int], cum_logprob: float, text: str | None = None):
        self.tokens = tokens
        self.cum_logprob = cum_logprob
        self.text = text


class NativeEngine:
    def __init__(self, beam_width: int = 4):
        self.server_args = SimpleNamespace(beam_width=beam_width)

    def generate(self, **kwargs):
        tok = FakeTokenizer()
        outs = []
        for seed in kwargs["prompt"]:
            seed_ids = tok.encode(seed, add_special_tokens=True)
            outs.append(
                {
                    "meta_info": {
                        "beam_search_outputs": SimpleNamespace(
                            sequences=[
                                FakeBeam(seed_ids + [1, 2], -0.1, text="ab"),
                                FakeBeam(seed_ids + [1, 3], -0.2, text="ac"),
                                FakeBeam(seed_ids + [2], -0.3, text="b"),
                            ]
                        )
                    }
                }
            )
        return outs


class FallbackEngine:
    def __init__(self):
        self.server_args = SimpleNamespace(beam_width=0)

    def generate(self, **kwargs):
        prompts = kwargs["prompt"]
        # One-step manual expansion shape expected by fallback beam path.
        return [
            {"meta_info": {"output_top_logprobs": [[(-0.1, 1), (-0.2, 0), (-0.3, 2)]]}}
            for _ in prompts
        ]


class TestNativeBeamSearch(unittest.TestCase):
    def test_uses_native_path_when_available(self):
        strategy = BeamSearchDecoding(beam_width=2, length_penalty=1.0)
        outputs = strategy.decode(
            engine=NativeEngine(beam_width=4),
            tokenizer=FakeTokenizer(),
            seeds=["hi"],
            max_new_tokens=2,
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].num_sequences, 2)
        self.assertTrue(outputs[0].strategy_params.get("native_beam_search"))
        self.assertEqual(outputs[0].sequences[0].generated_text, "ab")

    def test_falls_back_when_native_is_not_configured(self):
        strategy = BeamSearchDecoding(beam_width=2, length_penalty=1.0, max_batch_prompts=8)
        outputs = strategy.decode(
            engine=FallbackEngine(),
            tokenizer=FakeTokenizer(),
            seeds=["hi"],
            max_new_tokens=2,
            capture_top_k=2,
        )

        self.assertEqual(len(outputs), 1)
        self.assertGreaterEqual(outputs[0].num_sequences, 1)
        self.assertNotIn("native_beam_search", outputs[0].strategy_params)


if __name__ == "__main__":
    unittest.main()
