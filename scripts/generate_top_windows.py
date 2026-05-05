"""Greedy-generate continuations for our top high-p_z prefixes on Modal.

Loads the same Qwen2.5-Coder-32B-Instruct-GPTQ-Int4 used for the memorization
eval, and asks vLLM to greedy-decode 100 tokens per prefix. Output is printed
to stdout so we can paste it back into the analysis script and build a clean
prefix / target / model-output comparison.

Run: uv run python scripts/generate_top_windows.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


PROMPTS: list[dict[str, Any]] = [
    {
        "song": "Don't You Worry Baby — Tyler, the Creator",
        "p_z": 0.6361,
        "prefix": " move your hips\nDamn, girl, you better move your hips\nDamn, girl, you better move your hips\nDamn, gi",
        "target": "rl, you better move your hips\nDamn, girl, you better move your hips\nDamn, girl, you better move your",
    },
    {
        "song": "Don't You Worry Baby — Tyler, the Creator",
        "p_z": 0.5819,
        "prefix": " hips\nDamn, girl, you better move your hips\nDamn, girl, you better move your hips\nDamn, girl, you be",
        "target": "tter move your hips\nDamn, girl, you better move your hips\nDamn, girl, you better move your hips\nDamn",
    },
    {
        "song": "Big Poe — Tyler, the Creator",
        "p_z": 0.4163,
        "prefix": " wanna (Jump, jump)\nDon't this shit make a nigga wanna (Jump, jump)\nDon't this shit make a nigga wan",
        "target": "na (Jump, jump)\nDon't this shit make a nigga wanna (Jump, jump)\nDon't this shit make a nigga wanna (",
    },
    {
        "song": "Drive — The Weeknd",
        "p_z": 0.1177,
        "prefix": "ain\nAnd no child deserves suffering\n\nAnd\nOh, I just want to drive, I just want to drive\nOh, I just w",
        "target": "ant to drive, I just want to drive\nOh, I just want to drive, I just want to drive\nOh, I just want to",
    },
    {
        "song": "Butterflies — Justin Bieber",
        "p_z": 0.1152,
        "prefix": "ies (I'm 'bout to slip away, slippin' away)\nButterflies (I'm 'bout to slip away)\nButterflies (I'm 'b",
        "target": "out to slip away, slippin' away)\nButterflies (I'm 'bout to slip away)\nButterflies (I'm 'bout to slip",
    },
    {
        "song": "Munyun — Playboi Carti",
        "p_z": 0.1151,
        "prefix": "it on the floor (Swamp Izzo)\nCame a long way, still can't believe I made Forbes (Swamp Izzo)\nCame a ",
        "target": "long way, still can't believe I made Forbes (Swamp Izzo)\nCame a long way, still can't believe I made",
    },
]


def main() -> None:
    import modal
    from cleanslate.modal._runtime import HF_CACHE, HF_SECRET, build_eval_image

    image = build_eval_image()
    app = modal.App("qwen25-coder-32b-int4-generate", image=image)

    @app.function(
        gpu="H100",
        secrets=[HF_SECRET],
        volumes={"/model_cache": HF_CACHE},
        timeout=15 * 60,
        serialized=True,
    )
    def gen(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model="Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4",
            dtype="float16",
            max_model_len=512,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
            trust_remote_code=True,
            download_dir="/model_cache",
        )
        sp = SamplingParams(temperature=0.0, max_tokens=100, stop=None)
        out = llm.generate([p["prefix"] for p in prompts], sp)
        results = []
        for p, o in zip(prompts, out):
            generated = o.outputs[0].text
            results.append({
                "song": p["song"],
                "p_z": p["p_z"],
                "prefix": p["prefix"],
                "target": p["target"],
                "generated": generated,
            })
        return results

    with modal.enable_output():
        with app.run():
            results = gen.remote(PROMPTS)
    json.dump(results, sys.stdout, indent=2, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
