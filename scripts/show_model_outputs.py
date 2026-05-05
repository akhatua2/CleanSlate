"""Greedy-decode Llama-3.1-70B for the 5 sub-0.5-sim post-llama31 windows.
Prints prefix / actual lyric (target) / model's free greedy continuation.

    uv run modal run scripts/show_model_outputs.py
"""

from __future__ import annotations

import modal

from cleanslate.modal._runtime import HF_CACHE, HF_SECRET, build_eval_image

WINDOWS: list[tuple[str, str, str]] = [
    (
        "Red Terror — window 1 (p_z=0.0281)",
        "verything remains exactly how it was\n\nI am I, and you are you\nAnd the old life that we lived so fond",
        "ly together is untouched, unchanged\nWhatever we were to each other, that we are still\nCall me by the",
    ),
    (
        "Red Terror — window 2 (p_z=0.0250)",
        "happened\nEverything remains exactly how it was\n\nI am I, and you are you\nAnd the old life that we liv",
        "ed so fondly together is untouched, unchanged\nWhatever we were to each other, that we are still\nCall",
    ),
    (
        "Butterflies (Justin Bieber) — p_z=0.0033",
        "m 'bout to slip away, I'm 'bout to slip away, I'm 'bout to slip away)\n\nButterflies (I'm 'bout to sli",
        "p away)\nButterflies (I'm 'bout to slip away)\nButterflies (I'm 'bout to slip away)\nButterflies (I'm '",
    ),
    (
        "Garden of Eden (Lady Gaga) — window A, p_z=0.0016",
        " life (oh)\nTake you to the Garden of Eden\nPoisoned apple\nTake a bite\n\nI'll t-t-take you to the Garde",
        "n of Eden\nI'll t-t-take you to the Garden of Eden\nI'll t-t-take you to the Garden of Eden\nI'll t-t-t",
    ),
    (
        "Garden of Eden (Lady Gaga) — window B, p_z=0.0033",
        "ife\nTake you to the Garden of Eden\nPoisoned apple\nTake a bite\n\nI'll t-t-take you to the Garden of Ed",
        "en\nI'll t-t-take you to the Garden of Eden\nI'll t-t-take you to the Garden of Eden\nI'll t-t-take you",
    ),
]

app = modal.App("llama-3-1-70b-show-outputs", image=build_eval_image())


@app.function(
    gpu="H100:2",
    secrets=[HF_SECRET],
    volumes={"/model_cache": HF_CACHE},
    timeout=60 * 60,
    serialized=True,
)
def generate() -> list[dict[str, str]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="meta-llama/Llama-3.1-70B",
        tensor_parallel_size=2,
        dtype="bfloat16",
        gpu_memory_utilization=0.92,
        max_model_len=1024,
        enforce_eager=False,
    )
    params = SamplingParams(temperature=0.0, max_tokens=80)
    prompts = [w[1] for w in WINDOWS]
    outputs = llm.generate(prompts, params, use_tqdm=False)

    return [
        {
            "label": label,
            "prefix_tail": prefix[-100:],
            "target": target,
            "model_output": out.outputs[0].text,
        }
        for (label, prefix, target), out in zip(WINDOWS, outputs, strict=True)
    ]


@app.local_entrypoint()
def main() -> None:
    for r in generate.remote():
        print("=" * 90)
        print(r["label"])
        print("=" * 90)
        print("PREFIX (last 100 chars):")
        print(f"  {r['prefix_tail']!r}")
        print("\nTARGET (actual song lyric):")
        print(f"  {r['target']!r}")
        print("\nMODEL OUTPUT (free greedy continuation, 80 tokens):")
        print(f"  {r['model_output']!r}")
        print()
