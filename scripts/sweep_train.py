"""Local driver: fan out FSID training across (model × rule × objective) on Modal.

Usage:
    uv run modal run scripts/sweep_train.py

Edit ``MODELS``, ``RULES``, ``PRESETS`` below to expand the matrix. Combos whose
output repo already exists on Hub are skipped automatically by ``modal_train``.
"""

from __future__ import annotations

from itertools import product

import modal

from modal_train import app, train_1gpu, train_2gpu, train_8gpu

# ---------------------------------------------------------------------------
# Matrix (extend these to grow the sweep)
# ---------------------------------------------------------------------------

MODELS: list[str] = [
    "llama-8b",
    "qwen-8b",
    "olmo-7b",
    "nemotron-9b",
    "gemma-12b",
    "olmo-32b",
]

RULES: list[str] = ["baseline"]

# objective_label → preset config stem under cleanslate/fsid/configs/
PRESETS: dict[str, str] = {"simnpo": "simnpo"}

# Model alias → which Modal function to dispatch to.
GPU_FN: dict[str, modal.Function] = {
    "llama-8b": train_1gpu,
    "qwen-8b": train_1gpu,
    "olmo-7b": train_1gpu,
    "nemotron-9b": train_1gpu,
    "gemma-12b": train_2gpu,
    "olmo-32b": train_8gpu,
}


@app.local_entrypoint()
def main() -> None:
    combos = [
        (model, rule, objective, preset)
        for model, rule, (objective, preset) in product(MODELS, RULES, PRESETS.items())
    ]
    print(f"Launching {len(combos)} runs across {len(MODELS)} models, {len(RULES)} rules, {len(PRESETS)} presets")

    futures = []
    for model, rule, objective, preset in combos:
        fn = GPU_FN[model]
        f = fn.spawn(model, rule, objective, preset)
        futures.append((model, rule, preset, f))

    results: list[tuple[str, str, str, str]] = []
    for model, rule, preset, f in futures:
        try:
            outcome = f.get()
        except Exception as exc:  # noqa: BLE001 — fan-in error reporting
            outcome = f"FAIL: {type(exc).__name__}: {exc}"
        print(f"[{model}/{rule}/{preset}] {outcome}")
        results.append((model, rule, preset, outcome))

    ok = sum(1 for *_, r in results if r.startswith("OK") or r.startswith("SKIP"))
    print(f"\nDone: {ok}/{len(results)} succeeded (incl. skipped)")
