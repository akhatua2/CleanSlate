"""Scan results/validation/ and render a markdown results matrix.

Handles both layouts:
    results/validation/<model>.json                  (combined, {"benchmarks": {...}})
    results/validation/<model>/<benchmark>.json      (per-benchmark)

Writes results/validation/SUMMARY.md.

Usage:
    uv run python scripts/update_results_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MODELS = [
    ("llama-3_1-8b",         "Llama-3.1-8B"),
    ("qwen3-8b",             "Qwen3-8B"),
    ("olmo-3-7b",            "Olmo-3-7B"),
    ("nemotron-nano-9b-v2",  "Nemotron-9B"),
    ("gemma-3-12b",          "Gemma-3-12B"),
    ("olmo-3-32b",           "Olmo-3-32B"),
]

BENCHMARKS: list[tuple[str, str]] = [
    # (task, domain)
    ("mmlu",            "Knowledge"),
    ("triviaqa",        "Factual recall"),
    ("cleanslate_qa",   "Forget-set retain"),
    ("gsm8k",           "Math"),
    ("arc_challenge",   "Science reasoning"),
    ("hellaswag",       "Commonsense"),
    ("winogrande",      "Commonsense"),
    ("drop",            "Reading comp."),
    ("coqa",            "Reading comp."),
    ("bbh",             "Reasoning (held-out)"),
    ("humaneval_plus",  "Code"),
    ("lambada_openai",  "Language modeling"),
]


def _scores_for_model(root: Path, slug: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    # Per-benchmark nested files
    nested = root / slug
    if nested.is_dir():
        for f in nested.glob("*.json"):
            try:
                d = json.loads(f.read_text())
                for bname, bdata in d.get("benchmarks", {}).items():
                    scores[bname] = float(bdata["accuracy"])
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
    # Combined files at root (any that start with slug)
    for f in root.glob(f"{slug}*.json"):
        if f.is_file():
            try:
                d = json.loads(f.read_text())
                for bname, bdata in d.get("benchmarks", {}).items():
                    scores.setdefault(bname, float(bdata["accuracy"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
    return scores


def render_table(root: Path) -> str:
    all_scores = {slug: _scores_for_model(root, slug) for slug, _ in MODELS}

    header = ["Domain", "Benchmark"] + [label for _, label in MODELS]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for bench, domain in BENCHMARKS:
        row = [domain, bench]
        for slug, _ in MODELS:
            s = all_scores[slug].get(bench)
            row.append(f"{s:.4f}" if s is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    root = Path("results/validation")
    table = render_table(root)
    out = root / "SUMMARY.md"
    out.write_text("# Validation Results\n\n" + table + "\n")
    print(table)
    print(f"\n-> {out}")


if __name__ == "__main__":
    sys.exit(main())
