# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Overview

CleanSlate is a **sliding-window memorization evaluation and unlearning framework** for LLMs. It measures whether a model has memorized specific text by computing extraction probability (p_z) across overlapping character-level windows using vLLM, then drives an unlearning pipeline that removes memorized content while preserving utility.

Based on:
- Cooper et al. "Extracting memorized pieces of copyrighted books"
- Hayes et al. "Measuring memorization in language models"

Two workflows:
- **Memorization Evaluation** (`cleanslate eval`) — slide a window across text, compute p_z = P(suffix | prefix) via teacher forcing using vLLM prompt logprobs
- **FSID Unlearning** (`cleanslate fsid ...`) — build a forget-request pool from eval results, curate forget/retain training data, train an unlearned checkpoint, score pre vs post memorization

## Development Commands

### Setup
```bash
uv sync

# Validation benchmarks (adds lm-eval):
uv sync --extra eval

# Unlearning training (adds accelerate):
uv sync --extra unlearning

# Development (ruff, mypy, pytest):
uv sync --extra dev
```

### Primary CLI

```bash
# Sliding window memorization evaluation
cleanslate eval -d unlearning-cleanslate/cleanslate_dataset -m "Qwen/Qwen3-8B" -n 100 -o results/eval

# Validation benchmarks (requires [eval])
cleanslate validate -m "Qwen/Qwen3-8B" -B mmlu,hellaswag
cleanslate validate -m qwen-8b -B cleanslate_qa
# Pre/post comparison lives in `cleanslate score --pre-validation --post-validation`

# Two-phase form: generate once, judge repeatedly (and in parallel)
# Phase A persists samples_*.jsonl under <output>/generations/ so a flaky
# judge (Gemini 5xx, rate limits) can be retried without rerunning vLLM.
cleanslate generate -m qwen-8b -B cleanslate_qa -o results/validation/qwen-8b
cleanslate judge -i results/validation/qwen-8b -o results/validation/qwen-8b.json

# Publish generations as an HF dataset (one config per benchmark).
# Repo name: <hf-org>/generations-<model_clean>[-<push-suffix>]
cleanslate generate -m qwen-8b -B mmlu,cleanslate_qa -o results/validation/qwen-8b \
  --push-to-hub --push-suffix corpus_sweep_post_eval
cleanslate validate -m qwen-8b -B mmlu,cleanslate_qa --push-to-hub \
  --push-suffix corpus_sweep_post_eval -o results/validation/qwen-8b

# Task presets: `full` expands to the 12-benchmark retention suite
# (mmlu, triviaqa, cleanslate_qa, gsm8k, hellaswag, winogrande, drop,
#  arc_challenge, humaneval_plus, bbh, lambada_openai, coqa). Mixable
# with individual tasks: `-B full,mmlu_pro`.
cleanslate validate -m qwen-8b -B full -o results/validation/qwen-8b \
  --push-to-hub --push-suffix full_suite

# FSID unlearning pipeline (requires [unlearning])
cleanslate fsid build-pool --eval-dataset <hf_dataset> --threshold 0.30 -o data/fsid/pool
cleanslate fsid curate --rule baseline --forget-pool-path data/fsid/pool/forget.parquet -o data/fsid/curated/baseline
cleanslate fsid train --config cleanslate/fsid/configs/npo.yaml -m qwen-8b \
  --forget-path data/fsid/curated/baseline/forget_set.parquet \
  --retain-path data/fsid/curated/baseline/retain_set.parquet -o runs
cleanslate score --forget-path data/fsid/curated/baseline/forget_set.parquet \
  --pre-eval results/eval/pre.parquet --post-eval results/eval/post.parquet -o results/fsid/score.json
```

### Code Quality
```bash
mypy cleanslate          # Type check (strict mode)
ruff check cleanslate    # Lint
ruff format cleanslate   # Format
pytest tests/            # Run tests
```

## Architecture Notes

Contracts and gotchas that aren't visible from reading a single module — for API details, read the source (`cleanslate/evaluator.py`, `cleanslate/backends/`, `cleanslate/validation/`, `cleanslate/fsid/`).

- **FSID training is FSDP-only** — launch `cleanslate fsid train` with `torchrun --nproc_per_node=N` (N=1 works).
- **The Gemini judge fails loudly**: `is_correct` requires the `GEMINI_API_KEY` env var and raises the original exception when its retry budget is exhausted — never a silent `return False`. Generations are already persisted under `<output>/generations/`, so the fix is rerunning `cleanslate judge`, not regenerating.
- **`cleanslate judge` scoring split**: tasks in `_JUDGE_TASKS` (currently `cleanslate_qa`) go through the threadpooled Gemini judge; all other tasks replay lm-eval's own `task.process_results` on saved samples so scoring logic stays authoritative.
- **`validation/tasks/cleanslate_qa.yaml`'s `llm_judge_match` metric is legacy-path-only** (used by `run_validation`); the two-phase `judge_generations` path applies the judge directly.
- **QA generation pipeline** (`fsid/qa/`): `extract → generate → filter → push` stages build `unlearning-cleanslate/cleanslate_dataset:qa_benchmark` for `cleanslate validate -B cleanslate_qa`.

## Development Notes

### Type Strictness
`mypy --strict` is configured in `pyproject.toml`. All function parameters and return types must be explicitly typed. Use `X | Y` union syntax (Python 3.12+).

### Key Design Decisions
- **vLLM is the sole backend** — uses prompt logprobs for teacher-forced probability extraction
- **Character-space sliding window**: Text is sliced by character offsets (prefix_len, suffix_len, stride), not token offsets
- **Batch evaluation**: `sliding_window_batch()` collects all windows across texts and evaluates them in batched vLLM calls
- **Lazy loading**: Backend loads model on first `sliding_window()` call
- **YAML config**: Window parameters and vLLM settings live in `config.yaml`, not CLI flags
- **No generation**: v2 removed text generation, decoding strategies, and BLEU/ROUGE metrics — evaluation is purely probability-based
