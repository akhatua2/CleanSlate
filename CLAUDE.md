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

## Architecture

### Core: MemorizationEvaluator (`cleanslate/evaluator.py`)

Primary interface, exported from `cleanslate/__init__.py`.

- **`MemorizationResult`** (dataclass): Per-window result with `p_z`, `log_prob`, `is_memorized`, `num_target_tokens`, `target_log_probs`, `target_ranks`, metadata fields (`content_id`, `content_title`, `content_creators`, `content_year`). Method: `to_dict()`
- **`SlidingWindowResult`** (dataclass): Aggregate across windows — `max_p_z`, `mean_p_z`, `median_p_z`, `memorized_windows`, `coverage`, `total_windows`
- **`ExperimentSummary`** (dataclass): Batch experiment statistics — `memorized_count`, `memorized_fraction`, p_z stats, timing. Method: `to_dict()`
- **`_Window`** (internal dataclass): `prefix`, `suffix`, `start_char`, `end_char`
- **`_generate_windows()`**: Slices text into overlapping (prefix, suffix) pairs in character space
- **`_coverage()`**: Computes fraction of text covered by memorized windows
- **`MemorizationEvaluator`**: `__init__(model, config_path=None, backend=None)`, lazy-loads VLLMBackend
  - `sliding_window(text, name)` — evaluate single text
  - `sliding_window_batch(texts, names, batch_size=256)` — batch evaluation
  - `compute_summary(results, dataset_name)` — compute experiment statistics
- **`save_results()`** (module-level function): Saves results to Parquet + JSON metadata

### Configuration (`cleanslate/config.py` + `config.yaml`)

- `load_config(path)` — loads YAML config, defaults to project root `config.yaml`
- `get_eval_config(cfg)` — extracts `eval` section
- `get_vllm_config(cfg)` — extracts `vllm` section

Default `config.yaml`:
```yaml
eval:
  window_prefix_len: 100    # chars
  window_suffix_len: 100    # chars
  window_stride: 10         # chars
  tau_min: 0.001            # p_z threshold for "memorized"

vllm:
  num_gpus: 1
  gpu_memory_utilization: 0.90
  max_model_len: 4096
  dtype: bfloat16
```

### Backends (`cleanslate/backends/`)

Single backend: **vLLM** (requires vllm >= 0.19.0).

- `InferenceBackend` (ABC in `base.py`): Abstract properties `model_name`, `tokenizer`. Abstract method `compute(seed, target) -> ProbabilityResult`. Concrete method `compute_batch(pairs)` with default serial fallback.
- `ProbabilityResult` (dataclass in `base.py`): `log_prob`, `num_tokens`, `token_log_probs`, `token_ranks`
- `VLLMBackend` (`vllm_backend.py`): Initializes vLLM `LLM` with config options (trust_remote_code, dtype, gpu_memory_utilization, max_model_len, tensor_parallel_size, enforce_eager). Uses `SamplingParams(max_tokens=1, prompt_logprobs=1)` for teacher-forced probability extraction. Implements batched `compute_batch()`.

### Metrics (`cleanslate/metrics/`)

- `ExtractionResult` (dataclass in `probability.py`): `log_prob`, `prob`, `num_tokens`
- `compute_extraction_probability(token_log_probs)` — computes p_z = exp(sum(log_probs)), handles edge cases (empty → prob=1.0, non-finite → prob=0.0)

### CLI (`cleanslate/cli/`)

**Entry point**: `cleanslate = "cleanslate.cli:main"`

- `main.py`: Typer app setup, registers `eval`, `generate`, `judge`, `validate`, `score`, and the `fsid` sub-app (`build-pool`, `curate`, `train`)
- `eval.py`: `eval_cmd()` — loads HF dataset, filters items, runs `sliding_window_batch()`, saves to Parquet, optionally pushes to HF Hub
- `validate.py`: `validate_cmd()` — runs lm-eval benchmarks via vLLM, displays results table, saves JSON, optionally compares with a previous run
  - Options: `-d` (HF dataset), `-c` (HF config), `--text-col`, `-n` (limit), `--offset`, `-m` (model), `-o` (output dir), `--hf-org`, `--no-push`, `--config` (YAML path)
  - Default dataset: `unlearning-cleanslate/cleanslate_dataset`
  - Metadata columns: `content_id`, `content_title`, `content_creators`, `content_year`
- `aliases.py`: `MODEL_ALIASES` dict and `resolve_model(model) -> (hf_path, clean_name)`
- `fsid.py`: `fsid_app` Typer sub-app — `build_pool`, `curate`, `train` subcommands
- `score.py`: top-level `cleanslate score` — compares pre/post memorization artifacts (and optional pre/post validation JSON)

**Model aliases**:
```
llama-8b      → meta-llama/Llama-3.1-8B
qwen-8b       → Qwen/Qwen3-8B
qwen-8b-base  → Qwen/Qwen3-8B-Base
olmo-7b       → allenai/Olmo-3-1025-7B
olmo-32b      → allenai/Olmo-3-1125-32B
gemma-12b     → google/gemma-3-12b-pt
```

### Validation (`cleanslate/validation/`)

Integration with lm-evaluation-harness for benchmark validation. Three CLI verbs, two phases:

- `cleanslate generate` → `run_generate(model_name, tasks, output_dir, ...)` — invokes `lm_eval.simple_evaluate(..., predict_only=True, evaluation_tracker=EvaluationTracker(output_path=<dir>))`. Skips metrics entirely; writes per-sample JSONL to `<dir>/**/samples_{task}_{date}.jsonl` so vLLM output survives any judge failure.
- `cleanslate judge` → `judge_generations(samples_dir, model_name)` — reads the samples JSONL, dispatches per task:
  - Tasks in `_JUDGE_TASKS` (currently `{cleanslate_qa}`): `ThreadPoolExecutor(max_workers=50)` over Gemini calls via `is_correct`. Fails loudly (re-raises) if the judge itself errors after its retry budget — generations are already durable, so the fix is `cleanslate judge` again.
  - Other tasks (MCQ-style): replay lm-eval's own `task.process_results(doc, resps)` on each saved sample, so scoring logic stays authoritative.
- `cleanslate validate` — end-to-end wrapper: `generate` then `judge`. Same output shape as before; generations are persisted in between under `<output>/generations/` so a flaky judge no longer throws away hours of vLLM work.
- `run_validation(model_name, tasks, ...)` — the legacy one-shot path (lm-eval with metrics inline); still exported and still works if you want the old behavior.
- `BenchmarkResult` (dataclass in `base.py`): `benchmark_name`, `model_name`, `accuracy`, `num_correct`, `num_total`, `per_subset`, `eval_time_seconds`, `metadata`. Method: `to_dict()`
- `SubsetResult` (dataclass in `base.py`): `name`, `accuracy`, `num_correct`, `num_total`
- `save_validation()`, `load_validation()` — JSON persistence
- `compare_validations()` — returns list of `ValidationDelta` (pre/post accuracy comparison)
- `ValidationDelta` (dataclass in `results.py`): `benchmark_name`, `pre_accuracy`, `post_accuracy`, `delta_accuracy`, `relative_retention`
- `judge.py`: `is_correct(reference, model_answer, *, max_retries=5)` — binary QA judging via Gemini (`gemini-3.1-flash-lite-preview`) through LiteLLM. Retries transient errors (`ServiceUnavailableError`, `RateLimitError`, `APIConnectionError`, `Timeout`, `InternalServerError`, `json.JSONDecodeError`) with exponential backoff (`2**attempt`) and **raises the original exception on exhaustion** — no silent `return False`. Requires `GEMINI_API_KEY` env var.

**Custom lm-eval tasks** (`validation/tasks/`):
- `cleanslate_qa.yaml` — QA benchmark against `unlearning-cleanslate/cleanslate_dataset` (config: `qa_benchmark`). Uses `llm_judge_match` as the sole metric (Gemini-based semantic answer matching). Only relevant for the legacy `run_validation` path; the new `judge_generations` applies the threadpool judge directly.
- `utils.py` — `llm_judge_match(references, predictions)` metric function for lm-eval (legacy path only).

### FSID Module (`cleanslate/fsid/`)

Unlearning pipeline.

**Config** (`fsid/config.py`):
- `TrainerConfig` (dataclass): all training + data + logging knobs. Training is **FSDP-only** — launch with `torchrun --nproc_per_node=N` (N=1 works). Checkpointing: `save_every` (0 = final only) and `save_total_limit` (retention cap). Types: `ObjectiveType = Literal["ga", "npo", "simnpo", "rmu", "undial"]`, `FSDPShardingStrategyType = Literal["full_shard", "shard_grad_op"]`, `RetainLossType = Literal["nll", "kl", "embed_diff"]`.
- `load_config(yaml_path, *, cli_overrides)` — YAML + CLI merge. Raises on unknown keys and invalid values.

**Data** (`fsid/datasets/`):
- `io.py :: load_local_records(path)` — parquet / json / jsonl / HF dataset ID.
- `pool.py :: build_candidate_pool(...)`, `split_pool_records(...)`, `save_pool_datasets(...)` — stratified memorization pool, forget/retain split, parquet writes.
- `unlearning.py :: ForgetPairDataset`, `TextSequenceDataset`, `build_forget_pairs(...)`, `build_forget_pairs_from_windows(...)`, `tokenize_text_rows(...)`.

**Rules** (`fsid/rules/`):
- `FORGET_SET_RULES` registry: `baseline` (direct-lyric forget rows) and `corpus_sweep` (dolma-matched windows from `unlearning-cleanslate/corpus_sweep_cleanslate_dataset`).
- `baseline.py :: curate_baseline_sets(...)`, `save_curated_sets(output_dir, forget_rows, retain_rows, metadata)`.
- `corpus_sweep.py :: curate_corpus_sweep_sets(...)`.
- `_retain.py :: load_wikitext_retain_rows(...)` — shared retain loader.

**Training** (`fsid/training/`):
- `modeling.py :: load_tokenizer(model_name)`, `load_model(config, *, device_map)` — HF `AutoModelForCausalLM` + rope_scaling patch + FSDP-uniform dtype.
- `losses.py`: `compute_batch_nll`, `compute_npo_loss`, `compute_simnpo_loss`, `compute_undial_loss`, `compute_rmu_forget_loss`, `compute_activation_mse`, `compute_embed_diff_retain_loss`, `compute_kl_retain_loss`, `forward_with_cache`.  
- `data.py :: ForgetRetainDataset` (anchored pair dataset), `UnlearnDataCollator` (nested forget/retain collator).
- `trainer.py :: CleanSlateUnlearnTrainer` (extends HF `Trainer`, dispatches by objective), `train_unlearning(config)` entry point.

**Scoring** (`fsid/scoring.py`):
- `compute_score(*, forget_set_path, pre_eval_path, post_eval_path) -> UnlearningScore`
- `save_score(score, path)`
- Dataclasses: `WindowDelta`, `ContentDelta`, `UnlearningScore`.

**Configs** (`fsid/configs/`): `ga.yaml`, `npo.yaml`, `simnpo.yaml`, `simnpo_gentle.yaml`.

**QA generation pipeline** (`fsid/qa/` — unchanged from v2): `extract → generate → filter → push` stages that build `unlearning-cleanslate/cleanslate_dataset:qa_benchmark` for `cleanslate validate -B cleanslate_qa`.

## Dependencies

**Core** (always installed): `vllm==0.19.0`, `torch==2.10.0`, `transformers==4.57.6`, `pyarrow==23.0.1`, `pyyaml==6.0.3`, `datasets==4.7.0`, `typer==0.24.1`, `rich==14.3.3`, `tqdm==4.67.3`
**Validation** (`[eval]`): `lm-eval==0.4.11`, `litellm`, `python-dotenv`
**Unlearning** (`[unlearning]`): `accelerate>=0.34`
**Curation** (`[curation]`): `polars`, `zstandard`, `bm25s`, `numpy`
**Dev** (`[dev]`): `ruff`, `mypy`, `pytest`

Requires Python 3.12. Uses `hatchling` build backend. Managed with `uv` (lockfile: `uv.lock`).

## Development Notes

### Type Strictness
`mypy --strict` is configured in `pyproject.toml`. All function parameters and return types must be explicitly typed. Use `X | Y` union syntax (Python 3.12+).

### Testing
- `tests/test_windowing.py`: Character-based windowing tests (stride, prefix/suffix lengths, reconstruction, Unicode handling for Chinese, emoji, Arabic, mixed scripts)

### Key Design Decisions
- **vLLM is the sole backend** — uses prompt logprobs for teacher-forced probability extraction
- **Character-space sliding window**: Text is sliced by character offsets (prefix_len, suffix_len, stride), not token offsets
- **Batch evaluation**: `sliding_window_batch()` collects all windows across texts and evaluates them in batched vLLM calls
- **Lazy loading**: Backend loads model on first `sliding_window()` call
- **YAML config**: Window parameters and vLLM settings live in `config.yaml`, not CLI flags
- **No generation**: v2 removed text generation, decoding strategies, and BLEU/ROUGE metrics — evaluation is purely probability-based

### Adding New Backends
1. Create file in `cleanslate/backends/`
2. Inherit from `InferenceBackend`
3. Implement: `compute(seed, target) -> ProbabilityResult`, properties `model_name` and `tokenizer`
4. Optionally override `compute_batch()` for efficient batching
5. Export from `backends/__init__.py`
