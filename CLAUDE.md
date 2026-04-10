# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Overview

CleanSlate is a **sliding window memorization evaluation framework** for LLMs. It measures whether a model has memorized specific text by computing extraction probability (p_z) across overlapping character-level windows using vLLM for inference.

Based on:
- Cooper et al. "Extracting memorized pieces of copyrighted books"
- Hayes et al. "Measuring memorization in language models"

Primary workflow:
- **Memorization Evaluation** — Slide a window across text, compute p_z = P(suffix | prefix) via teacher forcing using vLLM prompt logprobs

FSID (Forget Set IDentification) unlearning pipeline is planned but not yet implemented in v2.

## Development Commands

### Setup
```bash
uv sync

# For unlearning (adds peft/LoRA + accelerate):
uv sync --extra unlearning

# For validation benchmarks (adds lm-eval):
uv sync --extra eval

# For development (ruff, mypy, pytest):
uv sync --extra dev
```

### Primary CLI

```bash
# Sliding window memorization evaluation
cleanslate eval \
  -d unlearning-cleanslate/cleanslate_dataset \
  -c default \
  -m "Qwen/Qwen3-8B" \
  -n 100 \
  -o results/eval
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

- `main.py`: Typer app setup, registers `eval` command
- `eval.py`: `eval_cmd()` — loads HF dataset, filters items, runs `sliding_window_batch()`, saves to Parquet, optionally pushes to HF Hub
  - Options: `-d` (HF dataset), `-c` (HF config), `--text-col`, `-n` (limit), `--offset`, `-m` (model), `-o` (output dir), `--hf-org`, `--no-push`, `--config` (YAML path)
  - Default dataset: `unlearning-cleanslate/cleanslate_dataset`
  - Metadata columns: `content_id`, `content_title`, `content_creators`, `content_year`
- `aliases.py`: `MODEL_ALIASES` dict and `resolve_model(model) -> (hf_path, clean_name)`

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

Integration with lm-evaluation-harness for benchmark validation.

- `run_validation(model_name, tasks, *, limit, num_fewshot, batch_size, gpu_memory_utilization, tp_size, max_model_len)` — runs lm-eval with vLLM backend, returns `dict[str, BenchmarkResult]`
- `BenchmarkResult` (dataclass in `base.py`): `benchmark_name`, `model_name`, `accuracy`, `num_correct`, `num_total`, `per_subset`, `eval_time_seconds`, `metadata`. Method: `to_dict()`
- `SubsetResult` (dataclass in `base.py`): `name`, `accuracy`, `num_correct`, `num_total`
- `save_validation()`, `load_validation()` — JSON persistence
- `compare_validations()` — returns list of `ValidationDelta` (pre/post accuracy comparison)
- `ValidationDelta` (dataclass in `results.py`): `benchmark_name`, `pre_accuracy`, `post_accuracy`, `delta_accuracy`, `relative_retention`

### FSID Module (`cleanslate/fsid/`)

Placeholder — unlearning pipeline planned but not yet implemented in v2.

## Dependencies

**Core** (always installed): `vllm==0.19.0`, `torch==2.10.0`, `transformers==4.57.6`, `pyarrow==23.0.1`, `pyyaml==6.0.3`, `datasets==4.7.0`, `typer==0.24.1`, `rich==14.3.3`, `tqdm==4.67.3`
**Unlearning** (`[unlearning]`): `peft==0.18.1`, `accelerate==1.13.0`
**Validation** (`[eval]`): `lm-eval==0.4.11`
**Dev** (`[dev]`): `ruff`, `mypy`, `pytest`

Requires Python >=3.12. Uses `hatchling` build backend. Managed with `uv` (lockfile: `uv.lock`).

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
