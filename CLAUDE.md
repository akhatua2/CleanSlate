# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

CleanSlate is a Python library for **sequence likelihood estimation** - measuring how likely an LLM is to generate specific text given a prompt/seed. It supports two operational modes:

1. **Standard Mode**: Computes metrics (perplexity, rank-1 fraction, avg rank) on seed/target pairs
2. **Generation Mode**: Generates multiple outputs per seed and saves raw data (tokens, logprobs, alternatives) for later analysis

## Development Commands

### Setup
```bash
# Install in editable mode with all dependencies
pip install -e ".[cli,export]"

# Upgrade dependencies
pip install --upgrade -e ".[cli,export]"
```

### Running Experiments

**Standard mode (compute metrics immediately):**
```bash
# Via CLI
cleanslate run -m "Qwen/Qwen2.5-0.5B-Instruct" -d shakespeare -o results.json

# Via Python
python run_trial.py

# Single seed/target pair
cleanslate probe "To be or not to be" ", that is the question" -m "Qwen/Qwen2.5-0.5B-Instruct"
```

**Generation mode (save raw data):**
```bash
# Via CLI
cleanslate generate -m "Qwen/Qwen2.5-0.5B-Instruct" -d shakespeare -n 3 -o generations.parquet

# Via Python - see examples in GENERATION_MODE.md
```

### Code Quality

The project uses:
- **Ruff** for linting (configured in `pyproject.toml`)
- **mypy** with strict mode for type checking

```bash
# Type check
mypy cleanslate

# Lint
ruff check cleanslate

# Format (Ruff also handles formatting)
ruff format cleanslate
```

## Architecture

### Core Components

**`Experiment` class** (`cleanslate/experiment.py:14`)
- Main harness for running likelihood experiments
- Lazy-loads inference backend on first use
- Methods: `run()` for batch experiments, `run_single()` for single pair, `compare_targets()` for comparing alternatives
- Backend selection: "huggingface" (recommended), "vllm", or "sglang"

**`InferenceBackend` abstract class** (`cleanslate/backends/base.py:39`)
- All backends inherit from this
- Key method: `compute(seed, target) -> ProbabilityResult`
- Returns: log probability, perplexity, token-level metrics, token ranks

**`ProbabilityResult` dataclass** (`cleanslate/backends/base.py:8`)
- Stores computation results
- Key properties: `log_prob`, `perplexity`, `rank_1_fraction`, `avg_rank`, `avg_log_prob`
- Per-token data: `token_log_probs`, `token_ranks`

**Results Storage**
- `SampleResult` (`cleanslate/results.py:9`): Single (seed, target) result
- `ExperimentResults` (`cleanslate/results.py:33`): Collection with aggregate stats, computed on init

### Backend Structure

All backends in `cleanslate/backends/`:
- **HuggingFace** (`huggingface.py`) - **RECOMMENDED**: Simple, memory-efficient, GPU-accelerated. Uses `device_map="auto"` for multi-GPU support. Both `compute()` and `generate()` fully implemented.
- **SGLang** (`sglang_backend.py`) - GPU-optimized backend with memory-efficient serving. Both `compute()` and `generate()` fully implemented. Captures token logprobs and top-k alternatives per position. Requires: `pip install cleanslate[sglang]` (builds from source).
- **vLLM** (`vllm_backend.py`) - Basic implementation; not recommended due to API changes (v0.15.0+) and memory overhead (see `BACKEND_RECOMMENDATIONS.md`)

### Data Flow

1. User provides `seed` (prefix) and `target` (continuation)
2. Backend tokenizes `seed + target`, gets logits from forward pass
3. Extracts log probabilities for target tokens only
4. Computes: total log-prob (sum), perplexity (exp(-avg)), ranks (position in top-k)
5. Returns `ProbabilityResult` with all metrics and per-token data

### Generation Mode

**Generation System** (`cleanslate/generation.py`)
- `GenerationRow`: Dataclass for flat Parquet/HuggingFace Datasets output (one row = one generation)
- Generates N outputs per seed using backend's `generate()` method
- Saves to Parquet (fast columnar format) or HuggingFace Datasets
- Captures full token sequence, logprobs, top-k alternatives per position
- Each backend (HuggingFace, SGLang) implements token-level probability capture

### Evaluation Metrics

**Metrics Module** (`cleanslate/metrics/`)
- `Metric` abstract base class for evaluation metrics
- **N-gram metrics**: BLEU (BLEU1, BLEU2), ROUGE (ROUGE1, ROUGE2, ROUGEL) for sequence similarity
- **Probability metrics** (`probability.py`): Hayes et al. extraction probability framework
  - `compute_extraction_probability()`: Compute probability of extracting target text
  - `prompts_for_probability()`: Generate prompts for probability estimation
  - `probability_after_n_prompts()`: Estimate extraction probability with N prompts
- All metrics return `MetricResult` with score and metadata

## Key Files & Patterns

**Sample datasets** (`cleanslate/samples/`)
- `SHAKESPEARE_SAMPLES`: Hamlet, Romeo & Juliet, Macbeth, etc.
- `BOOK_SAMPLES`: Classic literature (1984, Pride & Prejudice, etc.)
- Sample format: `{"name": str, "seed": str, "target": str}`

**CLI entry point** (`cleanslate/cli.py:19`)
- Built with Typer (requires `pip install -e ".[cli]"`)
- Commands: `run`, `generate`, `probe`, `show`, `compare`, `list-samples`
- All CLI commands can be discovered via `cleanslate --help`

**Configuration**
- No explicit config files; all configuration via CLI args or function parameters
- Backend selected at `Experiment` init time
- Device auto-detection (CUDA > MPS > CPU) with manual override via `--device`

## Generation Data Captured

When using `generate()` method on any backend, the following data is captured per generation:

- **Generated Text**: Full decoded output
- **Token IDs & Strings**: Token-by-token IDs and decoded strings
- **Token Log Probabilities**: Log probability of each generated token
- **Top-K Alternatives**: Per position, the top `capture_top_k` (default 50) token IDs and their log probabilities
- **Generation Parameters**: temperature, top_k, top_p, do_sample, max_new_tokens (for reproducibility)
- **Timing**: Generation time in milliseconds

This data is returned in `GenerationResult` dataclass and can be saved to Parquet or HuggingFace Datasets formats via `generation.py`.

## Important Design Decisions

### Why HuggingFace Backend?

See `BACKEND_RECOMMENDATIONS.md` for detailed analysis. TL;DR:
- Simple, no complex setup
- ~3GB memory vs vLLM's 40GB+ for same model
- Direct logits access, no hacks needed
- GPU-accelerated when available

### Computation Strategy

- **Per-sample** not batched: Each (seed, target) runs independently (TODO item in GENERATION_MODE.md:254)
- **Lazy loading**: Model loads on first `compute()` call, stays in memory
- **No caching**: Each experiment run computes fresh (appropriate for memorization testing)

## Development Notes

### Type Strictness

Project uses `mypy --strict`, so:
- All function parameters must be typed
- All return types explicit
- No implicit `Any` types
- Use union syntax `X | Y` (Python 3.12+)

### Testing

No formal test suite in the repo. Tests are:
- `run_trial.py`: Manual integration test with Shakespeare samples
- `test_generation.py` (untracked): Generation mode testing

### Adding New Backends

1. Create file in `cleanslate/backends/`
2. Inherit from `InferenceBackend`
3. Implement both abstract methods:
   - `compute(seed, target, **decoding_options)` -> `ProbabilityResult`: Compute log probability for target given seed
   - `generate(seed, max_new_tokens, temperature, top_k, top_p, do_sample, capture_top_k)` -> `GenerationResult`: Generate text and capture token logprobs + top-k alternatives
4. Implement properties: `device` (str), `model_name` (str)
5. Add backend type to `BackendType` literal in `experiment.py`
6. Add initialization logic in `Experiment._get_estimator()` method

### Adding New Sample Datasets

1. Add to appropriate file in `cleanslate/samples/`
2. Create list with `{"name": str, "seed": str, "target": str}` dicts
3. Export from `cleanslate/samples/__init__.py`
4. Update CLI choices if needed (`cli.py:29`)

## Dependencies

**Core**: torch, transformers, accelerate (HF ecosystem)
**Data**: pandas, polars, pyarrow, fastparquet (data handling)
**CLI**: typer, rich (if using CLI)
**Export**: datasets (HuggingFace Datasets format)

## Common Issues & Solutions

- **Import errors**: Ensure you've installed with `pip install -e ".[cli,export]"` for full functionality
- **GPU memory**: HuggingFace backend uses `device_map="auto"` to distribute across all available GPUs
- **Model loading**: First run is slow (downloads & caches model), subsequent runs reuse cached model
- **Output formats**: Generation mode saves to Parquet by default (fast, pandas/DuckDB compatible)
