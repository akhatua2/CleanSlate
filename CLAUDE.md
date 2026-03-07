# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Overview

CleanSlate is a **sliding window memorization evaluation and unlearning framework** for LLMs. It measures whether a model has memorized specific text by computing extraction probability (p_z) across overlapping windows, and provides a full FSID (Forget Set IDentification) pipeline for unlearning memorized content.

Based on:
- Cooper et al. "Extracting memorized pieces of copyrighted books"
- Hayes et al. "Measuring memorization in language models"

Two main workflows:
1. **Memorization Evaluation** — Slide a window across text, compute p_z = P(suffix | prefix) via teacher forcing, optionally generate and compare with BLEU/ROUGE
2. **FSID Unlearning** — Build forget/retain sets from eval results, train unlearning objectives (GA, NPO, SimNPO, GradDiff), score effectiveness, validate utility retention

## Development Commands

### Setup
```bash
pip install -e ".[cli,export]"

# For unlearning (adds peft/LoRA):
pip install -e ".[cli,export,unlearning]"

# For validation benchmarks (adds lm-eval):
pip install -e ".[cli,export,eval]"
```

### Primary CLI

```bash
# Sliding window memorization evaluation (main command)
cleanslate eval \
  -d unlearning-cleanslate/cleanslate_dataset \
  -c default \
  -m "Qwen/Qwen3-8B" \
  --window-size 50 --stride 25 \
  -n 100

# FSID unlearning workflow
cleanslate fsid build-pool --eval-dataset <hf_dataset> --threshold 0.10
cleanslate fsid curate --forget-set-rule baseline --forget-pool-path data/fsid/pool/forget.parquet
cleanslate fsid train -m <model> --forget-path <path> --objective npo
cleanslate fsid score --forget-path <path> --pre-eval <path> --post-eval <path>
cleanslate fsid validate -m <model> -B mmlu,gsm8k
```

### Code Quality
```bash
mypy cleanslate          # Type check (strict mode)
ruff check cleanslate    # Lint
ruff format cleanslate   # Format
```

## Architecture

### Core: MemorizationEvaluator (`cleanslate/evaluator.py`)

Primary interface, exported from `cleanslate/__init__.py`.

- **`MemorizationConfig`** (dataclass, line 35): `window_prefix_len`, `window_suffix_len`, `window_stride`, `tau_min` (0.1% default), `do_generate`, `decode_strategy` ("greedy"/"beam"), `beam_width`, `reproduction_threshold`
- **`MemorizationResult`** (dataclass, line 72): Per-window result with `p_z`, `log_prob`, `perplexity`, `is_memorized`, `bleu_1`, `rouge_l`, `is_reproduced`, beam search outputs
- **`SlidingWindowResult`** (dataclass, line 160): Aggregate across windows — `max_p_z`, `mean_p_z`, `memorized_windows`, `coverage`, `char_to_max_pz` heatmap data
- **`ExperimentSummary`** (dataclass, line 214): Batch experiment statistics
- **`MemorizationEvaluator`** (class, line 253): `__init__(model, backend, config)`, methods: `evaluate_single(seed, target)`, `sliding_window(text, name)`, `save_results()`
- **`save_results()`** (function): Saves results to Parquet + optional HF Hub push

### Legacy: Experiment & GenerationExperiment

- **`Experiment`** (`experiment.py:19`): Legacy batch likelihood computation. Methods: `run()`, `run_single()`, `compare_targets()`
- **`GenerationExperiment`** (`generation.py`): Batch generation with double-buffered I/O. `GenerationRow` dataclass for flat Parquet output

### Backends (`cleanslate/backends/`)

Factory: `create_backend(name, model_name, device=..., **kwargs)` in `__init__.py`

| Backend | File | Best For |
|---------|------|----------|
| **SGLang** (default for eval) | `sglang_backend.py` | GPU batched inference, supports `dp_size` for multi-GPU |
| **HuggingFace** | `huggingface.py` | Simple setup, CPU/GPU, `device_map="auto"` |
| **vLLM** | `vllm_backend.py` | Not recommended (API instability) |

Base classes in `base.py`:
- `InferenceBackend` (ABC): `compute(seed, target) -> ProbabilityResult`, `generate(seed, strategy, ...) -> DecodingOutput`
- `ProbabilityResult` (dataclass, line 49): `log_prob`, `perplexity`, `token_log_probs`, `token_ranks`
- `GenerationResult` (dataclass, line 14): `generated_text`, `token_log_probs`, `top_k_tokens`
- `SongGenerationOutput` (Pydantic model, line 6): Structured output for song generation

### Decoding Strategies (`cleanslate/decoding/`)

- `DecodingStrategy` (ABC in `base.py`): Base class
- `GreedyDecoding` (`greedy.py`): Always pick highest probability token
- `BeamSearchDecoding` (`beam.py`): Maintain top-k hypotheses with `beam_width` and `length_penalty`
- Factory: `create_strategy(name, **kwargs)` in `__init__.py`
- `DecodingOutput`: Contains list of `GenerationSequence` (one per beam)

### Metrics (`cleanslate/metrics/`)

- `Metric` (ABC in `base.py`): `compute(reference, hypothesis) -> MetricResult`
- `BLEU1`, `BLEU2` (`bleu.py`): N-gram precision
- `ROUGE1`, `ROUGEL` (`rouge.py`): Recall-oriented overlap
- `compute_extraction_probability()` (`probability.py`): Hayes et al. p_z from token log-probs

### FSID Module (`cleanslate/fsid/`)

Full unlearning pipeline:

**Config** (`config.py`):
- `TrainerConfig` (dataclass, line 33): All training parameters
  - Objectives: `ObjectiveType = "ga" | "grad_diff" | "npo" | "simnpo"`
  - Train modes: `"full" | "lora"`
  - Distributed: `"single" | "fsdp"` with `fsdp_sharding_strategy`, `fsdp_cpu_offload`
  - LoRA params: `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules`
  - Validation: `validation_path`, `eval_every`, `eval_max_samples`
- `load_config(yaml_path, cli_overrides)`: Load from YAML with CLI override merging

**Datasets** (`datasets/`):
- `pool.py`: `build_candidate_pool()` — stratified sampling from eval results above memorization threshold
- `unlearning.py`: `build_forget_pairs()` — sliding window (prefix, suffix) pairs for training; `ForgetPairDataset`, `TextSequenceDataset`
- `io.py`: `load_local_records()` — load Parquet/JSON/JSONL files

**Rules** (`rules/`):
- `baseline.py`: Default forget-set curation policy — uses direct prompt template, none seed mode

**Training** (`training/`):
- `trainer.py`: `CleanSlateUnlearnTrainer` (extends HF `Trainer`), `train_unlearning(config)` entry point
  - Objectives: GA (gradient ascent), GradDiff (gradient difference), NPO, SimNPO
- `losses.py`: `compute_batch_nll()`, `compute_npo_loss()`, `compute_simnpo_loss()`, `compute_kl_retain_loss()`
- `modeling.py`: `load_model()`, `load_tokenizer()` — model loading with dtype/device handling
- `data.py`: `ForgetRetainDataset` (interleaved), `UnlearnDataCollator`

**Scoring** (`scoring/benchmark.py`):
- `compute_score()`: Compare pre/post eval Parquets → `UnlearningScore` with forget_score, retain_penalty
- Window-level transitions: forgotten, retained_memorized, retained_clean, regressed

**Validation** (`validation/`):
- `run_validation()` (`__init__.py`): lm-eval-harness with SGLang backend
- `BenchmarkResult`, `SubsetResult` (`base.py`): Per-benchmark accuracy
- `compare_validations()`, `load_validation()`, `save_validation()` (`results.py`)

**YAML Configs** (`configs/`): `ga.yaml`, `npo.yaml`, `simnpo.yaml`, `grad_diff.yaml`

### Supporting Modules

- **`prompts.py`**: `PromptTemplate` dataclass with `format_messages()`. Multiple templates for lyrics extraction testing (direct, helpful_assistant, etc.). `ALL_TEMPLATES` dict, `SeedMode` literal
- **`chat_format.py`**: `format_chat(tokenizer, messages)` — applies chat template
- **`prompt_dataset.py`**: `build_prompt_splits()` — builds HF DatasetDict from templates x seed modes
- **`cli_utils.py`**: `run_parallel_eval()` (subprocess-based multi-GPU), `generate_hf_repo_name()`, `make_flame_graph()`, `estimate_windows()`
- **`samples/`**: `shakespeare.py`, `books.py`, `songs.py`, `billboard.py` — sample datasets with `{"name", "seed", "target"}` format
- **`results.py`**: `SampleResult` (line 12), `ExperimentResults` (line 72) — legacy result types

## CLI Details (`cleanslate/cli.py`)

**Entry point**: `cleanslate = "cleanslate.cli:main"` (also `python -m cleanslate`)

**Model aliases** (`MODEL_ALIASES` dict, line 56):
```
llama-8b      → meta-llama/Llama-3.1-8B
qwen-8b       → Qwen/Qwen3-8B
qwen-8b-base  → Qwen/Qwen3-8B-Base
olmo-7b       → allenai/Olmo-3-1025-7B
gemma-12b     → google/gemma-3-12b-pt
nemotron-9b   → nvidia/NVIDIA-Nemotron-Nano-9B-v2
trinity-nano  → arcee-ai/Trinity-Nano-Base
gpt-oss-20b   → openai/gpt-oss-20b
```

**`cleanslate eval`** (line 102): Main command. Key options: `-d` (HF dataset), `-c` (config), `-m` (model), `-w` (window size), `--stride`, `-t` (threshold), `--reproduce` (enable generation), `--decode` (greedy/beam), `--beam-width`, `-g` (num GPUs), `--hf-push`, `--no-push`

**`cleanslate fsid`** subcommands: `build-pool`, `curate`, `train`, `score`, `validate` — see FSID Module above

**Default dataset**: `unlearning-cleanslate/cleanslate_dataset`

**Metadata columns**: `content_id`, `content_title`, `content_creators`, `content_year` (also aliased as `song_id`/`song_title`)

## Dependencies

**Core**: `torch`, `transformers`, `accelerate`, `pydantic>=2`, `pyyaml`, `python-dotenv`
**Data**: `polars`, `pyarrow`, `fastparquet`, `datasets`
**CLI**: `typer`, `rich` (optional group `[cli]`)
**Export**: `pyarrow`, `datasets` (optional group `[export]`)
**Unlearning**: `peft>=0.11.0` (optional group `[unlearning]`)
**Validation**: `lm-eval` (optional group `[eval]`)
**Backends**: `sglang` (`[sglang]`), `vllm` (`[vllm]`)

Requires Python >=3.12. Uses `hatchling` build backend.

## Development Notes

### Type Strictness
`mypy --strict` is configured in `pyproject.toml`. All function parameters and return types must be explicitly typed. Use `X | Y` union syntax (Python 3.12+).

### Testing
- `test_evaluator.py`: Tests for MemorizationEvaluator
- `test_eval_hf.py`: End-to-end eval with HuggingFace backend

### Key Design Decisions
- **SGLang is default backend** for `cleanslate eval` — optimized for GPU batched inference
- **Sliding window approach**: Text is evaluated across overlapping windows (prefix + suffix), not as single seed/target pairs
- **Per-sample not batched**: Each sliding window computation runs independently
- **Lazy loading**: Backend loads model on first use
- **FSDP for large models**: 7B+ models use `accelerate launch` with `--distributed-strategy fsdp` and `--gradient-checkpointing`
- **No implicit distributed**: FSDP is opt-in via CLI flag, no automatic `device_map="auto"` for training
- **HF Trainer**: Unlearning uses HF Trainer with custom loss computation via `CleanSlateUnlearnTrainer`

### Adding New Backends
1. Create file in `cleanslate/backends/`
2. Inherit from `InferenceBackend`
3. Implement: `compute(seed, target) -> ProbabilityResult`, `generate(seed, strategy, ...) -> DecodingOutput`
4. Implement properties: `device`, `model_name`, `tokenizer`
5. Register in `BACKEND_REGISTRY` dict in `backends/__init__.py`

### Adding New Unlearning Objectives
1. Add loss function in `cleanslate/fsid/training/losses.py`
2. Add objective name to `ALLOWED_OBJECTIVES` list in `trainer.py`
3. Add case to `CleanSlateUnlearnTrainer.compute_loss()`
4. Add YAML config template in `cleanslate/fsid/configs/`
5. Update `ObjectiveType` literal in `config.py`
