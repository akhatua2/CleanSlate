# CleanSlate

Sliding window memorization evaluation for LLMs. Measures whether a model has memorized specific text by computing extraction probability (p_z) across overlapping character-level windows.

Based on:
- Cooper et al. "Extracting memorized pieces of copyrighted books"
- Hayes et al. "Measuring memorization in language models"

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync

# With validation benchmarks (lm-eval):
uv sync --extra eval

# With dev tools (ruff, mypy, pytest):
uv sync --extra dev
```

## Usage

### Memorization Evaluation

Slide a character-based window across text and compute p_z = P(suffix | prefix) via teacher forcing with vLLM.

```bash
# Evaluate against the default HuggingFace dataset
cleanslate eval \
  -m "Qwen/Qwen3-8B" \
  -d unlearning-cleanslate/cleanslate_dataset \
  -c default \
  -n 100 \
  -o results/eval

# Use a model alias
cleanslate eval -m qwen-8b -n 50

```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model` | HF model name or alias | `Qwen/Qwen3-8B` |
| `-d, --hf-dataset` | HuggingFace dataset | `unlearning-cleanslate/cleanslate_dataset` |
| `-c, --hf-config` | Dataset config | `default` |
| `--text-col` | Text column name | `reference_target` |
| `-n, --limit` | Max items to evaluate | all |
| `--offset` | Skip first N items | `0` |
| `-o, --output` | Output directory | `results/eval` |
| `--hf-org` | HF org for pushing | `unlearning-cleanslate` |
| `--no-push` | Don't push to HF Hub | `false` |
| `--config` | Path to YAML config | `config.yaml` |

### Validation Benchmarks

Run lm-evaluation-harness benchmarks via vLLM to measure model utility.

```bash
# Run default benchmarks (mmlu, hellaswag, arc_easy, arc_challenge)
cleanslate validate -m "Qwen/Qwen3-8B"

# Specific benchmarks with a sample limit for quick testing
cleanslate validate -m qwen-8b -B mmlu,gsm8k -n 100

# Compare against a previous run
cleanslate validate -m qwen-8b --compare results/validation/qwen3-8b.json

# Multi-GPU
cleanslate validate -m olmo-32b --tp-size 4
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model` | HF model name or alias (required) | — |
| `-B, --benchmarks` | Comma-separated lm-eval tasks | `mmlu,hellaswag,arc_easy,arc_challenge` |
| `-n, --limit` | Max items per task | all |
| `--num-fewshot` | Few-shot count override | task default |
| `--batch-size` | Batch size | `auto` |
| `--gpu-mem` | vLLM GPU memory fraction | `0.90` |
| `--tp-size` | Tensor parallel size | `1` |
| `--max-model-len` | Max sequence length | `4096` |
| `-o, --output` | Path to save JSON results | `results/validation/<model>.json` |
| `--compare` | Previous validation JSON for comparison | — |

### Model Aliases

Short names for commonly used models:

| Alias | Model |
|-------|-------|
| `llama-8b` | `meta-llama/Llama-3.1-8B` |
| `qwen-8b` | `Qwen/Qwen3-8B` |
| `qwen-8b-base` | `Qwen/Qwen3-8B-Base` |
| `olmo-7b` | `allenai/Olmo-3-1025-7B` |
| `olmo-32b` | `allenai/Olmo-3-1125-32B` |
| `gemma-12b` | `google/gemma-3-12b-pt` |

## Configuration

Window parameters and vLLM settings are configured in `config.yaml`:

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

Override with `--config path/to/custom.yaml`.

## Development

```bash
uv sync --extra dev

# Run tests
pytest tests/

# Type check
mypy cleanslate

# Lint and format
ruff check cleanslate
ruff format cleanslate
```
