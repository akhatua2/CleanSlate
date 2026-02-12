# CleanSlate

Sliding window memorization evaluation for LLMs. Detects whether a model has memorized text by computing extraction probability (p_z) across overlapping windows.

Based on the methodology from:
- Cooper et al. "Extracting memorized pieces of copyrighted books"
- Hayes et al. "Measuring memorization in language models"

## What It Does

For any text (song lyrics, book excerpts, etc.), CleanSlate:

1. **Slides a window** across the text (e.g., 50 tokens prefix + 50 tokens suffix)
2. **Computes p_z** for each window via teacher forcing: P(suffix | prefix)
3. **Generates** text from the prefix and compares with BLEU/ROUGE
4. **Aggregates** results: one row per text with summary metrics

## Install

```bash
pip install -e ".[cli,export]"
```

## Quick Start

### Probe a text for memorization

```bash
cleanslate probe "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer" \
  -m "Qwen/Qwen2.5-0.5B-Instruct" \
  --window-size 20
```

Output:
```
Windows: 1
Memorized: 1
Coverage: 93.0%
Max p_z: 0.018083  ← 1.8% (above 0.1% threshold = memorized!)
```

### Evaluate a HuggingFace dataset

```bash
cleanslate eval \
  -d unlearning-cleanslate/generated_songs \
  -c direct__first_line \
  -m "Qwen/Qwen3-8B" \
  --window-size 50 \
  --stride 25 \
  -n 100 \
  --hf-push unlearning-cleanslate/eval_results
```

**Note:** The `-c` option selects which HF dataset config to load from. The evaluation is done on raw text (no prompts/system messages).

## Output Schema

One row per text with:

| Column | Description |
|--------|-------------|
| `max_p_z` | Highest p_z across all windows |
| `mean_p_z` | Average p_z |
| `memorized_windows` | Count where p_z >= threshold |
| `coverage` | % of characters in memorized windows |
| `best_window_seed` | Prefix of most memorized window |
| `best_window_target` | Suffix of most memorized window |
| `best_window_generated` | Model's generation from that prefix |
| `best_window_bleu_1` | BLEU-1 score (generated vs target) |
| `windows` | Array of all window results |

## CLI Options

```bash
cleanslate eval --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `-d`, `--hf-dataset` | required | HuggingFace dataset ID |
| `-c`, `--hf-config` | required | Dataset config/split to load |
| `-m`, `--model` | Qwen/Qwen2.5-0.5B-Instruct | Model to evaluate |
| `-w`, `--window-size` | 50 | Window size in tokens (prefix & suffix) |
| `--stride` | 25 | Stride in tokens |
| `-t`, `--threshold` | 0.001 | Memorization threshold (0.1%) |
| `-n`, `--limit` | None | Limit number of texts to process |
| `--no-generate` | False | Skip generation, only compute p_z |
| `--hf-push` | None | Push results to HuggingFace |

## Python API

```python
from cleanslate import MemorizationConfig, MemorizationEvaluator

config = MemorizationConfig(
    window_prefix_len=50,
    window_suffix_len=50,
    window_stride=25,
    tau_min=0.001,  # 0.1% threshold
    do_generate=True,
)

evaluator = MemorizationEvaluator(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    backend="sglang",
    config=config,
)

# Sliding window on a text
result = evaluator.sliding_window(
    text="When you're weary, feeling small...",
    name="Bridge Over Troubled Water",
)

print(f"Windows: {result.total_windows}")
print(f"Memorized: {result.memorized_windows}")
print(f"Max p_z: {result.max_p_z:.6f}")
print(f"Coverage: {result.coverage:.1%}")
```

## Metrics

| Metric | Description |
|--------|-------------|
| **p_z** | Extraction probability: P(suffix \| prefix) via teacher forcing |
| **is_memorized** | True if p_z >= threshold (default 0.1%) |
| **coverage** | % of text characters in at least one memorized window |
| **perplexity** | exp(-avg_log_prob) per window |
| **BLEU-1** | Unigram overlap between generated and target |
| **ROUGE-L** | Longest common subsequence F1 |

## Backends

| Backend | Best For |
|---------|----------|
| **SGLang** (default) | GPU-optimized, batched inference |
| **HuggingFace** | Simple setup, CPU/GPU |

```bash
# Use SGLang (recommended for GPU)
cleanslate eval -d ... -b sglang

# Use HuggingFace
cleanslate eval -d ... -b huggingface
```

## How Sliding Window Works

```
Text: "When you're weary, feeling small, when tears are in your eyes..."

Window 0:
  prefix (seed):   "When you're weary, feeling small, when tears"  [50 tokens]
  suffix (target): "are in your eyes, I'll dry them all, I'm on"   [50 tokens]
  → compute p_z = P(suffix | prefix)
  → generate from prefix, compare with BLEU/ROUGE

Window 1 (stride 25):
  prefix (seed):   "feeling small, when tears are in your eyes"    [50 tokens]
  suffix (target): "I'll dry them all, I'm on your side, oh when"  [50 tokens]
  → compute p_z, generate, compare

... repeat across entire text
```

## Example: Full Pipeline

```bash
# 1. Evaluate songs dataset
CUDA_VISIBLE_DEVICES=0 cleanslate eval \
  -d unlearning-cleanslate/generated_songs \
  -c direct__first_line \
  -m "Qwen/Qwen3-8B" \
  --window-size 50 \
  -n 1000 \
  -o results/songs \
  --hf-push unlearning-cleanslate/memorization_eval

# 2. Results are saved locally and pushed to HF
# - results/songs/direct__first_line/results.parquet
# - results/songs/direct__first_line/config.json

# 3. Analyze results
python -c "
import pandas as pd
df = pd.read_parquet('results/songs/direct__first_line/results.parquet')
print(f'Songs with memorization: {(df.memorized_windows > 0).sum()}/{len(df)}')
print(f'Max p_z: {df.max_p_z.max():.6f}')
print(df.nlargest(10, 'max_p_z')[['song_title', 'max_p_z', 'coverage']])
"
```
