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

## FSID Workflow (Local Forget/Retain + Local PyTorch Training)

Use this when you already have eval datasets and want a local forget-set benchmark workflow.

### 1) Build a stratified candidate pool from eval outputs

```bash
cleanslate fsid build-pool \
  --eval-dataset unlearning-cleanslate/eval__olmo7b-base__sliding_window__all \
  --threshold 0.10 \
  --target-size 200 \
  --forget-fraction 0.30 \
  --min-size 50 \
  --max-size 300 \
  --output-dir data/fsid/olmo7b
```

This filters songs with `memorized_fraction >= threshold`, samples a stratified total pool, and then writes:
- `data/fsid/olmo7b/pool.parquet`: total sampled song pool
- `data/fsid/olmo7b/forget.parquet`: songs selected for unlearning
- `data/fsid/olmo7b/retain.parquet`: songs kept out of the forget set

### 2) Curate baseline forget/retain sets locally

```bash
cleanslate fsid curate \
  --forget-set-rule baseline \
  --forget-pool-path data/fsid/olmo7b/forget.parquet \
  --base-dataset allenai/dolma3_mix-6T \
  --output-dir data/fsid/curated/baseline/olmo7b
```

Baseline policy:
- `D_forget` uses the selected unlearn songs, formats them with the `direct` prompt template and `none` seed mode, and keeps the true lyrics as the target text
- `D_retain` is loaded from WikiText (general-domain text to preserve model quality). By default uses all filtered rows from `wikitext-103-raw-v1` validation split. Use `--retain-samples N` to subsample

### 3) Train unlearning model locally in PyTorch

```bash
cleanslate fsid train \
  -m "Qwen/Qwen2.5-0.5B-Instruct" \
  --forget-path data/fsid/curated/baseline/forget_set.parquet \
  --retain-path data/fsid/curated/baseline/retain_set.parquet \
  --objective npo \
  --train-mode full \
  --epochs 2 \
  --batch-size 2 \
  --grad-accum 16 \
  --output-dir results/fsid/unlearning/qwen_baseline
```

This does not call Open-Unlearning subprocesses. Training data is local and no HTTPS fetch happens in the training loop. Single-device training is the default; multi-GPU requires `accelerate launch` or `torchrun` with `--distributed-strategy fsdp`.

### 4) Multi-GPU FSDP2 training

Training uses HuggingFace Accelerate with FSDP2 when the model does not fit on one GPU:

```bash
accelerate launch --num_processes 10 -m cleanslate fsid train \
  --config cleanslate/fsid/configs/npo.yaml \
  -m "allenai/Olmo-3-1025-7B" \
  --forget-path data/fsid/curated/baseline/olmo7b/forget_set.parquet \
  --retain-path data/fsid/curated/baseline/olmo7b/retain_set.parquet \
  --objective npo \
  --distributed-strategy fsdp \
  --fsdp-sharding-strategy full_shard \
  --gradient-checkpointing \
  --output-dir results/fsid/unlearning/olmo3_npo_fsdp \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 2
```

`--gradient-checkpointing` is essential for 7B+ models on 16 GB GPUs — without it, activations for all decoder layers are held simultaneously and will OOM. Use `--batch-size 1` per GPU and compensate with `--grad-accum` for the desired effective batch size (here: 1 * 10 GPUs * 16 accum = 160).

`torchrun --nproc_per_node=10` also works as a launcher (Accelerate detects the torchrun environment automatically).

Constraints:
- FSDP is explicit only; there is no implicit `device_map="auto"` training path.
- FSDP currently supports `train_mode=full` only (no LoRA + FSDP).
- FSDP currently does not support the RMU objective.
- Forget, retain, and validation sets must still be materialized locally before training.

### 5) Score unlearning effectiveness

After training, evaluate the unlearned checkpoint on the same eval dataset used in step 1, then compare pre/post memorization:

```bash
# Run eval on the unlearned checkpoint (produces post-eval results)
cleanslate eval \
  -d unlearning-cleanslate/generated_songs \
  -c direct__first_line \
  -m results/fsid/unlearning/olmo3_npo_fsdp/<run_name> \
  --window-size 50 \
  -o results/fsid/post_eval

# Score: compare pre vs post memorization
cleanslate fsid score \
  --forget-path data/fsid/curated/baseline/olmo7b/forget_set.parquet \
  --pre-eval  results/pre_eval/direct__first_line/results.parquet \
  --post-eval results/fsid/post_eval/direct__first_line/results.parquet \
  -o results/fsid/score.json
```

The scorer computes:

| Metric | Description |
|--------|-------------|
| **Forget score** | Fraction of forget-set songs with reduced `memorized_fraction` (higher = better) |
| **Retain penalty** | Normalized mean |Δ| in `memorized_fraction` for non-forget songs (lower = better) |
| **Window forget rate** | Fraction of pre-memorized windows in forget set that became clean (higher = better) |
| **Window stability** | 1 − (regressed windows / total windows) in retain set (higher = better) |

**Window-level tracking:** When eval parquets contain a `windows` column (list of per-window dicts with `idx`, `p_z`, `is_memorized`), the scorer classifies each window into one of four transitions:

| Transition | Pre → Post | Meaning |
|------------|------------|---------|
| **forgotten** | memorized → clean | Desired for forget set |
| **retained_memorized** | memorized → memorized | Still memorized |
| **retained_clean** | clean → clean | No change |
| **regressed** | clean → memorized | Undesirable |

These appear in the CLI output and JSON report. If the `windows` column is absent, scoring falls back to song-level metrics only.

Input files (`--pre-eval` and `--post-eval`) must each have columns `song_id` (or `content_id`), `memorized_fraction`, and `max_p_z`, and must cover the exact same set of songs. Every song in `--forget-path` must appear in the eval universe.

### 6) Validate utility retention

Run standard NLP benchmarks before and after unlearning to measure utility retention. Uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with the SGLang backend — any of its 300+ tasks are available.

```bash
pip install -e ".[eval]"

# Pre-unlearning baseline
cleanslate fsid validate \
  -m allenai/OLMo-3-32B \
  -B mmlu,gsm8k \
  -o results/fsid/pre_val.json

# Post-unlearning (on checkpoint)
cleanslate fsid validate \
  -m results/fsid/unlearning/olmo3_npo_fsdp/<run_name> \
  -B mmlu,gsm8k \
  -o results/fsid/post_val.json

# Include in scoring
cleanslate fsid score \
  --forget-path data/fsid/curated/baseline/olmo7b/forget_set.parquet \
  --pre-eval  results/pre_eval/direct__first_line/results.parquet \
  --post-eval results/fsid/post_eval/direct__first_line/results.parquet \
  --pre-validation results/fsid/pre_val.json \
  --post-validation results/fsid/post_val.json \
  -o results/fsid/score.json
```

| Option | Default | Description |
|--------|---------|-------------|
| `-B`, `--benchmarks` | `mmlu,gsm8k` | Comma-separated lm-eval task names (e.g. `arc_easy`, `hellaswag`, `winogrande`) |
| `-n`, `--limit` | None | Limit items per task (for quick testing) |
| `--num-fewshot` | None | Override default few-shot count |
| `--batch-size` | `auto` | Batch size (`auto` or integer) |
| `--tp-size` | 1 | Tensor parallel size for multi-GPU |
| `--mem-fraction` | 0.85 | GPU memory fraction for SGLang engine |

## QA Benchmark

The QA benchmark evaluates **semantic knowledge retention** after unlearning. While p_z measures verbatim memorization, the QA benchmark tests whether the model still *knows* factual information about the content.

**Goal:** After unlearning, p_z (verbatim memorization) should DROP, but QA accuracy (semantic knowledge) should STAY THE SAME.

### Dataset

The QA benchmark is available on HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("unlearning-cleanslate/cleanslate_dataset", "qa_benchmark")
# 1616 validated QA pairs with columns: question, answer, difficulty, num_models_passed
```

**Difficulty levels** (based on how many base models answered correctly):
- `common`: 6/6 models (185 QAs)
- `general`: 4-5/6 models (955 QAs)
- `expert`: 3/6 models (476 QAs)

### Running with lm-eval

```bash
# Evaluate a model on the QA benchmark (HF backend)
lm_eval --model hf \
    --model_args pretrained=allenai/Olmo-3-1025-7B,dtype=bfloat16 \
    --tasks cleanslate_qa \
    --include_path cleanslate/fsid/qa/tasks \
    --batch_size 16

# With SGLang backend (faster)
lm_eval --model sglang \
    --model_args pretrained=allenai/Olmo-3-1025-7B,mem_fraction_static=0.85 \
    --tasks cleanslate_qa \
    --include_path cleanslate/fsid/qa/tasks \
    --batch_size auto
```

### Metrics

| Metric | Description |
|--------|-------------|
| `exact_match` | Strict string equality (usually ~0%) |
| `llm_judge_match` | Semantic correctness via Gemini LLM judge (recommended) |

The LLM judge checks if the model's answer conveys the same meaning as the reference, handling variations like "The 1980s" vs "1980s".

### Baseline Results

| Model | llm_judge_match |
|-------|-----------------|
| Qwen/Qwen3-8B | 63.37% |
| allenai/Olmo-3-1025-7B | 54.70% |

### QA Generation Pipeline

To generate new QA pairs for additional content:

```bash
# Generate QA candidates (uses Gemini)
cleanslate fsid qa generate \
  -d unlearning-cleanslate/cleanslate_dataset \
  -n 10 \
  -o data/fsid/qa/candidates.parquet

# Validate against base models (pass@k)
cleanslate fsid qa validate \
  -i data/fsid/qa/candidates.parquet \
  -k 5 \
  -o data/fsid/qa/validated.parquet

# Merge validated QA into the HF dataset (adds content_qa column)
cleanslate fsid qa merge \
  -i data/fsid/qa/validated.parquet \
  -d unlearning-cleanslate/cleanslate_dataset \
  --hf-push unlearning-cleanslate/cleanslate_dataset
```

QA pairs must be:
- **Atomic**: Exactly one unambiguous answer
- **Objective**: No subjective interpretation
- **Self-contained**: Include song/book title in the question

### QA Evaluation & Scoring

Evaluate a model's QA accuracy and score forget/retain splits using lm-eval with the SGLang backend:

```bash
# Evaluate pre-unlearning model
cleanslate fsid qa eval \
  -m allenai/Olmo-3-1025-7B \
  -d unlearning-cleanslate/cleanslate_dataset \
  -c qa_benchmark \
  --judge \
  -o results/fsid/qa_pre.json

# Evaluate post-unlearning checkpoint
cleanslate fsid qa eval \
  -m results/fsid/unlearning/olmo3_npo_fsdp/<run_name> \
  -d unlearning-cleanslate/cleanslate_dataset \
  -c qa_benchmark \
  --judge \
  -o results/fsid/qa_post.json

# Quick test with limited samples and no LLM judge
cleanslate fsid qa eval \
  -m allenai/Olmo-3-1025-7B \
  --no-judge \
  -n 50 \
  -o results/fsid/qa_quick.json

# Multi-GPU evaluation
cleanslate fsid qa eval \
  -m allenai/Olmo-3-1025-7B \
  --judge \
  -g 4 --tp-size 2 \
  -o results/fsid/qa_pre.json

# Score: compare forget vs retain QA accuracy change
cleanslate fsid qa score \
  --forget-path data/fsid/curated/baseline/olmo7b/forget_set.parquet \
  --pre-qa results/fsid/qa_pre.json \
  --post-qa results/fsid/qa_post.json \
  -o results/fsid/qa_score.json
```

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--model` | required | Model path or HuggingFace name |
| `-d`, `--dataset` | `unlearning-cleanslate/cleanslate_dataset` | HuggingFace dataset ID |
| `-c`, `--config` | `qa_benchmark` | Dataset config name |
| `--judge/--no-judge` | `--judge` | Enable parallel LLM judge (Gemini) for samples that don't pass `contains_match`. `--no-judge` uses `contains_match` only |
| `--max-judge-workers` | 32 | Max parallel threads for LLM judge API calls |
| `--mem-fraction` | 0.85 | GPU memory fraction for SGLang engine |
| `--tp-size` | 1 | Tensor parallel size |
| `-g`, `--num-gpus` | 1 | Number of GPUs (split across TP and DP) |
| `--gpu-ids` | None | Comma-separated GPU IDs |
| `--batch-size` | `auto` | Batch size (`auto` or integer) |
| `-n`, `--limit` | None | Limit number of QA pairs (for quick testing) |

`qa eval` runs the `cleanslate_qa_fast` lm-eval task (SGLang backend) to generate answers and compute `exact_match` + `contains_match` per sample. When `--judge` is enabled (default), samples that didn't pass `contains_match` are sent to the Gemini LLM judge in parallel (`--max-judge-workers` threads). Per-sample output includes all three metric scores (`exact_match`, `contains_match`, `llm_judge_match`) and the model's prediction.

`qa score` splits QA results by forget/retain content and reports per-subset accuracy deltas. After unlearning, forget-set QA accuracy should ideally stay the same (knowledge retention), while p_z should drop (verbatim memorization removed)
