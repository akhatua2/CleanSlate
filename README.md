# Cleanslate

Sequence likelihood estimation for LLMs. Measure how likely a model is to output specific text given a seed/prompt.

Based on (n, p)-discoverable extraction from Hayes et al.

Useful for:
- **Memorization detection** - Did the model memorize training data?
- **Perplexity analysis** - How "surprised" is the model by text?
- **Token-level analysis** - Which tokens does the model predict correctly?
- **Extraction probability** - How many prompts to extract with probability p?

## Install

```bash
pip install -e ".[cli]"
```

## Quick Start

### Python API

```python
from cleanslate import Experiment
from cleanslate.samples import SHAKESPEARE_SAMPLES

# Run experiment on Shakespeare samples
exp = Experiment(model="Qwen/Qwen3-0.6B", name="shakespeare_test")
results = exp.run(SHAKESPEARE_SAMPLES)

# View results
print(results.summary())

# Save to JSON
results.save("results/shakespeare.json")
```

### CLI

```bash
# Run experiment on a dataset
cleanslate run -m Qwen/Qwen3-0.6B -d shakespeare -o results/out.json

# With verbose token diff
cleanslate run -m Qwen/Qwen3-0.6B -d shakespeare -v

# Probe a single seed/target pair
cleanslate probe "To be or not to be" ", that is the question" -m Qwen/Qwen3-0.6B

# List available sample datasets
cleanslate list-samples

# Show saved results
cleanslate show results/out.json

# Compare two experiments
cleanslate compare results/model_a.json results/model_b.json
```

## Metrics

### Probability Metrics

| Metric | Description |
|--------|-------------|
| **Log-Likelihood** | Sum of log P(token_i \| context) - higher = more likely |
| **Perplexity** | exp(-avg_log_likelihood) - lower = more predictable/memorized |
| **Rank-1 Fraction** | % of tokens where model's top choice matched target |
| **Avg Rank** | Average rank of target tokens (1 = always top choice) |

### N-gram Metrics

| Metric | Description |
|--------|-------------|
| **BLEU-1** | Unigram precision - what fraction of predicted words appear in target |
| **ROUGE-1** | Unigram F1 - word overlap between predicted and target |
| **ROUGE-L** | Longest common subsequence - structural/fluency similarity |

### Extraction Metrics (from probability.py)

```python
from cleanslate.metrics import compute_extraction_probability

result = compute_extraction_probability(token_log_probs)
print(f"pz = {result.prob:.4f}")  # Single-prompt extraction probability
print(f"Prompts for 90% extraction: {result.prompts_needed(0.9)}")
```

## Decoding Options

Control how probabilities are computed:

| Option | Description | Default |
|--------|-------------|---------|
| `temperature` | Logit scaling. Higher = flatter distribution | 1.0 |
| `top_k` | Only consider top-k tokens | None (all) |
| `top_p` | Nucleus sampling - keep tokens with cumulative prob >= p | None |
| `repetition_penalty` | Penalize repeated tokens (>1 discourages) | 1.0 |

```python
# Default: temperature=1.0, full distribution
result = backend.compute(seed, target)

# With temperature scaling (higher = flatter distribution)
result = backend.compute(seed, target, temperature=0.7)

# With top-k filtering (only consider top-k tokens)
result = backend.compute(seed, target, top_k=40)

# With nucleus sampling (top-p)
result = backend.compute(seed, target, top_p=0.9)

# With repetition penalty
result = backend.compute(seed, target, repetition_penalty=1.2)

# Combined
result = backend.compute(seed, target, temperature=0.8, top_p=0.9, repetition_penalty=1.1)
```

## Example Output

```
Experiment: shakespeare_test
Model: Qwen/Qwen3-0.6B
Device: mps
Samples: 8

──────────────────────────────────────────────────────────────────────────────────────────────────
Sample                         │   Log-Lik │     PPL │ Rank-1 │ AvgRank │ BLEU-1 │ ROUGE-1 │ ROUGE-L
──────────────────────────────────────────────────────────────────────────────────────────────────
Hamlet - To be or not to be    │   -210.48 │    25.5 │   40%  │    36.7 │   0.40 │    0.42 │    0.38
Macbeth - Tomorrow and tomorro │   -119.11 │     9.1 │   56%  │    25.6 │   0.56 │    0.58 │    0.52
...
──────────────────────────────────────────────────────────────────────────────────────────────────
MEAN                           │   -207.28 │    81.9 │   30%  │   250.6 │   0.30 │    0.32 │    0.28
──────────────────────────────────────────────────────────────────────────────────────────────────
```

## Included Samples

- **Shakespeare**: Hamlet, Romeo & Juliet, Macbeth, Julius Caesar, Sonnets, etc.
- **Books**: 1984, Pride & Prejudice, Moby Dick, Declaration of Independence, etc.

## Device Support

Auto-detects available hardware:
- **CUDA** (NVIDIA GPU)
- **MPS** (Apple Silicon)
- **CPU** (fallback)

Override with `--device`:
```bash
cleanslate run -m Qwen/Qwen3-0.6B -d shakespeare --device cpu
```
