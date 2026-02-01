# Cleanslate

Sequence likelihood estimation for LLMs. Measure how likely a model is to output specific text given a seed/prompt.

Useful for:
- **Memorization detection** - Did the model memorize training data?
- **Perplexity analysis** - How "surprised" is the model by text?
- **Token-level analysis** - Which tokens does the model predict correctly?

## Install

```bash
pip install -e .

# With CLI support
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

| Metric | Description |
|--------|-------------|
| **Log-Likelihood** | Sum of log P(token_i \| context) for target tokens |
| **Perplexity** | exp(-avg_log_likelihood) - lower = more predictable |
| **Rank-1 Fraction** | % of tokens where model's top choice matched target |
| **Avg Rank** | Average rank of target tokens (1 = always top choice) |

## Example Output

```
Experiment: shakespeare_test
Model: Qwen/Qwen3-0.6B
Device: mps
Samples: 8

────────────────────────────────────────────────────────────────────────────
Sample                         │    Log-Lik │      PPL │   Rank-1 │  AvgRank
────────────────────────────────────────────────────────────────────────────
Hamlet - To be or not to be    │    -210.48 │    25.49 │   40.0%  │     36.7
Macbeth - Tomorrow and tomorro │    -119.11 │     9.08 │   55.6%  │     25.6
...
────────────────────────────────────────────────────────────────────────────
MEAN                           │    -207.28 │    81.94 │   30.2%  │    250.6
────────────────────────────────────────────────────────────────────────────
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
