# Cleanslate

Sequence likelihood estimation for LLMs. Measure how likely a model is to output specific text given a seed/prompt.

Useful for:
- **Memorization detection** - Did the model memorize training data?
- **Perplexity analysis** - How "surprised" is the model by text?
- **Token-level analysis** - Which tokens does the model predict correctly?
- **Generation analysis** - Generate multiple outputs and save everything for later analysis

## Two Modes

### 1. Standard Mode - Compute Metrics
Run experiments and get immediate metrics (perplexity, rank-1 fraction, etc.)

### 2. Generation Mode - Save Everything
Generate multiple outputs per seed and save all data (tokens, logprobs, alternatives) for later analysis.

## Install

```bash
pip install -e .

# With CLI support
pip install -e ".[cli]"

# With export support (Parquet/HF Datasets)
pip install -e ".[cli,export]"
```

## Quick Start

### Mode 1: Standard Mode (Compute Metrics)

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

### Mode 2: Generation Mode (Save Everything)

Generate multiple outputs per seed and save all data for later analysis.

#### Python API

```python
from cleanslate.generation import GenerationExperiment
from cleanslate.samples import SHAKESPEARE_SAMPLES

# Generate 3 outputs per seed
exp = GenerationExperiment(model="Qwen/Qwen3-0.6B")
generations = exp.run_generation(
    samples=SHAKESPEARE_SAMPLES,
    num_generations=3,
    max_tokens=100,
    temperature=0.8,
    output_path="generations.parquet",
)

# Load and analyze later
import pandas as pd
df = pd.read_parquet("generations.parquet")
print(df[["sample_name", "generated_text", "mean_logprob"]])
```

#### CLI

```bash
# Generate 3 outputs per seed, save to Parquet
cleanslate generate -m Qwen/Qwen3-0.6B -d shakespeare -n 3 -o generations.parquet

# Analyze later with pandas, DuckDB, etc.
```

**Saves per generation:**
- Generated text + tokens
- Log probabilities per token
- Top-10 alternative tokens at each position
- Generation parameters (temp, top-p, top-k)
- Timing and model info

**Output formats:**
- Parquet (fast, columnar, pandas/DuckDB compatible)
- HuggingFace Datasets
- Both

## Backends

| Backend | Install | Best For |
|---------|---------|----------|
| **HuggingFace** (default) | `pip install -e .` | Simple setup, CPU/GPU, single-sample probes |
| **SGLang** | `pip install -e ".[sglang]"` | GPU-optimized batch generation, continuous batching |
| **vLLM** | `pip install -e ".[vllm]"` | Basic support (not recommended) |

### SGLang Backend

SGLang uses continuous batching and PagedAttention for maximum GPU utilization.
Ideal for large-scale generation workloads.

```python
from cleanslate.generation import GenerationExperiment
from cleanslate.samples import SHAKESPEARE_SAMPLES

# SGLang backend with batch generation
exp = GenerationExperiment(model="Qwen/Qwen2.5-0.5B-Instruct", backend="sglang")
rows = exp.run(
    samples=SHAKESPEARE_SAMPLES,
    dataset_name="shakespeare",
    output_path="generations/shakespeare.parquet",
    num_generations=5,
    batch_size=32,
)
```

**Important:** Scripts using the SGLang backend must use `if __name__ == '__main__':` guard.

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
