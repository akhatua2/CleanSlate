# Generation Mode - User Guide

## What is Generation Mode?

Generation mode creates multiple outputs per seed/prompt and saves **everything** for later analysis:
- Generated text
- Token IDs and strings  
- Log probabilities per token
- Top-K alternative tokens at each position
- Generation parameters (temperature, top-p, top-k)
- Timing information

**No metrics are computed** - just raw data saved in HuggingFace-compatible formats.

## Why Use This?

1. **Deferred analysis** - Generate now, compute metrics later
2. **Multiple analysis passes** - Compute different metrics on same generations
3. **Comparison studies** - Compare outputs from different models/settings
4. **Dataset creation** - Build datasets of model generations
5. **Research** - Save everything for reproducible experiments

## Quick Start

### Python API

```python
from cleanslate.generation_experiment import GenerationExperiment
from cleanslate.samples import SHAKESPEARE_SAMPLES

exp = GenerationExperiment(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    name="my_generations",
)

generations = exp.run_generation(
    samples=SHAKESPEARE_SAMPLES,
    num_generations=3,  # 3 outputs per seed
    max_tokens=100,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    output_path="generations.parquet",
    save_format="parquet",  # or "hf_dataset" or "both"
)
```

### CLI

```bash
# Generate 3 outputs per seed from Shakespeare dataset
cleanslate generate \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset shakespeare \
    --num-generations 3 \
    --max-tokens 100 \
    --temperature 0.8 \
    --output generations.parquet
```

## What Gets Saved?

Each generation includes:

### Identification
- `sample_id`: Unique ID
- `sample_name`: Human-readable name
- `generation_index`: Which generation (0, 1, 2, ...)

### Input
- `seed`: The prompt text
- `original_target`: Original target (if doing memorization tests)

### Model Info
- `model_name`: e.g., "Qwen/Qwen2.5-0.5B-Instruct"
- `backend`: e.g., "huggingface"
- `device`: e.g., "cuda"

### Generation Parameters
- `temperature`: Sampling temperature used
- `top_p`: Nucleus sampling parameter
- `top_k`: Top-k sampling parameter
- `max_tokens`: Maximum tokens limit

### Output
- `generated_text`: The full generated text
- `num_tokens`: Number of tokens generated
- `finish_reason`: "stop", "length", or "eos"

### Token-Level Data
- `tokens`: List of token IDs
- `token_strings`: Decoded strings for each token
- `token_logprobs`: Log probability of each generated token

### Top-K Alternatives
- `top_logprobs_data`: For each position, the top 10 alternative tokens with their logprobs
  ```python
  [
    [  # Position 0
      {"token_id": 123, "token_str": " the", "logprob": -0.5},
      {"token_id": 456, "token_str": " a", "logprob": -1.2},
      ...
    ],
    [  # Position 1
      ...
    ]
  ]
  ```

### Aggregate Statistics
- `cumulative_logprob`: Sum of all token logprobs
- `mean_logprob`: Average logprob per token

### Metadata
- `timestamp`: When generated (ISO format)
- `generation_time_ms`: How long generation took

## Save Formats

### Parquet (Recommended)
```bash
--format parquet
```
- Fast to read/write
- Columnar format (efficient for analysis)
- Works with pandas, polars, DuckDB
- HuggingFace Datasets compatible

```python
import pandas as pd
df = pd.read_parquet("generations.parquet")
print(df[["sample_name", "generation_index", "generated_text", "mean_logprob"]])
```

### HuggingFace Dataset
```bash
--format hf_dataset
```
- Native HF Datasets format
- Good for training/fine-tuning pipelines
- Automatic caching

```python
from datasets import load_from_disk
dataset = load_from_disk("generations")
print(dataset[0])
```

### Both
```bash
--format both
```
- Saves in both formats
- Best for flexibility

## Example Use Cases

### 1. Compare Temperature Settings

```bash
# Low temperature (focused)
cleanslate generate -m Qwen/Qwen2.5-0.5B-Instruct \
    -d shakespeare -n 5 --temperature 0.3 -o gens_temp0.3.parquet

# High temperature (creative)
cleanslate generate -m Qwen/Qwen2.5-0.5B-Instruct \
    -d shakespeare -n 5 --temperature 1.2 -o gens_temp1.2.parquet
```

Then analyze:
```python
import pandas as pd

df_low = pd.read_parquet("gens_temp0.3.parquet")
df_high = pd.read_parquet("gens_temp1.2.parquet")

print(f"Low temp mean logprob: {df_low['mean_logprob'].mean():.3f}")
print(f"High temp mean logprob: {df_high['mean_logprob'].mean():.3f}")
```

### 2. Multi-Model Comparison

```bash
# Generate from multiple models
for model in "Qwen/Qwen2.5-0.5B-Instruct" "meta-llama/Llama-2-7b-hf"; do
    cleanslate generate -m "$model" -d shakespeare -n 3 \
        -o "gens_$(basename $model).parquet"
done
```

### 3. Build Training Dataset

```python
from cleanslate.generation_experiment import GenerationExperiment
from cleanslate.samples import BOOK_SAMPLES

exp = GenerationExperiment(model="Qwen/Qwen2.5-7B-Instruct")

# Generate diverse outputs
generations = exp.run_generation(
    samples=BOOK_SAMPLES,
    num_generations=10,  # 10 per seed for diversity
    temperature=1.0,
    max_tokens=200,
    output_path="training_data",
    save_format="hf_dataset",
)

# Now you have a dataset ready for training
```

### 4. Memorization Analysis

```python
import pandas as pd

df = pd.read_parquet("shakespeare_generations.parquet")

# Check if model reproduced original target
df["exact_match"] = df.apply(
    lambda row: row["original_target"] in row["generated_text"], 
    axis=1
)

print(f"Exact match rate: {df['exact_match'].mean():.1%}")

# Analyze by sample
memorization = df.groupby("sample_name")["exact_match"].mean()
print(memorization.sort_values(ascending=False))
```

## CLI Reference

```bash
cleanslate generate [OPTIONS]

Options:
  -m, --model TEXT              Model name [default: Qwen/Qwen2.5-0.5B-Instruct]
  -d, --dataset TEXT            Dataset: shakespeare, books, or all [default: shakespeare]
  -o, --output TEXT             Output file path (.parquet) [required]
  -n, --num-generations INT     Number of generations per seed [default: 3]
  --max-tokens INT              Maximum tokens per generation [default: 100]
  -t, --temperature FLOAT       Sampling temperature [default: 0.8]
  --top-p FLOAT                 Nucleus sampling (top-p) [default: 0.95]
  --top-k INT                   Top-k sampling [default: 50]
  --format TEXT                 Save format: parquet, hf_dataset, or both [default: parquet]
  --device TEXT                 Device: cpu, cuda, mps, or auto
  --help                        Show this message and exit
```

## Performance Tips

1. **Use Parquet** - Faster than HF dataset format for most use cases
2. **Batch size = 1** - Currently generates sequentially (TODO: add batching)
3. **GPU recommended** - Much faster than CPU
4. **Start small** - Test with `--num-generations 1` first

## Loading and Analysis

### With Pandas

```python
import pandas as pd

df = pd.read_parquet("generations.parquet")

# Basic stats
print(df.groupby("sample_name").agg({
    "mean_logprob": "mean",
    "num_tokens": "mean",
    "generation_time_ms": "mean",
}))

# Filter by generation quality
good_gens = df[df["mean_logprob"] > -2.0]
```

### With DuckDB (for large files)

```python
import duckdb

# Query without loading into memory
result = duckdb.query("""
    SELECT 
        sample_name,
        generation_index,
        generated_text,
        mean_logprob
    FROM 'generations.parquet'
    WHERE mean_logprob > -1.5
    ORDER BY mean_logprob DESC
    LIMIT 10
""").df()
```

### With HuggingFace Datasets

```python
from datasets import load_from_disk

dataset = load_from_disk("generations")

# Filter
good_gens = dataset.filter(lambda x: x["mean_logprob"] > -2.0)

# Map over dataset
dataset = dataset.map(lambda x: {
    **x,
    "perplexity": 2 ** (-x["mean_logprob"])
})
```

## Comparison: Generation Mode vs Standard Mode

| Feature | Standard Mode | Generation Mode |
|---------|---------------|-----------------|
| Purpose | Compute metrics | Save raw data |
| Output | Metrics (perplexity, rank, etc.) | Raw generations + logprobs |
| Multiple generations | No (1 per seed) | Yes (N per seed) |
| File format | JSON | Parquet / HF Dataset |
| Analysis timing | Immediate | Deferred |
| Storage size | Small | Larger |
| Flexibility | Fixed metrics | Compute any metric later |
| Use case | Quick evaluation | Research / comparison |

## Next Steps

After generating data:

1. **Explore** - Load in pandas/jupyter and visualize
2. **Compute metrics** - Calculate custom metrics on saved data
3. **Compare** - Analyze differences between models/settings
4. **Train** - Use as training data if desired
5. **Share** - Parquet files are portable and HF-compatible
