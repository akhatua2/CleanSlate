# CleanSlate Backend Recommendations

## Summary

After implementing and testing multiple backends for log probability computation:

## ✅ **RECOMMENDATION: Use HuggingFace Backend**

```python
from cleanslate import Experiment
from cleanslate.samples import SHAKESPEARE_SAMPLES

exp = Experiment(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    name="my_experiment",
    backend="huggingface"  # ← Use this
)

results = exp.run(SHAKESPEARE_SAMPLES)
```

### Why HuggingFace Backend?

1. **✅ Works reliably** - Tested and confirmed working
2. **✅ Simple** - No complex setup or configuration
3. **✅ Memory efficient** for this use case - Only uses ~2-3GB
4. **✅ Direct logits access** - Clean API, no hacks needed
5. **✅ GPU accelerated** - Still uses CUDA when available

**Results from test:**
```
Sample: Hamlet - To be or not to be
Log-Likelihood: -35.17
Perplexity: 1.72
Tokens: 65
Rank-1 Fraction: 86.2%
Avg Rank: 1.4
```

## ⚠️ Why NOT vLLM?

### Problems Found:

1. **API Changed in v0.15.0**
   - V1 engine doesn't expose model directly
   - Internal structure completely different
   - Would need hacky workarounds

2. **Memory Overhead**
   - Even with optimized settings: 38GB KV cache for a 2GB model
   - Designed for serving, not evaluation

3. **Architecture Mismatch**
   - Built for text generation with concurrent requests
   - You only need one forward pass per sample
   - All the scheduling/batching logic is wasted overhead

See `docs_vllm_vs_sglang.md` for detailed explanation.

## ⚠️ Why NOT SGLang?

- **Rust compiler required** to build `outlines-core` dependency
- Installation failed on your system
- Would need admin access to install Rust toolchain

## Performance Comparison

| Backend | Memory Used | Speed | Complexity | Status |
|---------|-------------|-------|------------|--------|
| **HuggingFace** | ~3GB | Fast | Simple | ✅ Works |
| vLLM | ~40GB+ | Similar | Complex | ❌ API issues |
| SGLang | N/A | N/A | Medium | ❌ Build failed |

## Usage Example

```python
#!/usr/bin/env python
from cleanslate import Experiment
from cleanslate.samples import SHAKESPEARE_SAMPLES

# Simple and reliable
exp = Experiment(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    name="shakespeare_memorization",
    backend="huggingface",  # Best choice
)

# Run on all Shakespeare samples
results = exp.run(SHAKESPEARE_SAMPLES)
print(results.summary())

# Or test a single sample
result = exp.run_single(
    seed="To be or not to be",
    target=", that is the question"
)
print(f"Perplexity: {result.perplexity:.2f}")
```

## Multi-GPU Support (If Needed)

If you want to use multiple GPUs with HuggingFace backend:

```python
# HuggingFace automatically uses all available GPUs with device_map="auto"
exp = Experiment(
    model="Qwen/Qwen2.5-7B",  # Larger model
    backend="huggingface",
    # device="auto" is default, will use all GPUs
)
```

The HuggingFace backend uses `device_map="auto"` which automatically:
- Detects available GPUs
- Distributes model layers across GPUs
- Handles multi-GPU tensor parallelism

## Batch Processing for Efficiency

```python
# Process multiple samples efficiently
samples = SHAKESPEARE_SAMPLES

for i, sample in enumerate(samples):
    print(f"[{i+1}/{len(samples)}] Processing {sample['name']}...")
    result = exp.run_single(sample['seed'], sample['target'])
    print(f"  Perplexity: {result.perplexity:.2f}")
```

## Bottom Line

**Use HuggingFace backend.** It's the sweet spot of:
- Simplicity
- Reliability  
- Performance
- GPU utilization

vLLM and SGLang are over-engineered for this use case.
