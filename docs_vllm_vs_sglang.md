# Why vLLM is Suboptimal for Log Probability Computation

## The Problem with vLLM for Log Probs

### 1. **Memory Management Design**
vLLM is optimized for **text generation**, not log probability extraction:

- **Huge KV Cache Allocation**: vLLM pre-allocates large KV caches for serving multiple concurrent generation requests
  - In your error: vLLM tried to allocate 66GB+ for KV cache even though the model is only 0.5B (~2GB)
  - From vLLM logs: `"GPU 0 has a total capacity of 79.14 GiB of which 2.07 GiB is free. Process has 66.07 GiB memory in use"`
  - The model itself uses ~1GB, but vLLM allocates 65GB+ for KV cache to serve many requests

- **Why?** vLLM's PagedAttention needs extensive memory for:
  - Caching attention key-value pairs for multiple sequences
  - Supporting high-throughput text generation with batching
  - Managing long context windows (32K tokens in your case)

### 2. **Architecture Mismatch**
From vLLM Architecture docs:

```
LLMEngine is responsible for:
- Input Processing (tokenization)
- Scheduling (chooses which requests to process)
- Model Execution (distributed execution)
- Output Processing (decoding tokens → text)
```

**For log probability computation, you only need:**
- Tokenization
- One forward pass to get logits
- Softmax + log computation

**vLLM adds unnecessary overhead:**
- Scheduling system for concurrent requests
- KV cache management
- Sampling and decoding logic
- Multi-step generation pipeline

### 3. **API Design**
vLLM's primary API is `llm.generate()`:
- Designed for **generating new tokens**
- Returns `RequestOutput` with generated text
- Logits/log probabilities are not directly exposed

To get logits with vLLM, you must:
1. Access internal `model_executor` → `driver_worker` → `model_runner` → `model`
2. Manually run forward pass
3. Extract logits from complex output structure
4. This is **not a supported API** and can break between versions

### 4. **Resource Utilization**
For your use case (computing log probs on fixed text):
- **vLLM**: Allocates 66GB for serving infrastructure you don't use
- **HuggingFace**: Uses ~2-3GB for model + minimal overhead
- **SGLang**: Uses ~5-10GB with better control

## Why SGLang is Better

### 1. **Flexible Memory Management**
```python
runtime = sgl.Runtime(
    model_path="model",
    mem_fraction_static=0.7  # Control memory usage
)
```
- You can tune memory allocation
- Doesn't over-allocate for generation you don't need
- Better for batch processing vs online serving

### 2. **Better API for Log Probs**
SGLang provides:
- Direct access to model forward pass
- Cleaner logits extraction
- Designed for both generation AND evaluation tasks

### 3. **Radix Attention** 
SGLang uses "Radix Attention" which is more memory-efficient than PagedAttention for:
- Prefix caching
- Structured generation
- Evaluation workloads

## Recommendations

### For Your Use Case (Log Probability Computation):

**1. Small-scale (< 5 samples/sec):**
- ✅ **HuggingFace backend** - Simplest, works everywhere, minimal memory

**2. Medium-scale (batch processing):**
- ✅ **SGLang** - Better GPU utilization, controllable memory, ~3x faster than HF

**3. High-throughput serving (100+ req/sec):**
- ⚠️ **vLLM** - Only if you ALSO need concurrent text generation
- But for pure log prob computation, still not ideal

### Why You're Seeing OOM with vLLM

```
max_model_len=32768  # vLLM allocates KV cache for 32K tokens
gpu_memory_utilization=0.9  # Uses 90% of 80GB = 72GB
```

Even with `gpu_memory_utilization=0.3`, vLLM still allocates:
- Model: ~2GB
- KV cache: 0.3 * 80GB * 0.8 = ~20GB (still huge for log prob task)
- Compilation overhead: ~5GB

## Bottom Line

**vLLM is a race car built for throughput in production serving.**
**You need a sedan for single-pass evaluation.**

Use SGLang for better GPU utilization without vLLM's overhead.
