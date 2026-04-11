# Resume Notes — Memorization Eval Sweep

## Status as of 2026-04-11

### Completed evals (results in results/eval/default/):
- ✓ llama-8b (meta-llama/Llama-3.1-8B)
- ✓ qwen-8b (Qwen/Qwen3-8B)
- ✓ olmo-7b (allenai/Olmo-3-1025-7B)
- ✓ gemma-12b (google/gemma-3-12b-pt)

### Remaining evals (not started or incomplete):
**~8B cross-family:**
- marin-8b (marin-community/marin-8b-base) — was running, killed at 7%

**Qwen version range:**
- qwen2-7b (Qwen/Qwen2-7B)
- qwen2.5-7b (Qwen/Qwen2.5-7B)
- qwen3.5-9b (Qwen/Qwen3.5-9B)

**Qwen2.5 training stages:**
- qwen2.5-7b-instruct (Qwen/Qwen2.5-7B-Instruct)
- qwen2.5-coder-7b (Qwen/Qwen2.5-Coder-7B)
- qwen2.5-coder-7b-instruct (Qwen/Qwen2.5-Coder-7B-Instruct)
- qwen2.5-math-7b (Qwen/Qwen2.5-Math-7B)
- qwen2.5-math-7b-instruct (Qwen/Qwen2.5-Math-7B-Instruct)

**~32B models (need TP=2, use config_32b.yaml):**
- olmo-32b (allenai/Olmo-3-1125-32B)
- qwen-32b (Qwen/Qwen3-32B)
- gemma-27b (google/gemma-3-27b-pt)
- marin-32b (marin-community/marin-32b-base)

### Model downloads
All 17 models fully downloaded to ~/.cache/huggingface/hub/ with snapshot symlinks.
Exception: qwen3-8b-base was deleted (not needed).

### How to resume
```bash
# Install deps
uv sync

# Install NVIDIA drivers if fresh VM
sudo apt-get install -y nvidia-driver-590-open linux-modules-nvidia-590-open-gcp
sudo modprobe nvidia

# Run remaining models (edit run_all_evals.sh to skip completed ones)
nohup bash run_all_evals.sh --no-push > results/logs/eval_sweep.log 2>&1 &

# Or run individual model:
uv run cleanslate eval -m <alias> -o results/eval --no-push
# For 32B models:
uv run cleanslate eval -m <alias> -o results/eval --no-push --config config_32b.yaml
```

### Key configs
- config.yaml: DP=2, TP=1, batch_size=512 (for ~8B models on 2 GPUs)
- config_32b.yaml: DP=1, TP=2, batch_size=256 (for ~32B models on 2 GPUs)
- For 8x GPU machine: set data_parallel=8 for 8B models, or TP=2 DP=4 for 32B

### Throughput benchmarks (2x H100)
- ~8B models: ~580-680 win/s, ~2hrs per model for full dataset (4M windows)
- ~12B (gemma): ~460 win/s, ~2.5hrs
- enforce_eager=false + CUDA graphs: ~10% speedup
- FP8 quantization: ~13% speedup (memory-bandwidth bound)

### OpenModal parallel eval
scripts/parallel_eval.py exists but Cloud Build image push was too slow for the 20GB image.
Fix options: use vllm base image, or fix openmodal to use local_build (see providers/gcp/build.py).

### Code changes (committed to new_v2, pushed)
- VLLMBackend: DP support, FP8 passthrough, clean shutdown, use_tqdm=False
- Aliases: 17 models across families, versions, training stages
- Eval CLI: batch_size config, summary.json output
- Evaluator: tqdm progress bar
