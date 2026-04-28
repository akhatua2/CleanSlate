source .venv/bin/activate

CUDA_VISIBLE_DEVICES=6,7 \
    torchrun \
    --nproc-per-node=2 \
    --master-port=$((29500 + RANDOM % 1000)) \
    -m cleanslate.cli fsid train \
    --config cleanslate/fsid/configs/simnpo_gentle.yaml \
    -m llama-8b \
    --forget-path "unlearning-cleanslate/fsid-curated-llama-8b:forget:baseline" \
    --retain-path "unlearning-cleanslate/fsid-curated-llama-8b:retain:baseline" \
    -o runs/llama-3_1-8b/baseline-simnpo_gentle 2>&1 | tee logs/fsid/llama-8b_baseline_simnpo_gentle.log


cleanslate eval \
    -m runs/llama-3_1-8b/baseline-simnpo_gentle/simnpo_Llama-3.1-8B_20260427_200535/checkpoint-134 \
    --config config_dp4_gpu2467.yaml \
    -o results/llama-debug/llama-8b-simnpo-baseline-w100-s10

