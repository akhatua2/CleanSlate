source .venv/bin/activate

cleanslate score \
    --forget-pool     "unlearning-cleanslate/fsid-curated-llama-8b:forget_pool" \
    --pre-eval        "unlearning-cleanslate/verif-llama-3.1-8b-w100-s10" \
    --post-eval       "unlearning-cleanslate/DEBUG-llama-8b-eval-checkpoint-134" \
    --pre-generations  "unlearning-cleanslate/generations-llama-3_1-8b-pre_val" \
    --post-generations "unlearning-cleanslate/generations-checkpoint-134-debug-checkpoint-134-llama" \
    --checkpoint "runs/llama-3_1-8b/baseline-simnpo_gentle/simnpo_Llama-3.1-8B_20260427_200535/checkpoint-134"\
    -o results/scores/llama-8b-simnpo-baseline/score_llama-8b-debug.json


cleanslate score \
    --forget-pool     "unlearning-cleanslate/fsid-curated-qwen3-8b:forget_pool" \
    --pre-eval        "unlearning-cleanslate/verif-qwen3-8b-w100-s10" \
    --post-eval       "unlearning-cleanslate/eval-simnpo_qwen3-8b_20260428_063109-debug-post-qwen" \
    --pre-generations  "unlearning-cleanslate/generations-qwen3-8b-pre_val" \
    --post-generations "unlearning-cleanslate/generations-simnpo_qwen3-8b_20260428_063109-debug-post-qwen" \
    --checkpoint "runs/qwen3-8b/baseline-simnpo_gentle-debug-qwen-8b/simnpo_Qwen3-8B_20260428_063109"\
    -o results/scores/qwen-8b-simnpo-baseline/score_qwen-8b-debug.json


cleanslate score \
    --forget-pool     "unlearning-cleanslate/fsid-curated-olmo-7b:forget_pool" \
    --pre-eval        "unlearning-cleanslate/eval-olmo-3-7b-w100-s10" \
    --post-eval       "unlearning-cleanslate/eval-simnpo_olmo-3-1025-7b_20260428_064411-debug-post-olmo" \
    --pre-generations  "unlearning-cleanslate/generations-olmo-3-7b-pre_val" \
    --post-generations "unlearning-cleanslate/generations-simnpo_olmo-3-1025-7b_20260428_064411-debug-post-olmo" \
    --checkpoint "runs/olmo-3-7b/baseline-simnpo_gentle-debug-olmo-7b/simnpo_Olmo-3-1025-7B_20260428_064411/checkpoint-127"\
    -o results/scores/olmo-7b-simnpo-baseline/score_olmo-7b-debug.json



