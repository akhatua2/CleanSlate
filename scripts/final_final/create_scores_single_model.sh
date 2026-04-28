source .venv/bin/activate

cleanslate score \
    --forget-pool     "unlearning-cleanslate/fsid-curated-llama-8b:forget_pool" \
    --pre-eval        "unlearning-cleanslate/verif-llama-3.1-8b-w100-s10" \
    --post-eval       "unlearning-cleanslate/DEBUG-llama-8b-eval-checkpoint-134" \
    --pre-generations  "unlearning-cleanslate/generations-llama-3_1-8b-pre_val" \
    --post-generations "unlearning-cleanslate/generations-checkpoint-134-debug-checkpoint-134-llama" \
    --checkpoint "runs/llama-3_1-8b/baseline-simnpo_gentle/simnpo_Llama-3.1-8B_20260427_200535/checkpoint-134"\
    -o results/scores/llama-8b-simnpo-baseline/score_llama-8b-debug.json
