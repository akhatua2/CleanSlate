source .venv/bin/activate

cleanslate score \
    --forget-path     "unlearning-cleanslate/fsid-curated-llama-8b:forget_pool:train" \
    --pre-eval        "unlearning-cleanslate/verif-llama-3.1-8b-w100-s10" \
    --post-eval       "unlearning-cleanslate/DEBUG-llama-8b-eval-checkpoint-134" \
    --pre-generations  "unlearning-cleanslate/generations-llama-3_1-8b-pre_val" \
    --post-generations "unlearning-cleanslate/generations-llama-3_1-8b-post_val" \
    -o results/post_val/llama-8b-simnpo-baseline/score_v2.json
