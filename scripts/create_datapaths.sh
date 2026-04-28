source .venv/bin/activate

python scripts/fsid_curate_and_push.py \
    --model qwen-8b \
    --hub-repo-id unlearning-cleanslate/fsid-curated-qwen3-8b \
    --eval-dataset unlearning-cleanslate/verif-qwen3-8b-w100-s10


python scripts/fsid_curate_and_push.py \
    --model gemma-12b \
    --hub-repo-id unlearning-cleanslate/fsid-curated-gemma-12b \
    --eval-dataset unlearning-cleanslate/verif-gemma-3-12b-pt-w100-s10


python scripts/fsid_curate_and_push.py \
    --model llama-8b \
    --hub-repo-id unlearning-cleanslate/fsid-curated-llama-8b \
    --eval-dataset unlearning-cleanslate/verif-llama-3.1-8b-w100-s10


python scripts/fsid_curate_and_push.py \
    --model olmo-32b \
    --hub-repo-id unlearning-cleanslate/fsid-curated-olmo-32b \
    --eval-dataset unlearning-cleanslate/verif-olmo-3-1125-32b-w100-s10


python scripts/fsid_curate_and_push.py \
    --model olmo-7b \
    --hub-repo-id unlearning-cleanslate/fsid-curated-olmo-7b \
    --eval-dataset unlearning-cleanslate/eval-olmo-3-7b-w100-s10

python scripts/fsid_curate_and_push.py \
    --model nemotron-9b \
    --hub-repo-id unlearning-cleanslate/fsid-curated-nemotron-9b \
    --eval-dataset unlearning-cleanslate/eval-nemotron-nano-9b-v2-w100-s10



