source .venv/bin/activate

cleanslate eval -m olmo-7b --config config_dp2.yaml \
    -o results/eval/verif-olmo-7b-w100-s10 \                                              
    --push-suffix w100-s10

cleanslate eval -m nemotron-9b --config config_dp2.yaml \
    -o results/eval/verif-nemotron-9b-w100-s10 \                                              
    --push-suffix w100-s10

cleanslate eval -m gemma-12b --config config_dp2.yaml \
    -o results/eval/verif-gemma-12b-w100-s10 \                                              
    --push-suffix w100-s10

cleanslate eval -m llama-8b --config config_dp2.yaml \
    -o results/eval/verif-llama-8b-w100-s10 \                                              
    --push-suffix w100-s10

cleanslate eval -m olmo-32b --config config_tp2dp2.yaml \
    -o results/eval/verif-olmo-32b-w100-s10 \                                              
    --push-suffix w100-s10

cleanslate eval -m qwen-8b --config config_dp2.yaml \
    -o results/eval/verif-qwen-8b-w100-s10 \                                              
    --push-suffix w100-s10