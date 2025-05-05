#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH=../configs/baseline_vllm.json
config_file=../samples/baseline.json
mkdir -p ../outputs/baseline_vllm
> ../outputs/baseline_vllm/vllm_msg.log
python ../examples/test_distN.py \
    --config_file=$config_file \
    --prefetch_mode=none \
    --flattened_cache=true \
    --merge-prefetch-buffer=true \
    --output_log=../outputs/baseline_vllm/output.log \


# python ../examples/test_distN.py \
#     --config_file=$config_file \
#     --prefetch_mode=static_req_wise \
#     --flattened_cache=true \
#     --merge-prefetch-buffer=true \
#     --output_log=../outputs/baseline_vllm/output.log \
# sed -i -E 's/^[A-Z]+ +[0-9-]+ [0-9:]+ [^ ]+:[0-9]+\] //' ../outputs/test_no_prefetch_flatten_kv/vllm_msg.log
