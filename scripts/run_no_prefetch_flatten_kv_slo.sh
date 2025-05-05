#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH=../configs/test_no_prefetch_flatten_kv_slo_logging.json
config_file=../samples/test_no_prefetch_flatten_kv_slo.json
mkdir -p ../outputs/test_no_prefetch_flatten_kv_slo
> ../outputs/test_no_prefetch_flatten_kv_slo/vllm_msg.log
python ../examples/test_distN.py \
    --config_file=$config_file \
    --prefetch_mode=static_req_wise \
    --flattened_cache=true \
    --merge-prefetch-buffer=true \
    --output_log=../outputs/test_no_prefetch_flatten_kv_slo/output.log \

# sed -i -E 's/^[A-Z]+ +[0-9-]+ [0-9:]+ [^ ]+:[0-9]+\] //' ../outputs/test_no_prefetch_flatten_kv/vllm_msg.log
