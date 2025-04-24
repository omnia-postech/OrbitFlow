#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH=../configs/test_no_prefetch_flatten_kv_logging.json
config_file=../examples/trace_single_chatbot_qa.json
mkdir -p ../outputs/test_no_prefetch_flatten_kv
> ../outputs/test_no_prefetch_flatten_kv/vllm_msg.log
python ../examples/test_distN.py \
    --config_file=$config_file \
    --prefetch_mode=none \
    --flattened_cache=true \
    --output_log=../outputs/test_no_prefetch_flatten_kv/output.log \

sed -i -E 's/^[A-Z]+ +[0-9-]+ [0-9:]+ [^ ]+:[0-9]+\] //' ../outputs/test_no_prefetch_flatten_kv/vllm_msg.log
