#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH=../configs/test_prefetch_2_flatten_kv_logging.json
config_file=../examples/trace_single_chatbot_qa_1.json
mkdir -p ../outputs/test_prefetch_2_flatten_kv
> ../outputs/test_prefetch_2_flatten_kv/vllm_msg.log
python ../examples/test_distN.py \
    --config_file=$config_file \
    --prefetch_mode=static \
    --prefetch_distance=2 \
    --flattened_cache=true \
    --merge_prefetch_buffer=true \
    --output_log=../outputs/test_prefetch_2_flatten_kv/output.log \

sed -i -E 's/^[A-Z]+ +[0-9-]+ [0-9:]+ [^ ]+:[0-9]+\] //' ../outputs/test_prefetch_2_flatten_kv/vllm_msg.log
