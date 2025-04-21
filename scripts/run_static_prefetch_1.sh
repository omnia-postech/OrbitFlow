#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH=../configs/test_static_prefetch_1_logging.json
config_file=../samples/trace_mix2_creative_proofreading.json
mkdir -p ../outputs/test_static_prefetch_1
> ../outputs/test_static_prefetch_1/vllm_msg.log
python ../examples/test_distN.py \
    --config_file=$config_file \
    --prefetch_mode=static \
    --prefetch_distance=1 \
    --output_log=../outputs/test_static_prefetch_1/output.log \