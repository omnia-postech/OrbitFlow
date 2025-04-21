#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH=../configs/test_no_prefetch_logging.json
config_file=../samples/trace_mix2_creative_proofreading.json
mkdir -p ../outputs/test_no_prefetch
> ../outputs/test_no_prefetch/vllm_msg.log
python ../examples/test_distN.py \
    --config_file=$config_file \
    --prefetch_mode=none \
    --output_log=../outputs/test_no_prefetch/output.log \