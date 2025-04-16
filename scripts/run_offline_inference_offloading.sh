#!/bin/bash

# export VLLM_TRACE_FUNCTION=1
# export VLLM_LOGGING_LEVEL=DEBUG

export CUDA_VISIBLE_DEVICES=0
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH=/home/xinyuema/vllm/configs/logging_to_file.json

python ../examples/offline_inference_offloading_long_out.py \
    --is-monolithic-distn False
    # > output/test_2k.txt 2>&1