#!/bin/bash

# export VLLM_TRACE_FUNCTION=1
# export VLLM_LOGGING_LEVEL=DEBUG

export CUDA_VISIBLE_DEVICES=0
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH=/home/xinyuema/vllm/configs/logging_to_file.json

python ../examples/offline_inference_offloading_long_out_mono_batched.py \
    --is-monolithic-distn=True  \
    # --output_log=output/test.log 