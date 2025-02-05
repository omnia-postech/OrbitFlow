#!/bin/bash

# export VLLM_TRACE_FUNCTION=1
# export VLLM_LOGGING_LEVEL=DEBUG

length=$1

export CUDA_VISIBLE_DEVICES=1

/home/jongseop/anaconda3/envs/vllm/bin/python ../examples/offline_inference_offloading_length.py \
    --input-len ${length}
    # > output/test_2k.txt 2>&1