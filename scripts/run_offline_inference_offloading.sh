#!/bin/bash

# export VLLM_TRACE_FUNCTION=1
# export VLLM_LOGGING_LEVEL=DEBUG

export CUDA_VISIBLE_DEVICES=3

/home/jongseop/anaconda3/envs/vllm/bin/python ../examples/offline_inference_offloading_long_out.py \
    --is-monolithic-distn False
    # > output/test_2k.txt 2>&1