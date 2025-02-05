#!/bin/bash

# export VLLM_TRACE_FUNCTION=1
# export VLLM_LOGGING_LEVEL=DEBUG

/home/jongseop/anaconda3/envs/vllm/bin/python ../examples/offline_inference_offloading.py \
    --input ../samples/128k.md \
    # > output/test_2k.txt 2>&1