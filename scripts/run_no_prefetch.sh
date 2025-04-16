#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH=/home/xinyuema/vllm/configs/test_no_prefetch_logging.json
config_file=/home/xinyuema/vllm/samples/large_new_request.json
mkdir -p /home/xinyuema/vllm/outputs/test_no_prefetch
> /home/xinyuema/vllm/outputs/test_no_prefetch/vllm_msg.log
python ../examples/test_distN.py \
    --config=$config_file \
    --prefetch_mode=none \
    --output_log=/home/xinyuema/vllm/outputs/test_no_prefetch/output.log \