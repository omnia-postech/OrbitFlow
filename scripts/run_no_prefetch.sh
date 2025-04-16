#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH=/home/heelim/vllm/configs/test_no_prefetch_logging.json
config_file=/home/heelim/vllm/samples/large_new_request.json
mkdir -p /home/heelim/vllm/outputs/test_no_prefetch
> /home/heelim/vllm/outputs/test_no_prefetch/vllm_msg.log
python ../examples/test_distN.py \
    --config=$config_file \
    --prefetch_mode=none \
    --output_log=/home/heelim/vllm/outputs/test_no_prefetch/output.log \