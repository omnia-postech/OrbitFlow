#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH=/home/heelim/vllm/configs/test_no_prefetch_seperated_logging.json
config_file=/home/heelim/vllm/samples/large_new_request_seperated.json
mkdir -p /home/heelim/vllm/outputs/test_no_prefetch_seperated
> /home/heelim/vllm/outputs/test_no_prefetch_seperated/vllm_msg.log
python ../examples/test_distN.py \
    --config_file=$config_file \
    --prefetch_mode=none \
    --output_log=/home/heelim/vllm/outputs/test_no_prefetch_seperated/output.log \