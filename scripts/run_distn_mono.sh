#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH=/home/xinyuema/vllm/configs/test_distn_mono_logging.json
config_file=/home/xinyuema/vllm/samples/large_new_request.json
mkdir -p /home/xinyuema/vllm/outputs/test_distn_mono
> /home/xinyuema/vllm/outputs/test_distn_mono/vllm_msg.log
python ../examples/test_distN.py \
    --config=$config_file \
    --prefetch_mode=distn \
    --is-monolithic-distn=True \
    --output_log=/home/xinyuema/vllm/outputs/test_distn_mono/output.log \