#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH=/home/xinyuema/vllm/configs/test_distn_dyn_logging.json
config_file=../samples/trace_mix2_creative_proofreading.json
mkdir -p /home/xinyuema/vllm/outputs/test_distn_dyn
> /home/xinyuema/vllm/outputs/test_distn_dyn/vllm_msg.log
python ../examples/test_distN.py \
    --config_file=$config_file \
    --prefetch_mode=distn \
    --is-monolithic-distn=False \
    --output_log=/home/xinyuema/vllm/outputs/test_distn_dyn/output.log \