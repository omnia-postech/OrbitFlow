#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH=/home/xinyuema/vllm/configs/test_disn_logging.json
> /home/xinyuema/vllm/outputs/test_distn/vllm_msg.log
python ../examples/test_distN.py 