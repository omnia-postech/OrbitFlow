#!/bin/bash

source /home/sychoy/anaconda3/etc/profile.d/conda.sh
conda activate vllm

export CUDA_VISIBLE_DEVICES=2
export VLLM_CONFIGURE_LOGGING=1
# take the skeleton for logging 
# change the path based on the experiment name 
# save as new .json in ../config, or maybe in ../outputs/ 
# pass to VLLM_LOGGING_CONFIG_PATH 
export VLLM_LOGGING_CONFIG_PATH=../configs/test_no_prefetch_logging.json
config_file=../samples/trace_mix2_creative_proofreading.json
mkdir -p ../outputs/test_no_prefetch
> ../outputs/test_no_prefetch/vllm_msg.log
python ../examples/test_distN.py \
    --config_file=$config_file \
    --prefetch_mode=none \
    --output_log=../outputs/test_no_prefetch/output.log 