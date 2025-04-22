#!/bin/bash

source /home/sychoy/anaconda3/etc/profile.d/conda.sh
conda activate vllm

export CUDA_VISIBLE_DEVICES=0
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH=../configs/test_distn_dyn_logging.json
config_file=../examples/trace_mix2_creative_proofreading.json
mkdir -p ../outputs/test_distn_dyn
> ../outputs/test_distn_dyn/vllm_msg.log
python ../examples/test_distN.py \
    --config_file=$config_file \
    --prefetch_mode=distn \
    --is-monolithic-distn=False \
    --output_log=../outputs/test_distn_dyn/output.log \