#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH=../configs/TP_offload_static.json
config_file=../samples/TP_test.json
mkdir -p ../outputs/TP_offload_static
> ../outputs/TP_offload_static/vllm_msg.log
python ../examples/test_distN.py \
    --config_file=$config_file \
    --prefetch_mode=static \
    --prefetch-distance=1 \
    --flattened_cache=true \
    --merge-prefetch-buffer=true \
    --pause-and-resume=false  \
    --output_log=../outputs/TP_offload_static/output.log

# sed -i -E 's/^[A-Z]+ +[0-9-]+ [0-9:]+ [^ ]+:[0-9]+\] //' ../outputs/test_no_prefetch_flatten_kv/vllm_msg.log
