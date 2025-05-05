# #!/bin/bash

# export CUDA_VISIBLE_DEVICES=3
# export VLLM_CONFIGURE_LOGGING=1
# export VLLM_LOGGING_CONFIG_PATH=../configs/test_output_logging.json
# config_file=../samples/output_test.json
# mkdir -p ../outputs/test_output
# > ../outputs/test_output/vllm_msg.log
# python ../examples/test_distN.py \
#     --config_file=$config_file \
#     --prefetch_mode=none \
#     --is-monolithic-distn=False \
#     --output_log=../outputs/test_output/output.log \

#!/bin/bash

# export VLLM_TRACE_FUNCTION=1
# export VLLM_LOGGING_LEVEL=DEBUG

# CUDA_VISIBLE_DEVICES=3

CUDA_VISIBLE_DEVICES=3 /home/heelim/anaconda3/envs/vllm/bin/python ../examples/offline_inference_offloading.py \
    --input ../samples/output_test.md \
    # > output/test_2k.txt 2>&1