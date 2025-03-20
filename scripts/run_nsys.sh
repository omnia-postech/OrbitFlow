#!/bin/bash

sudo /usr/local/cuda-12.1/bin/nsys profile --stats=true \
    --trace cuda,cudnn,nvtx \
    --force-overwrite=true \
    --output nsys/ap_expr_0.9 \
    ./run_offline_inference_offloading.sh \
    > output/ap_expr_0.9.txt 2>&1

# length test
# Define the input lengths to test
# input_lengths=(128 256 512 1024 2048 4096 8192 16384 32768 65536 131072)
# input_lengths=(128 256 512 1024 2048 4096 8192 16384 32768 65536)
# input_lengths=(131072)

# # Loop through each input length
# for input_len in "${input_lengths[@]}"; do
#     echo "Running inference for input length: $input_len"

#     # Run the nsys profiling command
#     sudo /usr/local/cuda-12.1/bin/nsys profile --stats=true \
#         --trace cuda,cudnn,nvtx \
#         --force-overwrite=true \
#         --output nsys/test_${input_len} \
#         ./run_offline_inference_offloading_length.sh "${input_len}" \
#         > output/test_${input_len}.txt 2>&1

#     # Check if the command was successful
#     if [ $? -eq 0 ]; then
#         echo "Profiling completed for input length: $input_len"
#     else
#         echo "Error during profiling for input length: $input_len"
#     fi
# done

# echo "All profiling tasks completed."