#!/bin/bash

# 테스트할 input length 리스트
lengths=(256 512 1024 2048 4096 8192 16384 32768 65536)
lengths=(131040)

# 리스트의 각 length에 대해 테스트 실행
for length in "${lengths[@]}"; do
    echo "Running inference with input length: $length"

    # nsys와 스크립트를 실행
    sudo /usr/local/cuda-12.1/bin/nsys profile --stats=true \
        --trace cuda,cudnn,nvtx \
        --force-overwrite=true \
        --output nsys/test_recomp_${length} \
        ./run_offline_inference_offloading_length.sh $length \
        > output/test_recomp_${length}.txt 2>&1

    sleep 10
done
