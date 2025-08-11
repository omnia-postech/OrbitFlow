#!/usr/bin/env bash

paths=(
    "/home/heelim/vllm/outputs/benchmark/paper_main_exp_32k/slo2.5/NextLayer/lambda2.0x_cv1"
    "/home/heelim/vllm/outputs/benchmark/paper_main_exp_32k/slo2.5/Static1/lambda2.0x_cv1"
    "/home/heelim/vllm/outputs/benchmark/paper_main_exp_32k/slo2.5/Flexgen/lambda2.0x_cv1"
    "/home/heelim/vllm/outputs/benchmark/paper_main_exp_32k/slo2.5/SelectN/lambda2.0x_cv1"
    "/home/heelim/vllm/outputs/benchmark/paper_main_exp_32k/slo2.5/DistNSingle/lambda2.0x_cv1"
    "/home/heelim/vllm/outputs/benchmark/paper_main_exp_32k/slo2.5/Ours/lambda2.0x_cv1"
)

for path in "${paths[@]}"; do
    echo "▶ 처리 중: $path"
    python /home/heelim/vllm/benchmark/data_analysis/data_parsing/get_mem_utilzation.py "$path"
done


for path in "${paths[@]}"; do
    echo "▶ 처리 중: $path"
    python /home/heelim/vllm/benchmark/data_analysis/data_parsing/make_mem_util.py "$path"
done
    
