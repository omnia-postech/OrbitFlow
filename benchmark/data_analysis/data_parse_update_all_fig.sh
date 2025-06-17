#!/bin/bash

# input_dir은 리스트 형태로 여러 경로를 받을 수 있음
input_dirs=(
    # "/path/to/dir1"
    # "/path/to/dir2"
    # "/path/to/dir3"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1.5/Ours_TP"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/Ours_TP"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/Ours_TP"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo4.5/Ours_TP"
    "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1/Flexgen"
    "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1.5/Flexgen"
    "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/Flexgen"
    "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/Flexgen"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1.5/SelectN_TP"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/SelectN_TP"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/SelectN_TP"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo4.5/SelectN_TP"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/Ours"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/Ours"
)

# 리스트를 공백으로 이어붙여 하나의 문자열로 만듦
args="${input_dirs[@]}"

# 모든 하위 디렉토리 경로를 담을 리스트
all_subdirs=()

# 각 input_dir에서 하위 디렉토리 수집
for root in "${input_dirs[@]}"; do
    if [ -d "$root" ]; then
        # 하위 항목 중 디렉토리만 필터링
        for subdir in "$root"/*; do
            if [ -d "$subdir" ]; then
                all_subdirs+=("$subdir")
            fi
        done
    else
        echo "경로가 유효하지 않음: $root"
        exit 1
    fi
done

# sim_slo_violation.py 실행
echo "Running sim_slo_violation.py ..."
python ./data_parsing/sim_slo_violation.py "${all_subdirs[@]}"
if [ $? -ne 0 ]; then
    echo "sim_slo_violation.py 실행 실패"
    exit 1
fi

# make_summerize.py 실행
echo "Running make_summerize.py ..."
python ./data_parsing/make_summerize.py $args
if [ $? -ne 0 ]; then
    echo "make_summerize.py 실행 실패"
    exit 1
fi

# design_validation.py 실행
echo "Running design_validation.py ..."
python ./draw_paper_graph/design_validation.py
if [ $? -ne 0 ]; then
    echo "design_validation.py 실행 실패"
    exit 1
fi

# draw_tp_tbt.py 실행
echo "Running draw_tp_tbt.py ..."
python ./draw_paper_graph/draw_tp_tbt.py
if [ $? -ne 0 ]; then
    echo "draw_tp_tbt.py 실행 실패"
    exit 1
fi

# p_slo_scale_bar.py 실행
echo "Running p_slo_scale_bar.py ..."
python ./draw_paper_graph/p_slo_scale_bar.py
if [ $? -ne 0 ]; then
    echo "p_slo_scale_bar.py 실행 실패"
    exit 1
fi

# throughput_total_compare.py 실행
echo "Running throughput_total_compare.py ..."
python ./draw_paper_graph/throughput_total_compare.py
if [ $? -ne 0 ]; then
    echo "throughput_total_compare.py 실행 실패"
    exit 1
fi

# tpot_tbt_total_compare_graph.py 실행
echo "Running tpot_tbt_total_compare_graph.py ..."
python ./draw_paper_graph/tpot_tbt_total_compare_graph.py
if [ $? -ne 0 ]; then
    echo "tpot_tbt_total_compare_graph.py 실행 실패"
    exit 1
fi

# batch_size.py 실행
echo "Running batch_size.py ..."
python ./draw_paper_graph/batch_size.py
if [ $? -ne 0 ]; then
    echo "batch_size.py 실행 실패"
    exit 1
fi

# context_length.py 실행
echo "Running context_length.py ..."
python ./draw_paper_graph/context_length.py
if [ $? -ne 0 ]; then
    echo "context_length.py 실행 실패"
    exit 1
fi