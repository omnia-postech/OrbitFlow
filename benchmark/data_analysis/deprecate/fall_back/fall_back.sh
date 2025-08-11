#!/usr/bin/env bash

# 공통 경로
base_path=(
    /home/heelim/vllm/outputs/benchmark/paper_main_exp_longest_xinyue
    /home/heelim/vllm/outputs/benchmark/paper_main_exp_random_xinyue
    /home/heelim/vllm/outputs/benchmark/paper_main_exp_shortest_xinyue
    # # "/home/heelim/vllm/outputs/benchmark/paper_main_exp_fallback_slo_strict"
)  # change here

# echo "Running change_slo_scale.py ..."
# python /home/heelim/vllm/benchmark/data_analysis/data_parsing/change_slo_scale.py \
#   --old-sc 2.5 \
#   --new-sc-list 3.3 4.3 5.4\
#   --base-path "${base_path}" \
#   --is-arrival \
#   --arrival-rate-list 2.0 \
#   --cv-list 1 \
#   --methods SelectN DistNSingle Flexgen Ours\
#   --arrival-tpl "lambda{rate}x_cv{cv}" # change here

# if [ $? -ne 0 ]; then
#     echo "❌ change_slo_scale.py 실행 실패"
# fi


# base_path 아래의 상대 경로만 나열 # change here

#   "slo2.5/Ours"


# # 절대 경로 배열 생성
all_roots=()
for rel in "${base_path[@]}"; do
  dir="$rel/slo1/Ours"
  if [ -d "$dir" ]; then
    echo "유효한 경로: $dir"
    all_roots+=("$dir")
  else
    echo "경로가 유효하지 않음: $dir" >&2
  fi
done

# lambda로 시작하는 하위 디렉토리만 골라서 all_subdirs 에 모으기
all_subdirs=()
for root in "${all_roots[@]}"; do
  for sub in "$root"/*; do
    [ -d "$sub" ] || continue
    all_subdirs+=("$sub")
    echo "유효한 하위 디렉토리: $sub"
  done
done


# sim_slo_violation.py 실행
echo "Running sim_slo_violation.py ..."
python /home/heelim/vllm/benchmark/data_analysis/data_parsing/sim_slo_violation_v2.py "${all_subdirs[@]}"
if [ $? -ne 0 ]; then
    echo "❌ sim_slo_violation.py 실행 실패"
fi

# make_arrival_rate_summerize.py 실행
echo "Running make_arrival_rate_summerize.py ..."
python /home/heelim/vllm/benchmark/data_analysis/data_parsing/make_arrival_rate_summerize_v2.py "${all_roots[@]}"
if [ $? -ne 0 ]; then
    echo "❌ make_arrival_rate_summerize.py 실행 실패"
    exit 1
fi

# arrival_rate_tbt_tpot_v2.py 실행
echo "Running fall_back_tbt.py ..."
python /home/heelim/vllm/benchmark/data_analysis/fall_back/fall_back_tbt.py
if [ $? -ne 0 ]; then
    echo "❌ fall_back_tbt.py 실행 실패"
fi