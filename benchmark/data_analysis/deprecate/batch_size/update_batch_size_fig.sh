#!/usr/bin/env bash

# 공통 경로
base_path="/home/heelim/vllm/outputs/benchmark/paper_main_exp_bs"  # change here

# echo "Running change_slo_scale.py ..."
# python /home/heelim/vllm/benchmark/data_analysis/data_parsing/change_slo_scale.py \
#   --old-sc 2.5 \
#   --new-sc-list 1\
#   --base-path "${base_path}" \
#   --is-arrival \
#   --arrival-rate-list 2.0 \
#   --cv-list 1\
#   --methods DistNSingle Flexgen\
#   --arrival-tpl "lambda{rate}x_cv{cv}_bs8" # change here

# if [ $? -ne 0 ]; then
#     echo "❌ change_slo_scale.py 실행 실패"
# fi

# base_path 아래의 상대 경로만 나열 # change here
subdirs=(
  # "slo1/Ours"
#   "slo1.5/Ours"
  # "slo2.5/Ours"

  # "slo1/DistNSingle"
  # "slo1.5/DistNSingle"
  # "slo2.5/DistNSingle"

  # "slo1/Flexgen"
  # "slo1.5/Flexgen"
  # "slo2.5/Flexgen"

  # "slo1/UniformSolver_TD_PR"
  # "slo1.5/UniformSolver_TD_PR"
  # "slo2.5/UniformSolver_TD_PR"

)

# # 절대 경로 배열 생성
all_roots=()
for rel in "${subdirs[@]}"; do
  dir="$base_path/$rel"
  if [ -d "$dir" ]; then
    all_roots+=("$dir")
  else
    echo "경로가 유효하지 않음: $dir" >&2
  fi
done

# lambda로 시작하는 하위 디렉토리만 골라서 all_subdirs 에 모으기
all_subdirs=()
for root in "${all_roots[@]}"; do
  for sub in "$root"/*lambda*; do
    [ -d "$sub" ] || continue
    all_subdirs+=("$sub")
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
echo "Running design_validation.py ..."
python /home/heelim/vllm/benchmark/data_analysis/bs/arrival_rate_bs.py "${base_path}"
if [ $? -ne 0 ]; then
    echo "❌ design_validation.py 실행 실패"
fi

