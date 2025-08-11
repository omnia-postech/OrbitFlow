#!/usr/bin/env bash

# 공통 경로
base_path="/home/heelim/vllm/outputs/benchmark/paper_main_exp_32k"  # change here

# echo "Running change_slo_scale.py ..."
# python /home/heelim/vllm/benchmark/data_analysis/data_parsing/change_slo_scale.py \
#   --old-sc 2.5 \
#   --new-sc-list 3\
#   --base-path "${base_path}" \
#   --is-arrival \
#   --arrival-rate-list 1.0 2.0 3.0 4.0 5.0 \
#   --cv-list 1 \
#   --methods NextLayer Flexgen Static1 SelectN DistNSingle\
#   --arrival-tpl "lambda{rate}x_cv{cv}" # change here

# if [ $? -ne 0 ]; then
#     echo "❌ change_slo_scale.py 실행 실패"
# fi


# base_path 아래의 상대 경로만 나열 # change here
subdirs=(
  # "slo1/Ours"
  # "slo1.5/Ours"
  # "slo2/Ours"
  # "slo2.5/Ours"

  # "slo1/Flexgen"
  # "slo1.5/Flexgen"
  # "slo2/Flexgen"
  # "slo2.5/Flexgen"
  # "slo3/Flexgen"
  # "slo3.5/Flexgen"

  # "slo1/Static1"
  # "slo1.5/Static1"
  # "slo2/Static1"
  # "slo2.5/Static1"
  # "slo3/Static1"
  # "slo3.5/Static1"

  # "slo1/SelectN"
  # "slo1.5/SelectN"
  # "slo2/SelectN"
  # "slo2.5/SelectN"
  # "slo3/SelectN"
  # "slo3.5/SelectN"

  # "slo1/NextLayer"
  # "slo1.5/NextLayer"
  # "slo2/NextLayer"
  # "slo2.5/NextLayer"
  # "slo3/NextLayer"
  # "slo3.5/NextLayer"

  # "slo1/DistNSingle"
  # "slo1.5/DistNSingle"
  # "slo2/DistNSingle"
  # "slo2.5/DistNSingle"
  # "slo3/DistNSingle"
  # "slo3.5/DistNSingle"
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


# # sim_slo_violation.py 실행
# echo "Running sim_slo_violation.py ..."
# python /home/heelim/vllm/benchmark/data_analysis/data_parsing/sim_slo_violation_v2.py "${all_subdirs[@]}"
# if [ $? -ne 0 ]; then
#     echo "❌ sim_slo_violation.py 실행 실패"
# fi

# # make_arrival_rate_summerize.py 실행
# echo "Running make_arrival_rate_summerize.py ..."
# python /home/heelim/vllm/benchmark/data_analysis/main_result/make_arrival_rate_summerize_v2.py "${all_roots[@]}"
# if [ $? -ne 0 ]; then
#     echo "❌ make_arrival_rate_summerize.py 실행 실패"
#     exit 1
# fi

# arrival_rate_tbt_tpot_v2.py 실행
echo "Running arrival_rate_tbt_tpot_v2.py ..."
python /home/heelim/vllm/benchmark/data_analysis/data_parsing/make_arrival_rate_summerize_v2.py "${base_path}"
if [ $? -ne 0 ]; then
    echo "❌ arrival_rate_tbt_tpot_v2.py 실행 실패"
fi

# arrival_rate_tbt_tpot_v2.py 실행
echo "Running p_slo_scale_bar.py ..."
python /home/heelim/vllm/benchmark/data_analysis/draw_paper_graph/p_slo_scale_bar.py "${base_path}"
if [ $? -ne 0 ]; then
    echo "❌ p_slo_scale_bar.py 실행 실패"
fi
