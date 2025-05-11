#!/usr/bin/env bash
# set -euo pipefail

###############################################################################
# CONSTANTS (edit these lists only)                                           #
###############################################################################
ROOT="/home/xinyuema/vllm"
LOGGING_LEVEL=CRITICAL
FIGURE_ONLY=0

EXP_LIST=(Debug)
METHOD_LIST=(NoPrefetch Flexgen Static8 SelectN Ours DistNSingle)
TRACE_CFG_DIR="${ROOT}/benchmark/test_traces/test_best_worst"
TRACE_LIST=(test_longshort_enough)

METHOD_CFG_FILE="${ROOT}/benchmark/scripts/supported_methods.json"
BASE_LOG="${ROOT}/configs/test_no_prefetch_logging.json"
PLOTTER="${ROOT}/benchmark/data_analysis/metrics_plot.py"
OVERVIEW_PLOTTER="${ROOT}/benchmark/data_analysis/plot_overview.py"

###############################################################################
# Split method list into 3 parts                                              #
###############################################################################
split_methods() {
  local -n arr=$1
  local total=${#arr[@]}
  local part_size=$(( (total + 2) / 3 ))  # ceil(total/3)
  GPU0=("${arr[@]:0:part_size}")
  GPU1=("${arr[@]:part_size:part_size}")
  GPU2=("${arr[@]:2*part_size}")
}

# Split METHOD_LIST into 3 parts
METHOD_LEN=${#METHOD_LIST[@]}
PART_SIZE=$(( (METHOD_LEN + 2) / 3 ))

GPU0_METHODS=("${METHOD_LIST[@]:0:PART_SIZE}")
GPU1_METHODS=("${METHOD_LIST[@]:PART_SIZE:PART_SIZE}")
GPU2_METHODS=("${METHOD_LIST[@]:2*PART_SIZE}")

# split_and_run 1 "${GPU0_METHODS[@]}" &
# split_and_run 2 "${GPU1_METHODS[@]}" &
# split_and_run 3 "${GPU2_METHODS[@]}" &

# echo ./run_grid_runner.sh 1 "${GPU0_METHODS[@]}"

# ./run_grid_runner.sh 1 "${GPU0_METHODS[@]}" &
# ./run_grid_runner.sh 2 "${GPU0_METHODS[@]}" &
# ./run_grid_runner.sh 3 "${GPU0_METHODS[@]}" &

wait

EXP_OUT="${ROOT}/outputs/benchmark/Debug"

plot_global() {
  
  python $OVERVIEW_PLOTTER \
    --input-dir $EXP_OUT \
    --x-labels "${METHOD_LIST[@]}" \
    --metric "slo_99" \
    --trace "${TRACE_LIST[0]}" \
    --xlabel "Methods" \
    --ylabel "99th decode latency / SLO" \
    --title "99th decode latency / SLO" \
    --output "${EXP_OUT}/slo_99th.pdf"
}


plot_global