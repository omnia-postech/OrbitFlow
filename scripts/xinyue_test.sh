#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_CONFIGURE_LOGGING=1

EXP="xinyue_test_TP_distance2" # <---- Change this to the experiment name 
ROOT="/home/heelim/vllm"
BASE_LOG="${ROOT}/configs/test_no_prefetch_logging.json"
NEW_LOG="${ROOT}/configs/test${EXP}.json"          # will be overwritten per-run
OUT_DIR="${ROOT}/outputs/benchmark/${EXP}"
mkdir -p "${OUT_DIR}"

CFG_DIR="${ROOT}/scripts/benchmark/test_traces"
CFG_LIST=(S04)

for T in "${CFG_LIST[@]}"; do
    RUN_DIR="${OUT_DIR}/${T}"           # one sub-folder per cfg
    mkdir -p "${RUN_DIR}"

    # regenerate logging config for this run
    sed '16s#"filename":.*#"filename": "../outputs/benchmark/'"${EXP}/${T}"'/vllm_msg.log"#' \
        "${BASE_LOG}" > "${NEW_LOG}"
    export VLLM_LOGGING_CONFIG_PATH="${NEW_LOG}"
    : > "${RUN_DIR}/vllm_msg.log"

    # Change option below to match with the experiment name

    python ../examples/test_distN.py \
        --config_file="${CFG_DIR}/${T}.json" \
        --prefetch_mode=static \
        --prefetch_distance=2 \
        --flattened_cache=true \
        --merge-prefetch-buffer=true \
        --output_log=../outputs/TP_no_offload/output.log
done