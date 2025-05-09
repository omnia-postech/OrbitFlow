#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export VLLM_CONFIGURE_LOGGING=1

EXP="Solver" # <---- Change this to the experiment name 
ROOT="/home/heelim/vllm"
BASE_LOG="${ROOT}/configs/test_no_prefetch_logging.json"
NEW_LOG="${ROOT}/configs/test_${EXP}.json"          # will be overwritten per-run
OUT_DIR="${ROOT}/outputs/benchmark/${EXP}"
OUT_DIR="${ROOT}/outputs/${EXP}"
mkdir -p "${OUT_DIR}"

# CFG_DIR="${ROOT}/scripts/benchmark/test_traces"
# CFG_LIST=(S04)
CFG_DIR="${ROOT}/samples"
CFG_LIST=(TP_test)

for T in "${CFG_LIST[@]}"; do
    RUN_DIR="${OUT_DIR}/${T}"           # one sub-folder per cfg
    mkdir -p "${RUN_DIR}"

    # regenerate logging config for this run
    sed '16s#"filename":.*#"filename": "../outputs/'"${EXP}/${T}"'/vllm_msg.log"#' \
        "${BASE_LOG}" > "${NEW_LOG}"
    export VLLM_LOGGING_CONFIG_PATH="${NEW_LOG}"
    : > "${RUN_DIR}/vllm_msg.log"

    # Change option below to match with the experiment name

    python ../examples/test_distN.py \
        --config-file "${CFG_DIR}/${T}.json" \
        --prefetch-mode solver \
        --prefetch-distance 1 \
        --flattened-cache true \
        --merge-prefetch-buffer true \
        --pause-and-resume \
        --output-log ../outputs/${EXP}/${T}/output.log
done