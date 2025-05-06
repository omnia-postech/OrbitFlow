#!/usr/bin/env bash
set -euo pipefail

── Experiment setup ──────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0
export VLLM_CONFIGURE_LOGGING=1

EXP="No_offload" # <---- Change this to the experiment name 
ROOT="/home/xinyuema/vllm"
BASE_LOG="${ROOT}/configs/test_no_prefetch_logging.json"
NEW_LOG="${ROOT}/configs/test${EXP}.json"          # will be overwritten per-run
OUT_DIR="${ROOT}/outputs/benchmark/${EXP}"
mkdir -p "${OUT_DIR}"

── Trace files ───────────────────────────────────────────────────────────────
CFG_DIR="${ROOT}/scripts/benchmark/test_traces"
List of traces to test
CFG_LIST=(S00 S03 S04)

── Run each trace ────────────────────────────────────────────────────────────
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
        --prefetch_mode=none \
        --flattened_cache=true \
        --merge_prefetch_buffer=true \
        --output_log="${RUN_DIR}/output.log"
done