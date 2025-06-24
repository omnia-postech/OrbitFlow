#!/usr/bin/env bash
###############################################################################
# run_benchmarks.sh
#
# One-stop script for vLLM benchmark execution **and** figure generation.
#
# USAGE
#   ./run_benchmarks.sh [FIGURE_ONLY]
#     FIGURE_ONLY = 0 (default) → run + plot
#     FIGURE_ONLY = 1           → *skip* execution, **only** plot existing CSVs
#
# To tweak what gets run (methods, traces, SLOs…), simply edit the CONSTANTS
# section below – **no other code changes are needed.**
###############################################################################

set -euo pipefail              # fail fast on any error
IFS=$'\n\t'                    # safer word-splitting

###############################################################################
# 1. CONSTANTS – edit freely ✏️                                                #
###############################################################################
export CUDA_VISIBLE_DEVICES=1
export VLLM_CONFIGURE_LOGGING=1        # 0 → minimal, 1 → user-configurable

LOGGING_LEVEL=INFO                 # CRITICAL│ERROR│WARNING│INFO│DEBUG
ROOT="/home/heelim/vllm"               # project root

profiled_path="/home/heelim/vllm/benchmark/scripts/profiled_results_A5000.json"
FIGURE_ONLY="${1:-0}"                  # default = 0 (run + plot)

EXP_LIST=(paper_main_exp_48k)              # high-level experiment names
METHOD_LIST=(Flexgen)                  # see supported_methods.json for keys
TRACE_LIST=(test)     # trace JSONs (basename only)
# TRACE_LIST=(48k_lambda4.0x_cv1 48k_lambda3.5x_cv1 48k_lambda3.0/x_cv1 48k_lambda2.5x_cv1 48k_lambda2.0x_cv1 48k_lambda1.5x_cv1 48k_lambda1.0x_cv1)     # trace JSONs (basename only)

TRACE_CFG_DIR="${ROOT}/benchmark/selected_traces"
METHOD_CFG_FILE="${ROOT}/benchmark/scripts/supported_methods.json"
BASE_LOG="${ROOT}/configs/test_no_prefetch_logging.json"
PLOTTER="${ROOT}/benchmark/data_analysis/metrics_plot.py"

SLO_RATIO_LIST=(2.5)                   # e.g. 1.5 2.0 2.5 …

###############################################################################
# 2. UTILITY FUNCTIONS                                                         #
###############################################################################

# ---------------------------------------------------------------------------
# load_method_args <method_key>
#   Reads CLI argument list for <method_key> from supported_methods.json
#   -> prints an *array* usable in "${array[@]}"
# ---------------------------------------------------------------------------
load_method_args() {
  local key="$1"
  mapfile -t args < <(
    python - "$METHOD_CFG_FILE" "$key" <<'PY'
import json, sys, pathlib, textwrap
cfg_path, key = sys.argv[1:]
cfg = json.load(open(cfg_path))
vals = cfg.get(key)
if vals is None:
    sys.exit(f"[CFG ERROR] '{key}' not found in {cfg_path}")
print("\n".join(vals))
PY
  )
  echo "${args[@]}"
}

# ---------------------------------------------------------------------------
# make_logging_cfg <dest_json> <run_log_path>
#   Creates a copy of BASE_LOG with level+filename patched in
# ---------------------------------------------------------------------------
make_logging_cfg() {
  local dest="$1" run_log="$2"
  sed \
    -e '15s#"level": *"INFO"#"level\": \"'"${LOGGING_LEVEL}"'"#' \
    -e '16s#"filename":.*#"filename\": \"'"${run_log}"'"#' \
    "$BASE_LOG" > "$dest"
}

# ---------------------------------------------------------------------------
# plot_results <csv_path>
#   Generates five standard plots for given CSV (stats & TBT)
# ---------------------------------------------------------------------------
plot_results() {
  local csv="$1"
  python "$PLOTTER" stats      "$csv"
  python "$PLOTTER" tbt_wc     "$csv"
  python "$PLOTTER" tbt        "$csv"
  python "$PLOTTER" tbt_err    "$csv"
  python "$PLOTTER" tbt_err_wc "$csv"
}

###############################################################################
# 3. MAIN LOOP                                                                 #
###############################################################################
for SLO in "${SLO_RATIO_LIST[@]}"; do
  echo -e "\n================ SLO_RATIO = ${SLO} ================"

  for EXP in "${EXP_LIST[@]}"; do
    EXP_OUT="${ROOT}/outputs/benchmark/${EXP}/slo${SLO}"
    mkdir -p "$EXP_OUT"

    for METHOD in "${METHOD_LIST[@]}"; do
      METHOD_OUT="${EXP_OUT}/${METHOD}"
      mkdir -p "$METHOD_OUT"

      # 3-a) CLI arguments for this METHOD
      IFS=$' ' read -r -a EXP_ARGS <<<"$(load_method_args "$METHOD")"

      for TRACE in "${TRACE_LIST[@]}"; do
        RUN_DIR="${METHOD_OUT}/${TRACE}"
        mkdir -p "$RUN_DIR"
        : > "${RUN_DIR}/vllm_msg.log"      # truncate per-run log

        # 3-b) generate per-run logging config
        NEW_LOG="${RUN_DIR}/logging_cfg.json"
        make_logging_cfg "$NEW_LOG" "${RUN_DIR}/vllm_msg.log"
        export VLLM_LOGGING_CONFIG_PATH="$NEW_LOG"

        # 3-c) pretty banner
        echo -e ">>> SLO=${SLO} | EXP=${EXP} | METHOD=${METHOD} | TRACE=${TRACE}"
        printf '    CLI: %q %q --slo-ratio %s\n' \
               "${EXP_ARGS[@]}" "${TRACE}" "$SLO"

        # 3-d) run vLLM (unless figure-only)
        if (( FIGURE_ONLY )); then
          echo "    ↳ FIGURE_ONLY=1 → skipping execution"
        else
          echo "    ↳ running..."
          python "${ROOT}/examples/test_distN.py" \
            --config-file "${TRACE_CFG_DIR}/${TRACE}.json" \
            "${EXP_ARGS[@]}" \
            --profiled-results $profiled_path \
            --slo-ratio "$SLO" \
            --output-log "${RUN_DIR}/outputs.log"
        fi

        # 3-e) post-processing (plots)
        CSV="${RUN_DIR}/outputs.csv"
        if [[ -f "$CSV" ]]; then
          echo "    ↳ plotting figures for $(basename "$CSV")"
          plot_results "$CSV"
        else
          echo "    ⚠️  $(basename "$CSV") not found – skipping plots" >&2
        fi
      done
    done
  done
done
