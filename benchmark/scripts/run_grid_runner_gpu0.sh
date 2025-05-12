#!/usr/bin/env bash
# set -euo pipefail

###############################################################################
# CONSTANTS (edit these lists only)                                           #
###############################################################################
export CUDA_VISIBLE_DEVICES=$1
export VLLM_CONFIGURE_LOGGING=1
LOGGING_LEVEL=CRITICAL
ROOT="/home/heelim/vllm"

FIGURE_ONLY=0
EXP_LIST=(Debug)                         # ← your “experiments”
shift
METHOD_LIST=("$@")      # ← indexes into JSON above

TRACE_CFG_DIR="${ROOT}/benchmark/test_traces/test_best_worst"
TRACE_LIST=(new_test)

METHOD_CFG_FILE="${ROOT}/benchmark/scripts/supported_methods.json"
BASE_LOG="${ROOT}/configs/test_no_prefetch_logging.json"
PLOTTER="${ROOT}/benchmark/data_analysis/metrics_plot.py"
###############################################################################
# MAIN LOOP: EXP ➔ METHOD ➔ TRACE                                             #
###############################################################################
for EXP in "${EXP_LIST[@]}"; do
  EXP_OUT="${ROOT}/outputs/benchmark/${EXP}"
  mkdir -p "${EXP_OUT}"

  for METHOD in "${METHOD_LIST[@]}"; do
    METHOD_OUT="${EXP_OUT}/${METHOD}"
    mkdir -p "${METHOD_OUT}"

    if ! mapfile -t EXP_ARGS < <(
        python - "$METHOD_CFG_FILE" "$METHOD" <<'PY'
import json, sys, os
cfg_path, key = sys.argv[1:]
try:
    cfg = json.load(open(cfg_path))
except Exception as e:
    print(f"[JSON ERROR] {e}", file=sys.stderr); sys.exit(1)
vals = cfg.get(key)
if vals is None:
    print(f"[CFG ERROR] Method {key!r} not found in {cfg_path}", file=sys.stderr)
    sys.exit(1)
print("\n".join(vals))
PY
    ); then
        echo "Aborting: failed to obtain CLI args for METHOD=${METHOD}" >&2
        exit 1
    fi

    for TRACE in "${TRACE_LIST[@]}"; do
      RUN_DIR="${METHOD_OUT}/${TRACE}"
      mkdir -p "${RUN_DIR}"
      : > "${RUN_DIR}/vllm_msg.log"            # fresh per-run log

      # -------- logging JSON **per TRACE** -----------------------------------
      NEW_LOG="${RUN_DIR}/logging_cfg.json"
      sed  -e '15s#"level": *"INFO"#"level\": \"'"${LOGGING_LEVEL}"'"#' \
           -e '16s#"filename":.*#"filename\": \"'"${RUN_DIR}/vllm_msg.log"'"#' \
           "$BASE_LOG" > "$NEW_LOG"
      export VLLM_LOGGING_CONFIG_PATH="$NEW_LOG"

      echo ">>> EXP=${EXP}  METHOD=${METHOD}  TRACE=${TRACE}"
      # ---- DEBUG: show the CLI we’re about to use -----------------------------
      printf 'EXP_ARGS for METHOD=%s:\n  ' "$METHOD"
      printf '%q ' "${EXP_ARGS[@]}"; echo            # newline

        ### execution
      if ((FIGURE_ONLY)); then
        echo "  ↳ Skipping execution (FIGURE_ONLY)"
      else
        echo "  ↳ Running ${METHOD} on ${TRACE}"
        python "${ROOT}/examples/test_distN.py" \
            --config-file "${TRACE_CFG_DIR}/${TRACE}.json" \
            "${EXP_ARGS[@]}" \
            --output-log "${RUN_DIR}/outputs.log"
        ### execution
      fi 
      CSV="${RUN_DIR}/outputs.csv"
      if [[ -f "$CSV" ]]; then         # sanity-check: file must exist
          echo "  ↳ Plotting stats and TBT for $(basename "$CSV")"
          python "$PLOTTER" stats "$CSV"
          python "$PLOTTER" tbt_wc  "$CSV"
      else
          echo "  [WARN] $CSV not found -- skipping plots" >&2
      fi

    done
  done
done