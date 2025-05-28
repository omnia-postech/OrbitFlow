#!/usr/bin/env bash
# set -euo pipefail

###############################################################################
# CONSTANTS (edit these lists only)                                           #
###############################################################################
export CUDA_VISIBLE_DEVICES=3
export VLLM_CONFIGURE_LOGGING=1

# LOGGING_LEVEL=DEBUG
LOGGING_LEVEL=CRITICAL

ROOT="/home/heelim/vllm"

FIGURE_ONLY=$1                      # 0 → 실행 + 그림, 1 → 그림만
EXP_LIST=(paper_main_exp)
METHOD_LIST=(NoPrefetch)
TRACE_CFG_DIR="${ROOT}/benchmark/selected_traces/"
# TRACE_LIST=(bim50_hi_ov78_scaled_debugging)
# TRACE_LIST=(batch_dyn_low batch_dyn_mid batch_dyn_high batch_dyn_veryhigh)
# TRACE_LIST=(token_dyn_low token_dyn_mid token_dyn_high token_dyn_veryhigh)
TRACE_LIST=(both_dyn_low both_dyn_mid both_dyn_high both_dyn_veryhigh)

METHOD_CFG_FILE="${ROOT}/benchmark/scripts/supported_methods.json"
BASE_LOG="${ROOT}/configs/test_no_prefetch_logging.json"
PLOTTER="${ROOT}/benchmark/data_analysis/metrics_plot.py"

# ★ 실험할 slo_ratio 값들만 여기에 나열하면 됩니다.
# SLO_RATIO_LIST=(1.5 2.0 2.5 3.0)
SLO_RATIO_LIST=(2.5)
###############################################################################
# MAIN LOOP: SLO ➔ EXP ➔ METHOD ➔ TRACE                                      #
###############################################################################
for SLO in "${SLO_RATIO_LIST[@]}"; do
  echo -e "\n========== SLO_RATIO = ${SLO} =========="

  for EXP in "${EXP_LIST[@]}"; do
    EXP_OUT="${ROOT}/outputs/benchmark/${EXP}/slo${SLO}"
    mkdir -p "${EXP_OUT}"

    for METHOD in "${METHOD_LIST[@]}"; do
      METHOD_OUT="${EXP_OUT}/${METHOD}"
      mkdir -p "${METHOD_OUT}"

      # ── 1) JSON → EXP_ARGS 배열로 읽어오기 ────────────────────────────────
      if ! mapfile -t EXP_ARGS < <(
          python - "$METHOD_CFG_FILE" "$METHOD" <<'PY'
import json, sys
cfg_path, key = sys.argv[1:]
cfg = json.load(open(cfg_path))
vals = cfg.get(key)
if vals is None:
    print(f"[CFG ERROR] {key} not found", file=sys.stderr); sys.exit(1)
print("\n".join(vals))
PY
      ); then
          echo "Aborting: failed to obtain CLI args for METHOD=${METHOD}" >&2
          exit 1
      fi

      # ── 2) SLO별 반복: TRACE마다 실행 ────────────────────────────────────
      for TRACE in "${TRACE_LIST[@]}"; do
        RUN_DIR="${METHOD_OUT}/${TRACE}"
        mkdir -p "${RUN_DIR}"
        : > "${RUN_DIR}/vllm_msg.log"          # fresh per-run log

        # ---------- logging JSON **per TRACE** ------------------------------
        NEW_LOG="${RUN_DIR}/logging_cfg.json"
        sed -e '15s#"level": *"INFO"#"level\": \"'"${LOGGING_LEVEL}"'"#' \
            -e '16s#"filename":.*#"filename\": \"'"${RUN_DIR}/vllm_msg.log"'"#' \
            "$BASE_LOG" > "$NEW_LOG"
        export VLLM_LOGGING_CONFIG_PATH="$NEW_LOG"

        echo ">>> SLO=${SLO}  EXP=${EXP}  METHOD=${METHOD}  TRACE=${TRACE}"
        printf 'CLI for %s:\n  ' "$METHOD"
        printf '%q ' "${EXP_ARGS[@]}"; echo -n " --slo-ratio $SLO"; echo

        if ((FIGURE_ONLY)); then
          echo "  ↳ Skipping execution (FIGURE_ONLY)"
        else
          echo "  ↳ Running ${METHOD} on ${TRACE} (slo_ratio=${SLO})"
          python "${ROOT}/examples/test_distN.py" \
              --config-file "${TRACE_CFG_DIR}/${TRACE}.json" \
              "${EXP_ARGS[@]}" \
              --slo-ratio "$SLO" \
              --output-log "${RUN_DIR}/outputs.log"
        fi

        CSV="${RUN_DIR}/outputs.csv"
        if [[ -f "$CSV" ]]; then
            echo "  ↳ Plotting stats and TBT for $(basename "$CSV")"
            python "$PLOTTER" stats      "$CSV"
            python "$PLOTTER" tbt_wc     "$CSV"
            python "$PLOTTER" tbt        "$CSV"
            python "$PLOTTER" tbt_err    "$CSV"
            python "$PLOTTER" tbt_err_wc "$CSV"
        else
            echo "  [WARN] $CSV not found -- skipping plots" >&2
        fi
      done
    done
  done
done
