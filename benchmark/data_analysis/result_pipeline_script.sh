#!/usr/bin/env bash

# base_paths: Support for multiple experimental routes
base_paths=(
  ../../outputs/benchmark/paper_main_exp_32k
)
reference_root="../selected_traces/lambda2.0x_cv1.json"

# ───────────────────────────────────────────────
# Step 1. Change SLO scale Only For Baselines
# ───────────────────────────────────────────────
echo "📐 Recalculating SLO scale for baselines..."
for base_path in "${base_paths[@]}"; do
  python ./data_analysis/data_parsing/slo_data_rescaler.py \
    --old-sc 2.5 \
    --new-sc-list 1 1.5 2 \
    --base-path "${base_path}" \
    --is-arrival \
    --arrival-rate-list 1 1.5 2 \
    --cv-list 2 \
    --methods DistNSingle Flexgen \
    --arrival-tpl "lambda{rate}x_cv{cv}"
done

# ───────────────────────────────────────────────
# Step 2. Collect all method/trace directories
# ───────────────────────────────────────────────
subdirs=(
  "slo1/Ours"
  "slo1.5/Ours"
  "slo2.5/Ours"

  "slo1/DistNSingle"
  "slo1.5/DistNSingle"
  "slo2.5/DistNSingle"

  "slo1/Flexgen"
  "slo1.5/Flexgen"
  "slo2.5/Flexgen"
)

declare -a all_roots all_subdirs

for base_path in "${base_paths[@]}"; do
  for rel in "${subdirs[@]}"; do
    dir="$base_path/$rel"
    if [ -d "$dir" ]; then
      all_roots+=("$dir")
    else
      echo "⚠️ Directory not found: $dir" >&2
    fi
  done
done

for root in "${all_roots[@]}"; do
  for sub in "$root"/*; do
    [ -d "$sub" ] || continue
    all_subdirs+=("$sub")
  done
done

# ───────────────────────────────────────────────
# Step 2-1. Calculate SLO Violation and Arrival Rate
# ───────────────────────────────────────────────
echo "📊 Simulating SLO violation per token..."
python ./data_analysis/data_parsing/calculate_simulated_slo_violations.py "${all_subdirs[@]}" --reference-root "${reference_root}"
if [ $? -ne 0 ]; then
    echo "❌ calculate_simulated_slo_violations.py 실행 실패"
fi

echo "📈 Generating arrival-based summary CSV..."
python ./data_analysis/data_parsing/generate_slo_summary_csv.py "${all_roots[@]}" --reference-root "${reference_root}"
if [ $? -ne 0 ]; then
    echo "❌ generate_slo_summary_csv.py 실행 실패"
    exit 1
fi

# ───────────────────────────────────────────────
# Step 2-2. Compute memory utilization
# ───────────────────────────────────────────────
echo "💾 Computing memory utilization from logs..."
python ./data_analysis/data_parsing/compute_memory_util_from_log.py "${all_subdirs[@]}"
if [ $? -ne 0 ]; then
    echo "❌ compute_memory_util_from_log.py 실행 실패"
fi