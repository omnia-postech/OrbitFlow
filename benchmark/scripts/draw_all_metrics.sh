#!/bin/bash

source /home/sychoy/anaconda3/etc/profile.d/conda.sh
conda activate vllm

export CUDA_VISIBLE_DEVICES=2
export VLLM_CONFIGURE_LOGGING=1

# 설정
ROOT="/home/sychoy/vllm"
OUTPUT_BASE="${ROOT}/outputs/benchmark"
PYTHON_SCRIPT="${ROOT}/benchmark/data_analysis/plots_per_metric.py"

# 실험 디렉토리
EXP_LIST=("NotEnough")

# trace list 하드코딩
TRACE_LIST=("test_trace1_10_not_enough")  # ← 여기에 trace 이름들 적기 (확장자 없이)

# (선택사항) 직접 method 지정 (안 하면 자동 감지)
# METHOD_LIST=("Static1" "Static2")

# TRACE별 루프
for TRACE in "${TRACE_LIST[@]}"; do
    echo "🔍 Processing trace: $TRACE"

    csv_paths=()
    method_labels=()

    for EXP in "${EXP_LIST[@]}"; do
        EXP_DIR="${OUTPUT_BASE}/${EXP}"

        # METHOD_LIST가 비어있으면 EXP_DIR 내부 디렉토리 탐색
        if [ ${#METHOD_LIST[@]} -eq 0 ]; then
            mapfile -t DETECTED_METHODS < <(find "$EXP_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort)
        else
            DETECTED_METHODS=("${METHOD_LIST[@]}")
        fi

        
        for METHOD in "${DETECTED_METHODS[@]}"; do
            csv_file="${EXP_DIR}/${METHOD}/${TRACE}/outputs.csv"
            if [ -f "$csv_file" ]; then
                csv_paths+=("$csv_file")
                method_labels+=("$METHOD")
            else
                echo "❌ Missing: $csv_file"
            fi
        done
    done

    if [ "${#csv_paths[@]}" -ge 2 ]; then
        output_image="${EXP_DIR}/${TRACE}.jpg"
        echo "✅ Generating comparison for $TRACE"
        echo "    ➤ CSVs: ${csv_paths[*]}"
        echo "    ➤ Labels: ${method_labels[*]}"
        echo "    ➤ Output: $output_image"
        python "$PYTHON_SCRIPT" --out "$output_image" --trace "$TRACE" "${csv_paths[@]}" 
    else
        echo "⚠️ Skipping $TRACE — need at least 2 output.csv files"
    fi
done
