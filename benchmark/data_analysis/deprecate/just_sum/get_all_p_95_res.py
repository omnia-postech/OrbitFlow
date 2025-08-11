import pandas as pd
from pathlib import Path

# 1. 기준 디렉토리
BASE_DIRs = [
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp_32k/slo1"),
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp_32k/slo1.5"),
    # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp_32k/slo2"),
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp_32k/slo2.5"),
]

# 2. 필요한 컬럼
required_columns = [
    "slo", "method", "arrival_rate", "cv_num", "p95_ratio"
]

merged_rows = []

# 3. 파일 순회 및 필터링
for BASE_DIR in BASE_DIRs:
    for csv_path in BASE_DIR.rglob("arrival_summerizev2.csv"):
        try:
            df = pd.read_csv(csv_path)

            # 필요한 컬럼이 없으면 스킵
            if not all(col in df.columns for col in required_columns):
                print(f"[SKIP] Missing columns in {csv_path}")
                continue

            # 필요한 컬럼만 선택
            df = df[required_columns].copy()

            # 컬럼 이름 변경
            df = df.rename(columns={"p95_ratio": "p95_tail_latency"})

            merged_rows.append(df)

        except Exception as e:
            print(f"[ERROR] Failed to process {csv_path}: {e}")

# 4. 병합 및 저장
if merged_rows:
    merged_df = pd.concat(merged_rows, ignore_index=True)
    merged_df.to_csv("merged_p95_res.csv", index=False)
    print("✅ 병합 완료 → merged_main_res.csv")
else:
    print("⚠️ 병합할 데이터가 없습니다.")
