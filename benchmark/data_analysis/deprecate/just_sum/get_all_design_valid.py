import pandas as pd
from pathlib import Path

# 1. 기준 디렉토리
BASE_DIRs = [
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp_design_validation"),
]

# 2. 필요한 컬럼
required_columns = [
    "slo", "method", "arrival_rate", "cv_num",
    "tbt_attainment_with_TD",
    "tbt_attainment_no_TD"
]

# 3. 분류 기준
method_list = [
    "UniformSolver", "UniformSolver_TD",
    "UniformSolver_TD_PR", "Ours"
]

merged_rows = []

# 4. 파일 순회
for BASE_DIR in BASE_DIRs:
    for csv_path in BASE_DIR.rglob("arrival_summerizev2.csv"):
        try:
            df = pd.read_csv(csv_path)

            if not all(col in df.columns for col in required_columns):
                print(f"[SKIP] Missing columns in {csv_path}")
                continue

            df = df[required_columns].copy()

            # 필요한 method만 필터링
            df = df[df["method"].isin(method_list)]

            # TBT 계산
            def compute_tbt(row):
                if row["method"] == "UniformSolver":
                    return row["tbt_attainment_no_TD"]
                else:  # 모든 나머지(TD 포함) 케이스
                    return row["tbt_attainment_with_TD"]

            df["tbt_attainment"] = df.apply(compute_tbt, axis=1)

            # 불필요 컬럼 제거
            df = df.drop(columns=["tbt_attainment_with_TD", "tbt_attainment_no_TD"])

            # 빈 or 전부 NA인 DataFrame 제거
            if not df.empty and not df.isna().all().all():
                merged_rows.append(df)

        except Exception as e:
            print(f"[ERROR] Failed to process {csv_path}: {e}")

# 5. 병합 및 저장
if merged_rows:
    merged_df = pd.concat(merged_rows, ignore_index=True)
    output_path = "merged_design_valid.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"✅ 최종 병합 완료 → {output_path}")
else:
    print("⚠️ 병합할 데이터가 없습니다.")
