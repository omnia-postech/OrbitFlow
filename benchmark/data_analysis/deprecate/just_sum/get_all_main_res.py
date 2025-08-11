import pandas as pd
from pathlib import Path

# 1. 기준 디렉토리
BASE_DIRs = [
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp_32k/slo1"),
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp_32k/slo1.5"),
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp_32k/slo2.5"),
]

# 2. 필요한 컬럼
required_columns = [
    "slo", "method", "arrival_rate", "cv_num",
    "tpot_attainment",
    "tbt_attainment_with_TD",
    "tbt_attainment_no_TD"
]

# 3. 분류 기준
no_td_methods = {"Flexgen", "Static1", "SelectN", "DistNSingle", "NextLayer"}

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


            def compute_tbt(row):
                if row["method"] in no_td_methods:
                    return row["tbt_attainment_no_TD"]
                elif row["method"].startswith("Ours"):
                    return row["tbt_attainment_with_TD"]
                else:
                    return pd.NA  # 처리하지 않는 method는 NaN

            df["tbt_attainment"] = df.apply(compute_tbt, axis=1)

            # 불필요 컬럼 제거
            df = df.drop(columns=["tbt_attainment_with_TD", "tbt_attainment_no_TD"])

            merged_rows.append(df)

        except Exception as e:
            print(f"[ERROR] Failed to process {csv_path}: {e}")

# 5. 병합 및 저장
if merged_rows:
    merged_df = pd.concat(merged_rows, ignore_index=True)
    merged_df.to_csv("merged_main_res.csv", index=False)
    print("✅ 최종 병합 완료 → merged_arrival_summary_with_tbt.csv")
else:
    print("⚠️ 병합할 데이터가 없습니다.")