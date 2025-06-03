import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. 설정: 그래프에서 사용했던 slo_scales, trace_list, metric_list, method_list
# ─────────────────────────────────────────────────────────────────────────────
slo_scales = [10, 4.5, 3.5, 2.5, 1.5, 1]
trace_list = ["both_static", "token_dyn", "both_dyn"]
metric_list = ["low", "mid", "high", "veryhigh"]
method_list = ["Flexgen", "SelectN", "Ours"]

# ─────────────────────────────────────────────────────────────────────────────
# 2. 빈 리스트를 만들어서 각 조합에 대한 결과를 기록
# ─────────────────────────────────────────────────────────────────────────────
records = []

for sc in slo_scales:
    for method in method_list:
        # summary CSV 파일 경로 (slo{sc} 폴더 아래)
        summary_path = Path(
            f"/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo{sc}/{method}/summerize.csv"
        )
        # summary파일이 없으면 모든 조합에 대해 NaN을 기록
        if not summary_path.exists():
            for trace in trace_list:
                for metric in metric_list:
                    records.append({
                        "slo": sc,
                        "trace": trace,
                        "metric": metric,
                        "method": method,
                        "tpot": np.nan,
                        "tbt": np.nan
                    })
            continue

        # summary 파일을 불러온 뒤, slo와 method 컬럼은 이미 동일하므로 trace/metric만 필터링
        summary_df = pd.read_csv(summary_path)

        for trace in trace_list:
            for metric in metric_list:
                # 해당 조합이 있는지 확인
                row = summary_df[
                    (summary_df["slo"] == sc) &
                    (summary_df["method"] == method) &
                    (summary_df["trace"] == trace) &
                    (summary_df["metric"] == metric)
                ]
                if len(row) == 1:
                    tpot_val = float(row["tpot_attainment"].iloc[0])
                    tbt_val  = float(row["tbt_attainment"].iloc[0])
                else:
                    tpot_val = np.nan
                    tbt_val  = np.nan

                records.append({
                    "slo": sc,
                    "trace": trace,
                    "metric": metric,
                    "method": method,
                    "tpot": tpot_val,
                    "tbt": tbt_val
                })

# ─────────────────────────────────────────────────────────────────────────────
# 3. pandas DataFrame으로 변환 후, 필요 시 CSV로 저장
# ─────────────────────────────────────────────────────────────────────────────
df_out = pd.DataFrame(records)

# 결과 예시 출력
# print(df_out.head(12))  # 상위 12개 행 확인

# 필요하면 CSV로 저장
out_csv = Path("solver_tpot_tbt_data.csv")
df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"데이터가 '{out_csv}'에 저장되었습니다.")
