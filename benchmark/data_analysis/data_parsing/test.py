import pandas as pd
from pathlib import Path

# 1. 파일 경로 설정 및 읽기
csv_path = Path("/home/heelim/vllm/benchmark/data_analysis/solver_summary_table_slo_total.csv")
if not csv_path.exists():
    raise FileNotFoundError(f"파일을 찾을 수 없습니다: {csv_path}")

df = pd.read_csv(csv_path)

# 2. 그룹화하기 전에 컬럼 이름이 긴 경우 편하게 다루도록 리네임(선택 사항)
df = df.rename(
    columns={
        "ratio_solver_cnt/total_decode": "solver_cnt_ratio",
        "ratio_sum_solver_time/mean_e2e_time": "solver_time_e2e_ratio",
        "get_ratio_solver_cnt_request": "solver_cnt_per_request"
    }
)

# 3. trace별로 평균 계산
grouped = df.groupby("trace", as_index=False).agg({
    "solver_cnt_ratio": "mean",
    "solver_time_e2e_ratio": "mean",
    "solver_cnt_per_request": "mean"
})

# 4. 결과 확인 (옵션)
print(grouped)

# 5. 결과를 CSV로 저장
out_path = Path("/home/heelim/vllm/benchmark/data_analysis/solver_summary_by_trace_merge.csv")
grouped.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"Trace별 평균값이 '{out_path}'에 저장되었습니다.")
