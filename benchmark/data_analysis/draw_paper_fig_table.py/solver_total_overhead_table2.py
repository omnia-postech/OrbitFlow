import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import ast
import ast, json, logging

# ─────────────────────────────────────────────────────────────────────────────
# 0. 사용자 정의: 처리할 SLO와 Trace/Metric 이름 리스트
#    - slo_list: slo 폴더 이름들 (예: ["slo1", "slo2.5", ...])
#    - trace_metric_list: 각 trace_metric은 "trace_metric" 식으로 입력
#      (예: ["both_dyn_veryhigh", "token_dyn_veryhigh", "batch_dyn_high", ...])
#
#  예시:
slo_list = [1, 1.5, 2, 2.5, 3, 3.5, 4.5, 10]  # 실제 사용하려는 slo 폴더들로 교체하세요
trace_list = [
    "both_static",
    "batch_dyn",
    "token_dyn",
    "both_dyn"
    # 필요에 따라 추가
]
metric_list =[
    "low",
    "mid",
    "high",
    "veryhigh"
]
#
#  기타:
#    - method_name: 각 slo/trace 아래 “Ours” 하위 폴더를 대상으로 처리한다고 가정
#      (필요에 따라 "NoPrefetch", "Flexgen" 등으로 확장 가능합니다)
# ─────────────────────────────────────────────────────────────────────────────

def get_ratio_solver_cnt_request(solver_df, req_len) -> float:
    """
    solver_summerize.csv 데이터프레임(solver_df)에서
    trace 컬럼이 trace_name이고 metric 컬럼이 metric인 행을 찾아
    solver_cnt / total_decode_cnt * 100 값을 반환합니다.
    """
    mask = (solver_df["trace"] == trace_name) & (solver_df["metric"] == metric)
    if mask.sum() != 1:
        # 결과값을 없음을 표시하거나 0으로 처리할 수 있습니다.
        return 0.0
    row = solver_df.loc[mask].iloc[0]
    solver_cnt = float(row["solver_cnt"])

    return solver_cnt / req_len

def get_ratio_solver_cnt_total_decode(solver_df) -> float:
    """
    solver_summerize.csv 데이터프레임(solver_df)에서
    trace 컬럼이 trace_name이고 metric 컬럼이 metric인 행을 찾아
    solver_cnt / total_decode_cnt * 100 값을 반환합니다.
    """
    mask = (solver_df["trace"] == trace_name) & (solver_df["metric"] == metric)
    if mask.sum() != 1:
        # 결과값을 없음을 표시하거나 0으로 처리할 수 있습니다.
        return 0.0
    row = solver_df.loc[mask].iloc[0]
    solver_cnt = float(row["solver_cnt"])
    total_decode_cnt = float(row["total_decode_cnt"])
    return (solver_cnt / total_decode_cnt) * 100

def get_ratio_mean_solver_time_mean_tbt(output_df) -> float:
    """
    outputs.csv 데이터프레임(output_df)에서
    - solver_time 리스트 요소가 0인 것은 제외 후
    - 같은 인덱스의 time_between_tokens 를 짝지어
    solver_time이 0이 아닌 부분만 모아 각각 평균을 계산하여
    mean(solver_time_nonzero) / mean(tbt_corresponding) * 100 반환
    """
    solver_vals = []
    tbt_vals = []

    for _, row in output_df.iterrows():
        tbt_list = row.get("time_between_tokens", [])
        solver_time_list = row.get("solver_time", [])

        # 문자열로 저장된 경우 이미 파싱되었으므로, 이 예시에서는 그대로 리스트라고 가정
        # if not isinstance(tbt_list, (list, np.ndarray)) or not isinstance(solver_time_list, (list, np.ndarray)):
        #     continue

        # 같은 인덱스끼리 짝지어, solver_time != 0인 값만 모음
        for st, tbt in zip(solver_time_list, tbt_list):
            if st != 0:
                solver_vals.append(st)
                tbt_vals.append(tbt)

    if not solver_vals or not tbt_vals:
        return 0.0

    mean_solver_nonzero = np.mean(solver_vals)
    mean_tbt_corresponding = np.mean(tbt_vals)
    res = (mean_solver_nonzero / mean_tbt_corresponding) * 100
    # print(res)
    return res

def get_ratio_sum_solver_time_mean_e2e(output_df) -> float:
    """
    outputs.csv 데이터프레임(output_df)에서
    각 행마다 sum(solver_time_list) / end_to_end_time 을 계산한 뒤,
    그 전체 평균 * 100을 반환
    """
    per_row_ratio = []
    if "end_to_end_time" not in output_df.columns:
        return 0.0

    for _, row in output_df.iterrows():
        solver_time_list = row.get("solver_time", [])
        if not isinstance(solver_time_list, (list, np.ndarray)):
            continue

        # 만약 end_to_end_time이 NaN 또는 0이면 건너뛰기
        e2e = row.get("end_to_end_time", np.nan)
        try:
            e2e = float(e2e)
        except:
            continue
        if e2e == 0 or np.isnan(e2e):
            continue

        per_row_ratio.append(sum(solver_time_list) / e2e)

    if not per_row_ratio:
        return 0.0

    return np.mean(per_row_ratio) * 100

def get_ratio_estimated_minus_between(output_df) -> float:
    """
    outputs.csv 데이터프레임(output_df)에서
    solver_estimated_time == 100인 값과 solver_estimated_time == 0인 값은 모두 건너뛴 뒤,
    같은 인덱스짜리 time_between_tokens 와 짝지어
    abs(tbt - estimate) / tbt 비율을 구해서 중간값 * 100을 반환
    """
    ratios = []
    for _, row in output_df.iterrows():
        tbt_list = row.get("time_between_tokens", [])
        estimate_list = row.get("solver_estimated_time", [])

        if not isinstance(tbt_list, (list, np.ndarray)) or not isinstance(estimate_list, (list, np.ndarray)):
            continue

        for tbt, est in zip(tbt_list, estimate_list):
            # solver_estimated_time이 100 또는 0인 것은 스킵, tbt가 0이면 계산 불가
            if est in (0, 100) or tbt == 0:
                continue
            ratios.append(abs(tbt - est) / tbt)

    if not ratios:
        return 0.0

    return np.median(ratios) * 100

# 결과를 저장할 리스트
total = []
for trace_name in trace_list:
    results = []
    val1_list = []
    val2_list = []
    val3_list = []
    for slo in slo_list:
        for metric in metric_list:
            # ─────────────────────────────────────────────────────────────
            # 1. base_path 구성
            #    경로 예시: /home/heelim/vllm/outputs/benchmark/paper_main_exp/{slo}/{method_name}/{trace_metric}
            # ─────────────────────────────────────────────────────────────
            base_path = Path(f"/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo{slo}/Ours/{trace_name}_{metric}")
            if not base_path.exists():
                print(f"[경고] base_path가 존재하지 않습니다: {base_path}")
                continue

            print(f"Start {base_path}")

            # ─────────────────────────────────────────────────────────────
            # 2. outputs.csv 읽기 및 전처리
            # ─────────────────────────────────────────────────────────────
            output_path = base_path / "outputs.csv"
            if not output_path.exists():
                print(f"[경고] outputs.csv 가 없습니다: {output_path}")
                continue

            output_df = pd.read_csv(output_path)

            # 숫자로 변환할 컬럼들: num_cols
            num_cols = [
                "arrival_time", "first_scheduled_time", "finished_time",
                "time_to_first_token", "slo_threshold", "slo_violations",
                "stall_duration", "decode_length", "end_to_end_time",
                "decode_time", "time_per_output_token"
            ]
            for col in num_cols:
                if col in output_df.columns:
                    output_df[col] = pd.to_numeric(output_df[col], errors="coerce")

            # 문자열 형태 리스트로 저장된 컬럼들: ast.literal_eval
            for col in ("time_between_tokens", "solver_time", "solver_estimated_time"):
                if col in output_df.columns:
                    output_df[col] = output_df[col].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                    )

            # ─────────────────────────────────────────────────────────────
            # 3. solver_summerize.csv 읽기
            # ─────────────────────────────────────────────────────────────
            solver_path = base_path.parent / "solver_summerize.csv"
            if not solver_path.exists():
                print(f"[경고] solver_summerize.csv 가 없습니다: {solver_path}")
                continue

            solver_df = pd.read_csv(solver_path)

            ref_json = Path("/home/heelim/vllm/benchmark/selected_traces") / f"{trace_name}_{metric}.json"
            with open(ref_json, "r") as f:
                reference = json.load(f)
            ref_reqs = reference.get("requests", {})

            # total_decode = 0
            # for id in range(len(ref_reqs)):
            #     total_decode += ref_reqs[f"request_{id}"]["output_length"]


            # ─────────────────────────────────────────────────────────────
            # 5. 각 비율을 계산하여 결과 목록에 추가
            # ─────────────────────────────────────────────────────────────
            val1_list.append(get_ratio_mean_solver_time_mean_tbt(output_df))
            val2_list.append(get_ratio_sum_solver_time_mean_e2e(output_df))
            val3_list.append(get_ratio_solver_cnt_request(solver_df, len(ref_reqs)))

    results.append({
        "trace": trace_name,
        "ratio_mean_solver/time_mean_tbt": np.mean(val1_list),
        "ratio_sum_solver_time/mean_e2e_time": np.mean(val2_list),
        "get_ratio_solver_cnt_request": np.mean(val3_list)
    })

    # ─────────────────────────────────────────────────────────────────────────────
    # 6. 결과를 Pandas DataFrame으로 변환 후 CSV로 저장
    # ─────────────────────────────────────────────────────────────────────────────
    if results:
        result_df = pd.DataFrame(results)
        out_csv = Path(f"solver_summary_table_slo{slo}.csv")
        # result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"결과가 '{out_csv}'에 저장되었습니다.")
        total.extend(results)
    else:
        print("저장할 결과가 없습니다. 처리 대상이 없었거나 경로 오류를 확인하세요.")

if total:
    result_df = pd.DataFrame(total)
    out_csv = Path(f"solver_summary_table_slo_total.csv")
    result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"결과가 '{out_csv}'에 저장되었습니다.")
else:
    print("저장할 결과가 없습니다. 처리 대상이 없었거나 경로 오류를 확인하세요.")