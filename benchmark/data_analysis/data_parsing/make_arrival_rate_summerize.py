import pandas as pd
import numpy as np
import ast
from pathlib import Path
from typing import List
import sys
import ast, json, logging
import re


# ───────────────────────────────────────────────
# 1. load_metrics 정의 (기존 함수 활용)
def load_output_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    num_cols = [
        "arrival_time", "first_scheduled_time", "finished_time",
        "time_to_first_token", "slo_threshold", "slo_violations",
        "stall_duration", "decode_length", "end_to_end_time",
        "decode_time", "time_per_output_token"
    ]
    for col in num_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("time_between_tokens", "stall_times", "stall_durations"):
        if col in df:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
    return df

# ───────────────────────────────────────────────
def _extract_request_tpot_attainment(
        output_df: pd.DataFrame, 
        slo_df: pd.DataFrame, 
        total_decode: int,
        penalty_tbt: float = 1000.0
    ) -> float:
    # 1) 실패한 request_id 집합 만들기  (slo_violation.csv 기준)
    if {"failed", "request_id"}.issubset(slo_df.columns):
        failed_reqs = set(slo_df.loc[slo_df["failed"], "request_id"])
    else:
        failed_reqs = set()

    # 2) 요청별 TPOT 배열 생성
    tpot_list = []
    tpot_median_list = []
    req_ids   = (
        slo_df["request_id"]
        if "request_id" in slo_df.columns
        else [f"request_{i}" for i in range(len(slo_df))]
    )

    for rid in req_ids:
        if rid in failed_reqs:
            tpot_list.append(penalty_tbt) 
        else :
            vals = output_df.loc[output_df["request_id"] == rid, "time_between_tokens"]
            tpot_list.append(np.mean(vals.iloc[0]))
            tpot_median_list.append(np.median(vals.iloc[0]))


    # for rid, tbt_series in zip(req_ids, output_df["time_between_tokens"]):
    #     if rid in failed_reqs:
    #         tpot_list.append(penalty_tbt)          # 실패 → 패널티
    #     else:
    #         tpot_list.append(np.mean(tbt_series))  # 정상 → 실제 TPOT

    tpot = np.asarray(tpot_list, dtype=float)

    # 3) SLO 임계값(요청별 값이 모두 동일하면 평균으로 충분)
    thr = pd.to_numeric(output_df["slo_threshold"],
                        errors="coerce").to_numpy().mean()

    valid_mask = ~np.isnan(tpot)          # 실패 요청 포함
    attain_cnt = (tpot[valid_mask] <= thr).sum()

    tpot_mean_res = (attain_cnt / valid_mask.sum() * 100
            if valid_mask.sum() else 0.0)

    # ---------------------------------------------------

    tpot_median = np.asarray(tpot_median_list, dtype=float)

    valid_mask_median = ~np.isnan(tpot_median)          # 실패 요청 포함
    attain_cnt_median = (tpot_median[valid_mask_median] <= thr).sum()

    tpot_median_res = (attain_cnt_median / valid_mask_median.sum() * 100
            if valid_mask_median.sum() else 0.0)

    return tpot_mean_res, tpot_median_res

def _extract_token_slo_attainment(
        output_df: pd.DataFrame, 
        slo_df: pd.DataFrame,
        total_decoded: int
    ) -> float:
    # total_decoded = int(output_df.get("decode_length", pd.Series(0)).sum())
    viol = int(slo_df.get("slo_violation_with_TD", pd.Series(0)).sum())
    return ((total_decoded - viol) / total_decoded * 100 if total_decoded else 0.0)

def tbt_no_Td(
        output_df: pd.DataFrame, 
        slo_df: pd.DataFrame,
        total_decoded: int
    ) -> float:
    # total_decoded = int(output_df.get("decode_length", pd.Series(0)).sum())
    viol = int(slo_df.get("slo_violation_no_TD", pd.Series(0)).sum())
    return ((total_decoded - viol) / total_decoded * 100 if total_decoded else 0.0)

def compute_throughput(
        df: pd.DataFrame,
        # total_decode: int
    ) -> float:
    # 총 토큰 수 = input_length(존재 시) + decode_length
    total_input = df.get("input_length", pd.Series(0)).sum()
    total_decode = df["decode_length"].sum()
    total_tokens = total_input + total_decode

    # 전체 wall-clock 시간
    wall_time = df["finished_time"].max()

    # throughput = 총 토큰 수 / 전체 시간
    return total_tokens / wall_time if wall_time and total_tokens else 0.0

def _extract_percentile_ratio_lists(
    df: pd.DataFrame, percentiles: List[int], total_decode:int
) -> List[List[float]]:
    """
    For each percentile, return a list of Pxx/SLO ratios (one per request).
    
    Request‐wise: use (expected_output_length − 1) as the index into that request’s
    sorted time_between_tokens list; if out‐of‐range, append np.nan.
    
    System‐wide: sum all (expected_output_length − 1) across requests to get a total
    index length, then index into the flattened, sorted union of all time_between_tokens.
    """
    import numpy as np
    
    lists: List[List[float]] = [[] for _ in percentiles]
    # need expected_output_length column as well
    if {"time_between_tokens", "slo_threshold", "expected_output_length"}.issubset(df.columns):
        # parse thresholds
        thr_vals = pd.to_numeric(df["slo_threshold"], errors="coerce").to_numpy()
        
        flat = []
        total_intervals = 0
        for tbt_list, exp_len in zip(df["time_between_tokens"], df["expected_output_length"]):
            if not tbt_list or exp_len is None:
                continue
            flat.extend(tbt_list)
            total_intervals += int(exp_len) - 1
        if total_intervals <= 0:
            return lists
        sorted_flat = sorted(flat)
        thr = thr_vals[0]
        for idx, pct in enumerate(percentiles):
            pos = int(np.floor(pct/100 * total_decode))
            if 0 <= pos < len(sorted_flat):
                px = sorted_flat[pos]
                lists[idx] = [px / thr]
                if px == 100:
                    lists[idx] = [np.nan]
                # print(f"{px}, {thr}, {lists[idx]}")
            else:
                lists[idx] = [np.nan]
    return lists

def extract_metrics(input_base_path, total_decode):
    output_df = load_output_metrics(Path(input_base_path, "outputs.csv"))
    slo_df = pd.read_csv(Path(input_base_path, "slo_violation.csv"))

    tpot_slo, tpot_median_slo = _extract_request_tpot_attainment(output_df, slo_df, total_decode)
    tbt_slo_with_TD = _extract_token_slo_attainment(output_df, slo_df, total_decode)
    tbt_slo_no_TD = tbt_no_Td(output_df, slo_df, total_decode)
    throughput = compute_throughput(output_df)

    slo_thr = pd.to_numeric(output_df["slo_threshold"], errors="coerce").to_numpy().mean()

    pct_lists = _extract_percentile_ratio_lists(output_df, [90, 95, 99], total_decode)
    ratio_means = [np.mean(lst) if lst else 0.0 for lst in pct_lists]
    p90_ratio, p95_ratio, p99_ratio = ratio_means

    return tpot_slo, tbt_slo_with_TD, throughput, slo_thr, p90_ratio, p95_ratio, p99_ratio, tpot_median_slo


REFERENCE_ROOT = Path("/home/heelim/vllm/benchmark/selected_traces")
pattern = re.compile(r"^lambda(?P<rate>\d+(?:\.\d+)?)x_cv(?P<cv_num>\d+)$")


def make_summerize(input_paths: list):
    results = []
    for path in input_paths:
        slo = path.parts[-3].replace('slo', '')
        method = path.parts[-2]

        match = pattern.match(path.name)

        rate = float(match.group("rate"))
        cv_num = int(match.group("cv_num"))

        ref_json = REFERENCE_ROOT / f"{path.name}.json"

        if not ref_json.exists():
            # log.warning(f"reference JSON missing: {ref_json}")
            return

        with open(ref_json, "r") as f:
            reference = json.load(f)
        ref_reqs = reference.get("requests", {})

        total_decode = 0
        for id in range(len(ref_reqs)):
            total_decode += ref_reqs[f"request_{id}"]["output_length"]

        # print(f"{trace} {metric}")

        # extract_metrics(path) -> (tpot_slo, tbt_slo, throughput, slo_thr, p90, p95, p99)
        tpot_slo, tbt_slo, throughput, slo_thr, p90_ratio, p95_ratio, p99_ratio, tpot_median_slo = extract_metrics(path, total_decode)
        results.append({
            'slo': slo,
            'method': method,
            'arrival_rate': rate,
            'cv_num': cv_num,
            'tpot_attainment': tpot_slo,
            'tbt_attainment': tbt_slo,
            'throughput_tokens_per_sec': throughput,
            'slo_threshold_mean': slo_thr,
            'p90_ratio': p90_ratio,
            'p95_ratio': p95_ratio,
            'p99_ratio': p99_ratio,
            'tpot_median_slo': tpot_median_slo,
        })

    # DataFrame 생성 및 저장
    df_result = pd.DataFrame(results)
    df_result.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    for i in input_paths:
        print(f"{i}")
    print(f"Aggregated CSV saved to {OUTPUT_CSV}")
    print()


import pandas as pd
from pathlib import Path

# ──────────────────────────────────────────────────────
# 설정: paper_main_exp 아래 모든 slo/<scale>/<method>/<trace_metric> 경로 자동 수집
BASE_DIR = Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp")

input_base_paths = [
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1/Ours"),
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1.5/Ours"),
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/Ours"),
    # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/Ours"),
    # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/Ours"),
    # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1.5/Ours")
    # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1/NoPrefetch")
]

if len(sys.argv) > 1:
    input_base_paths = [Path(p) for p in sys.argv[1:] if Path(p).exists()]

if len(input_base_paths) > 0:
    
    for method_dir in input_base_paths:
        input_paths = []    
        OUTPUT_CSV = method_dir / "arrival_summerize.csv"

        # 각 trace_metric 디렉터리
        for trace_dir in sorted(method_dir.iterdir(), key=lambda p: p.name):
            if not trace_dir.is_dir():
                continue

            if not pattern.match(trace_dir.name):
                continue

            print(f"start: {trace_dir}")

            outputs_csv = trace_dir / 'outputs.csv'
            slo_violation = trace_dir / 'slo_violation.csv'
            if outputs_csv.exists() and slo_violation.exists():
                input_paths.append(trace_dir)
        
        make_summerize(input_paths)
    
    sys.exit()  # 프로그램을 완전히 종료

# ──────────────────────────────────────────────────────
# input_paths: slo 디렉터리 하위의 method별 trace_metric 디렉터리 필터링
    
print("Run All Start")
print()
for slo_dir in BASE_DIR.glob('slo*'):
    if not slo_dir.is_dir():
        continue
    for method_dir in slo_dir.iterdir():
        input_paths = []    
        OUTPUT_CSV = method_dir / "arrival_summerize.csv"

        if not method_dir.is_dir():
            continue

        skip_paths = [
        # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/both_dyn_mid"),
        # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/token_dyn_low"),
        # 추가하고 싶은 경로들...
        ]
        # 각 trace_metric 디렉터리
        for trace_dir in sorted(method_dir.iterdir(), key=lambda p: p.name):
            if not trace_dir.is_dir():
                continue

            if not pattern.match(trace_dir.name):
                continue

            print(f"start: {trace_dir}")

            outputs_csv = trace_dir / 'outputs.csv'
            slo_violation = trace_dir / 'slo_violation.csv'
            if outputs_csv.exists() and slo_violation.exists():
                if outputs_csv.parent in skip_paths:
                    continue
                input_paths.append(trace_dir)
        
        make_summerize(input_paths)

        # ──────────────────────────────────────────────────────
        # 추출 함수가 정의된 모듈 import
        # from your_module import extract_metrics

        # ──────────────────────────────────────────────────────
        # 데이터 수집 및 저장
        

