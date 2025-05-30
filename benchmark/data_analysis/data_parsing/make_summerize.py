import pandas as pd
import numpy as np
import ast
from pathlib import Path
from typing import List

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
def _extract_request_tpot_attainment(df: pd.DataFrame) -> float:
    tpot = np.array([np.mean(tbt) for tbt in df["time_between_tokens"]])
    thr = pd.to_numeric(df["slo_threshold"], errors="coerce").to_numpy().mean()
    valid = ~np.isnan(tpot)
    attain = (tpot[valid] <= thr).sum()
    return (attain / valid.sum() * 100 if valid.sum() else 0.0)

def _extract_token_slo_attainment(output_df: pd.DataFrame, slo_df: pd.DataFrame) -> float:
    total_decoded = int(output_df.get("decode_length", pd.Series(0)).sum())
    viol = int(slo_df.get("slo_violation", pd.Series(0)).sum())
    return ((total_decoded - viol) / total_decoded * 100 if total_decoded else 0.0)

def compute_throughput(df: pd.DataFrame) -> float:
    # 총 토큰 수 = input_length(존재 시) + decode_length
    total_input = df.get("input_length", pd.Series(0)).sum()
    total_decode = df["decode_length"].sum()
    total_tokens = total_input + total_decode

    # 전체 wall-clock 시간
    wall_time = df["finished_time"].max()

    # throughput = 총 토큰 수 / 전체 시간
    return total_tokens / wall_time if wall_time and total_tokens else 0.0

def _extract_percentile_ratio_lists(
    df: pd.DataFrame, percentiles: List[int]
) -> List[List[float]]:
    """For each percentile, return a list of Pxx/SLO ratios (one per request)."""
    lists = [[] for _ in percentiles]
    if {"time_between_tokens", "slo_threshold"}.issubset(df.columns):
        # system-wise SLO if all thresholds equal
        thr_vals = pd.to_numeric(df["slo_threshold"], errors="coerce").to_numpy()
        if np.all(thr_vals == thr_vals[0]):
            flat_tbt = np.concatenate(df["time_between_tokens"].values)
            thr = thr_vals[0]
            for idx, pct in enumerate(percentiles):
                px = np.percentile(flat_tbt, pct)
                lists[idx] = [px / thr]
        else:
            for tbt_list, thr_item in zip(df["time_between_tokens"], df["slo_threshold"]):
                if not tbt_list: 
                    continue
                thr_val = np.mean(thr_item) if isinstance(thr_item, (list, tuple, np.ndarray)) else float(thr_item)
                if thr_val <= 0:
                    continue
                for idx, pct in enumerate(percentiles):
                    px = np.percentile(tbt_list, pct)
                    lists[idx].append(px / thr_val)
    return lists

def extract_metrics(input_base_path):
    output_df = load_output_metrics(Path(input_base_path, "outputs.csv"))
    slo_df = pd.read_csv(Path(input_base_path, "slo_violation.csv"))

    tpot_slo = _extract_request_tpot_attainment(output_df)
    tbt_slo = _extract_token_slo_attainment(output_df, slo_df)
    throughput = compute_throughput(output_df)

    slo_thr = pd.to_numeric(output_df["slo_threshold"], errors="coerce").to_numpy().mean()

    pct_lists = _extract_percentile_ratio_lists(output_df, [90, 95, 99])
    ratio_means = [np.mean(lst) if lst else 0.0 for lst in pct_lists]
    p90_ratio, p95_ratio, p99_ratio = ratio_means

    return tpot_slo, tbt_slo, throughput, slo_thr, p90_ratio, p95_ratio, p99_ratio


# ───────────────────────────────────────────────
# 3. 입력 경로 리스트 및 출력 파일 경로 (하드코딩)
# Ours
# input_paths = [
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/Ours/both_static_low"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/Ours/both_static_mid"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/Ours/both_static_high"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/Ours/both_static_veryhigh"),

#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/Ours/batch_dyn_low"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/Ours/batch_dyn_mid"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/Ours/batch_dyn_high"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/Ours/batch_dyn_veryhigh"),

#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/Ours/token_dyn_low"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/Ours/token_dyn_mid"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/Ours/token_dyn_high"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/Ours/token_dyn_veryhigh"),

#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/Ours/both_dyn_low"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/Ours/both_dyn_mid"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/Ours/both_dyn_high",),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/Ours/both_dyn_veryhigh",),
# ]

# output_csv = input_paths[0].parent / "summerize.csv"


# FlexGen
# input_paths = [
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/DistNSingle_TP/both_static_low"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/DistNSingle_TP/both_static_mid"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/DistNSingle_TP/both_static_high"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/DistNSingle_TP/both_static_veryhigh"),

    # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/DistNSingle/batch_dyn_low"),
    # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/DistNSingle/batch_dyn_mid"),
    # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/DistNSingle/batch_dyn_high"),
    # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/DistNSingle/batch_dyn_veryhigh"),

    # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/SelectN/token_dyn_low"),
    # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/SelectN/token_dyn_mid"),
    # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/SelectN/token_dyn_high"),
    # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/SelectN/token_dyn_veryhigh"),

    # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/Flexgen/both_dyn_low"),
    # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/Flexgen/both_dyn_mid"),
    # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/Flexgen/both_dyn_high"),
    # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/Flexgen/both_dyn_veryhigh"),
# ]

# output_csv = input_paths[0].parent / "summerize.csv"

# NoPrefetch
# input_paths = [
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/both_static_low"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/both_static_mid"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/both_static_high"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/both_static_veryhigh"),

#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/batch_dyn_low"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/batch_dyn_mid"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/batch_dyn_high"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/batch_dyn_veryhigh"),

#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/token_dyn_low"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/token_dyn_mid"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/token_dyn_high"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/token_dyn_veryhigh"),

#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/both_dyn_low"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/both_dyn_mid"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/both_dyn_high"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/both_dyn_veryhigh"),
# ]

# output_csv = input_paths[0].parent / "summerize.csv"

# SelectN
# input_paths = [
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/SelectN/both_static_low"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/SelectN/both_static_mid"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/SelectN/both_static_high"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/SelectN/both_static_veryhigh"),

#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/SelectN/batch_dyn_low"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/SelectN/batch_dyn_mid"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/SelectN/batch_dyn_high"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/SelectN/batch_dyn_veryhigh"),

    # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo4.5/SelectN/token_dyn_low"),
    # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo4.5/SelectN/token_dyn_mid"),
    # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/SelectN/token_dyn_high"),
    # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo4.5/SelectN/token_dyn_veryhigh"),

#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/SelectN/both_dyn_low"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/SelectN/both_dyn_mid"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/SelectN/both_dyn_high"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/SelectN/both_dyn_veryhigh"),
# ]

# output_csv = input_paths[0].parent / "summerize.csv"

import pandas as pd
from pathlib import Path

# ──────────────────────────────────────────────────────
# 설정: paper_main_exp 아래 모든 slo/<scale>/<method>/<trace_metric> 경로 자동 수집
BASE_DIR = Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp")

# ──────────────────────────────────────────────────────
# input_paths: slo 디렉터리 하위의 method별 trace_metric 디렉터리 필터링
for slo_dir in BASE_DIR.glob('slo*'):
    if not slo_dir.is_dir():
        continue
    for method_dir in slo_dir.iterdir():
        input_paths = []    
        OUTPUT_CSV = method_dir / "summerize.csv"

        if not method_dir.is_dir():
            continue
        # 각 trace_metric 디렉터리
        for trace_dir in sorted(method_dir.iterdir(), key=lambda p: p.name):
            if not trace_dir.is_dir():
                continue
            outputs_csv = trace_dir / 'outputs.csv'
            slo_violation = trace_dir / 'slo_violation.csv'
            if outputs_csv.exists() and slo_violation.exists():
                input_paths.append(trace_dir)

        # ──────────────────────────────────────────────────────
        # 추출 함수가 정의된 모듈 import
        # from your_module import extract_metrics

        # ──────────────────────────────────────────────────────
        # 데이터 수집 및 저장
        results = []
        for path in input_paths:
            slo = path.parts[-3].replace('slo', '')
            method = path.parts[-2]
            trace_metric = path.name  # 'both_static_low'
            trace, metric = trace_metric.rsplit('_', 1)

            # extract_metrics(path) -> (tpot_slo, tbt_slo, throughput, slo_thr, p90, p95, p99)
            tpot_slo, tbt_slo, throughput, slo_thr, p90_ratio, p95_ratio, p99_ratio = extract_metrics(path)
            results.append({
                'slo': slo,
                'method': method,
                'trace': trace,
                'metric': metric,
                'tpot_attainment': tpot_slo,
                'tbt_attainment': tbt_slo,
                'throughput_tokens_per_sec': throughput,
                'slo_threshold_mean': slo_thr,
                'p90_ratio': p90_ratio,
                'p95_ratio': p95_ratio,
                'p99_ratio': p99_ratio,
            })

        # DataFrame 생성 및 저장
        df_result = pd.DataFrame(results)
        df_result.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        for i in input_paths:
            print(f"{i}")
        print(f"Aggregated CSV saved to {OUTPUT_CSV}")
        print()
