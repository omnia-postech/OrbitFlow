import pandas as pd
import numpy as np
import ast
from pathlib import Path

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

def extract_metrics(input_base_path):
    output_df = load_output_metrics(Path(input_base_path, "outputs.csv"))
    slo_df = pd.read_csv(Path(input_base_path, "slo_violation.csv"))

    tpot_slo = _extract_request_tpot_attainment(output_df)
    tbt_slo = _extract_token_slo_attainment(output_df, slo_df)
    throughput = compute_throughput(output_df)

    slo_thr = pd.to_numeric(output_df["slo_threshold"], errors="coerce").to_numpy().mean()

    return tpot_slo, tbt_slo, throughput, slo_thr


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

#     # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/Ours/both_dyn_low"),
#     # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/Ours/both_dyn_mid"),
#     # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/Ours/both_dyn_high",),
#     # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/Ours/both_dyn_veryhigh",),
# ]

# output_csv = input_paths[0].parent / "summerize.csv"


# FlexGen
# input_paths = [
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/Flexgen/both_static_low"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/Flexgen/both_static_mid"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/Flexgen/both_static_high"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/Flexgen/both_static_veryhigh"),

#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/Flexgen/batch_dyn_low"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/Flexgen/batch_dyn_mid"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/Flexgen/batch_dyn_high"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/Flexgen/batch_dyn_veryhigh"),

#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/Flexgen/token_dyn_low"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/Flexgen/token_dyn_mid"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/Flexgen/token_dyn_high"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/Flexgen/token_dyn_veryhigh"),

#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/Flexgen/token_dyn_low"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/Flexgen/token_dyn_mid"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/Flexgen/token_dyn_high"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/Flexgen/token_dyn_veryhigh"),
# ]

# output_csv = input_paths[0].parent / "summerize.csv"

# NoPrefetch
input_paths = [
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/both_static_low"),
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/both_static_mid"),
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/both_static_high"),
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/both_static_veryhigh"),

    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/batch_dyn_low"),
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/batch_dyn_mid"),
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/batch_dyn_high"),
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/batch_dyn_veryhigh"),

    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/token_dyn_low"),
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/token_dyn_mid"),
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/token_dyn_high"),
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/token_dyn_veryhigh"),

    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/both_dyn_low"),
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/both_dyn_mid"),
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/both_dyn_high"),
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/both_dyn_veryhigh"),
]

output_csv = input_paths[0].parent / "summerize.csv"

# SelectN
# input_paths = [
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo4.5/SelectN/both_static_low"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo4.5/SelectN/both_static_mid"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo4.5/SelectN/both_static_high"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo4.5/SelectN/both_static_veryhigh"),

#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo4.5/SelectN/batch_dyn_low"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo4.5/SelectN/batch_dyn_mid"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo4.5/SelectN/batch_dyn_high"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo4.5/SelectN/batch_dyn_veryhigh"),

#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo4.5/SelectN/token_dyn_low"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo4.5/SelectN/token_dyn_mid"),
#     # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/SelectN/token_dyn_high"),
#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo4.5/SelectN/token_dyn_veryhigh"),

#     Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo4.5/SelectN/both_dyn_low"),
# ]

# output_csv = input_paths[0].parent / "summerize.csv"


# ───────────────────────────────────────────────
# 4. 데이터 수집 및 저장
results = []
for path in input_paths:
    # 파일 구조에서 trace, metric, method 추출 (예: /.../{method}/{trace}_{metric})
    slo = path.parts[-3].replace("slo", "")
    method = path.parts[-2]
    trace_metric = path.stem  # "both_static_low" 형태
    trace, metric = trace_metric.rsplit("_", 1)

    tpot_slo, tbt_slo, throughput, slo_thr = extract_metrics(path)
    results.append({
        "slo": slo,
        "method": method,
        "trace": trace,
        "metric": metric,
        "tpot_attainment": tpot_slo,
        "tbt_attainment": tbt_slo,
        "throughput_tokens_per_sec": throughput,
        "slo_threshold_mean": slo_thr
    })

df_result = pd.DataFrame(results)
df_result.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"Aggregated CSV saved to {output_csv}")
