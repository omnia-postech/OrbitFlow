import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import ast
import json

# ── 기존 load 함수 재사용 ──
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
    for col in ("time_between_tokens", "stall_times", "stall_durations", "solver_time"):
        if col in df:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
    return df

# ── 데이터 로드 ──
output_df = load_output_metrics(
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/"
         "slo2.5/DistNSingle/lambda3.0x_cv1/outputs.csv")
)

def save_tbts_to_txt(tbts: list, filepath: str):
    """tbts 리스트의 값을 한 줄씩 filepath에 저장."""
    with open(filepath, "w") as f:
        for t in tbts:
            f.write(f"{t}\n")
    print(f"Saved {len(tbts)} values → {filepath}")

def get_tbts(df, request_id):
    # 해당 request_id 행의 리스트 반환 (없으면 빈 리스트)
    vals = df.loc[df["request_id"] == request_id, "time_between_tokens"]
    if vals.empty:
        return []
    tbts = vals.iloc[0]
    return tbts if isinstance(tbts, list) else list(tbts)

def get_solver_time(df, request_id):
    # 해당 request_id 행의 리스트 반환 (없으면 빈 리스트)
    vals = df.loc[df["request_id"] == request_id, "solver_time"]
    if vals.empty:
        return []
    solvers = vals.iloc[0]
    return solvers if isinstance(solvers, list) else list(solvers)

# ── 플롯 ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, req in zip(axes, ["request_8", "request_43"]):
    rtbts = np.asarray(get_tbts(output_df, req), float)
    solvers = np.asarray(get_solver_time(output_df, req), float)
    tbts = rtbts - solvers
    save_tbts_to_txt(tbts, f"{req}.txt")
    ax.plot(tbts,
            linestyle="-",     # 선만
            marker="",         # 마커 없음
            linewidth=2,
            color="#4C9085")   # 원하시는 색으로 변경 가능
    ax.axhline(0.13, color="red", linestyle="--", linewidth=1)
    ax.set_title(req, fontsize=14)
    ax.set_xticks([])        # x축 눈금 레이블 숨기기
    ax.set_xlabel("")        # x축 라벨 제거

axes[0].set_ylabel("Time Between Tokens (s)", fontsize=12)

plt.savefig("figures/failure.jpg", bbox_inches="tight", dpi=300)
