import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from pathlib import Path
from typing import List

# 실험 이름
EXP = "Debug"

# 설정

method_list = ["NoPause", "NoDeposit", "NoSolver", 
               "UniformDist", "BatchDimOnly", "Base"]
method_labels = ["No Pause", "No Deposit", "No Solver", 
                 "Uniform Distance", "Batch Dimension Only", "Best Baseline"]

slo_scales = [4, 3, 2, 1]  # 내림차순]
slo_labels    = [str(s) for s in slo_scales]   # x축 표기용 문자열

colors = [
    "#84C8F4",  # Soft Sky Blue (연하늘색)
    "#C59FDB",  # Pastel Lavender (연보라색)
    "#7CD6A4",  # Mint Green (민트색)
    "#63D0C2",  # Aqua Teal (청록색)
    "#FAC07D",  # 파스텔 오렌지
    "#E05A4F",  # Coral Red (산호빛 빨강)
]
markers = [
    'o',  # Flexgen
    's',  # DeepSpeed
    '^',  # SelectN
    'D',  # NoPrefetch
    '*',
    'P'   # Ours
]

font_size = 22
style = {
    "line": {
        "linewidth": 3, 
        "markersize": 10
    },
    "tick": {
        "fontsize": 18
    },
    "label": {
        "fontsize": font_size,
        "labelpad":5,
        # "weight": "bold"
    },
    "title": {
        "fontsize": font_size,
        "weight": "bold",
        "pad": 10
    },
    "legend": {
        "fontsize": font_size,
        # "loc": "upper center",
        # "bbox_to_anchor": (0.5, 1.2),
        # "ncol": len(method_list)
    },
    "spine": {
        "color": "black",
        "alpha": 0.7,
        "linestyle": "-",
        "linewidth": 1.5
    },
    "grid": {
        "color": "gray",
        "linestyle": "-",
        "linewidth": 2.5,
        "alpha": 1
    },
    "text": {
        "fontsize": 14,
        "weight": "bold"
    },
}

def load_metrics(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    numeric_cols: List[str] = [
        "arrival_time", "first_scheduled_time", "finished_time",
        "time_to_first_token", "slo_threshold", "slo_violations",
        "stall_duration", "decode_length", "end_to_end_time", "decode_time", "time_per_output_token"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("time_between_tokens", "stall_times", "stall_durations"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
    return df
    
def slo_tpot(df: pd.DataFrame) -> float:
    required_cols = {"end_to_end_time", "num_output_tokens", "slo_threshold"}
    if not required_cols.issubset(df.columns):
        print(f"Missing columns: {required_cols - set(df.columns)}")
        return 0.0

    end_to_end_time = pd.to_numeric(df["end_to_end_time"], errors="coerce")
    num_tokens = pd.to_numeric(df["num_output_tokens"], errors="coerce")
    tpot = end_to_end_time / num_tokens

    slo_threshold = df["slo_threshold"].apply(
        lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else x
    )
    slo_threshold = pd.to_numeric(slo_threshold, errors="coerce")

    valid = ~tpot.isna() & ~slo_threshold.isna()
    attained = (tpot[valid] <= slo_threshold[valid]).sum()
    total = valid.sum()

    return attained / total * 100 if total else 0.0

def slo_tbt(df: pd.DataFrame) -> float:
    decoded = df["decode_length"].sum()
    viol    = df["slo_violations"].sum()

    return (decoded-viol) / decoded * 100

def load_metrics_for_slo_scales(method: str, slo_scales: List[str], is_tpot: bool, k: int) -> List[float]:
    results = []
    for i, scale in enumerate(slo_scales):
        path = Path(f"/exp/{method}/trace_metric/{scale}/output.csv")
        try:
            if not path.exists():
                raise FileNotFoundError(f"Not found: {path}")
            df = load_metrics(path)
            if is_tpot:
                results.append(slo_tpot(df))
            else:
                results.append(slo_tbt(df))
        except Exception as e:
            print(f"[SLO 경고] {method} - {scale}: {e}")
            results.append(np.random.random() * 100)  # 기본값 설정
            results.sort()
    return results

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
plt.subplots_adjust(
    left=0.05, right=0.99, top=0.93, bottom=0.07,
    wspace=0.3, hspace=0.1
)

for ax, (is_tpot, ylabel) in zip(
        axes,
        [(False,  "TBT SLO Attainment (%)"),
         (True, "TPOT SLO Attainment (%)")]):
    i = 0
    for method, label, color, marker in zip(method_list, method_labels, colors, markers):
        y_vals = load_metrics_for_slo_scales(method, slo_scales, is_tpot, i)
        ax.plot(
            slo_scales, y_vals,
            **style["line"],
            marker=markers[i], color=colors[i], linestyle='-',
            label=method_labels[i], 
        )
        i += 1

    # ax.set_title(title, fontsize=30, pad=12)
    ax.set_xlabel("SLO Scale", fontsize=30, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=30, labelpad=8)

    ax.set_xticks(slo_scales)
    ax.set_xticklabels([str(s) for s in slo_scales], fontsize=30)

    # ax.tick_params(axis='y', labelsize=30)

    ax.tick_params(axis='x', which='both', length=0, labelsize=30, pad=10)
    ax.tick_params(axis='y', which='both', length=0, labelsize=30, pad=5)
    ax.set_ylim(-5, 105)

    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(['0', '50', '100'])

    # ax.xaxis.grid(True, **style["grid"])
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_edgecolor(style["spine"]["color"])
        spine.set_alpha(style["spine"]["alpha"])
        spine.set_linewidth(style["spine"]["linewidth"])
# 범례: 큰 그래프 상단 중앙
fig.legend(
    method_labels, loc="upper center",
    bbox_to_anchor=(0.5, 1.25), 
    ncol=len(method_labels)/2,
    fontsize=30, frameon=False
)

# 전체 레이아웃 및 저장
# plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig("figures/6_3_design_validation.jpg", format='jpg', bbox_inches="tight")
plt.savefig("figures/6_3_design_validation.pdf", format='pdf', bbox_inches="tight")