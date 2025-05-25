import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
import ast

trace_list = [
    "test_fit_static_0", 
    "test_shortshort_enough", 
    "test_shortlong_less", 
]
trace_labels = [
    "(a) Trace 1",
    "(b) Trace 2",
    "(c) Trace 3",
]
method_list = ["Flexgen", "DeepSpeed", "SelectN", "NoPrefetch", "Ours"]
method_labels = ["Flexgen", "DeepSpeed", "Placeholder(SelectN)", "No Prefetch", "Ours"]
metric_list = ["Low", "Mid", "High"]
metric_list_labels = ["Low", "Mid", "High"]
slo_scales = [4, 3, 2, 1]
slo_labels    = [str(s) for s in slo_scales]   # x축 표기용 문자열

colors = [
    "#84C8F4",  # Soft Sky Blue (연하늘색)
    "#C59FDB",  # Pastel Lavender (연보라색)
    "#7CD6A4",  # Mint Green (민트색)
    "#63D0C2",  # Aqua Teal (청록색)
    "#E05A4F",  # Coral Red (산호빛 빨강)
]
markers = [
    'o',  # Flexgen
    's',  # DeepSpeed
    '^',  # SelectN
    'D',  # NoPrefetch
    'P'   # Ours
]


font_size = 18
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
        "color": "gray",
        "alpha": 0.5,
        "linestyle": "-",
        "linewidth": 2
    },
    "grid": {
        "color": "gray",
        "linestyle": "--",
        "linewidth": 2,
        "alpha": 0.5
    },
    "text": {
        "fontsize": 14,
        "weight": "bold"
    },
}


# ───────────────────────────────────────────────
# 1. CSV 로드 
def load_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 필요한 컬럼 타입 변환
    for c in ["decode_length", "arrival_time", "finished_time", "time_to_first_token", "solver_time"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            df[c] = pd.to_numeric(df[c].explode()).groupby(level=0).sum() if c=="solver_time" else pd.to_numeric(df[c], errors="coerce")
    return df

# ───────────────────────────────────────────────
# 2. Throughput 계산 함수 (간소화 버전)
def compute_throughput(df: pd.DataFrame) -> float:
    # 총 토큰 수 = input_length(존재 시) + decode_length
    total_input = df.get("input_length", pd.Series(0)).sum()
    total_decode = df["decode_length"].sum()
    total_tokens = total_input + total_decode

    # 전체 wall-clock 시간
    wall_time = df["finished_time"].max()

    # throughput = 총 토큰 수 / 전체 시간
    return total_tokens / wall_time if wall_time and total_tokens else 0.0


def load_metrics_for_slo_scales(method: str, trace: str, metric: str, slo_scales: List[str], k: int) -> List[float]:
    results = []
    for i, scale in enumerate(slo_scales):
        path = Path(f"/exp/{method}/{trace}_{metric}/{scale}/output.csv")
        try:
            if not path.exists():
                raise FileNotFoundError(f"Not found: {path}")
            df = load_metrics(path)
            results.append(compute_throughput(df))
        except Exception as e:
            print(f"[SLO 경고] {method} - {trace} - {metric} - {scale}: {e}")
            results.append(10.0 * i * k)  # 기본값 설정
    return results

# ───────────────────────────────────────────────
# 3. 플롯 설정

fig, axes = plt.subplots(
    nrows=len(metric_list), 
    ncols=len(trace_list),
    figsize=(15, 3.5*len(metric_list)),
    squeeze=False, 
)
# y축 정렬 및 간격 조정
plt.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.08, wspace=0.3, hspace=0.4)

for i, metric in enumerate(metric_list):
    for j, (trace, trace_label) in enumerate(zip(trace_list, trace_labels)):
        ax = axes[i][j]

        for m, method in enumerate(method_list):
            thr_vals = load_metrics_for_slo_scales(method, trace, metric, slo_scales, m)
            
            ax.plot(
                slo_scales, thr_vals,
                **style["line"],
                marker=markers[m], color=colors[m], linestyle='-',
                label=method_labels[m], 
            )


        ax.set_xticks(slo_scales)  # 원래 순서 그대로
        ax.set_xticklabels([str(s) for s in reversed(slo_scales)], 
                               fontsize=style["tick"]["fontsize"])  # 라벨만 뒤집기

        ax.tick_params(axis='x', labelsize=style["tick"]["fontsize"])
        ax.tick_params(axis='y', labelsize=style["tick"]["fontsize"])

        ax.tick_params(axis='x', which='both', length=0)
        ax.tick_params(axis='y', which='both', length=0)

        ax.set_axisbelow(True)
        ax.set_yticks([])


        for spine in ax.spines.values():
            spine.set_edgecolor(style["spine"]["color"])
            spine.set_alpha(style["spine"]["alpha"])
            spine.set_linewidth(style["spine"]["linewidth"])

        # 오른쪽 y축 설정
        ax_right = ax.twinx()
        # ax_right.set_ylim(0, 100)
        # ax_right.set_yticks([0, 50, 100])

        # if i == len(metric_list) - 1:
        #     ax.set_xlabel(trace_label, fontsize=style["label"]["fontsize"], labelpad=10)
        
        if j == 0:
            ax.set_ylabel(metric_list_labels[i], fontsize=style["label"]["fontsize"], labelpad=10)
        if j == len(trace_list) - 1:
            ax_right.set_ylabel("Throughput\n(tokens / second)", 
                                fontsize=style["label"]["fontsize"], 
                                labelpad=30,
                                rotation=270,
                                )
            
            # 레이블을 오른쪽 축으로 지정
            ax_right.yaxis.set_label_position("right")
            # 레이블 축 좌표(x, y)를 (1.1, 0.5)로 설정 → 오른쪽으로 10% 더 이동
            ax_right.yaxis.set_label_coords(1.35, 0.5)
        else:
            ax.set_yticklabels([])
            ax.set_yticks([])
            # ax_right.set_yticklabels([])
            # ax_right.set_yticks([])

        ax_right.tick_params(axis='y', labelsize=style["tick"]["fontsize"])
    
        # 모든 서브플롯에 x축 레이블 "SLO Scale" 추가
        if i == len(metric_list) - 1:
            ax.set_xlabel(f"SLO Scale", fontsize=style["label"]["fontsize"], labelpad=10)
        else:
            ax.set_xlabel("")
        # ax.set_xlabel("SLO Scale", fontsize=style["label"]["fontsize"], labelpad=10)
            
        if i == len(metric_list) - 1:
            ax.text(
                0.5, -0.45, trace_label,
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=style["label"]["fontsize"],
                # weight='bold'
            )

# 범례
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(
    handles, labels, 
    loc="upper center",
    # bbox_to_anchor=(0.5,1.02), 
    ncol=len(method_list),
    fontsize=style["legend"]["fontsize"],
    frameon=False
)

# 전체 y축 라벨 
fig.text(0.02, 0.5, "Memory Pressure", va='center', rotation='vertical', fontsize=font_size, 
        #  weight='bold'
         )



plt.savefig("figures/throughput_total_compare.jpg", format='jpg', bbox_inches="tight")
# plt.savefig("figures/tbt_total_compare.pdf", format='pdf', bbox_inches="tight")