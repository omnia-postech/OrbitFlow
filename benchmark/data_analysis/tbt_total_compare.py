import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from pathlib import Path
from typing import List

# 실험 이름
EXP = "Debug"

# 설정
trace_list = [
    "test_fit_static_0", 
    "test_shortshort_enough", 
    "test_shortlong_less", 
    "test_shortlong_enough",
]
trace_labels = [
    "(a) Trace 1",
    "(b) Trace 2",
    "(c) Trace 3",
    "(d) Trace 4",
]
method_list = ["Flexgen", "DeepSpeed", "SelectN", "NoPrefetch", "Ours"]
method_labels = ["Flexgen", "DeepSpeed", "Placeholder(SelectN)", "No Prefetch", "Ours"]
metric_list = ["Low", "Mid", "High"]
metric_list_labels = ["Low", "Mid", "High"]
slo_scales = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, ]

colors = [
    # Static (파란색 계열 - 부드럽고 시인성 높음)
    "#84C8F4",  # NextLayer - 부드러운 파란색
    "#C59FDB",  # Flexgen - 연한 보라색
    "#7CD6A4",  # NoPrefetch - 연한 청록색
    "#63D0C2",  # Static2 - 중간 밝기의 민트

    # # Dynamic (따뜻한 계열, Ours는 강조)
    # "#FAC07D",  # SelectN - 파스텔 오렌지
    # "#F29E9E",  # DistNSingle - 연한 코랄

    "#E05A4F"   # Ours - 강조용 진한 살구+레드 (단독 대비 확보)
]

markers = [
    'o',  # Flexgen
    'o',  # DeepSpeed
    'o',  # SelectN
    'o',  # NoPrefetch
    '*'   # Ours
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
    
def compute_slo_attainment(df: pd.DataFrame) -> float:
    if {"decode_length", "slo_violations"}.issubset(df.columns):
        total_decoded = int(df.get("decode_length", pd.Series(0)).sum())
        viol = int(df.get("slo_violations", pd.Series(0)).sum())
        return total_decoded - viol, total_decoded
    else:
        print("Missing columns: decode_length, slo_violations")
        return 0.1

def load_metrics_for_slo_scales(method: str, trace: str, metric: str, slo_scales: List[str], k: int) -> List[float]:
    results = []
    for i, scale in enumerate(slo_scales):
        path = Path(f"/exp/{method}/{trace}_{metric}/{scale}/output.csv")
        try:
            if not path.exists():
                raise FileNotFoundError(f"Not found: {path}")
            df = load_metrics(path)
            results.append(compute_slo_attainment(df))
        except Exception as e:
            print(f"[SLO 경고] {method} - {trace} - {metric} - {scale}: {e}")
            results.append(10.0 * i * k)  # 기본값 설정
    return results

# 시각화: 큰 그래프는 metric(y축) vs trace(x축), 작은 그래프는 slo_scale(x축) vs slo_attainment(y축)
fig, axes = plt.subplots(
    len(metric_list), 
    len(trace_list), 
    figsize=(5 * len(trace_list), 3.5 * len(metric_list)),  # 직사각형 비율로 조정
    squeeze=False
)

for i, metric in enumerate(metric_list):
    for j, (trace, trace_label) in enumerate(zip(trace_list, trace_labels)):
        ax = axes[i][j]

        for k, method in enumerate(method_list):
            y_values = load_metrics_for_slo_scales(method, trace, metric, slo_scales, k)
            ax.plot(
                slo_scales, y_values, 
                **style["line"],
                marker=markers[k], color=colors[k], linestyle='-',
                label=method_labels[k], 
            )

            # 🔽 90% 선과 교차하는 구간 찾기 및 x축에 같은 색으로 선 그리기
            for idx in range(len(slo_scales) - 1):
                y1, y2 = y_values[idx], y_values[idx + 1]
                x1, x2 = float(slo_scales[idx]), float(slo_scales[idx + 1])

                if (y1 - 90) * (y2 - 90) <= 0 and y2 != y1:
                    x_cross = x1 + (90 - y1) * (x2 - x1) / (y2 - y1)
                    y_cross = 90
                    ax.vlines(
                        x=x_cross, ymin=0, ymax=y_cross,
                        colors=colors[k], linestyles="--", linewidth=2, alpha=0.8
                    )

                    # if i == 0:
                    #     ax.set_title(f"{trace_label}", **style["title"])

        ax.set_xticks(slo_scales)
        ax.set_xticklabels([str(s) for s in slo_scales])

        ax.tick_params(axis='x', labelsize=style["tick"]["fontsize"])
        ax.tick_params(axis='y', labelsize=style["tick"]["fontsize"])

        ax.tick_params(axis='x', which='both', length=0)
        ax.tick_params(axis='y', which='both', length=0)

        ax.set_axisbelow(True)
        ax.set_ylim(0, 100)
        ax.set_yticks([])


        ax.axhline(90, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

        for spine in ax.spines.values():
            spine.set_edgecolor(style["spine"]["color"])
            spine.set_alpha(style["spine"]["alpha"])
            spine.set_linewidth(style["spine"]["linewidth"])

        # 오른쪽 y축 설정
        ax_right = ax.twinx()
        ax_right.set_ylim(0, 100)
        ax_right.set_yticks([0, 50, 100])

        # if i == len(metric_list) - 1:
        #     ax.set_xlabel(trace_label, fontsize=style["label"]["fontsize"], labelpad=10)
        
        if j == 0:
            ax.set_ylabel(metric_list_labels[i], fontsize=style["label"]["fontsize"], labelpad=10)
        if j == len(trace_list) - 1:
            ax_right.set_ylabel("SLO attainment(%)", fontsize=style["label"]["fontsize"], labelpad=10)
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
    loc='upper center', 
    # bbox_to_anchor=(0.5, 0.98),
    ncol=len(method_list), fontsize=style["legend"]["fontsize"],
    frameon=False  # ← 테두리 제거
)

# 전체 y축 라벨 
fig.text(0.02, 0.5, "Memory Pressure", va='center', rotation='vertical', fontsize=font_size, 
        #  weight='bold'
         )

# y축 정렬 및 간격 조정
plt.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.08, wspace=0.2, hspace=0.4)

plt.savefig("figures/tbt_total_compare.jpg", format='jpg', bbox_inches="tight")
plt.savefig("figures/tbt_total_compare.pdf", format='pdf', bbox_inches="tight")