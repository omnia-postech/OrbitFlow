import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import sys

# ───────────────────────────────────────────────
# 실험 설정
trace = "batch_dyn"
method_list = [
    "OursMinusPause", "DistNSingle",
    "OursUniformSolver", "Ours"
]
method_labels = [
    "No Pause", "Dynamic Heuristic",
    "Uniform Dist", "OrbitFlow"
]
metric_list = ["veryhigh",]
metric_labels = ["",]
slo_scales = [1.5, 1.25, 1]
slo_labels = [str(s) for s in slo_scales]

colors = [
    "#8B5E3C",  # Dark Brown (짙은 갈색)
    "#4C9085",  # Teal (중간 톤 청록색)
    "#7F8C8D",  # Medium Gray (중간 회색)
    "#FF8C69"   # Coral Orange (기존 유지)
]

style = {
    "line":   {"linewidth":3, "markersize":10},
    "spine":  {"color":"black", "alpha":0.7, "linewidth":1.5},
    "title":  {"fontsize":32, "pad":8},
    "label":  {"fontsize":45, "labelpad":10},
    "legend": {"fontsize":44, "padding":5},
    "tick":   {"labelsize":40},
}

base_dir = Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp")

# ───────────────────────────────────────────────
# 플롯 준비: 메트릭별 서브플롯
fig, axes = plt.subplots(1, len(metric_list), figsize=(14, 10), sharey=True)
plt.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.12, wspace=0.25)

# 바 폭 및 위치 계산
bar_width = 0.18
x_base = np.arange(len(slo_scales))
total_methods = len(method_list)
x_offsets = [x_base + (i - total_methods / 2) * (bar_width + 0.01) + bar_width / 2 for i in range(total_methods)]

# 범례 핸들 수집용 dict
method_handles = {}

for i, metric in enumerate(metric_list):
    ax = axes

    for idx, (method, label, color) in enumerate(zip(method_list, method_labels, colors)):
        y_vals = []
        for sc in slo_scales:
            summary_path = base_dir / f"slo{sc}" / method / "summerize.csv"
            if summary_path.exists():
                df_sum = pd.read_csv(summary_path)
                sel = df_sum[
                    (df_sum["slo"] == sc) &
                    (df_sum["method"] == method) &
                    (df_sum["trace"] == trace) &
                    (df_sum["metric"] == metric)
                ]
                if len(sel) == 1:
                    val = float(sel["tbt_attainment"].iloc[0])
                else:
                    val = 0.0
            else:
                val = 0.0
            y_vals.append(val)

        print(f"[{method}] trace={trace}, metric={metric}, tbt_attainment: {y_vals}")

        bar = ax.bar(x_offsets[idx], y_vals, width=bar_width, color=color, label=label)
        if label not in method_handles:
            method_handles[label] = bar[0]

    # 90% 점선 라인 추가
    ax.axhline(y=90, color='black', linestyle='--', linewidth=2)

    # 축 및 눈금 설정
    ax.set_xticks(x_base)
    ax.set_xticklabels(slo_labels, fontsize=style["tick"]["labelsize"])
    ax.tick_params(axis='x', length=0, labelsize=style["tick"]["labelsize"])
    ax.tick_params(axis='y', length=5, labelsize=style["tick"]["labelsize"])
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 50, 100])

    # 제목 및 레이블
    ax.set_title(metric_labels[i], **style["title"])
    if i == 0:
        ax.set_ylabel("TBT SLO Attainment (%)", **style["label"])
    ax.set_xlabel("SLO Scale", **style["label"])

    # 스파인 스타일
    for spine in ax.spines.values():
        spine.set_edgecolor(style["spine"]["color"])
        spine.set_alpha(style["spine"]["alpha"])
        spine.set_linewidth(style["spine"]["linewidth"])

# ───────────────────────────────────────────────
# 범례
ordered_labels = method_labels + ["SLO 90%"]
ordered_handles = [method_handles[label] for label in method_labels] + [
    Line2D([0], [0], color='black', linestyle='--', linewidth=2)
]

fig.legend(ordered_handles, ordered_labels,
           loc='upper center',
           ncol=3,
           bbox_to_anchor=(0.5, 1.12),
           fontsize=style["legend"]["fontsize"],
           frameon=False)

# 저장
plt.savefig("figures/6_3_design_validation.jpg", bbox_inches="tight")
plt.savefig("figures/6_3_design_validation.pdf", bbox_inches="tight")
plt.show()