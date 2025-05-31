import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ───────────────────────────────────────────────
# 실험 설정
trace          = "both_static"
method_list    = ["OursMinusPause", "NoDeposit", "DistNSingle", 
                  "UniformDist", "BatchDimOnly", "Base"]
method_labels  = ["No Pause", "No Deposit", "DistNSingle", 
                  "Uniform Distance", "Batch Dimension Only", "Best Baseline"]
metric_list    = ["low", "mid", "high", "veryhigh"]
metric_labels  = ["Low", "Mid", "High", "Very High"]
slo_scales     = [10, 4.5, 3.5, 2.5, 1]
slo_labels     = [str(s) for s in slo_scales]

colors   = ["#84C8F4", "#C59FDB", "#7CD6A4", "#63D0C2", "#FAC07D", "#E05A4F"]

colors = [
    "#4DA6FF",  # Sky Blue
    "#3CC58F",  # Mint Green
    "#9F79C1",  # Lavender Purple
    "#C59FDB",  # Pastel Lavender
    "#FAC07D",
    "#FF8C69"   # Coral Orange
]

markers  = ['o', 's', '^', 'D', '*', 'P']

style = {
    "line":   {"linewidth":3, "markersize":10},
    "tick":   {"labelsize":18},
    "label":  {"fontsize":22, "labelpad":5},
    "title":  {"fontsize":22, "weight":"bold", "pad":10},
    "legend": {"fontsize":22},
    "spine":  {"color":"black","alpha":0.7,"linewidth":1.5},
}

base_dir = Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp")

# ───────────────────────────────────────────────
# 플롯 준비: 메트릭별 서브플롯
fig, axes = plt.subplots(1, len(metric_list),
                         figsize=(21, 5),
                         sharey=True)
plt.subplots_adjust(left=0.07, right=0.99,
                    top=0.88, bottom=0.12,
                    wspace=0.25)

for i, metric in enumerate(metric_list):
    ax = axes[i]
    for method, label, color, marker in zip(method_list, method_labels, colors, markers):
        y_vals = []
        for sc in slo_scales:
            summary_path = base_dir / f"slo{sc}" / method / "summerize.csv"
            if summary_path.exists():
                df_sum = pd.read_csv(summary_path)
                sel = df_sum[
                    (df_sum["slo"]    == sc) &
                    (df_sum["method"] == method) &
                    (df_sum["trace"]  == trace) &
                    (df_sum["metric"] == metric)
                ]
                y_vals.append(float(sel["tbt_attainment"].iloc[0]) if len(sel)==1 else 0.0)
                print(f"[Open] summary: {summary_path}")
            else:
                # print(f"[Warning] Missing summary: {summary_path}", file=sys.stderr)
                y_vals.append(0.0)
        ax.plot(slo_scales, y_vals,
                label=label, color=color, marker=marker,
                **style["line"])

    # 축 및 눈금 설정
    ax.set_xticks(slo_scales)
    ax.set_xticklabels(slo_labels, fontsize=style["tick"]["labelsize"])
    ax.tick_params(axis='x', length=0, labelsize=style["tick"]["labelsize"])
    ax.tick_params(axis='y', length=5, labelsize=style["tick"]["labelsize"])
    ax.set_ylim(-5, 105)
    ax.set_yticks([0, 50, 100])

    # 제목 및 레이블
    ax.set_title(metric_labels[i], **style["title"])
    if i == 0:
        ax.set_ylabel("TBT SLO Attainment (%)", **style["label"])
    # if i == len(metric_list) - 1:
    ax.set_xlabel("SLO Scale", **style["label"])

    # 스파인 스타일
    for spine in ax.spines.values():
        spine.set_edgecolor(style["spine"]["color"])
        spine.set_alpha(style["spine"]["alpha"])
        spine.set_linewidth(style["spine"]["linewidth"])

# 범례
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="upper center",
           bbox_to_anchor=(0.5, 1.2),
           ncol=6,
           **style["legend"])

# 전체 y축 텍스트
# fig.text(0.02, 0.5, "Memory Pressure",
#          va='center', rotation='vertical',
#          fontsize=style["label"]["fontsize"])

# 저장

plt.savefig("figures/6_3_design_validation.jpg", bbox_inches="tight")
plt.savefig("figures/6_3_design_validation.pdf", bbox_inches="tight")
