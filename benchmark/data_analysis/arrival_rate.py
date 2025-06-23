import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path

# ───────────────────────────────────────────────
# 실험 설정
method_list    = ["Ours"]
method_labels  = ["OrbitFlow"]

arrival_rate   = [1.0, 2.0, 3.0, 4.0, 5.0]
arrival_labels = [str(r) for r in arrival_rate]
slo_scales     = [2.5, 1.5, 1]
slo_labels     = [str(s) for s in slo_scales]

colors = [
    "#8B5E3C",  # Dark Brown
    "#4C9085",  # Teal
    "#7F8C8D",  # Gray
    "#FF8C69",  # Coral
    "#ded8a0",
]

style = {
    "spine":  {"color":"black", "alpha":0.7, "linewidth":1.5},
    "title":  {"fontsize":32, "pad":8},
    "label":  {"fontsize":45, "labelpad":10},
    "legend": {"fontsize":44, "padding":5},
    "tick":   {"labelsize":40},
}

base_dir = Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp")

# ───────────────────────────────────────────────
fig, axes = plt.subplots(1, len(slo_scales), figsize=(24, 10), sharey=True)
plt.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.12, wspace=0.25)

bar_width = 0.8
x = np.arange(len(arrival_rate))

for i, sc in enumerate(slo_scales):
    ax = axes[i]
    # slo별로 각 arrival rate에 대한 TBT attainment 수집
    y_vals = []
    summary_path = base_dir / f"slo{sc}" / "Ours" / "arrival_summerize.csv"
    if summary_path.exists():
        df_sum = pd.read_csv(summary_path)
        for rate in arrival_rate:
            sel = df_sum[(df_sum["slo"] == sc) 
                         & (df_sum["arrival_rate"] == rate)
                         & (df_sum["cv_num"] == 1)
                         ]
            y_vals.append(float(sel["tbt_attainment"].iloc[0]) if len(sel)==1 else 0.0)
    else:
        y_vals = [0.0] * len(arrival_rate)

    # 바 차트
    ax.bar(x, y_vals, width=bar_width, color=colors[:len(arrival_rate)])

    # 90% SLO 라인
    ax.axhline(90, color='black', linestyle='--', linewidth=2)

    # 축 설정
    ax.set_xticks(x)
    ax.set_xticklabels(arrival_labels, fontsize=style["tick"]["labelsize"])
    ax.tick_params(axis='x', length=0, labelsize=style["tick"]["labelsize"])
    ax.tick_params(axis='y', length=5, labelsize=style["tick"]["labelsize"])
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 50, 100])

    # 제목 & 레이블
    ax.set_title(f"SLO Scale = {sc}", **style["title"])
    if i == 0:
        ax.set_ylabel("TBT SLO Attainment (%)", **style["label"])
    ax.set_xlabel("Arrival Rate", **style["label"])

    # 스파인 스타일
    for spine in ax.spines.values():
        spine.set_edgecolor(style["spine"]["color"])
        spine.set_alpha(style["spine"]["alpha"])
        spine.set_linewidth(style["spine"]["linewidth"])

# ───────────────────────────────────────────────
# 범례: arrival rate 컬러 + SLO 90% 선
patches = [Patch(facecolor=colors[j], label=arrival_labels[j]) for j in range(len(arrival_rate))]
patches.append(Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='SLO 90%'))

# fig.legend(handles=patches,
#            loc='upper center',
#            ncol=len(patches),
#            bbox_to_anchor=(0.5, 1.12),
#            fontsize=style["legend"]["fontsize"],
#            frameon=False)

# 저장 및 출력
plt.savefig("figures/arrival_rate.jpg", bbox_inches="tight")
plt.savefig("figures/arrival_rate.pdf", bbox_inches="tight")
plt.show()
