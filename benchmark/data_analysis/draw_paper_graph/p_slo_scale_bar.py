import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ───────────────────────────────────────────────
# 1. 설정
METHODS      = ["Flexgen", "SelectN", "OursTD"]
METHOD_LABS  = ["FlexGen", "SLO-aware Offloading", "OrbitFlow"]

arrival_rate   = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
arrival_labels = [str(r) for r in arrival_rate]

cv_rate = 1

SLO_SCALES   = [2.5, 1.5, 1]
SLO_LABELS   = [f"SLO Scales = {str(s)}" for s in SLO_SCALES]
BASE_DIR     = Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp")

style = {
    "line":   {"linewidth":3, "markersize":10},
    "spine":  {"color":"black", "alpha":0.7, "linewidth":1.5},
    "title":  {"fontsize":40, "pad":8},
    "label":  {"fontsize":40, "labelpad":8},
    "legend": {"fontsize":40},
    "tick":   {"labelsize":27},
}

colors = [
    "#4DA6FF",  # Sky Blue
    "#9F79C1",  # Lavender Purple
    "#FF8C69"   # Coral Orange
]

bar_width = 0.8 / len(METHODS)
positions = np.arange(len(arrival_rate))

# ───────────────────────────────────────────────
# 2. 플롯 초기화 (sharey=False)
fig, axes = plt.subplots(1, len(SLO_SCALES), figsize=(21, 5), sharey=False)
plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.12, wspace=0.13)

# ───────────────────────────────────────────────
# 3. 서브플롯별 데이터 플로팅 (바 차트)
for ax, sc, sc_label in zip(axes, SLO_SCALES, SLO_LABELS):
    for i, (method, label) in enumerate(zip(METHODS, METHOD_LABS)):
        y_vals = []
        
        if method.endswith("TD"):
            summary_path = BASE_DIR / f"slo{sc}" / method[:-2] / "arrival_summerizev2.csv"
        else:
            summary_path = BASE_DIR / f"slo{sc}" / method / "arrival_summerizev2.csv"

        df_sum = pd.read_csv(summary_path)
        for rate in arrival_rate:
            sel = df_sum[(df_sum["slo"] == sc) 
                        & (df_sum["arrival_rate"] == rate)
                        & (df_sum["cv_num"] == cv_rate)
                        ]
            value = float(sel["p95_ratio"].iloc[0]) if len(sel)==1 else np.nan
            y_vals.append(value)


        # ─────────────── 데이터 출력

        offsets = (i - (len(METHODS)-1)/2) * bar_width
        ax.bar(positions + offsets, y_vals,
               width=bar_width,
               label=label,
               color=colors[i],
               edgecolor="white")

    # 제목 & 레이블
    ax.set_title(sc_label, **style["title"])
    ax.set_xlabel("Arrival Rate", **style["label"])
    if ax is axes[0]:
        ax.set_ylabel("SLO", **style["label"])

    # x축 눈금
    ax.set_xticks(positions)
    ax.set_xticklabels(arrival_labels, fontsize=style["tick"]["labelsize"])
    ax.tick_params(axis='x', length=0)

    # Y축 범위 및 눈금 설정
    heights = [bar.get_height() for bar in ax.patches]
    max_y = max(heights) if heights else 0.0
    ax.set_ylim(0, max_y + 0.5)
    ticks = np.arange(0, int(np.floor(max_y + 0.5)) + 1, 1)
    ax.set_yticks(ticks)
    tick_labels = ['' if t == 0 else f"{t}x" for t in ticks]
    ax.set_yticklabels(tick_labels)
    ax.tick_params(axis='y', labelsize=style["tick"]["labelsize"], length=5)

    # 스파인 스타일
    for spine in ax.spines.values():
        spine.set_edgecolor(style["spine"]["color"])
        spine.set_alpha(style["spine"]["alpha"])
        spine.set_linewidth(style["spine"]["linewidth"])

# ───────────────────────────────────────────────
# 4. 범례 (상단 중앙)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="upper center", bbox_to_anchor=(0.5, 1.3),
           ncol=len(METHODS), **style["legend"],
           frameon=False)

# ───────────────────────────────────────────────
# 5. 저장
plt.savefig("figures/6_2_tail_p95_by_slo.jpg", bbox_inches="tight")
plt.savefig("figures/6_2_tail_p95_by_slo.pdf", bbox_inches="tight")