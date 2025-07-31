import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ───────────────────────────────────────────────
# 1. 설정
METHODS      = ["Flexgen", "DistNSingle", "OursTD"]
METHOD_LABS  = ["FlexGen+", "Dynamic Heuristic", "OrbitFlow"]
METRICS      = [32, 128]
METRIC_LABS  = [f"{str(s)}k" for s in METRICS]
SLO_SCALES   = [2.5, 1.5]
SLO_LABELS   = [str(s) for s in SLO_SCALES]
BASE_DIR     = Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp_context_length")

arrival_rate = 2.0
cv_rate      = 1

style = {
    "line":   {"linewidth":3, "markersize":10},
    "spine":  {"color":"black", "alpha":0.7, "linewidth":1.5},
    "title":  {"fontsize":40, "pad":8},
    "label":  {"fontsize":40, "labelpad":8},
    "legend": {"fontsize":40},
    "tick":   {"labelsize":33},
    "grid": {
        "color": "gray",
        "linestyle": "-",
        "linewidth": 3,
        "alpha": 0.2
    },
}

# colors = ["#4DA6FF", "#76C7AE", "#508776", "#9F79C1", "#FFB3BA", "#FF8C69"]

colors = [
    "#76C7AE",  # Sky Blue
    "#FFB3BA",  # Lavender Purple
    "#FF8C69"   # Coral Orange
]

# ───────────────────────────────────────────────
# 2. 플롯 초기화
fig, axes = plt.subplots(1, len(METRICS), figsize=(18, 6.5), sharey=True)

x_positions = np.arange(len(SLO_SCALES))  # [0, 1]
bar_width = 0.30

# ───────────────────────────────────────────────
# 3. 각 subplot (context length 기준)
for i, (length, ax) in enumerate(zip(METRICS, axes)):
    for m_idx, method in enumerate(METHODS):
        y_vals = []

        for sc in SLO_SCALES:
            try:
                if method.endswith("TD"):
                    summary_path = BASE_DIR / f"slo{sc}" / method[:-2] / "arrival_summerizev2.csv"
                else:
                    summary_path = BASE_DIR / f"slo{sc}" / method / "arrival_summerizev2.csv"
                df_sum = pd.read_csv(summary_path)
                sel = df_sum[
                    (df_sum["slo"] == sc) &
                    (df_sum["length"] == length) &
                    (df_sum["arrival_rate"] == arrival_rate) &
                    (df_sum["cv_num"] == cv_rate)
                ]

                if method.endswith("TD"):
                    value = float(sel["tbt_attainment_with_TD"].iloc[0]) if len(sel) == 1 else np.nan
                    # print(f"{length} {df_sum['length']}")
                else:
                    value = float(sel["tbt_attainment_no_TD"].iloc[0]) if len(sel) == 1 else np.nan

                y_vals.append(value)
            except Exception as e:
                print(e)
                y_vals.append(np.nan)

        # Bar 위치 보정
        offset = (m_idx - (len(METHODS) - 1) / 2) * bar_width
        ax.bar(x_positions + offset, y_vals,
               width=bar_width,
               label=METHOD_LABS[m_idx],
               color=colors[m_idx],
               edgecolor="white")

    # 스타일 설정
    ax.set_title(METRIC_LABS[i], fontsize=style["title"]["fontsize"])
    ax.set_xticks(x_positions)
    ax.set_xticklabels(SLO_LABELS, fontsize=style["tick"]["labelsize"])
    ax.tick_params(axis='y', labelsize=style["tick"]["labelsize"], length=5)
    for spine in ax.spines.values():
        spine.set_edgecolor(style["spine"]["color"])
        spine.set_alpha(style["spine"]["alpha"])
        spine.set_linewidth(style["spine"]["linewidth"])

    # ax.axhline(90, color="gray", ls="--", lw=style["line"]["linewidth"], label="")
    ax.set_xlabel("SLO Scale", labelpad=style["label"]["labelpad"], fontsize=style["title"]["fontsize"])
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])

    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0)
    ax.yaxis.grid(True, **style["grid"])
# 공통 y축 레이블
axes[0].set_ylabel("TBT SLO (%)", **style["label"])

# 범례
handles, labels = axes[0].get_legend_handles_labels()

# "SLO 90%"를 마지막으로 이동
# if "SLO 90%" in labels:
#     idx = labels.index("SLO 90%")
#     slo_handle = handles.pop(idx)
#     slo_label = labels.pop(idx)
#     handles.append(slo_handle)
#     labels.append(slo_label)


for spine in ax.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

fig.legend(handles, labels,
           loc="upper center", bbox_to_anchor=(0.5, 1.06),
           ncol=len(METHODS)+1, **style["legend"], 
           columnspacing = 0.9,
           frameon=False)

plt.subplots_adjust(left=0.09, right=0.99,
                    top=0.81, bottom=0.165,
                    wspace=0.1,)
# plt.tight_layout(w_pad=3)
plt.savefig("/home/heelim/vllm/benchmark/data_analysis/figures/6_3_context_length_heelim.jpg")#, bbox_inches="tight")
plt.savefig("/home/heelim/vllm/benchmark/data_analysis/figures/6_3_context_length_heelim.pdf")#, bbox_inches="tight")
