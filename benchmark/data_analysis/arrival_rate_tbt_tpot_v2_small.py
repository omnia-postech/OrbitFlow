import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from pathlib import Path
from matplotlib.patches import Patch
import numpy as np
import os

# ───────────────────────────────────────────────
# 1. 설정 -------------------------------------------------------
method_list   = ["NextLayer", "Flexgen", "SelectN", "DistNSingle", "OursTD",]
method_labels = ["DeepSpeed", "FlexGen+batch", "SLO-aware Offloading", "Dynamic Heuristic", "OrbitFlow"]

arrival_rate   = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
arrival_labels = [str(r) for r in arrival_rate]

cv_rate = 1

slo_scales     = [2.5, 1.5, 1]
slo_labels     = [str(s) for s in slo_scales]

colors = [
    "#4DA6FF",  # Sky Blue
    "#76C7AE",  # Pastel Mint
    # "#508776",  # Pastel Mint Green
    "#9F79C1",  # Lavender Purple
    "#FFB3BA",  # Pastel Pink
    "#FF8C69",   # Coral Orange,

]
markers = ['o','s','^', 'D','p']


font_size = 30
style = {
    "title":  {"fontsize":40, "pad":8},
    "line":   {"linewidth":4,"markersize":15},
    "tick":   {"fontsize":33},
    "label":  {"fontsize":40,"labelpad":5},
    "legend": {"fontsize":40},
    "spine": {
        "color": "black",
        "alpha": 0.7,
        "linestyle": "-",
        "linewidth": 2
    },
}

base_dir = Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp_32k")

# 🔽 Legend용 90% 기준선 객체 생성
# threshold_line = Line2D([0], [0], color='gray', ls='--', lw=style["line"]["linewidth"], label="SLO 90%")

# ───────────────────────────────────────────────
# 2. Figure & GridSpec ---------------------------
N_SLO_SCALE = len(slo_scales)

fig, axes = plt.subplots(2, N_SLO_SCALE,
                         figsize=(22, 12),
                         sharey=True)
plt.subplots_adjust(left=0.07, right=0.99,
                    top=0.78, bottom=0.12,
                    wspace=0.1, hspace=0.25)

# ───────────────────────────────────────────────
# 3. 플롯 루프 -----------------------------------
for i, sc in enumerate(slo_scales):
    ax_TBT = axes[0][i]   # TBT
    ax_TPOT = axes[1][i]       # TPOT

    for m, method in enumerate(method_list):
        # TBT 데이터
        y_tbt = []
        if method.endswith("TD"):
            summary_path = base_dir / f"slo{sc}" / method[:-2] / "arrival_summerizev2.csv"
        else:
            summary_path = base_dir / f"slo{sc}" / method / "arrival_summerizev2.csv"
        try:
            df_sum = pd.read_csv(summary_path)

            for rate in arrival_rate:
                sel = df_sum[(df_sum["slo"] == sc) 
                            & (df_sum["arrival_rate"] == rate)
                            & (df_sum["cv_num"] == cv_rate)
                            ]
                if method.endswith("TD"):
                    value = float(sel["tbt_attainment_with_TD"].iloc[0]) if len(sel)==1 else np.nan
                else:
                    value = float(sel["tbt_attainment_no_TD"].iloc[0]) if len(sel)==1 else np.nan
                y_tbt.append(value)
                # print(f"get {value}")
        except:
            y_tbt = [np.nan] * len(arrival_rate)

        ax_TBT.plot(arrival_labels, y_tbt, **style["line"],
                    marker=markers[m], color=colors[m], label=method_labels[m])

        # y_tbt_zero = [0 if np.isnan(val) else np.nan for val in y_tbt]
        ax_TBT.plot(
            arrival_labels,
            y_tbt,
            linestyle="",
            marker=markers[m],
            color=colors[m],
            markersize=15
        )

        
        # TPOT 데이터
        y_tpot = []
        try:
            df_sum = pd.read_csv(summary_path)
            for rate in arrival_rate:
                sel = df_sum[(df_sum["slo"] == sc) 
                            & (df_sum["arrival_rate"] == rate)
                            & (df_sum["cv_num"] == cv_rate)
                            ]
                y_tpot.append(float(sel["tpot_attainment"].iloc[0]) if len(sel)==1 else np.nan)
        except:
            y_tpot = [np.nan] * len(arrival_rate)

        ax_TPOT.plot(arrival_labels, y_tpot, **style["line"],
                    marker=markers[m], color=colors[m])
        
        # y_tpot_zero = [0 if np.isnan(val) else np.nan for val in y_tpot]
        ax_TPOT.plot(
            arrival_labels,
            y_tpot,
            linestyle="",
            marker=markers[m],
            color=colors[m],
            markersize=15
        )

    # 공통 축 스타일
    for ax in (ax_TBT, ax_TPOT):
        ax.set_xticks(arrival_labels)
        ax.set_xticklabels(arrival_labels)
        ax.set_ylim(-5, 105)
        ax.axhline(90, color="gray", ls="--", lw=style["line"]["linewidth"])
        ax.tick_params(axis='both',
                        labelsize=style["tick"]["fontsize"],
                        length=0, pad=5)

    # y-틱 표시
    # if i == N_SLO_SCALE - 1:
    for ax in (ax_TBT, ax_TPOT):
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels(['0','50','100'])
        # ax.yaxis.tick_right()
        # ax.yaxis.set_label_position("right")
    # else:
    #     ax_TBT.set_yticks([]); ax_TPOT.set_yticks([])

    # row-label
    if i == 0:
        ax_TBT.set_ylabel("TBT SLO (%)", **style["label"])
        ax_TPOT.set_ylabel("TPOT SLO (%)", **style["label"])

    # TBT y-label (마지막 열의 오른쪽)
    # if c == N_TRACE - 1:
        # ax_R.set_ylabel("SLO attainment (%)", fontsize=30,
        #                 labelpad=style["label"]["labelpad"],
        #                 rotation=270)
        # ax_R.yaxis.set_label_coords(1.28, 0.5)

    # x-label
    ax_TBT.set_xlabel("", **style["label"])
    ax_TPOT.set_xlabel("Arrival Rate", **style["label"])

    ax_TBT.set_title(f"SLO Scales = {sc}", **style["title"])

    # Trace 라벨
    # if r == N_METRIC - 1:
    #     for ax, tl in [(ax_L, trace_labels[c]), (ax_R, trace_labels[c])]:
    #         ax.text(0.5, -0.45, tl, transform=ax.transAxes,
    #                 ha='center', va='top', fontsize=style["label"]["fontsize"] + 10)

# ───────────────────────────────────────────────
# 스파인 스타일
for row in axes:
    for ax in row:
        for spine in ax.spines.values():
            spine.set_edgecolor(style["spine"]["color"])
            spine.set_alpha(style["spine"]["alpha"])
            spine.set_linewidth(style["spine"]["linewidth"])

# 중앙 구분선 및 범례
# fig.add_artist(Line2D([0.52,0.52], [0.1,0.9], transform=fig.transFigure,
#                       color="black", lw=2))

handles, labels = axes[0][0].get_legend_handles_labels()
# handles.append(threshold_line)
# labels.append("SLO 90%")

fig.legend(handles, labels, loc='upper center', 
           ncol=3, 
           bbox_to_anchor=(0.5, 1.00),
           columnspacing=0.9,
           fontsize=style["legend"]["fontsize"], frameon=False)

# patches = [Patch(facecolor=colors[j], label=method[j]) for j in range(len(method))]
# patches.append(Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='SLO 90%'))

# fig.legend(handles=patches,
#            loc='upper center',
#            ncol=len(patches),
#            bbox_to_anchor=(0.5, 0.9),
#            fontsize=style["legend"]["fontsize"],
#            frameon=False)

# 저장
plt.savefig("figures/arrival_rate_tbt_tpot.jpg", bbox_inches="tight", dpi=300)
plt.savefig("figures/arrival_rate_tbt_tpot.pdf", bbox_inches="tight")
