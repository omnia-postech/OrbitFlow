import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from pathlib import Path
import numpy as np

# ───────────────────────────────────────────────
# 1. 설정 -------------------------------------------------------
trace_list   = ["both_static", "batch_dyn", "token_dyn", "both_dyn"]
trace_labels = ["(a) Both Static", "(b) Batch dynamic", "(c) Token dynamic", "(d) Both dynamic"]

method_list   = [
    # "NoPrefetch",
    "Flexgen", "SelectN", 
                 "Ours"
                 ]
method_labels = [
    # "No Prefetch",
      "Flexgen", "Placeholder(SelectN)", 
                 "Ours"
                 ]

metric_list   = ["low","mid","high", "veryhigh"]
metric_labels = ["Low","Mid","High", "Very High"]

slo_scales  = [10, 4.5, 3.5, 2.5, 1.5, 1]
slo_labels  = [str(s) for s in slo_scales]

colors = [
    "#4DA6FF",  # Sky Blue
    # "#3CC58F",  # Mint Green
    "#9F79C1",  # Lavender Purple
    "#FF8C69"   # Coral Orange
]
markers = ['o','s','^',
        #    'D',
           'P']

font_size = 35
style = {
    "line":   {"linewidth":4,"markersize":15},
    "tick":   {"fontsize":30},
    "label":  {"fontsize":font_size,"labelpad":5},
    "legend": {"fontsize":font_size},
    "spine": {
        "color": "black",
        "alpha": 0.7,
        "linestyle": "-",
        "linewidth": 2
    },
}

# ───────────────────────────────────────────────
# 2. Figure & GridSpec ---------------------------
N_TRACE, N_METRIC = len(trace_list), len(metric_list)
fig = plt.figure(figsize=(55, 5.5 * N_METRIC))
gs  = GridSpec(
    nrows=N_METRIC, ncols=N_TRACE*2 + 1,
    width_ratios=[2]*N_TRACE + [0.5] + [2]*N_TRACE,
    wspace=0.1, hspace=0.3
)

# 축 생성
axes = [[None]*(N_TRACE*2) for _ in range(N_METRIC)]
for i in range(N_METRIC):
    for j in range(N_TRACE):
        axes[i][2*j]   = fig.add_subplot(gs[i, j])               # TPOT
        axes[i][2*j+1] = fig.add_subplot(gs[i, j+N_TRACE+1])     # TBT

# ───────────────────────────────────────────────
# 3. 플롯 루프 -----------------------------------
for r, metric in enumerate(metric_list):
    for c, trace in enumerate(trace_list):
        ax_L = axes[r][2*c]       # TPOT
        ax_R = axes[r][2*c + 1]   # TBT

        for m, method in enumerate(method_list):
            # TPOT 데이터
            yL = []
            for sc in slo_scales:
                try: 
                    summary_path = Path(
                        f"/home/heelim/vllm/outputs/benchmark/paper_main_exp/"
                        f"slo{sc}/{method}/summerize.csv"
                    )
                    summary_df = pd.read_csv(summary_path)
                    row = summary_df[
                        (summary_df["slo"] == sc) &
                        (summary_df["method"] == method) &
                        (summary_df["trace"] == trace) &
                        (summary_df["metric"] == metric)
                    ]
                    if len(row) == 1:
                        yL.append(float(row["tpot_attainment"].iloc[0]))
                    else:
                        yL.append(np.nan)
                except Exception as e:
                    print(f"[No File] {e}")
                    yL.append(np.nan)
            ax_L.plot(slo_labels, yL, **style["line"],
                      marker=markers[m], color=colors[m], label=method_labels[m])

            # ———— 마커(점)만 찍기: yL이 np.nan인 위치만 골라서 Y=0 에 점을 찍음
            #     1) yL_zero: yL이 NaN인 곳은 0, 그렇지 않은 곳은 NaN으로 두기
            yL_zero = [0 if np.isnan(val) else np.nan for val in yL]
            #     2) marker만 찍으므로 linestyle="" (또는 'None') 지정
            ax_L.plot(
                slo_labels,
                yL_zero,
                linestyle="",            # 선은 그리지 않고, 마커만 표시
                marker=markers[m],
                color=colors[m],
                markersize=15
            )

            # TBT 데이터
            yR = []
            for sc in slo_scales:
                try:
                    summary_path = Path(
                        f"/home/heelim/vllm/outputs/benchmark/paper_main_exp/"
                        f"slo{sc}/{method}/summerize.csv"
                    )
                    summary_df = pd.read_csv(summary_path)
                    row = summary_df[
                        (summary_df["slo"] == sc) &
                        (summary_df["method"] == method) &
                        (summary_df["trace"] == trace) &
                        (summary_df["metric"] == metric)
                    ]
                    if len(row) == 1:
                        yR.append(float(row["tbt_attainment"].iloc[0]))
                    else:
                        yR.append(np.nan)
                except:
                    yR.append(np.nan)
            ax_R.plot(slo_labels, yR, **style["line"],
                      marker=markers[m], color=colors[m])
            
            yR_zero = [0 if np.isnan(val) else np.nan for val in yR]
            ax_R.plot(
                slo_labels,
                yR_zero,
                linestyle="",
                marker=markers[m],
                color=colors[m],
                markersize=15
            )
            

        # 공통 축 스타일
        for ax in (ax_L, ax_R):
            ax.set_xticks(slo_labels)
            ax.set_xticklabels(slo_labels)
            ax.set_ylim(-5, 105)
            ax.axhline(90, color="gray", ls="--", lw=style["line"]["linewidth"])
            ax.tick_params(axis='both',
                           labelsize=style["tick"]["fontsize"],
                           length=0)

        # y-틱 표시
        if c == N_TRACE - 1:
            for ax in (ax_L, ax_R):
                ax.set_yticks([0, 50, 100])
                ax.set_yticklabels(['0','50','100'])
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
        else:
            ax_L.set_yticks([]); ax_R.set_yticks([])

        # row-label
        if c == 0:
            ax_L.set_ylabel(metric_labels[r], **style["label"])

        # TBT y-label (마지막 열의 오른쪽)
        if c == N_TRACE - 1:
            ax_R.set_ylabel("SLO attainment (%)", fontsize=30,
                            labelpad=style["label"]["labelpad"],
                            rotation=270)
            ax_R.yaxis.set_label_coords(1.28, 0.5)

        # x-label
        if r == N_METRIC - 1:
            ax_L.set_xlabel("SLO Scale", **style["label"])
            ax_R.set_xlabel("SLO Scale", **style["label"])
        else:
            ax_L.set_xlabel(""); ax_R.set_xlabel("")

        # Trace 라벨
        if r == N_METRIC - 1:
            for ax, tl in [(ax_L, trace_labels[c]), (ax_R, trace_labels[c])]:
                ax.text(0.5, -0.45, tl, transform=ax.transAxes,
                        ha='center', va='top', fontsize=style["label"]["fontsize"])

# 스파인 스타일
for row in axes:
    for ax in row:
        for spine in ax.spines.values():
            spine.set_edgecolor(style["spine"]["color"])
            spine.set_alpha(style["spine"]["alpha"])
            spine.set_linewidth(style["spine"]["linewidth"])

# 중앙 구분선 및 범례
fig.add_artist(Line2D([0.52,0.52], [-0.07,0.9], transform=fig.transFigure,
                      color="black", lw=2))
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(method_list),
           fontsize=style["legend"]["fontsize"], frameon=False)

# 공통 텍스트
fig.text(0.1, 0.5, "Memory Pressure", va='center',
         rotation='vertical', fontsize=font_size)
fig.text(0.31, -0.065, "TPOT", ha='center', fontsize=43, weight='bold')
fig.text(0.715, -0.065, "TBT", ha='center', fontsize=43, weight='bold')

# 저장
plt.savefig("figures/6_2_tpot_tbt_combined.jpg", bbox_inches="tight", dpi=300)
plt.savefig("figures/6_2_tpot_tbt_combined.pdf", bbox_inches="tight")
