import pandas as pd, matplotlib.pyplot as plt, numpy as np, ast
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ─── 공통 설정 ──────────────────────────────────────────────────
trace_list   = ["test_fit_static_0","test_shortshort_enough",
                "test_shortlong_less","test_shortlong_enough"]
trace_labels = ["(a) Trace 1","(b) Trace 2","(c) Trace 3","(d) Trace 4"]

method_list   = ["Flexgen","DeepSpeed","SelectN","NoPrefetch","Ours"]
method_labels = ["Flexgen","DeepSpeed","Placeholder(SelectN)",
                 "No Prefetch","Ours"]

metric_list   = ["Low","Mid","High"]
metric_labels = ["Low","Mid","High"]

slo_scales  = [1.5,1.4,1.3,1.2,1.1,1.0]
slo_labels  = [str(s) for s in slo_scales]   # 이제 뒤집지 않고 그대로 사용
print(slo_labels)

colors  = ["#84C8F4","#C59FDB","#7CD6A4","#63D0C2","#E05A4F"]
markers = ['o','o','o','o','*']

font_size = 22
style = {
    "line":   {"linewidth":3,"markersize":10},
    "tick":   {"fontsize":18},
    "label":  {"fontsize":font_size,"labelpad":5},
    "legend": {"fontsize":font_size},
}

# ───────────────────────────────────────────────
# (load_metrics, slo_tpot, slo_tbt 함수 생략 — 그대로)

# ───────────────────────────────────────────────
N_TRACE, N_METRIC = len(trace_list), len(metric_list)
fig = plt.figure(figsize=(48, 4 * N_METRIC))
gs  = GridSpec(
    nrows=N_METRIC, ncols=N_TRACE*2 + 1,
    width_ratios=[2]*N_TRACE + [0.05] + [2]*N_TRACE,
    wspace=0.25, hspace=0.3
)

axes = [[None]*(N_TRACE*2) for _ in range(N_METRIC)]
for i in range(N_METRIC):
    for j in range(N_TRACE):
        axes[i][2*j]   = fig.add_subplot(gs[i, j])               # TPOT
        axes[i][2*j+1] = fig.add_subplot(gs[i, j+N_TRACE+1])     # TBT

# ───────────────────────────────────────────────
for r, metric in enumerate(metric_list):
    for c, trace in enumerate(trace_list):
        ax_L = axes[r][2*c]
        ax_R = axes[r][2*c + 1]

        # (데이터 로드 및 plot 부분 생략 — 그대로)

        # ── 축 공통 서식 ──────────────────────────
        for ax in (ax_L, ax_R):
            ax.set_xticks(slo_scales)
            ax.set_xticklabels(slo_labels,
                               fontsize=style["tick"]["fontsize"])
            # print(slo_scales)
            # print(slo_labels)

            # 왼쪽=1.5, 오른쪽=1.0 으로 축을 뒤집음
            ax.set_xlim(slo_scales[0], slo_scales[-1])
            ax.set_ylim(0, 100)   # y-limit 확실히 설정
            ax.axhline(90, color="gray", ls="--", lw=1.3, alpha=.6)
            ax.tick_params(axis='both', labelsize=style["tick"]["fontsize"], length=0)

        # ── y-tick 설정 ─────────────────────────────
        # TPOT: 첫 번째 열만 왼쪽 y-tick 표시
        if c == 0:
            ax_L.set_yticks([0, 50, 100])
            ax_L.set_yticklabels(['0', '50', '100'])
        else:
            ax_L.set_yticks([])
            ax_L.set_yticklabels([])

        # TBT: 마지막 열만 오른쪽 y-tick 표시, 왼쪽 레이블 제거
        if c == N_TRACE - 1:
            ax_R.set_yticks([0, 50, 100])
            ax_R.set_yticklabels(['0', '50', '100'])
            ax_R.yaxis.set_ticks_position("right")
            ax_R.tick_params(axis='y', labelleft=False, labelright=True)
        else:
            ax_R.set_yticks([])
            ax_R.set_yticklabels([])
            ax_R.tick_params(axis='y', labelleft=False, labelright=False)

        # ── row-label 왼쪽 첫 패널 ───────────────
        if c == 0:
            ax_L.set_ylabel(metric_labels[r], **style["label"])

        # ── SLO attainment(%) 오른쪽 y-라벨 (Trace 4만) ─
        if c == N_TRACE - 1:
            ax_R.set_ylabel("SLO attainment (%)", **style["label"])
            ax_R.yaxis.set_label_position("right")

        # ── Trace 라벨 (아랫쪽) ────────────────────
        if r == N_METRIC - 1:
            for ax, idx in [(ax_L, c), (ax_R, c)]:
                ax.text(0.5, -0.25, trace_labels[idx],
                        transform=ax.transAxes,
                        ha='center', va='top',
                        fontsize=style["label"]["fontsize"])

# (나머지 범례, 중앙선, 저장 부분 그대로)


# ───────────────────────────────────────────────
# 5. 중앙 세로선, 범례, 공통 텍스트 ---------------
fig.add_artist(Line2D([0.51, 0.51], [0.0, 0.92],
                      transform=fig.transFigure, color="black", lw=2))

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(method_list),
           fontsize=style["legend"]["fontsize"], frameon=False)

fig.text(0.085, 0.5, "Memory Pressure", va='center',
         rotation='vertical', fontsize=font_size)

fig.text(0.31, 0.015, "TPOT", ha='center',
         fontsize=style["label"]["fontsize"], weight='bold')
fig.text(0.715, 0.015, "TBT", ha='center',
         fontsize=style["label"]["fontsize"], weight='bold')

# ───────────────────────────────────────────────
# 6. 저장 ---------------------------------------
plt.savefig("figures/tpot_tbt_combined.jpg", dpi=300, bbox_inches="tight")
# plt.savefig("figures/tpot_tbt_combined.pdf", format='pdf', bbox_inches="tight")