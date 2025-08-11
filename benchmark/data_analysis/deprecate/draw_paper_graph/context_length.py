import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ───────────────────────────────────────────────
# 1. 설정
TRACE        = "both_dyn_veryhigh"
METHODS      = ["Flexgen", "Ours"]
METHOD_LABS  = ["FlexGen", "OrbitFlow"]
METRICS      = ["8k", "32k", "128k"]
METRIC_LABS  = ["8k", "32k", "128k"]
SLO_SCALES   = [2.5, 1.5]
SLO_LABELS   = [str(s) for s in SLO_SCALES]
BASE_DIR     = Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp")

style = {
    "line":   {"linewidth":3, "markersize":10},
    "spine":  {"color":"black", "alpha":0.7, "linewidth":1.5},
    "title":  {"fontsize":40, "pad":8},
    "label":  {"fontsize":25, "labelpad":8},
    "legend": {"fontsize":40},
    "tick":   {"labelsize":33},
}

colors = [
    "#4DA6FF",  # Sky Blue
    "#FF8C69"   # Coral Orange
]

# ───────────────────────────────────────────────
# 2. 플롯 초기화
fig, axes = plt.subplots(1, len(METRICS), figsize=(18, 5.5), sharey=True)

# x축 위치 고정 (등간격)
x_positions = np.arange(len(SLO_LABELS))
bar_width = 0.30

# ───────────────────────────────────────────────
# 3. 각 subplot
for i, (metric, ax) in enumerate(zip(METRICS, axes)):
    for m_idx, method in enumerate(METHODS):
        y_vals = []
        for slo in SLO_SCALES:
            summary_path = BASE_DIR / f"slo{slo}" / method / "summerize.csv"
            tbt_attainment = 0.0
            if summary_path.exists():
                df_sum = pd.read_csv(summary_path)
                sel = df_sum[
                    (df_sum["slo"] == slo) &
                    (df_sum["method"] == method) &
                    (df_sum["trace"] == TRACE) &
                    (df_sum["metric"] == metric)
                ]
                if len(sel) == 1:
                    tbt_attainment = float(sel["tbt_attainment"].iloc[0])
                else:
                    print(f"[Warning] Entry not found in {summary_path}", file=sys.stderr)
            else:
                print(f"[Warning] File not found: {summary_path}", file=sys.stderr)
            y_vals.append(tbt_attainment)

        # 추가: 그래프에 들어간 값 출력
        print(f"[{method}] trace={TRACE}, metric={metric}: {y_vals}")

        offset = (m_idx - (len(METHODS) - 1) / 2) * bar_width
        ax.bar(x_positions + offset, y_vals,
               width=bar_width,
               label=METHOD_LABS[m_idx],
               color=colors[m_idx],
               edgecolor="white")

    # x/y축 및 스타일
    ax.set_title(METRIC_LABS[i], fontsize=style["title"]["fontsize"])
    ax.set_xticks(x_positions)
    ax.set_xticklabels(SLO_LABELS, fontsize=style["tick"]["labelsize"])
    ax.tick_params(axis='y', labelsize=style["tick"]["labelsize"], length=5)
    for spine in ax.spines.values():
        spine.set_edgecolor(style["spine"]["color"])
        spine.set_alpha(style["spine"]["alpha"])
        spine.set_linewidth(style["spine"]["linewidth"])

    ax.axhline(90, color="gray", ls="--", lw=style["line"]["linewidth"], label="SLO 90%")

# 공통 y축 레이블
axes[0].set_ylabel("TBT SLO Attainment (%)", **style["label"])
# 공통 x축 레이블
for ax in axes:
    ax.set_xlabel("SLO Scale", labelpad = style["label"]["labelpad"], fontsize = 40)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 50, 100])

# 범례
handles, labels = axes[0].get_legend_handles_labels()

# "SLO 90%"를 마지막으로 이동
if "SLO 90%" in labels:
    idx = labels.index("SLO 90%")
    slo_handle = handles.pop(idx)
    slo_label = labels.pop(idx)
    handles.append(slo_handle)
    labels.append(slo_label)

fig.legend(handles, labels,
           loc="upper center", bbox_to_anchor=(0.5, 1.2),
           ncol=len(METHODS)+1, **style["legend"], frameon=False)

plt.tight_layout(w_pad=3)
plt.savefig("figures/6_3_context_length.jpg", bbox_inches="tight")
plt.savefig("figures/6_3_context_length.pdf", bbox_inches="tight")