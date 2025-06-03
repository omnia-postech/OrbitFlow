import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ───────────────────────────────────────────────
# 1. 설정
TRACE        = "both_dyn_veryhigh"
METHODS      = ["Ours", "Flexgen"]
METHOD_LABS  = ["Ours", "Flexgen"]
METRICS      = ["bs2", "bs4", "bs8"]
METRIC_LABS  = ["bs2", "bs4", "bs8"]
SLO_SCALES   = [3.5, 2.5, 1.5, 1]  # 실제 데이터용
SLO_LABELS   = ["3.5", "2.5", "1.5", "1"]  # categorical label
BASE_DIR     = Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp")

style = {
    "line":   {"linewidth":3, "markersize":10},
    "spine":  {"color":"black", "alpha":0.7, "linewidth":1.5},
    "title":  {"fontsize":32, "pad":8},
    "label":  {"fontsize":30, "labelpad":8},
    "legend": {"fontsize":20},
    "tick":   {"labelsize":24},
}

colors = [
    "#4DA6FF",  # Ours
    "#FF8C69",  # Flexgen
]

# ───────────────────────────────────────────────
# 2. 플롯 초기화
fig, axes = plt.subplots(1, len(METRICS), figsize=(18, 6), sharey=True)

# x축 위치 고정 (등간격)
x_positions = np.arange(len(SLO_LABELS))

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

        ax.plot(x_positions, y_vals,
                label=METHOD_LABS[m_idx],
                color=colors[m_idx],
                marker='o',
                **style["line"])

    # x/y축 및 스타일
    ax.set_title(METRIC_LABS[i], fontsize=style["title"]["fontsize"])
    ax.set_xticks(x_positions)
    ax.set_xticklabels(SLO_LABELS, fontsize=style["tick"]["labelsize"])
    ax.tick_params(axis='y', labelsize=style["tick"]["labelsize"], length=5)
    for spine in ax.spines.values():
        spine.set_edgecolor(style["spine"]["color"])
        spine.set_alpha(style["spine"]["alpha"])
        spine.set_linewidth(style["spine"]["linewidth"])

# 공통 y축 레이블
axes[0].set_ylabel("TBT SLO Attainment (%)", **style["label"])
# 공통 x축 레이블
for ax in axes:
    ax.set_xlabel("SLO Scale", **style["label"])
    ax.set_ylim(0, 100)

# 범례
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="upper center", bbox_to_anchor=(0.5, 1.1),
           ncol=len(METHODS), **style["legend"], frameon=False)

plt.tight_layout()
plt.savefig("figures/6_3_batch_size.jpg", bbox_inches="tight")
plt.savefig("figures/6_3_batch_size.pdf", bbox_inches="tight")