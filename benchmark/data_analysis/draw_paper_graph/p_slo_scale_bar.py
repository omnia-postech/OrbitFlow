import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ───────────────────────────────────────────────
# 1. 설정
TRACE        = "both_dyn"
METHODS      = ["Flexgen", "SelectN", "Ours"]
METHOD_LABS  = ["Flexgen", "SelectN", "Ours"]
METRICS      = ["low", "mid", "high", "veryhigh"]
METRIC_LABS  = ["Low", "Mid", "High", "Very High"]
SLO_SCALES   = [3.5, 2.5, 1.5]
SLO_LABELS   = [str(s) for s in SLO_SCALES]
BASE_DIR     = Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp")

style = {
    "line":   {"linewidth":3, "markersize":10},
    "spine":  {"color":"black", "alpha":0.7, "linewidth":1.5},
    "title":  {"fontsize":32, "pad":8},
    "label":  {"fontsize":30, "labelpad":8},
    "legend": {"fontsize":32},
    "tick":   {"labelsize":28},
}

colors = [
    "#4DA6FF",  # Sky Blue
    # "#3CC58F",  # Mint Green
    "#9F79C1",  # Lavender Purple
    "#FF8C69"   # Coral Orange
]


bar_width = 0.8 / len(METHODS)
positions = np.arange(len(SLO_SCALES))

# ───────────────────────────────────────────────
# 2. 플롯 초기화 (sharey=False)
fig, axes = plt.subplots(1, len(METRICS), figsize=(22, 5), sharey=False)
plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.12, wspace=0.2)

# ───────────────────────────────────────────────
# 3. 서브플롯별 데이터 플로팅 (바 차트)
for ax, metric, title in zip(axes, METRICS, METRIC_LABS):
    for i, (method, label) in enumerate(zip(METHODS, METHOD_LABS)):
        y_vals = []
        for sc in SLO_SCALES:
            summary_path = BASE_DIR / f"slo{sc}" / method / "summerize.csv"
            p99 = 0.0
            if summary_path.exists():
                df_sum = pd.read_csv(summary_path)
                sel = df_sum[
                    (df_sum["slo"]    == sc) &
                    (df_sum["method"] == method) &
                    (df_sum["trace"]  == TRACE) &
                    (df_sum["metric"] == metric)
                ]
                if len(sel) == 1:
                    p99 = float(sel["p99_ratio"].fillna(0).iloc[0])
                else :
                    print(f"[Warning] Missing {summary_path}", file=sys.stderr)
            else:
                print(f"[Warning] Missing {summary_path}", file=sys.stderr)
            y_vals.append(p99)

        # x 위치: 클러스터 중심에서 offset
        offsets = (i - (len(METHODS)-1)/2) * bar_width
        ax.bar(positions + offsets, y_vals,
               width=bar_width,
               label=label,
               color=colors[i],
               edgecolor="white")

    # 제목 & 레이블
    ax.set_title(title, **style["title"])
    ax.set_xlabel("SLO Scale", **style["label"])
    if ax is axes[0]:
        ax.set_ylabel("SLO", **style["label"])

    # x축 눈금
    ax.set_xticks(positions)
    ax.set_xticklabels(SLO_LABELS, fontsize=style["tick"]["labelsize"])
    ax.tick_params(axis='x', length=0)

    # Compute max over all plotted lines
    heights = [bar.get_height() for bar in ax.patches]
    max_y = max(heights) if heights else 0.0
    # 1) Set limit from 0 to max_y + 0.1
    ax.set_ylim(0, max_y + 0.5)

    # 2) Create integer ticks 0,1,2,… up to ceil(max_y + 0.1)
    ticks = np.arange(0, int(np.floor(max_y + 0.5)) + 1, 1)
    ax.set_yticks(ticks)

    # 3) Hide the '0' label, and format the rest as '1x', '2x', …
    tick_labels = ['' if t == 0 else f"{t}x" for t in ticks]
    ax.set_yticklabels(tick_labels)

    # 4) Ensure label font size via tick_params
    ax.tick_params(axis='y', labelsize=style["tick"]["labelsize"], length=5)


    # 스파인 스타일
    for spine in ax.spines.values():
        spine.set_edgecolor(style["spine"]["color"])
        spine.set_alpha(style["spine"]["alpha"])
        spine.set_linewidth(style["spine"]["linewidth"])

# ───────────────────────────────────────────────
# 4. 범례 (상단 중앙으로 올리기)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="upper center", bbox_to_anchor=(0.5, 1.2),
           ncol=len(METHODS), **style["legend"],
           frameon=False
           )

fig.suptitle("P99", fontsize=35, y=1.25)

# ───────────────────────────────────────────────
# 5. 저장
# out_dir = Path("figures")
# out_dir.mkdir(exist_ok=True, parents=True)
plt.savefig("figures/6_2_tail_p99_by_slo.jpg", bbox_inches="tight")
plt.savefig("figures/6_2_tail_p99_by_slo.pdf", bbox_inches="tight")

