import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ───────────────────────────────────────────────
# 1. 설정
TRACE        = "both_dyn"
SC           = 2.5  # single SLO scale for tail percentiles
METHODS      = ["NoPrefetch", "Flexgen", "SelectN", "Ours"]
METHOD_LABS  = ["No Prefetch", "Flexgen", "SelectN", "Ours"]
METRICS      = ["low", "mid", "high", "veryhigh"]
METRIC_LABS  = ["Low", "Mid", "High", "Very High"]
PCT_KEYS     = ["p90_ratio", "p95_ratio", "p99_ratio"]
PCT_LABELS   = ["p90", "p95", "p99"]
x_positions  = list(range(len(PCT_KEYS)))
base_dir     = Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp")

style = {
    "line":  {"linewidth":3, "markersize":10},
    "spine": {"color":"black", "alpha":0.7, "linewidth":1.5},
    "title": {"fontsize":22, "weight":"bold", "pad":8},
    "label": {"fontsize":20, "labelpad":8},
    "tick":  {"labelsize":18},
}

colors  = ["#84C8F4","#C59FDB","#7CD6A4","#E05A4F"]
markers = ['o','s','^','P']

# ───────────────────────────────────────────────
# 2. 플롯 초기화 (sharey=False로 y축 독립)
fig, axes = plt.subplots(1, len(METRICS), figsize=(22, 5), sharey=False)
plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.12, wspace=0.24)

# ───────────────────────────────────────────────
# 3. 서브플롯별 데이터 플로팅
for ax, metric, mlabel in zip(axes, METRICS, METRIC_LABS):
    for method, mlab, color, marker in zip(METHODS, METHOD_LABS, colors, markers):
        summary_path = base_dir / f"slo{SC}" / method / "summerize.csv"
        print(summary_path)
        ratios = [0.0] * len(PCT_KEYS)
        if summary_path.exists():
            df_sum = pd.read_csv(summary_path)
            sel = df_sum[
                (df_sum["slo"]    == SC) &
                (df_sum["method"] == method) &
                (df_sum["trace"]  == TRACE) &
                (df_sum["metric"] == metric)
            ]
            if len(sel) == 1:
                ratios = [float(sel[k].fillna(0).iloc[0]) for k in PCT_KEYS]
                check = [float(sel[k].iloc[0]) for k in PCT_KEYS]

                # print(f"{method} : {check}")
            else :
                ratios = [0,0,0]    
        else:
            print(f"[Warning] Missing summary: {summary_path}", file=sys.stderr)
            ratios = [0,0,0]

        ax.plot(x_positions, ratios,
                label=mlab, color=color,
                marker=marker, **style["line"])
    
    # 설정: 제목, 눈금, 레이블
    ax.set_title(mlabel, **style["title"])
    ax.set_xticks(x_positions)
    ax.set_xticklabels(PCT_LABELS, fontsize = style["tick"]["labelsize"])
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=5, labelsize=style["tick"]["labelsize"])

    # ax.set_ylim(0, 5)

    # # y축 범위 설정: 각 서브플롯 독립적으로
    # all_y = [line.get_ydata() for line in ax.get_lines()]
    # ymax = max(np.max(y) for y in all_y) if all_y else 1.0
    # ax.set_ylim(0, ymax + 0.1)
    ax.set_ylim(0, 4)
    # ax.set_yticks(np.linspace(0, ymax + 0.1, 5))

    if ax is axes[0]:
        ax.set_ylabel("TBT/SLO Ratio", **style["label"])
    if ax is axes[-1]:
        ax.set_xlabel("Percentile", **style["label"])

    # 스파인 스타일
    for spine in ax.spines.values():
        spine.set_edgecolor(style["spine"]["color"])
        spine.set_alpha(style["spine"]["alpha"])
        spine.set_linewidth(style["spine"]["linewidth"])

# ───────────────────────────────────────────────
# 4. 범례 (더 위로 이동)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="upper center", bbox_to_anchor=(0.5, 1.2),
           ncol=len(METHODS), fontsize=style["label"]["fontsize"], frameon=False)

# ───────────────────────────────────────────────
# 5. 저장

plt.savefig("./figures/6_2_tail_tbt.jpg", bbox_inches="tight")
plt.savefig("./figures/6_2_tail_tbt.pdf", bbox_inches="tight")
