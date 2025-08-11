import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# ───────────────────────────────────────────────
# 1. 설정
METHODS      = ["Flexgen_orig", "Flexgen", "SelectN", "DistNSingle", "OursTD"]
METHOD_LABS  = ["FlexGen", "FlexGen+batch", "SLO-aware Offloading", "Dynamic Heuristic", "OrbitFlow"]

arrival_rate   = [1.0, 2.0, 3.0, 4.0, 5.0]
arrival_labels = [str(r) for r in arrival_rate]

cv_rate = 1
SLO_SCALE = 1  # 고정된 SLO scale 사용

parser = argparse.ArgumentParser()
parser.add_argument(
    "base_dir",
    nargs="?",
    type=Path,
    default=Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp_32k"),
    help="실험 결과가 들어있는 최상위 디렉토리 (기본값 사용 시 생략)"
)
args = parser.parse_args()
BASE_DIR = args.base_dir

style = {
    "line":   {"linewidth": 3, "markersize": 10},
    "spine":  {"color": "black", "alpha": 0.7, "linewidth": 1.5},
    "title":  {"fontsize": 30, "pad": 8},
    "label":  {"fontsize": 30, "labelpad": 8},
    "legend": {"fontsize": 30},
    "tick":   {"labelsize": 27},
    "grid": {
        "color": "gray",
        "linestyle": "--",
        "linewidth": 3,
        "alpha": 0.7
    },
}

colors = ["#76C7AE", "#508776", "#9F79C1", "#FFB3BA", "#FF8C69"]
markers = ['o', 's', '^', 'D', 'p']

# ───────────────────────────────────────────────
# 2. 플롯 초기화
fig, ax = plt.subplots(1, 1, figsize=(15, 5), sharey=False)
plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.12, wspace=0.13)

# ───────────────────────────────────────────────
# 3. 각 method에 대해 선 그래프 그리기
for i, (method, label) in enumerate(zip(METHODS, METHOD_LABS)):
    y_vals = []
    for ar in arrival_rate:
        if method.endswith("TD"):
            summary_path = BASE_DIR / f"slo{SLO_SCALE}" / method[:-2] / "arrival_summerizev2.csv"
        else:
            summary_path = BASE_DIR / f"slo{SLO_SCALE}" / method / "arrival_summerizev2.csv"

        try:
            df_sum = pd.read_csv(summary_path)
            sel = df_sum[(df_sum["slo"] == SLO_SCALE)
                         & (df_sum["arrival_rate"] == ar)
                         & (df_sum["cv_num"] == cv_rate)]
            value = float(sel["p95_ratio"].iloc[0]) if len(sel) == 1 else np.nan
            y_vals.append(value)
        except Exception as e:
            y_vals.append(np.nan)

    ax.plot(arrival_rate, y_vals,
            label=label,
            color=colors[i],
            marker=markers[i],
            linewidth=style["line"]["linewidth"],
            markersize=style["line"]["markersize"])

# ───────────────────────────────────────────────
# 4. 축 및 눈금 설정
ax.set_xlabel("Arrival Rate", **style["label"])
ax.set_ylabel("P95 TBT (ms)", **style["label"])
ax.set_xticks(arrival_rate)
ax.set_xticklabels(arrival_labels, fontsize=style["tick"]["labelsize"])
ax.tick_params(axis='x', length=0)

# ───── y축 눈금: 정수만, 최대 4개, 0은 제외
heights = [pt for line in ax.lines for pt in line.get_ydata() if not np.isnan(pt)]
max_y = max(heights) if heights else 1.0

max_tick_count = 4
raw_step = max_y / max_tick_count

def round_step(x):
    if x <= 1: return 1
    elif x <= 2: return 2
    elif x <= 5: return 5
    elif x <= 10: return 10
    else: return int(np.ceil(x / 10.0)) * 10

step = round_step(raw_step)
ticks = list(range(step, int(np.ceil(max_y)) + 1, step))  # 0은 제외
ax.set_yticks(ticks)
ax.set_yticklabels([f"{t * 1000:.0f}" for t in ticks], fontsize=style["tick"]["labelsize"])

# ───── 스파인 및 격자
ax.yaxis.grid(True, **style["grid"])
for spine in ax.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# ───────────────────────────────────────────────
# 5. 범례
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels,
           loc="upper center", bbox_to_anchor=(0.5, 1.3),
           ncol=3, **style["legend"],
           frameon=False)

# ───────────────────────────────────────────────
# 6. 저장
Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/6_2_tail_p95_by_arrival.jpg", bbox_inches="tight")
plt.savefig("figures/6_2_tail_p95_by_arrival.pdf", bbox_inches="tight")
