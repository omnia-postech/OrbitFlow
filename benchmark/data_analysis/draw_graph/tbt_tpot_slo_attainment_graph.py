import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import argparse

# ───────────────────────────────────────────────
# CLI 인자
parser = argparse.ArgumentParser()
parser.add_argument("base_dir", nargs="?", type=Path, help="실험 결과가 들어있는 최상위 디렉토리")
parser.add_argument("--output-dir", type=Path, default=Path("./figures"), help="결과 그래프 저장 경로 (기본: ./figures)")
args = parser.parse_args()
base_dir = args.base_dir
output_dir = args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)

# ───────────────────────────────────────────────
# 설정
method_list   = ["NextLayer", "Static1", "Flexgen", "SelectN", "DistNSingle", "OursTD"]
method_labels = ["DeepSpeed", "FlexGen", "FlexGen+", "SLO-aware Offloading", "Dynamic Heuristic", "OrbitFlow"]
pick_x_method = "DistNSingle"
arrival_rate = [1.0, 2.0, 3.0, 4.0, 5.0]
arrival_labels = [str(r) for r in arrival_rate]
cv_rate = 1
slo_scales = [2.5, 1.5, 1]
slo_labels = [str(s) for s in slo_scales]

colors = ["#4DA6FF", "#76C7AE", "#508776", "#9F79C1", "#FFB3BA", "#FF8C69"]
markers = ['o','D', 'p', '>', '^', 's']
font_size = 30
style = {
    "title":  {"fontsize":32, "pad":8},
    "line":   {"linewidth":4,"markersize":16.5},
    "tick":   {"fontsize":30},
    "label":  {"fontsize":font_size,"labelpad":5},
    "legend": {"fontsize":font_size},
    "spine": {
        "color": "black",
        "alpha": 0.7,
        "linestyle": "-",
        "linewidth": 2
    },
    "grid": {
        "color": "gray",
        "linestyle": "-",
        "linewidth": 3,
        "alpha": 0.2
    },
}

# ───────────────────────────────────────────────
# Figure & Layout
fig, axes = plt.subplots(2, len(slo_scales), figsize=(22, 11), sharey=False)
plt.subplots_adjust(left=0.07, right=0.99, top=0.78, bottom=0.12, wspace=0.18, hspace=0.25)

# ───────────────────────────────────────────────
# Plot Loop
for i, sc in enumerate(slo_scales):
    ax_TBT = axes[0][i]
    ax_TPOT = axes[1][i]

    # x축: req/min
    x_tbt = []
    try:
        path_x = base_dir / f"slo{sc}" / pick_x_method / "arrival_summerizev2.csv"
        df_sum = pd.read_csv(path_x)
        x_base = df_sum[(df_sum["slo"] == sc) &
                        (df_sum["arrival_rate"] == min(arrival_rate)) &
                        (df_sum["cv_num"] == cv_rate)]['req_per_sec'].iloc[0]
        x_tbt = [x_base * rate * 60 for rate in arrival_rate]
    except Exception as e:
        print(f"X축 로딩 실패: {e}")
        x_tbt = [0] * len(arrival_rate)

    for m, method in enumerate(method_list):
        y_tbt, y_tpot = [], []

        path = base_dir / f"slo{sc}" / (method[:-2] if method.endswith("TD") else method) / "arrival_summerizev2.csv"
        try:
            df = pd.read_csv(path)
            for rate in arrival_rate:
                sel = df[(df["slo"] == sc) & (df["arrival_rate"] == rate) & (df["cv_num"] == cv_rate)]
                if len(sel) == 1:
                    y_tbt.append(sel["tbt_attainment_with_TD" if method.endswith("TD") else "tbt_attainment_no_TD"].iloc[0])
                    y_tpot.append(sel["tpot_attainment"].iloc[0])
                else:
                    y_tbt.append(np.nan)
                    y_tpot.append(np.nan)
        except Exception as e:
            print(f"파일 로딩 실패: {path}, 오류: {e}")
            y_tbt = [np.nan] * len(arrival_rate)
            y_tpot = [np.nan] * len(arrival_rate)

        # TBT plot
        ax_TBT.plot(x_tbt, y_tbt, marker=markers[m], color=colors[m], label=method_labels[m], **style["line"])
        ax_TBT.plot(x_tbt, y_tbt, linestyle="", marker=markers[m], color=colors[m], markersize=15)

        # TPOT plot
        ax_TPOT.plot(x_tbt, y_tpot, marker=markers[m], color=colors[m], **style["line"])
        ax_TPOT.plot(x_tbt, y_tpot, linestyle="", marker=markers[m], color=colors[m], markersize=15)

    # 공통 스타일
    for ax in (ax_TBT, ax_TPOT):
        ax.set_xlim(0.4, 2.6)
        ax.set_xticks(np.arange(0.5, 3.0, 0.5))
        ax.set_xticklabels([f"{x:.1f}" for x in np.arange(0.5, 3.0, 0.5)], fontsize=style["tick"]["fontsize"])
        ax.set_ylim(-5, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels([str(y) for y in [0, 25, 50, 75, 100]])
        ax.tick_params(axis='both', labelsize=style["tick"]["fontsize"], length=0, pad=5)
        ax.xaxis.grid(True, **style["grid"])
        ax.yaxis.grid(True, **style["grid"])

    if i == 0:
        ax_TBT.set_ylabel("TBT SLO (%)", **style["label"])
        ax_TPOT.set_ylabel("TPOT SLO (%)", **style["label"])

    ax_TBT.set_title(f"SLO Scales = {sc}", **style["title"])
    ax_TPOT.set_xlabel("request/min", **style["label"])

# ───────────────────────────────────────────────
# 스타일 및 범례
for row in axes:
    for ax in row:
        for spine in ax.spines.values():
            spine.set_edgecolor(style["spine"]["color"])
            spine.set_alpha(style["spine"]["alpha"])
            spine.set_linewidth(style["spine"]["linewidth"])

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center',
           ncol=len(method_list),
           bbox_to_anchor=(0.5, 0.93),
           columnspacing=0.9,
           fontsize=style["legend"]["fontsize"],
           frameon=False)

# ───────────────────────────────────────────────
# 저장
plt.savefig(output_dir / "arrival_rate_tbt_tpot.jpg", bbox_inches="tight", dpi=300)
plt.savefig(output_dir / "arrival_rate_tbt_tpot.pdf", bbox_inches="tight")
print(output_dir / "arrival_rate_tbt_tpot.jpg")
