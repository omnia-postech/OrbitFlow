import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# ───────────────────────────────────────────────
# CLI: 인자 처리
parser = argparse.ArgumentParser(description="Design validation plot")
parser.add_argument("base_dir_uniform", type=Path, help="Base directory for UniformSolver-related results")
parser.add_argument("base_dir_token", type=Path, help="Base directory for Token Deposit comparison")
parser.add_argument("--output-dir", type=Path, default=Path("./figures"), help="Output directory to save plots (default: ./figures)")
args = parser.parse_args()
base_dir_uniform = args.base_dir_uniform
base_dir_token = args.base_dir_token
output_dir = args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)

# ───────────────────────────────────────────────
# 공통 설정
arrival_rate = 2.0
cv_rate = 1
slo_scales = [2.5, 1]
slo_labels = [str(s) for s in slo_scales]

style = {
    "line": {"linewidth": 3, "markersize": 10},
    "spine": {"color": "black", "alpha": 0.7, "linewidth": 1.5},
    "title": {"fontsize": 30, "pad": 8},
    "label": {"fontsize": 30, "labelpad": 10},
    "legend": {
        "fontsize": 30,
        "loc": 'upper center',
        "ncol": 2,
        "bbox_to_anchor": (0.5, 1.45),
        "handletextpad": 0.4,
        "columnspacing": 0.9,
        "frameon": False
    },
    "tick": {"labelsize": 30},
    "grid": {
        "color": "gray",
        "linestyle": "-",
        "linewidth": 3,
        "alpha": 0.2
    },
}

# ───────────────────────────────────────────────
# 플롯 초기화
fig, [ax_ours, ax_base] = plt.subplots(1, 2, figsize=(20, 7), sharey=False)
plt.subplots_adjust(left=0.07, right=0.99, top=0.78, bottom=0.12, wspace=0.35)

# (a) 개별 기법
method_list_ours = ["UniformSolver", "UniformSolver_TD", "UniformSolver_TD_PR", "Ours"]
method_labels_ours = ["Uniform Distance", "+Token Deposit", "+Pause&Resume", "OrbitFlow"]
colors_ours = ["#8B5E3C", "#4C9085", "#7F8C8D", "#FF8C69"]

bar_width = 0.18
x_base = np.arange(len(slo_scales))
total_methods = len(method_list_ours)
x_offsets = [x_base + (i - total_methods / 2) * (bar_width + 0.01) + bar_width / 2 for i in range(total_methods)]

method_handles = {}
for idx, (method, label, color) in enumerate(zip(method_list_ours, method_labels_ours, colors_ours)):
    y_vals = []
    for sc in slo_scales:
        summary_path = base_dir_uniform / f"slo{sc}" / method / "arrival_summerizev2.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            sel = df[(df["slo"] == sc) & (df["arrival_rate"] == arrival_rate) & (df["cv_num"] == cv_rate)]
            val = sel["tbt_attainment_with_TD"].iloc[0] if len(sel) == 1 and method != "UniformSolver" else \
                  sel["tbt_attainment_no_TD"].iloc[0] if len(sel) == 1 else 0.0
        else:
            val = 0.0
        y_vals.append(val)

    bars = ax_ours.bar(x_offsets[idx], y_vals, width=bar_width, color=color, label=label)
    method_handles[label] = bars[0]

ax_ours.set_xticks(x_base)
ax_ours.set_xticklabels(slo_labels, fontsize=style["tick"]["labelsize"])
ax_ours.set_ylim(0, 105)
ax_ours.set_yticks([0, 25, 50, 75, 100])
ax_ours.set_ylabel("TBT SLO (%)", **style["label"])
ax_ours.set_xlabel("SLO Scale", **style["label"])
for spine in ax_ours.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])
ax_ours.tick_params(axis='x', which='both', length=0)
ax_ours.tick_params(axis='y', which='both', length=0)
ax_ours.yaxis.grid(True, **style["grid"])
ax_ours.text(0.5, -0.35, "(a) Individual Component", transform=ax_ours.transAxes, fontsize=30, ha='center')

# 범례
ordered_labels = method_labels_ours
ordered_handles = [method_handles[label] for label in method_labels_ours] + [
    Line2D([0], [0], color='black', linestyle='--', linewidth=2)
]

ax_ours.legend(ordered_handles, ordered_labels, **style["legend"])


# (b) Token Deposit 비교
method_list_token = ["Flexgen", "DistNSingle", "Ours"]
method_labels_token = ["FlexGen+", "Dynamic Heuristic", "OrbitFlow"]
colors_base = ["#76C7AE", "#FFB3BA", "#FF8C69"]
colors_dp = ["#006D5B", "#B53050", "#A64500"]
hatches = ['*', 'O', '//']

x_offsets = [x_base + (i - len(method_list_token) / 2) * (bar_width + 0.01) + bar_width / 2 for i in range(len(method_list_token))]
method_handles = {}

for idx, (method, label, color) in enumerate(zip(method_list_token, method_labels_token, colors_base)):
    y_base = []
    y_delta = []
    for sc in slo_scales:
        summary_path = base_dir_token / f"slo{sc}" / method / "arrival_summerizev2.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            sel = df[(df["slo"] == sc) & (df["arrival_rate"] == arrival_rate) & (df["cv_num"] == cv_rate)]
            base_val = sel["tbt_attainment_no_TD"].iloc[0] if len(sel) == 1 else 0.0
            delta_val = sel["tbt_attainment_with_TD"].iloc[0] - base_val if len(sel) == 1 else 0.0
        else:
            base_val = 0.0
            delta_val = 0.0
        y_base.append(base_val)
        y_delta.append(delta_val)

    bars = ax_base.bar(x_offsets[idx], y_base, width=bar_width, color=color, label=label)
    ax_base.bar(x_offsets[idx], y_delta, bottom=y_base, width=bar_width,
                color=colors_dp[idx], hatch=hatches[idx], edgecolor="white")
    method_handles[label] = bars[0]

ax_base.set_xticks(x_base)
ax_base.set_xticklabels(slo_labels, fontsize=style["tick"]["labelsize"])
ax_base.set_ylim(0, 105)
ax_base.set_yticks([0, 25, 50, 75, 100])
ax_base.set_ylabel("TBT SLO (%)", **style["label"])
ax_base.set_xlabel("SLO Scale", **style["label"])
for spine in ax_base.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])
ax_base.tick_params(axis='x', which='both', length=0)
ax_base.tick_params(axis='y', which='both', length=0)
ax_base.yaxis.grid(True, **style["grid"])
ax_base.text(0.5, -0.35, "(b) Token Deposit", transform=ax_base.transAxes, fontsize=30, ha='center')

# 범례
handles = [method_handles[label] for label in method_labels_token]
ax_base.legend(handles, method_labels_token, **style["legend"])

# 저장
plt.savefig(output_dir / "6_3_design_validation.jpg", bbox_inches="tight")
plt.savefig(output_dir / "6_3_design_validation.pdf", bbox_inches="tight")
print(output_dir / "6_3_design_validation.jpg")
