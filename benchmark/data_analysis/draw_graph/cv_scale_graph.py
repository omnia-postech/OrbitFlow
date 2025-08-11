import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import argparse

# Settings
method_list = ["Flexgen", "DistNSingle", "OursTD"]
method_labels = ["FlexGen+", "Dynamic Heuristic", "OrbitFlow"]
pick_x_method = "DistNSingle"
arrival_rate = 2.0
cv_rates = [1, 2, 4, 6]
cv_labels = [str(r) for r in cv_rates]
slo_scales = [1.5]
slo_labels = [str(s) for s in slo_scales]
colors = ["#508776", "#FFB3BA", "#FF8C69"]
markers = ['p', '^', 's']

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

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("base_dir", nargs="?", type=Path,)
parser.add_argument("--output-dir", type=Path, default=Path("./figures"),
                    help="Directory to save plots (default: ./figures)")
args = parser.parse_args()
base_dir = args.base_dir
output_dir = args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)

# Create figure
fig, axes = plt.subplots(1, 1, figsize=(7, 5))
plt.subplots_adjust(left=0.07, right=0.99, top=0.78, bottom=0.12)

# Plotting loop
for i, sc in enumerate(slo_scales):
    ax_TBT = axes
    x_tbt = cv_labels

    for m, method in enumerate(method_list):
        y_tbt = []
        summary_path = base_dir / f"slo{sc}" / (method[:-2] if method.endswith("TD") else method) / "arrival_summerizev2.csv"
        try:
            df_sum = pd.read_csv(summary_path)
            for cv_rate in cv_rates:
                sel = df_sum[(df_sum["slo"] == sc) &
                             (df_sum["arrival_rate"] == arrival_rate) &
                             (df_sum["cv_num"] == cv_rate)]
                col = "tbt_attainment_with_TD" if method.endswith("TD") else "tbt_attainment_no_TD"
                value = float(sel[col].iloc[0]) if len(sel) == 1 else np.nan
                y_tbt.append(value)
        except Exception as e:
            print(f"NO: {summary_path}, Error: {e}")
            y_tbt = [np.nan] * len(cv_rates)

        ax_TBT.plot(x_tbt, y_tbt, **style["line"], marker=markers[m], color=colors[m], label=method_labels[m])

    # Axis styling
    ax_TBT.set_ylim(-5, 105)
    ax_TBT.tick_params(axis='both', labelsize=style["tick"]["fontsize"], length=0, pad=5)
    ax_TBT.xaxis.grid(True, **style["grid"])
    ax_TBT.yaxis.grid(True, **style["grid"])
    ax_TBT.set_yticks([0, 25, 50, 75, 100])
    ax_TBT.set_yticklabels([str(y) for y in [0, 25, 50, 75, 100]])
    ax_TBT.set_ylabel("TBT SLO (%)", **style["label"])
    ax_TBT.set_xlabel("CV Scale", **style["label"])

# Spine styling
for spine in axes.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# Legend
handles, labels = axes.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(method_list),
           bbox_to_anchor=(0.5, 0.98), columnspacing=0.9,
           fontsize=style["legend"]["fontsize"], frameon=False)

# Save
plt.savefig(output_dir / "cv_tbt_tpot.jpg", bbox_inches="tight", dpi=300)
plt.savefig(output_dir / "cv_tbt_tpot.pdf", bbox_inches="tight")
print(output_dir / "cv_tbt_tpot.jpg")
