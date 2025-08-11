import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Settings
method_list = ["Flexgen", "DistNSingle", "OursTD"]
method_labels = ["FlexGen+", "Dynamic Heuristic", "OrbitFlow"]

bs_path = Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp_bs")
base = Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp_32k")

slo_scales = [1, 2.5]
cv_rate = 1
arrival_rate = 2.0
batch_sizes = [2, 4, 8]

colors = ["#508776", "#FFB3BA", "#FF8C69"]
font_size = 30
bar_width = 0.25

style = {
    "title":  {"fontsize": 32, "pad": 8},
    "tick":   {"fontsize": 30},
    "label":  {"fontsize": font_size, "labelpad": 5},
    "legend": {"fontsize": font_size},
    "spine": {
        "color": "black", "alpha": 0.7,
        "linestyle": "-", "linewidth": 2
    },
    "grid": {
        "color": "gray", "linestyle": "-",
        "linewidth": 3, "alpha": 0.2
    },
}

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(21, 5), sharey=False)
plt.subplots_adjust(left=0.07, right=0.99, top=0.78, bottom=0.12, wspace=0.25)

# Plotting per batch size
for idx_bs, bs in enumerate(batch_sizes):
    ax = axes[idx_bs]
    bar_positions = np.arange(len(slo_scales))
    
    for m, method in enumerate(method_list):
        method_dir = bs_path if bs != 4 else base
        folder_name = method[:-2] if method.endswith("TD") else method
        y_vals = []

        for sc in slo_scales:
            try:
                summary_path = method_dir / f"slo{sc}" / folder_name / "arrival_summerizev2.csv"
                # print(f"Processing: {summary_path}")
                df_sum = pd.read_csv(summary_path)
                sel = df_sum[(df_sum["slo"] == sc) &
                                (df_sum["arrival_rate"] == arrival_rate) &
                                (df_sum["cv_num"] == cv_rate)]
                if sel.empty:
                    name = f"lambda{arrival_rate}x_cv{cv_rate}_bs{bs}"
                    print(summary_path)
                    print(f"name : {name}")
                    
                    sel = df_sum[(df_sum["name"] == name)]
                    # print(f"sel: {len(sel)}")    
                col = "tbt_attainment_with_TD" if method.endswith("TD") else "tbt_attainment_no_TD"
                value = float(sel[col].iloc[0]) if len(sel) == 1 else np.nan
                y_vals.append(value)
            except Exception as e:
                print(f"NO: {summary_path}, Error: {e}")
                y_vals.append(np.nan)

        offset = (m - 1) * bar_width
        ax.bar(bar_positions + offset, y_vals, width=bar_width, color=colors[m],
               label=method_labels[m], zorder=3)

    # Axis styling
    ax.set_ylim(-5, 105)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels([str(s) for s in slo_scales], fontsize=style["tick"]["fontsize"])
    ax.set_title(f"Batch Size {bs}", fontsize=style["title"]["fontsize"])
    ax.set_xlabel("SLO Scale", **style["label"])
    if idx_bs == 0:
        ax.set_ylabel("TBT SLO (%)", **style["label"])

    ax.tick_params(axis='y', labelsize=style["tick"]["fontsize"], length=0, pad=5)
    ax.yaxis.grid(True, **style["grid"])
    for spine in ax.spines.values():
        spine.set_edgecolor(style["spine"]["color"])
        spine.set_alpha(style["spine"]["alpha"])
        spine.set_linewidth(style["spine"]["linewidth"])

# Legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(method_list),
           bbox_to_anchor=(0.5, 0.98), columnspacing=0.9,
           fontsize=style["legend"]["fontsize"], frameon=False)

# Save
plt.savefig("/home/heelim/vllm/benchmark/data_analysis/figures/bs_tbt_comparison_bar.jpg", bbox_inches="tight", dpi=300)
plt.savefig("/home/heelim/vllm/benchmark/data_analysis/figures/bs_tbt_comparison_bar.pdf", bbox_inches="tight")
print("/home/heelim/vllm/benchmark/data_analysis/figures/bs_tbt_comparison_bar.jpg")
