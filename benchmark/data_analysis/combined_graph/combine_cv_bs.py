# Re-import modules after state reset
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# ───────────────────────────────────────────────
# 공통 설정
method_list = ["Flexgen", "DistNSingle", "OursTD"]
method_labels = ["FlexGen+", "Dynamic Heuristic", "OrbitFlow"]
colors = ["#508776", "#FFB3BA", "#FF8C69"]  
colorsfd = ["#6B705C", "#A47149", "#FF8C69"]
markers = ['p', '^', 's']
font_size = 30

style = {
    "title": {"fontsize": 32, "pad": 8},
    "line": {"linewidth": 4, "markersize": 16.5},
    "tick": {"fontsize": 30},
    "label": {"fontsize": font_size, "labelpad": 5},
    "legend": {"fontsize": font_size},
    "spine": {"color": "black", "alpha": 0.7, "linestyle": "-", "linewidth": 2},
    "grid": {"color": "gray", "linestyle": "-", "linewidth": 3, "alpha": 0.2},
}

# ───────────────────────────────────────────────
# 데이터 설정
arrival_rate = 2.0
cv_rates = [1, 2, 4, 6]
cv_labels = [str(r) for r in cv_rates]
slo_cv = 1.5
base_dir_cv = Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp_CV")

batch_sizes = [2, 4, 8]
batch_labels = [str(bs) for bs in batch_sizes]
slo_bs = 1
cv_rate_bs = 1
bs_path = Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp_bs")
base_bs = Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp_32k")

slo_scales = [1]
slo_labels = [str(s) for s in slo_scales]
base_dirs = [
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp_random_xinyue"),
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp_shortest_xinyue"),
    Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp_longest_xinyue"),
]
base_labels = ["Random", "Shortest", "Longest"]
method_fallback = "Ours"

# ───────────────────────────────────────────────
# 전체 Figure 설정
fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=False)
plt.subplots_adjust(left=0.07, right=0.99, top=0.78, bottom=0.12, wspace=0.35)

# 1. CV Scale vs TBT SLO

print("Start CV Scale")
ax_cv = axes[1]
for m, method in enumerate(method_list):
    y_tbt = []
    summary_path = base_dir_cv / f"slo{slo_cv}" / (method[:-2] if method.endswith("TD") else method) / "arrival_summerizev2.csv"
    try:
        df_sum = pd.read_csv(summary_path)
        for cv_rate in cv_rates:
            sel = df_sum[(df_sum["slo"] == slo_cv) &
                         (df_sum["arrival_rate"] == arrival_rate) &
                         (df_sum["cv_num"] == cv_rate)]
            value = float(sel["tbt_attainment_with_TD" if method.endswith("TD") else "tbt_attainment_no_TD"].iloc[0]) if len(sel) == 1 else np.nan
            print(f"Method: {method}, CV Rate: {cv_rate}, Value: {value}")
            y_tbt.append(value)
    except Exception as e:
        y_tbt = [np.nan] * len(cv_rates)

    ax_cv.plot(cv_labels, y_tbt, **style["line"], marker=markers[m], color=colors[m], label=method_labels[m])

ax_cv.set_ylim(-5, 105)
ax_cv.set_xticks(cv_labels)
ax_cv.set_xlabel("(b) CV Scale", **style["label"])
ax_cv.set_ylabel("TBT SLO (%)", **style["label"])
ax_cv.tick_params(axis='both', labelsize=style["tick"]["fontsize"], length=0, pad=5)
ax_cv.xaxis.grid(True, **style["grid"])
ax_cv.yaxis.grid(True, **style["grid"])
for spine in ax_cv.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

print("CV Scale plot completed.")
print("Start Batch Size")
# 2. Batch Size vs TBT SLO
ax_bs = axes[2]
for m, method in enumerate(method_list):
    y_vals = []
    for bs in batch_labels:
        method_dir = base_bs if bs == "4" else bs_path
        folder_name = method[:-2] if method.endswith("TD") else method
        try:
            summary_path = method_dir / f"slo{slo_bs}" / folder_name / "arrival_summerizev2.csv"
            df_sum = pd.read_csv(summary_path)
            sel = df_sum[(df_sum["slo"] == slo_bs) &
                         (df_sum["arrival_rate"] == arrival_rate) &
                         (df_sum["cv_num"] == cv_rate_bs)]
            if sel.empty:
                name = f"lambda{arrival_rate}x_cv{cv_rate_bs}_bs{bs}"
                sel = df_sum[df_sum["name"] == name]
            col = "tbt_attainment_with_TD" if method.endswith("TD") else "tbt_attainment_no_TD"
            value = float(sel[col].iloc[0]) if len(sel) == 1 else np.nan
            print(f"Method: {method}, Batch Size: {bs}, Value: {value}")
            y_vals.append(value)
        except Exception as e:
            y_vals.append(np.nan)

    ax_bs.plot(batch_labels, y_vals, **style["line"], marker=markers[m], color=colors[m], label=method_labels[m])

ax_bs.set_ylim(-5, 105)
ax_bs.set_xticks(batch_labels)
ax_bs.set_xlabel("(c) Batch Size", **style["label"])
ax_bs.tick_params(axis='both', labelsize=style["tick"]["fontsize"], length=0, pad=5)
ax_bs.xaxis.grid(True, **style["grid"])
ax_bs.yaxis.grid(True, **style["grid"])
for spine in ax_bs.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

print("Batch Size plot completed.")
print("Start Fallback Strategy")
# 3. Fallback vs TBT SLO (bar chart)
ax_fb = axes[0]
bar_width = 0.18
x_base = np.arange(len(slo_scales))
total_methods = len(base_dirs)
x_offsets = [x_base + (i - total_methods / 2) * (bar_width + 0.01) + bar_width / 2 for i in range(total_methods)]

for idx, (base_dir, label, color) in enumerate(zip(base_dirs, base_labels, colorsfd)):
    y_vals = []
    for sc in slo_scales:
        summary_path = base_dir / f"slo{sc}" / method_fallback / "arrival_summerizev2.csv"
        if summary_path.exists():
            try:
                df_sum = pd.read_csv(summary_path)
                sel = df_sum[(df_sum["slo"] == sc) & (df_sum["name"] == "fallback_xinyue")]
                val = float(sel["tbt_attainment_with_TD"].iloc[0]) if len(sel) == 1 else 0.0
            except Exception:
                val = 0.0
        else:
            val = 0.0
        print(f"Fallback Strategy: {label}, SLO Scale: {sc}, Value: {val}")
        y_vals.append(val)

    ax_fb.bar(x_offsets[idx], y_vals, width=bar_width, color=color, label=label)

ax_fb.set_xticks(x_base)
ax_fb.set_xticklabels(" ", fontsize=style["tick"]["fontsize"])
ax_fb.tick_params(axis='x', length=0, labelsize=style["tick"]["fontsize"])
ax_fb.tick_params(axis='y', length=5, labelsize=style["tick"]["fontsize"])
ax_fb.set_ylim(0, 105)
# ax_fb.set_ylabel("TBT SLO (%)", **style["label"])
ax_fb.set_xlabel("(a) Fallback Strategy", **style["label"])
ax_fb.yaxis.grid(True, **style["grid"])
for spine in ax_fb.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])


for ax in axes:
    ax.tick_params(axis='both', length=0, labelsize=style["tick"]["fontsize"])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("TBT SLO (%)", **style["label"])
    for spine in ax.spines.values():
        spine.set_edgecolor(style["spine"]["color"])
        spine.set_alpha(style["spine"]["alpha"])
        spine.set_linewidth(style["spine"]["linewidth"])

# 범례
# 범례
handles, method_legend_labels = ax_cv.get_legend_handles_labels()
fallback_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colorsfd]

fig.legend(
    fallback_handles + handles,
    base_labels + method_labels,
    loc='upper center',
    ncol=len(method_labels) + len(base_labels),
    bbox_to_anchor=(0.5, 1.02),
    fontsize=style["legend"]["fontsize"],
    handletextpad=0.2,
    columnspacing=0.9,
    frameon=False
)


# 저장
output_path = "/home/heelim/vllm/benchmark/data_analysis/figures/tbt_cv_bs_fallback"
plt.savefig(f"{output_path}.jpg", bbox_inches="tight", dpi=300)
plt.savefig(f"{output_path}.pdf", bbox_inches="tight")

print(output_path + ".jpg")
