# Re-import modules after code execution state reset
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# 공통 설정
method_list = ["Flexgen", "DistNSingle", "OursTD"]
method_labels = ["FlexGen+", "Dynamic Heuristic", "OrbitFlow"]
colors = ["#508776", "#FFB3BA", "#FF8C69"]
markers = ['p', '^', 's']
font_size = 30

style = {
    "title":  {"fontsize": 32, "pad": 8},
    "line":   {"linewidth": 4, "markersize": 16.5},
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

# 전체 figure 설정
fig, (ax_cv, ax_bs) = plt.subplots(1, 2, figsize=(14, 5))
plt.subplots_adjust(left=0.07, right=0.99, top=0.78, bottom=0.12, wspace=0.25)

# 1. CV Scale 그래프
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
            y_tbt.append(value)
    except Exception as e:
        print(f"NO: {summary_path}, Error: {e}")
        y_tbt = [np.nan] * len(cv_rates)

    ax_cv.plot(cv_labels, y_tbt, **style["line"], marker=markers[m], color=colors[m], label=method_labels[m])

ax_cv.set_ylim(-5, 105)
ax_cv.set_xticks(cv_labels)
ax_cv.set_xlabel("(a) CV Scale", **style["label"])
ax_cv.set_ylabel("TBT SLO (%)", **style["label"])
ax_cv.tick_params(axis='both', labelsize=style["tick"]["fontsize"], length=0, pad=5)
ax_cv.xaxis.grid(True, **style["grid"])
ax_cv.yaxis.grid(True, **style["grid"])
# ax_cv.set_title("SLO Scale 1.5", **style["title"])
for spine in ax_cv.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# 2. Batch Size 그래프
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
            y_vals.append(value)
        except Exception as e:
            print(f"NO: {summary_path}, Error: {e}")
            y_vals.append(np.nan)

    ax_bs.plot(batch_labels, y_vals, **style["line"], marker=markers[m], color=colors[m], label=method_labels[m])

ax_bs.set_ylim(-5, 105)
ax_bs.set_xticks(batch_labels)
ax_bs.set_xlabel("(b) Batch Size", **style["label"])
ax_bs.tick_params(axis='both', labelsize=style["tick"]["fontsize"], length=0, pad=5)
# ax_bs.set_title("SLO Scale 1", **style["title"])
ax_bs.xaxis.grid(True, **style["grid"])
ax_bs.yaxis.grid(True, **style["grid"])
for spine in ax_bs.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# 범례 공통 설정
handles, labels = ax_cv.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(method_list),
           bbox_to_anchor=(0.5, 0.98), columnspacing=0.9,
           fontsize=style["legend"]["fontsize"], frameon=False)

# 저장
output_base = "/home/heelim/vllm/benchmark/data_analysis/figures/tbt_cv_bs"
plt.savefig(f"{output_base}.jpg", bbox_inches="tight", dpi=300)
plt.savefig(f"{output_base}.pdf", bbox_inches="tight")
print(f"{output_base}.jpg")
