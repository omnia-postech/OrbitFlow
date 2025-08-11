import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from pathlib import Path
from matplotlib.patches import Patch
import numpy as np
import argparse

# ───────────────────────────────────────────────
# 1. 설정 -------------------------------------------------------
method_list = ["NextLayer", "Static1", "Flexgen", "SelectN", "DistNSingle", "OursTD"]
method_labels = ["DeepSpeed", "FlexGen", "FlexGen+", "SLO-aware Offloading", "Dynamic Heuristic", "OrbitFlow"]
pick_x_method = "DistNSingle"
arrival_rate = [1.0, 2.0, 3.0, 4.0, 5.0]
arrival_labels = [str(r) for r in arrival_rate]
cv_rate = 1
slo_scales = [2.5, 1.5, 1]
slo_labels = [str(s) for s in slo_scales]

colors = ["#4DA6FF","#508776", "#76C7AE",  "#9F79C1", "#FFB3BA", "#FF8C69"]
markers = ['o', 'D', 'p', '>', '^', 's']

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

parser = argparse.ArgumentParser()
parser.add_argument(
    "base_dir",
    nargs="?",
    type=Path,
    default=Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp_32k"),
    help="실험 결과가 들어있는 최상위 디렉토리 (기본값 사용 시 생략)"
)
args = parser.parse_args()
base_dir = args.base_dir

# ───────────────────────────────────────────────
# 2. Figure & GridSpec ---------------------------
N_SLO_SCALE = len(slo_scales)
fig = plt.figure(figsize=(28, 9))  # Adjusted width to accommodate extra column
gs = GridSpec(2, N_SLO_SCALE, width_ratios=[1]*N_SLO_SCALE, height_ratios=[1, 1], figure=fig)
plt.subplots_adjust(left=0.05, right=0.6, top=0.78, bottom=0.12, wspace=0.08, hspace=0.25)

# TBT and TPOT axes
axes = [[fig.add_subplot(gs[i, j]) for j in range(N_SLO_SCALE)] for i in range(2)]

# Nested GridSpec for ax1 and ax2 in the last column, centered vertically
gs1 = GridSpec(1, 1, left=0.64, right=0.85, top=0.69, bottom=0.22)
ax1 = fig.add_subplot(gs1[0, 0])

# 두 번째 GridSpec (우측 1/3 영역)
gs2 = GridSpec(1, 1, left=0.9, right=1, top=0.69, bottom=0.22)
ax2 = fig.add_subplot(gs2[0, 0])
ax2.set_title("This is ax2")

# ───────────────────────────────────────────────
# 3. TBT, TPOT 플롯 -----------------------------------
for i, sc in enumerate(slo_scales):
    ax_TBT = axes[0][i]   # TBT
    ax_TPOT = axes[1][i]  # TPOT

    x_tbt = []
    if pick_x_method.endswith("TD"):
        summary_path = base_dir / f"slo{sc}" / pick_x_method[:-2] / "arrival_summerizev2.csv"
    else:
        summary_path = base_dir / f"slo{sc}" / pick_x_method / "arrival_summerizev2.csv"
    try:
        df_sum = pd.read_csv(summary_path)
        x_base_sel = df_sum[(df_sum["slo"] == sc) 
                            & (df_sum["arrival_rate"] == min(arrival_rate))
                            & (df_sum["cv_num"] == cv_rate)]
        x_base = x_base_sel['req_per_sec'].iloc[0]
        for rate in arrival_rate:
            x_tbt.append(x_base * rate * 60) if len(x_base_sel) == 1 else 0.0
    except Exception as e:
        print(f"NO: {e}")
        x_tbt = [0] * len(arrival_rate)
    
    for m, method in enumerate(method_list):
        # TBT 데이터
        y_tbt = []
        if method.endswith("TD"):
            summary_path = base_dir / f"slo{sc}" / method[:-2] / "arrival_summerizev2.csv"
        else:
            summary_path = base_dir / f"slo{sc}" / method / "arrival_summerizev2.csv"
        try:
            df_sum = pd.read_csv(summary_path)
            for rate in arrival_rate:
                sel = df_sum[(df_sum["slo"] == sc) 
                            & (df_sum["arrival_rate"] == rate)
                            & (df_sum["cv_num"] == cv_rate)]
                if method.endswith("TD"):
                    value = float(sel["tbt_attainment_with_TD"].iloc[0]) if len(sel) == 1 else np.nan
                else:
                    value = float(sel["tbt_attainment_no_TD"].iloc[0]) if len(sel) == 1 else np.nan
                y_tbt.append(value)
        except:
            print(f"NO: {summary_path}")
            y_tbt = [np.nan] * len(arrival_rate)

        ax_TBT.plot(x_tbt, y_tbt, **style["line"], marker=markers[m], color=colors[m], label=method_labels[m])
        ax_TBT.plot(x_tbt, y_tbt, linestyle="", marker=markers[m], color=colors[m], markersize=15)

        # TPOT 데이터
        y_tpot = []
        try:
            df_sum = pd.read_csv(summary_path)
            for rate in arrival_rate:
                sel = df_sum[(df_sum["slo"] == sc) 
                            & (df_sum["arrival_rate"] == rate)
                            & (df_sum["cv_num"] == cv_rate)]
                y_tpot.append(float(sel["tpot_attainment"].iloc[0]) if len(sel) == 1 else np.nan)
        except:
            y_tpot = [np.nan] * len(arrival_rate)

        ax_TPOT.plot(x_tbt, y_tpot, **style["line"], marker=markers[m], color=colors[m])
        ax_TPOT.plot(x_tbt, y_tpot, linestyle="", marker=markers[m], color=colors[m], markersize=15)

    # 공통 축 스타일
    for ax in (ax_TBT, ax_TPOT):
        ax.set_xlim(0.4, 2.6)
        x_ticks = np.arange(0.5, 3, 0.5)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{x:.1f}" for x in x_ticks], fontsize=style["tick"]["fontsize"])
        ax.set_ylim(-5, 105)
        ax.tick_params(axis='both', labelsize=style["tick"]["fontsize"], length=0, pad=5)
        ax.xaxis.grid(True, **style["grid"])
        ax.yaxis.grid(True, **style["grid"])

    # y-틱 표시
    ys = [0, 25, 50, 75, 100]
    for ax in (ax_TBT, ax_TPOT):
        ax.set_yticks(ys)

    # row-label
    if i == 0:
        for ax in (ax_TBT, ax_TPOT):
            ax.set_yticklabels([str(y) for y in ys])
        ax_TBT.set_ylabel("TBT SLO (%)", **style["label"])
        ax_TPOT.set_ylabel("TPOT SLO (%)", **style["label"])
    else:
        ax_TBT.set_yticklabels([])
        ax_TPOT.set_yticklabels([])

    ax_TBT.set_xlabel("", **style["label"])
    ax_TPOT.set_xlabel("request per min", **style["label"])
    ax_TBT.set_title(f"SLO Scales = {sc}", **style["title"])

axes[1][0].text(1.5, -1.7, "(a) TBT & TPOT SLO Attainment",
                transform=axes[0][0].transAxes,
                fontsize=style["title"]["fontsize"], 
                ha="center", va="top")

# ───────────────────────────────────────────────
# 4. P95 TBT, Mean Utilization 플롯 ----------------------
# P95 TBT 플롯
bar_width = 0.9 / len(method_list)
positions = np.arange(len(slo_labels))
y_max = 3.4

for i, (method, label) in enumerate(zip(method_list, method_labels)):
    y_vals = []
    for sc in slo_scales:
        if method.endswith("TD"):
            summary_path = base_dir / f"slo{sc}" / method[:-2] / "arrival_summerizev2.csv"
        else:
            summary_path = base_dir / f"slo{sc}" / method / "arrival_summerizev2.csv"
        try:
            df_sum = pd.read_csv(summary_path)
            sel = df_sum[(df_sum["slo"] == sc) &
                         (df_sum["arrival_rate"] == 2.0) &
                         (df_sum["cv_num"] == cv_rate)]
            value = float(sel["p95_ratio"].iloc[0]) if len(sel) == 1 else np.nan
            y_vals.append(value)
        except:
            y_vals.append(np.nan)

    offsets = (i - (len(method_list)-1)/2) * bar_width
    bars = ax1.bar(positions + offsets, y_vals, width=bar_width, label=label, color=colors[i], edgecolor="white")
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height) and height > y_max:
            ax1.text(bar.get_x() + bar.get_width() / 2, y_max - 0.35, f"{height*1000:.0f} ms",
                     ha='center', va='bottom', fontsize=style["tick"]["fontsize"]-6, color="black")

ax1.set_xlabel("SLO Scale", **style["label"])
ax1.set_ylabel("P95 TBT (sec)", **style["label"])
ax1.set_xticks(positions)
ax1.set_xticklabels(slo_labels, fontsize=style["tick"]["fontsize"])
ax1.tick_params(axis='both', length=0, labelsize=style["tick"]["fontsize"])
max_y = y_max
raw_step = max_y / 5
def round_step(x):
    if x <= 1: return 1
    elif x <= 2: return 2
    elif x <= 5: return 5
    elif x <= 10: return 10
    else: return int(np.ceil(x / 10.0)) * 10
step = round_step(raw_step)
ticks = list(range(step, int(np.ceil(max_y)) + 1, step))
ax1.set_yticks(ticks)
ax1.set_yticklabels([f"{t}" for t in ticks], fontsize=style["tick"]["fontsize"])
ax1.set_ylim(0, y_max)
ax1.yaxis.grid(True, **style["grid"])
# ax1.set_title("(a) P95 TBT SLO Attainment", **style["title"])
ax1.text(-2.3, -0.5, "(b) P95 TBT Latency",
         transform=ax2.transAxes,
         fontsize=style["title"]["fontsize"],
         ha="left",
          va="top")
ax1.set_title("")

# Mean Utilization 플롯
sc = 2.5
x_center = 0
for i, (method, label) in enumerate(zip(method_list, method_labels)):
    if method.endswith("TD"):
        summary_path = base_dir / f"slo{sc}" / method[:-2] / f"lambda{2.0}x_cv{cv_rate}" / "mem_util_output.csv"
    else:
        summary_path = base_dir / f"slo{sc}" / method / f"lambda{2.0}x_cv{cv_rate}" / "mem_util_output.csv"
    try:
        df_util = pd.read_csv(summary_path)
        df_util['utilization'] = df_util['used_num'] / df_util['total_num'] * 100
        mean_util = df_util['utilization'].mean()
    except:
        print(f"⚠️ 실패: {summary_path}")
        mean_util = np.nan
    offset = (i - (len(method_list) - 1) / 2) * bar_width
    ax2.bar(x_center + offset, mean_util, width=bar_width, label=label, color=colors[i], edgecolor="white")

ax2.set_xticks([])
ax2.tick_params(axis='x', length=0, labelsize=style["tick"]["fontsize"])
ax2.set_ylim(0, 105)
ax2.set_yticks([20, 40, 60, 80, 100])
ax2.set_yticklabels([f"{v:.0f}" for v in [20, 40, 60, 80, 100]], fontsize=style["tick"]["fontsize"])
ax2.set_ylabel("Mean Util. (%)", fontsize = style["label"]["fontsize"], labelpad=-8)
ax2.tick_params(axis='y', length=0, labelsize=style["tick"]["fontsize"])
ax2.yaxis.grid(True, **style["grid"])
ax2.text(-0.25, -0.5, "(c) GPU Mem Util.",
         transform=ax2.transAxes,
         fontsize=style["title"]["fontsize"],
         ha="left", va="top")
ax2.set_title("")

# ───────────────────────────────────────────────
# 5. 스파인 스타일 및 범례 -----------------------------------
for row in axes:
    for ax in row:
        for spine in ax.spines.values():
            spine.set_edgecolor(style["spine"]["color"])
            spine.set_alpha(style["spine"]["alpha"])
            spine.set_linewidth(style["spine"]["linewidth"])
for ax in (ax1, ax2):
    for spine in ax.spines.values():
        spine.set_edgecolor(style["spine"]["color"])
        spine.set_alpha(style["spine"]["alpha"])
        spine.set_linewidth(style["spine"]["linewidth"])

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(method_list), 
           bbox_to_anchor=(0.5, 0.95), columnspacing=0.9, fontsize=style["legend"]["fontsize"], frameon=False)

# ───────────────────────────────────────────────
# 6. 저장 ---------------------------------------
output_dir = "/home/heelim/vllm/benchmark/data_analysis/figures"
plt.savefig(f"{output_dir}/arrival_rate_tbt_tpot.jpg", bbox_inches="tight", dpi=300)
plt.savefig(f"{output_dir}/arrival_rate_tbt_tpot.pdf", bbox_inches="tight")
print(f"✅ 그래프 저장 완료: {output_dir}/arrival_rate_tbt_tpot.jpg")