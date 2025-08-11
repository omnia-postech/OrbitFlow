import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def round_step(x):
    if x <= 1:
        return 1
    elif x <= 2:
        return 2
    elif x <= 5:
        return 5
    elif x <= 10:
        return 10
    else:
        return int(np.ceil(x / 10.0)) * 10

def plot_metrics(base_dir: Path, output_dir: Path):
    METHODS = ["NextLayer", "Static1", "Flexgen", "SelectN", "DistNSingle", "OursTD"]
    METHOD_LABS = ["DeepSpeed", "FlexGen", "FlexGen+", "SLO-aware Offloading", "Dynamic Heuristic", "OrbitFlow"]
    arrival_rate = 2.0
    cv_rate = 1
    SLO_SCALES = [2.5, 1.5, 1]
    SLO_LABELS = [f"{str(s)}" for s in SLO_SCALES]

    style = {
        "line": {"linewidth": 3, "markersize": 10},
        "spine": {"color": "black", "alpha": 0.7, "linewidth": 1.5},
        "title": {"fontsize": 25, "pad": 8},
        "label": {"fontsize": 25, "labelpad": 8},
        "legend": {"fontsize": 25},
        "tick": {"labelsize": 27},
        "grid": {
            "color": "gray",
            "linestyle": "-",
            "linewidth": 3,
            "alpha": 0.2
        },
    }

    colors = ["#4DA6FF", "#76C7AE", "#508776", "#9F79C1", "#FFB3BA", "#FF8C69"]
    bar_width = 0.9 / len(METHODS)
    positions = np.arange(len(SLO_LABELS))
    y_max = 3.4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [3, 1]})
    plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.12, wspace=0.3)

    # Plot (a): P95 TBT
    for i, (method, label) in enumerate(zip(METHODS, METHOD_LABS)):
        y_vals = []
        for sc in SLO_SCALES:
            method_dir = method[:-2] if method.endswith("TD") else method
            summary_path = base_dir / f"slo{sc}" / method_dir / "arrival_summerizev2.csv"
            try:
                df_sum = pd.read_csv(summary_path)
                sel = df_sum[
                    (df_sum["slo"] == sc) &
                    (df_sum["arrival_rate"] == arrival_rate) &
                    (df_sum["cv_num"] == cv_rate)
                ]
                value = float(sel["p95_ratio"].iloc[0]) if len(sel) == 1 else np.nan
            except:
                value = np.nan
            y_vals.append(value)

        offsets = (i - (len(METHODS)-1)/2) * bar_width
        bars = ax1.bar(positions + offsets, y_vals,
                       width=bar_width,
                       label=label,
                       color=colors[i],
                       edgecolor="white")
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height) and height > y_max:
                ax1.text(bar.get_x() + bar.get_width() / 2, y_max - 0.35,
                         f"{height*1000:.0f} ms", ha='center', va='bottom',
                         fontsize=style["tick"]["labelsize"]-6, color="black")

    ax1.set_xlabel("SLO Scale", **style["label"])
    ax1.set_ylabel("P95 TBT (ms)", **style["label"])
    ax1.set_xticks(positions)
    ax1.set_xticklabels(SLO_LABELS, fontsize=style["tick"]["labelsize"])
    ax1.tick_params(axis='x', length=0, labelsize=style["tick"]["labelsize"])
    ax1.tick_params(axis='y', length=0, labelsize=style["tick"]["labelsize"])
    step = round_step(y_max / 5)
    ticks = list(range(step, int(np.ceil(y_max)) + 1, step))
    ax1.set_yticks(ticks)
    ax1.set_yticklabels([f"{t*1000}" for t in ticks], fontsize=style["tick"]["labelsize"])
    ax1.set_ylim(0, y_max)
    ax1.yaxis.grid(True, **style["grid"])
    for spine in ax1.spines.values():
        spine.set_edgecolor(style["spine"]["color"])
        spine.set_alpha(style["spine"]["alpha"])
        spine.set_linewidth(style["spine"]["linewidth"])

    # Plot (b): Mean Utilization at SLO 2.5
    sc = 2.5
    x_center = 0
    for i, (method, label) in enumerate(zip(METHODS, METHOD_LABS)):
        summary_path = base_dir / f"slo{sc}" / method / f"lambda{arrival_rate}x_cv{cv_rate}" / "mem_util_output.csv"
        try:
            df_util = pd.read_csv(summary_path)
            df_util['utilization'] = df_util['used_num'] / df_util['total_num'] * 100
            mean_util = df_util['utilization'].mean()
        except:
            print(f"⚠️ 실패: {summary_path}")
            mean_util = np.nan

        offset = (i - (len(METHODS) - 1) / 2) * bar_width
        ax2.bar(x_center + offset, mean_util, width=bar_width,
                label=label, color=colors[i], edgecolor="white")

    ax2.set_xticks([])
    ax2.tick_params(axis='x', length=0, labelsize=style["tick"]["labelsize"])
    ax2.set_ylim(0, 100)
    ax2.set_yticks([20, 40, 60, 80, 100])
    ax2.set_yticklabels([f"{v:.0f}" for v in ax2.get_yticks()], fontsize=style["tick"]["labelsize"])
    ax2.set_ylabel("Mean Utilization (%)", **style["label"])
    ax2.tick_params(axis='y', length=0, labelsize=style["tick"]["labelsize"])
    ax2.yaxis.grid(True, **style["grid"])
    for spine in ax2.spines.values():
        spine.set_edgecolor(style["spine"]["color"])
        spine.set_alpha(style["spine"]["alpha"])
        spine.set_linewidth(style["spine"]["linewidth"])

    ax1.text(0.5, -0.38, "(a) P95 TBT SLO Attainment", fontsize=style["title"]["fontsize"],
             ha='center', va='bottom', transform=ax1.transAxes)
    ax2.text(0.5, -0.38, "(b) GPU Utilization", fontsize=style["title"]["fontsize"],
             ha='center', va='bottom', transform=ax2.transAxes)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center", bbox_to_anchor=(0.5, 1.15),
               ncol=3, **style["legend"], columnspacing=0.9, frameon=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "6_2_tail_p95_by_slo.jpg", bbox_inches="tight")
    plt.savefig(output_dir / "6_2_tail_p95_by_slo.pdf", bbox_inches="tight")
    print(f"✅ 그래프 저장 완료: {output_dir / '6_2_tail_p95_by_slo.jpg'}")

def main():
    parser = argparse.ArgumentParser(description="Plot P95 TBT and GPU utilization by SLO")
    parser.add_argument("base_dir", type=Path, help="Base directory of experiment results")
    parser.add_argument("--output-dir", type=Path, default=Path("./figures"),
                        help="Directory to save the output plots (default: ./figures)")
    args = parser.parse_args()
    plot_metrics(args.base_dir, args.output_dir)

if __name__ == "__main__":
    main()
