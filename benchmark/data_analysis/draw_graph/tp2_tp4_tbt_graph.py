import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def plot_tp_tbt(base_dirs, output_dir):
    method_list   = ["Flexgen", "SelectN", "DistNSingle", "Ours"]
    method_labels = ["FlexGen+", "SLO-aware Offloading", "Dynamic Heuristic", "OrbitFlow"]
    arrival_rate = 2.0
    cv_rate = 1

    slo_scales_2 = [[5.4, 4.3, 3.3, 2.5], [2.5, 2, 1.5, 1.25]]
    slo_labels   = ["2.5", "2", "1.5", "1"]
    tp_labels    = ["TP-2", "TP-4"]

    colors  = ["#76C7AE", "#9F79C1", "#FFB3BA", "#FF8C69"]
    markers = ['D', 'o', 's', 'p']

    style = {
        "line":   {"linewidth":3, "markersize":15},
        "tick":   {"labelsize":33},
        "label":  {"fontsize":40, "labelpad":8},
        "title":  {"fontsize":40, "pad":10},
        "legend": {"fontsize":35},
        "spine":  {"color":"black","alpha":0.7,"linewidth":1.5},
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5), sharey=True)

    for ax, base_dir, tp_label, slo_scales in zip(axes, base_dirs, tp_labels, slo_scales_2):
        for method, label, color, marker in zip(method_list, method_labels, colors, markers):
            y_vals = []
            for sc in slo_scales:
                summary_path = base_dir / f"slo{sc}" / method / "arrival_summerizev2.csv"
                try:
                    if summary_path.exists():
                        df_sum = pd.read_csv(summary_path)
                        sel = df_sum[
                            (df_sum["slo"] == sc)
                            & (df_sum["arrival_rate"] == arrival_rate)
                            & (df_sum["cv_num"] == cv_rate)
                        ]
                        if method == "Ours":
                            value = float(sel["tbt_attainment_with_TD"].iloc[0]) if len(sel)==1 else 0.0
                        else:
                            value = float(sel["tbt_attainment_no_TD"].iloc[0]) if len(sel)==1 else 0.0
                    else:
                        print(f"File not found: {summary_path}")
                        value = np.nan
                except Exception as e:
                    print(f"Error reading {summary_path}: {e}")
                    value = np.nan
                y_vals.append(value)
                print(f"tp_label: {tp_label} slo: {sc}, method: {method}, value: {value}")

            ax.plot(slo_labels, y_vals,
                    label=label, color=color, marker=marker,
                    **style["line"])

        ax.axhline(90, color="gray", linestyle="--", linewidth=2, label="")
        ax.set_xticks(slo_labels)
        ax.set_xticklabels(slo_labels, fontsize=style["tick"]["labelsize"])
        ax.tick_params(axis='x', length=0, labelsize=style["tick"]["labelsize"])
        ax.tick_params(axis='y', length=5, labelsize=style["tick"]["labelsize"])
        ax.set_ylim(-5, 105)
        ax.set_yticks([0, 50, 100])
        ax.set_title(tp_label, **style["title"])
        if base_dir == base_dirs[0]:
            ax.set_ylabel("TBT SLO (%)", **style["label"])
        ax.set_xlabel("SLO Scale", **style["label"])
        for spine in ax.spines.values():
            spine.set_edgecolor(style["spine"]["color"])
            spine.set_alpha(style["spine"]["alpha"])
            spine.set_linewidth(style["spine"]["linewidth"])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center",
               bbox_to_anchor=(0.5, 1.06),
               ncol=2,
               frameon=False,
               columnspacing=1.6,
               labelspacing=0.1,
               handletextpad=0.3,
               **style["legend"])

    for ax in axes:
        ax.tick_params(axis='x', which='both', length=0)
        ax.tick_params(axis='y', which='both', length=0)

    plt.subplots_adjust(left=0.125, right=0.99,
                        top=0.75, bottom=0.18,
                        wspace=0.1)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "6_3_tp_tbt.jpg", bbox_inches="tight")
    plt.savefig(output_dir / "6_3_tp_tbt.pdf", bbox_inches="tight")
    print(output_dir / "6_3_tp_tbt.jpg")


def main():
    parser = argparse.ArgumentParser(description="Plot TBT SLO vs TP level comparison")
    parser.add_argument("base_dirs", type=Path, nargs=2,
                        help="Two base directories for TP-2 and TP-4 experiments")
    parser.add_argument("--output-dir", type=Path, default=Path("./figures"),
                        help="Directory to save output plots (default: ./figures)")
    args = parser.parse_args()
    plot_tp_tbt(args.base_dirs, args.output_dir)


if __name__ == "__main__":
    main()
