import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import argparse

def plot_tbt_slo_comparison(bs_path: Path, base_path: Path, output_dir: Path):
    method_list = ["Flexgen", "DistNSingle", "OursTD"]
    method_labels = ["FlexGen+", "Dynamic Heuristic", "OrbitFlow"]

    slo_scale = 1
    cv_rate = 1
    arrival_rate = 2.0
    batch_sizes = [2, 4, 8]
    batch_labels = [str(bs) for bs in batch_sizes]

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

    fig, ax = plt.subplots(figsize=(7, 5))
    plt.subplots_adjust(left=0.12, right=0.98, top=0.78, bottom=0.12)

    for m, method in enumerate(method_list):
        y_vals = []

        for bs in batch_labels:
            method_dir = base_path if bs == "4" else bs_path
            folder_name = method[:-2] if method.endswith("TD") else method
            try:
                summary_path = method_dir / f"slo{slo_scale}" / folder_name / "arrival_summerizev2.csv"
                df_sum = pd.read_csv(summary_path)
                sel = df_sum[(df_sum["slo"] == slo_scale) &
                             (df_sum["arrival_rate"] == arrival_rate) &
                             (df_sum["cv_num"] == cv_rate)]
                if sel.empty:
                    name = f"lambda{arrival_rate}x_cv{cv_rate}_bs{bs}"
                    sel = df_sum[df_sum["name"] == name]
                col = "tbt_attainment_with_TD" if method.endswith("TD") else "tbt_attainment_no_TD"
                value = float(sel[col].iloc[0]) if len(sel) == 1 else np.nan
                print(f" method: {method} bs: {bs}, Value: {value}")
                y_vals.append(value)
            except Exception as e:
                print(f"NO: {summary_path}, Error: {e}")
                y_vals.append(np.nan)

        ax.plot(batch_labels, y_vals,
                label=method_labels[m],
                color=colors[m],
                marker=markers[m],
                **style["line"])

    ax.set_ylim(-5, 105)
    ax.set_xticks(batch_labels)
    ax.set_xticklabels(batch_labels, fontsize=style["tick"]["fontsize"])
    ax.tick_params(axis='y', labelsize=style["tick"]["fontsize"], length=0, pad=5)
    ax.yaxis.grid(True, **style["grid"])
    ax.xaxis.grid(True, **style["grid"])
    ax.set_ylabel("TBT SLO (%)", **style["label"])
    ax.set_xlabel("Batch Size", **style["label"])

    for spine in ax.spines.values():
        spine.set_edgecolor(style["spine"]["color"])
        spine.set_alpha(style["spine"]["alpha"])
        spine.set_linewidth(style["spine"]["linewidth"])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(method_list),
               bbox_to_anchor=(0.5, 1.02), columnspacing=0.9,
               fontsize=style["legend"]["fontsize"], frameon=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "bs_tbt_comparison_bar.jpg", bbox_inches="tight", dpi=300)
    plt.savefig(output_dir / "bs_tbt_comparison_bar.pdf", bbox_inches="tight")
    print(f"Saved to: {output_dir / 'bs_tbt_comparison_bar.jpg'}")


def main():
    parser = argparse.ArgumentParser(description="Plot TBT SLO comparison over batch sizes")
    parser.add_argument("bs_path", type=Path, help="Path to batch size experiments (e.g., paper_main_exp_bs)")
    parser.add_argument("base_path", type=Path, help="Path to base experiments (e.g., paper_main_exp_32k)")
    parser.add_argument("--output-dir", type=Path, default=Path("./figures"),
                        help="Directory to save plots (default: ./figures)")
    args = parser.parse_args()
    plot_tbt_slo_comparison(args.bs_path, args.base_path, args.output_dir)


if __name__ == "__main__":
    main()
