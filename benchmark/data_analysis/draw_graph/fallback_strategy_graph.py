import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import argparse

def plot_fallback_comparison(base_dirs, output_dir):
    method_fallback = "Ours"
    base_labels = ["Random", "Shortest", "Longest"]
    colorsfd = ["#6B705C", "#A47149", "#FF8C69"]
    slo_scales = [1]
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

    fig_fb, ax_fb = plt.subplots(figsize=(6.5, 5.5))
    bar_width = 0.18
    x_base = np.arange(len(slo_scales))
    total_methods = len(base_dirs)
    x_offsets = [
        x_base + (i - total_methods / 2) * (bar_width + 0.01) + bar_width / 2
        for i in range(total_methods)
    ]

    for idx, (base_dir, label, color) in enumerate(zip(base_dirs, base_labels, colorsfd)):
        y_vals = []
        for sc in slo_scales:
            summary_path = base_dir / f"slo{sc}" / method_fallback / "arrival_summerizev2.csv"
            if summary_path.exists():
                try:
                    df_sum = pd.read_csv(summary_path)
                    sel = df_sum[
                        (df_sum["slo"] == sc) &
                        (df_sum["name"] == "fallback_xinyue")
                    ]
                    val = float(sel["tbt_attainment_with_TD"].iloc[0]) if len(sel) == 1 else 0.0
                except Exception:
                    val = 0.0
            else:
                val = 0.0
            print(f"Fallback Strategy: {label}, SLO Scale: {sc}, Value: {val}")
            y_vals.append(val)

        ax_fb.bar(x_offsets[idx], y_vals, width=bar_width, color=color, label=label)

    ax_fb.set_xticks(x_base)
    ax_fb.set_xticklabels([" "] * len(slo_scales), fontsize=style["tick"]["fontsize"])
    ax_fb.set_ylim(0, 105)
    ax_fb.set_xlabel("(a) Fallback Strategy", **style["label"])
    ax_fb.set_ylabel("TBT SLO (%)", **style["label"])
    ax_fb.set_yticks([0, 25, 50, 75, 100])
    ax_fb.tick_params(axis='both', length=0, labelsize=style["tick"]["fontsize"])

    ax_fb.yaxis.grid(True, **style["grid"])
    for spine in ax_fb.spines.values():
        spine.set_edgecolor(style["spine"]["color"])
        spine.set_alpha(style["spine"]["alpha"])
        spine.set_linewidth(style["spine"]["linewidth"])

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colorsfd]
    ax_fb.legend(
        handles,
        base_labels,
        loc='upper center',
        ncol=len(base_labels),
        bbox_to_anchor=(0.5, 1.22),
        fontsize=style["legend"]["fontsize"],
        handletextpad=0.2,
        columnspacing=0.9,
        frameon=False
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "fall_back.jpg", bbox_inches="tight")
    plt.savefig(output_dir / "fall_back.pdf", bbox_inches="tight")
    print(f"✅ 그래프 저장 완료: {output_dir / 'fall_back.jpg'}")


def main():
    parser = argparse.ArgumentParser(description="Plot fallback strategy comparison")
    parser.add_argument(
        "base_dirs", type=Path, nargs=3,
        help="Exactly 3 base directories for Random, Shortest, and Longest fallback strategies"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./figures"),
        help="Directory to save the output plots (default: ./figures)"
    )
    args = parser.parse_args()
    plot_fallback_comparison(args.base_dirs, args.output_dir)


if __name__ == "__main__":
    main()
