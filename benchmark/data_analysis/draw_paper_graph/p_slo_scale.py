import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ast

# ───────────────────────────────────────────────
# 2. CSV 로드 및 synthetic fallback
def load_metrics(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        df["slo_threshold"] = pd.to_numeric(df["slo_threshold"], errors="coerce")
        df["time_between_tokens"] = df["time_between_tokens"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        return df
    except FileNotFoundError:
        print(f"[Warning] File not found: {path}, using synthetic fallback.")
        N = 50
        tbt = [np.random.uniform(0, 2.5, np.random.randint(5,20)) for _ in range(N)]
        return pd.DataFrame({
            "time_between_tokens": tbt,
            "slo_threshold": np.ones(N, dtype=float)
        })
    
style = {
    "spine": {
        "color": "black",
        "alpha": 0.7,
        "linestyle": "-",
        "linewidth": 1.5
    },
}

# ───────────────────────────────────────────────
# 3. 플롯 설정
method_list   = ["NoPrefetch", "Flexgen", "SelectN", "Ours"]
method_labels = ["No Prefetch", "Flexgen", "Placeholder(SelectN)", "Ours"]
colors = [
    "#84C8F4",  # Soft Sky Blue
    "#C59FDB",  # Pastel Lavender
    "#7CD6A4",  # Mint Green
    # "#63D0C2",  # Aqua Teal
    "#E05A4F",  # Coral Red
]
markers = ['o','s','^',
        #    'D',
           'P']

TRACE  = "both_dyn"

metric_list   = ["low","mid","high", "veryhigh"]
metric_labels = ["Low","Mid","High", "Very High"]

sc = 2.5

PERCENTILES  = [90, 95, 99]
x_positions  = range(len(PERCENTILES))

fig, axes = plt.subplots(1, len(metric_list), figsize=(21, 5), sharey=True)
plt.subplots_adjust(
    left=0.05, right=0.99, top=0.93, bottom=0.07,
    wspace=0.075, hspace=0.1
)

for ax, metric, metric_label in zip(axes, metric_list, metric_labels):
    for method, label, color, marker in zip(method_list, method_labels, colors, markers):
        summary_path = Path(
            f"/home/heelim/vllm/outputs/benchmark/paper_main_exp/"
            f"slo{sc}/{method}/summerize.csv"
        )
        ratio_lists = []
        try:
            summary_df = pd.read_csv(summary_path)
            row = summary_df[
                (summary_df["slo"] == sc) &
                (summary_df["method"] == method) &
                (summary_df["trace"] == TRACE) &
                (summary_df["metric"] == metric)
            ]
            ratio_lists.append(float(row["p90_ratio"].iloc[0]))
            ratio_lists.append(float(row["p95_ratio"].iloc[0]))
            ratio_lists.append(float(row["p99_ratio"].iloc[0]))
        except :
            ratio_lists.append(0)
            ratio_lists.append(0)
            ratio_lists.append(0)

        ax.plot(
            x_positions,
            ratio_lists,
            label=label,
            color=color,
            marker=marker,
            linewidth=3,
            markersize=10
        )
    ax.set_title(metric_label, fontsize=35)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"p{p}" for p in PERCENTILES], fontsize=35)
    # ax.set_xlabel("Percentile", fontsize=35)
    # ax.grid(alpha=0.3)


# y축 tick 간격 0.5로 설정, 글자 크기 30
max_ylim = axes[0].get_ylim()[1]
y_ticks = np.arange(0, max_ylim+0.5, 0.5)
axes[0].set_yticks(y_ticks)
for ax in axes:
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)

    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0)

    for spine in ax.spines.values():
        spine.set_edgecolor(style["spine"]["color"])
        spine.set_alpha(style["spine"]["alpha"])
        spine.set_linewidth(style["spine"]["linewidth"])
    
    ax.set_ylim(-0.15, max_ylim + 0.15)

# 공통 y축 레이블 & 범례
axes[0].set_ylabel("SLO Scale", fontsize=35, labelpad=15)
fig.legend(method_labels, loc='upper center', 
           bbox_to_anchor=(0.5, 1.3),
           ncol=len(method_labels),
           fontsize=35, frameon=False)

# 폴더 생성 및 저장

plt.savefig("figures/6_2_tail_tbt.jpg", format='jpg', bbox_inches="tight")
plt.savefig("figures/6_2_tail_tbt.pdf", format='pdf', bbox_inches="tight")
