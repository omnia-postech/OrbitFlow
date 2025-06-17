import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ───────────────────────────────────────────────
# 실험 설정
trace          = "both_dyn"
method_list    = ["Flexgen_TP", "SelectN_TP", "Ours_TP"]
method_labels  = ["FlexGen", "SLO-aware Offloading", "OrbitFlow"]
metric_list    = ["veryhigh",]
metric_labels  = ["",]
slo_scales     = [4.5, 3.5, 2.5, 1.5]
slo_labels     = [str(s) for s in slo_scales]

colors = [
    "#4DA6FF",  # Sky Blue
    "#9F79C1",  # Lavender Purple
    "#FF8C69"   # Coral Orange
]

markers  = ['o', 's', '^']

style = {
    "line":   {"linewidth":3, "markersize":10},
    "tick":   {"labelsize":25},
    "label":  {"fontsize":25, "labelpad":8},
    "title":  {"fontsize":32, "pad":10},
    "legend": {"fontsize":23},
    "spine":  {"color":"black","alpha":0.7,"linewidth":1.5},
}

base_dir = Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp")

# ───────────────────────────────────────────────
# 플롯 준비: 메트릭별 서브플롯
fig, axes = plt.subplots(1, len(metric_list),
                         figsize=(7, 6),
                         sharey=True)
plt.subplots_adjust(left=0.07, right=0.99,
                    top=0.78, bottom=0.12,
                    wspace=0.25)

for i, metric in enumerate(metric_list):
    ax = axes
    for method, label, color, marker in zip(method_list, method_labels, colors, markers):
        y_vals = []
        for sc in slo_scales:
            summary_path = base_dir / f"slo{sc}" / method / "summerize.csv"
            if summary_path.exists():
                df_sum = pd.read_csv(summary_path)
                sel = df_sum[
                    (df_sum["slo"]    == sc) &
                    (df_sum["method"] == method) &
                    (df_sum["trace"]  == trace) &
                    (df_sum["metric"] == metric)
                ]
                y_vals.append(float(sel["tbt_attainment"].iloc[0]) if len(sel)==1 else 0.0)
            else:
                y_vals.append(0.0)
        ax.plot(slo_labels, y_vals,
                label=label, color=color, marker=marker,
                **style["line"])

        print(f"[{method}] trace={trace}, metric={metric}, method={label}: {y_vals}")

    # 90% 점선 추가
    ax.axhline(90, color="gray", linestyle="--", linewidth=2, label="SLO 90%")

    # 축 및 눈금 설정
    ax.set_xticks(slo_labels)
    ax.set_xticklabels(slo_labels, fontsize=style["tick"]["labelsize"])
    ax.tick_params(axis='x', length=0, labelsize=style["tick"]["labelsize"])
    ax.tick_params(axis='y', length=5, labelsize=style["tick"]["labelsize"])
    ax.set_ylim(-5, 105)
    ax.set_yticks([0, 50, 100])

    ax.set_title(metric_labels[i], **style["title"])
    if i == 0:
        ax.set_ylabel("TBT SLO Attainment (%)", **style["label"])
    ax.set_xlabel("SLO Scale", **style["label"])

    for spine in ax.spines.values():
        spine.set_edgecolor(style["spine"]["color"])
        spine.set_alpha(style["spine"]["alpha"])
        spine.set_linewidth(style["spine"]["linewidth"])

# 범례
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels,
           loc="upper center",
           bbox_to_anchor=(0.5, 0.94),
           ncol=6,
           frameon=False,
           **style["legend"])

# 저장
plt.savefig("figures/6_3_tp_tbt.jpg", bbox_inches="tight")
plt.savefig("figures/6_3_tp_tbt.pdf", bbox_inches="tight")