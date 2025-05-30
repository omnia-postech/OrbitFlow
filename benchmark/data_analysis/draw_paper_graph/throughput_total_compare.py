import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ───────────────────────────────────────────────
# 설정
trace = "both_dyn"
method_list   = ["NoPrefetch", "Flexgen", "SelectN", "Ours"]
method_labels = ["No Prefetch", "Flexgen", "Placeholder(SelectN)", "Ours"]
metric_list   = ["low","mid","high","veryhigh"]
metric_labels = ["Low","Mid","High","Very High"]

slo_scales = [5.5, 4.5, 3.5, 2.5, 1.5]
slo_labels = [str(s) for s in slo_scales]

colors  = ["#84C8F4","#C59FDB","#7CD6A4","#E05A4F"]
markers = ['o','s','^','P']

style = {
    "line":   {"linewidth":3,"markersize":10},
    "tick":   {"fontsize":18},
    "label":  {"fontsize":19,"labelpad":5},
    "legend": {"fontsize":19},
    "spine":  {"color":"black","alpha":0.7,"linewidth":1.5},
}

base_dir = Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp")

# ───────────────────────────────────────────────
# 플롯 준비 (sharey=False로 모든 y축 표시)
fig, axes = plt.subplots(1, len(metric_list),
                         figsize=(21, 5),
                         sharey=False)
plt.subplots_adjust(left=0.07, right=0.99,
                    top=0.80, bottom=0.12,
                    wspace=0.25)

# ───────────────────────────────────────────────
for i, metric in enumerate(metric_list):
    ax = axes[i]
    for m, method in enumerate(method_list):
        thr_vals = []
        for sc in slo_scales:
            summary_path = base_dir / f"slo{sc}" / method / "summerize.csv"
            if summary_path.exists():
                df = pd.read_csv(summary_path)
                sel = df[
                    (df.slo==sc)&
                    (df.method==method)&
                    (df.trace==trace)&
                    (df.metric==metric)
                ]
                thr_vals.append(
                    sel["throughput_tokens_per_sec"].iloc[0]
                    if len(sel)==1 else 0.0
                )
            else:
                thr_vals.append(0.0)
        ax.plot(slo_scales, thr_vals,
                label=method_labels[m],
                color=colors[m],
                marker=markers[m],
                **style["line"])

    # x축 설정
    ax.set_xticks(slo_scales)
    ax.set_xticklabels(slo_labels,
                       fontsize=style["tick"]["fontsize"])
    ax.tick_params(axis='x', length=0)
    # y축 눈금 표시
    ax.tick_params(axis='y',
                   labelsize=style["tick"]["fontsize"],
                   length=5)

    # 타이틀
    ax.set_title(metric_labels[i],
                 fontsize=style["label"]["fontsize"],
                 pad=10)

    # y축 레이블: 첫 서브플롯에만
    if i == 0:
        ax.set_ylabel("Throughput\n(tokens/s)",
                      fontsize=style["label"]["fontsize"],
                      labelpad=10)

    # x축 레이블: 마지막 서브플롯에만
    # if i == len(metric_list) - 1:
    ax.set_xlabel("SLO Scale",
                      fontsize=style["label"]["fontsize"],
                      labelpad=10)

    # 스파인 스타일
    for spine in ax.spines.values():
        spine.set_edgecolor(style["spine"]["color"])
        spine.set_alpha(style["spine"]["alpha"])
        spine.set_linewidth(style["spine"]["linewidth"])

# 범례를 더 위로
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="upper center",
           bbox_to_anchor=(0.5, 1.05),
           ncol=len(method_list),
           **style["legend"])

# 전체 y축 텍스트
# fig.text(0.02, 0.5, "Memory Pressure",
#          va='center', rotation='vertical',
#          fontsize=19)

plt.savefig("figures/6_3_throughput_total_compare.jpg",bbox_inches="tight")
plt.savefig("figures/6_3_throughput_total_compare.pdf",bbox_inches="tight")

