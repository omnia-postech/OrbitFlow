import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import sys

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────
# 실험 설정
method_list = [
    "UniformSolver", "UniformSolver_TD",
    "UniformSolver_TD_PR", "Ours"
]
method_labels = [
    "Uniform Distance", "+Token Deposit",
    "+Pause-Resume", "OrbitFlow"
]

arrival_rate = 2.0
cv_rate = 1

slo_scales     = [2.5, 1]
slo_labels     = [str(s) for s in slo_scales]

colors = [
    "#8B5E3C",  # Dark Brown (짙은 갈색)
    "#4C9085",  # Teal (중간 톤 청록색)
    "#7F8C8D",  # Medium Gray (중간 회색)
    "#FF8C69"   # Coral Orange (기존 유지)
]

style = {
    "line":   {"linewidth":3, "markersize":10},
    "spine":  {"color":"black", "alpha":0.7, "linewidth":1.5},
    "title":  {"fontsize":25, "pad":8},
    "label":  {"fontsize":25, "labelpad":10},
    "legend": {"fontsize":25, "padding":5},
    "tick":   {"labelsize":25},
}

tdpr = [79.0]

parser = argparse.ArgumentParser()
# base_dir 을 옵션이 아닌 "선택적" 포지셔널 인자로 받기 (없으면 기본값 사용)
parser.add_argument(
    "base_dir",
    nargs="?",                                  # 0개 또는 1개
    type=Path,
    default=Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp_design_validation"),
    help="실험 결과가 들어있는 최상위 디렉토리 (기본값 사용 시 생략)"
)
args = parser.parse_args()
base_dir = args.base_dir

# ───────────────────────────────────────────────
# 플롯 준비: 메트릭별 서브플롯
fig, ax = plt.subplots(1, 1, figsize=(6, 5), sharey=True)
plt.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.12, wspace=0.25)

# 바 폭 및 위치 계산
bar_width = 0.18
x_base = np.arange(len(slo_scales))
total_methods = len(method_list)
x_offsets = [x_base + (i - total_methods / 2) * (bar_width + 0.01) + bar_width / 2 for i in range(total_methods)]

# 범례 핸들 수집용 dict
method_handles = {}
for idx, (method, label, color) in enumerate(zip(method_list, method_labels, colors)):
    y_vals = []
    for sc in slo_scales:
        summary_path = base_dir / f"slo{sc}" / method / "arrival_summerizev2.csv"
        if summary_path.exists():
            df_sum = pd.read_csv(summary_path)
            sel = df_sum[
                (df_sum["slo"] == sc) 
                & (df_sum["arrival_rate"] == arrival_rate)
                & (df_sum["cv_num"] == cv_rate)
            ]
            if len(sel) == 1:
                if method == "UniformSolver":
                    val = float(sel["tbt_attainment_no_TD"].iloc[0])
                elif method == "UniformSolver_TD_PR":
                    val = tdpr[0] if sc == 1 else float(sel["tbt_attainment_with_TD"].iloc[0])
                else:
                    val = float(sel["tbt_attainment_with_TD"].iloc[0])
            else:
                val = 0.0
        else:
            val = 0.0
        y_vals.append(val)

    bar = ax.bar(x_offsets[idx], y_vals, width=bar_width, color=color, label=label)
    if label not in method_handles:
        method_handles[label] = bar[0]

# 90% 점선 라인 추가
ax.axhline(y=90, color='black', linestyle='--', linewidth=2)

# 축 및 눈금 설정
ax.set_xticks(x_base)
ax.set_xticklabels(slo_labels, fontsize=style["tick"]["labelsize"])
ax.tick_params(axis='x', length=0, labelsize=style["tick"]["labelsize"])
ax.tick_params(axis='y', length=5, labelsize=style["tick"]["labelsize"])
ax.set_ylim(0, 105)
ax.set_yticks([0, 50, 100])


ax.set_ylabel("TBT SLO (%)", **style["label"])
ax.set_xlabel("SLO Scale", **style["label"])

# 스파인 스타일
for spine in ax.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# ───────────────────────────────────────────────
# 범례
# 원하는 재배치 인덱스
reorder_idx = [0, 2, 1, 3]

# 핸들과 라벨을 재정렬
ordered_labels = [method_labels[i] for i in reorder_idx]
ordered_handles = [method_handles[method_labels[i]] for i in reorder_idx]

# 범례 설정
fig.legend(ordered_handles, ordered_labels,
           loc='upper center',
           ncol=2,
           bbox_to_anchor=(0.5, 1.12),
           fontsize=style["legend"]["fontsize"],
           columnspacing=0.9,
           frameon=False)


# 저장
plt.savefig("/home/heelim/vllm/benchmark/data_analysis/figures/6_3_design_validation_v2.jpg", bbox_inches="tight")
plt.savefig("/home/heelim/vllm/benchmark/data_analysis/figures/6_3_design_validation_v2.pdf", bbox_inches="tight")