import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────
# 1. 설정
METHODS      = [
    "NextLayer", 
    "Static1", "Flexgen", "SelectN", "DistNSingle", "OursTD",]
METHOD_LABS  = [
    "DeepSpeed", 
    "FlexGen", "FlexGen+", "SLO-aware Offloading", "Dynamic Heuristic", "OrbitFlow"]

arrival_rate  = 2.0

cv_rate = 1

SLO_SCALES   = [2.5, 1.5, 1]
SLO_LABELS   = [f"{str(s)}" for s in SLO_SCALES]
parser = argparse.ArgumentParser()
# base_dir 을 옵션이 아닌 "선택적" 포지셔널 인자로 받기 (없으면 기본값 사용)
parser.add_argument(
    "base_dir",
    nargs="?",                                  # 0개 또는 1개
    type=Path,
    default=Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp_32k"),
    help="실험 결과가 들어있는 최상위 디렉토리 (기본값 사용 시 생략)"
)
args = parser.parse_args()
BASE_DIR = args.base_dir

style = {
    "line":   {"linewidth":3, "markersize":10},
    "spine":  {"color":"black", "alpha":0.7, "linewidth":1.5},
    "title":  {"fontsize":25, "pad":8},
    "label":  {"fontsize":25, "labelpad":8},
    "legend": {"fontsize":25},
    "tick":   {"labelsize":27},
    "grid": {
        "color": "gray",
        "linestyle": "-",
        "linewidth": 3,
        "alpha": 0.2
    },
}

colors = [
    "#4DA6FF",  # Sky Blue
    "#76C7AE",  # Pastel Mint
    "#508776",  # Pastel Mint Green
    "#9F79C1",  # Lavender Purple
    "#FFB3BA",  # Pastel Pink
    "#FF8C69",   # Coral Orange,
]

bar_width = 0.9 / len(METHODS)
positions = np.arange(len(SLO_LABELS))
y_max = 3.4

# ───────────────────────────────────────────────
# 2. 플롯 초기화 (sharey=False)
fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=False)
plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.12, wspace=0.23)

# ───────────────────────────────────────────────
# 3. 서브플롯별 데이터 플로팅 (바 차트)
for i, (method, label) in enumerate(zip(METHODS, METHOD_LABS)):
    y_vals = []
    for sc, sc_label in zip(SLO_SCALES, SLO_LABELS):
        if method.endswith("TD"):
            summary_path = BASE_DIR / f"slo{sc}" / method[:-2] / "arrival_summerizev2.csv"
        else:
            summary_path = BASE_DIR / f"slo{sc}" / method / "arrival_summerizev2.csv"
        try:
            df_sum = pd.read_csv(summary_path)
            sel = df_sum[(df_sum["slo"] == sc) 
                        & (df_sum["arrival_rate"] == arrival_rate)
                        & (df_sum["cv_num"] == cv_rate)
                        ]
            slo_thr = float(sel["slo_threshold_mean"].iloc[0]) if len(sel)==1 else np.nan
            if len(sel)==1:
                value = float(sel["p95_ratio"].iloc[0])  
                print(f"Method: {method}, SLO: {sc}, arrival_rate: {arrival_rate}, Value: {value}")
            else:
                value = np.nan
                print(f"Method: {method}, SLO: {sc}, arrival_rate: {arrival_rate}, Value: None")
            y_vals.append(value)
        except:
            print(summary_path)
            y_vals.append(np.nan)

    offsets = (i - (len(METHODS)-1)/2) * bar_width
    bars = ax.bar(positions + offsets, y_vals,
                  width=bar_width,
                  label=label,
                  color=colors[i],
                  edgecolor="white")

    # y_max보다 큰 값에 대해 텍스트 추가
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height) and height > y_max:
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # x좌표: 막대 중심
                y_max - 0.35,                             # y좌표: y_max 위치
                f"{height*1000:.0f} ms",             # 값: ms 단위로 변환 (소수점 없음)
                ha='center',                      # 수평 정렬: 중심
                va='bottom',                      # 수직 정렬: 막대 위
                fontsize=style["tick"]["labelsize"]-6,  # 폰트 크기
                color="black"
            )


ax.set_xlabel("SLO Scale", **style["label"])

ax.set_ylabel("P95 TBT (ms)", **style["label"])

# x축 눈금
ax.set_xticks(positions)
ax.set_xticklabels(SLO_LABELS, fontsize=style["tick"]["labelsize"])
ax.tick_params(axis='x', length=0)

# 유효한 bar height만 가져옴
# heights = [bar.get_height() for bar in ax.patches if not np.isnan(bar.get_height())]
max_y = y_max

# 최대 tick 수를 제한
max_tick_count = 5

# 적절한 간격 계산 (1, 2, 5, 10, ...)
raw_step = max_y / max_tick_count
# step을 정수이면서 보기 좋게 반올림 (1, 2, 5, 10...)
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

step = round_step(raw_step)

# 실제 눈금 리스트 생성 (1부터 시작, 0은 제외)
ticks = list(range(step, int(np.ceil(max_y)) + 1, step))
ax.set_yticks(ticks)
ax.set_yticklabels([f"{t*1000}" for t in ticks], fontsize=style["tick"]["labelsize"])
ax.set_ylim(0, y_max)

ax.yaxis.grid(True, **style["grid"])
# 스파인 스타일
for spine in ax.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# ───────────────────────────────────────────────
# 4. 범례 (상단 중앙)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels,
           loc="upper center", bbox_to_anchor=(0.5, 1.15),
           ncol=3, 
           **style["legend"],
           columnspacing=0.9,
           frameon=False)

# ───────────────────────────────────────────────
# 5. 저장
plt.savefig("/home/heelim/vllm/benchmark/data_analysis/figures/6_2_tail_p95_by_slo.jpg", bbox_inches="tight")
plt.savefig("/home/heelim/vllm/benchmark/data_analysis/figures/6_2_tail_p95_by_slo.pdf", bbox_inches="tight")