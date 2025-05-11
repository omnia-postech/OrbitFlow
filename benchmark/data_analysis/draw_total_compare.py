import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from pathlib import Path
from typing import List, Optional

import matplotlib.patches as mpatches

# 실험 이름
EXP = "Debug"

# 설정
#데이터를 가져올 떄 쓸 path에서 mehthod 와 trace의 naming
static_methods = ["NoPrefetch", "NextLayer", "Static2"]
dynamic_methods = ["SelectN", "Flexgen", "DistNSingle", "Ours"]
method_list = static_methods + dynamic_methods

trace_list = ["test_fit_static_0", "test_fit_static_2", 
              "test_shortshort_enough", 
              "test_shortlong_less", "test_shortlong_enough", 
              "test_longshort_less", "test_longshort_enough", 
              "test_longlong_less", "test_longlong_enough", 
              "test_mix4_less", "test_mix4_enough",
            ]
trace_labels = [
    "StaticFit-0",        # test_fit_static_0
    "StaticFit-2",        # test_fit_static_2

    "Short-Short (Enough)",   # test_shortshort_enough

    "Short-Long (Low)",       # test_shortlong_less
    "Short-Long (Enough)",       # test_shortlong_enough

    "Long-Short (Low)",       # test_longshort_less
    "Long-Short (Enough)",       # test_longshort_enough

    "Long-Long (Low)",        # test_longlong_less
    "Long-Long (Enough)",        # test_longlong_enough

    "Mix-4 (Low)",            # test_mix4_less
    "Mix-4 (Enough)",            # test_mix4_enough
]

def get_csv_path(exp: str, method: str, trace: str) -> Path:
    return Path(f"/home/xinyuema/vllm/outputs/benchmark/{exp}/{method}/{trace}/outputs.csv")

# 그래프에 나타나는 label naming
static_methods_labels = ["No Prefetch", "Static0", "Static2"]
dynamic_methods_labels = ["SelectN", "Flexgen", "DistNSingle", "Ours"]
method_labels = static_methods_labels + dynamic_methods_labels

metric_list = ["TBT attainment", "TPOT", "E2E throughput", "# Violations"]
metric_list_lables = ["TBT attainment\n(%)", "TPOT\n(token/s)", "E2E throughput\n(token/s)", "# Violations"]

colors = [
    # Static (파란색 계열 - 부드럽고 시인성 높음)
    "#84C8F4",  # NextLayer - 부드러운 파란색
    "#7CD6A4",  # NoPrefetch - 연한 청록색
    "#63D0C2",  # Static2 - 중간 밝기의 민트

    # Dynamic (따뜻한 계열, Ours는 강조)
    "#FAC07D",  # SelectN - 파스텔 오렌지
    "#C59FDB",  # Flexgen - 연한 보라색
    "#F29E9E",  # DistNSingle - 연한 코랄

    "#E05A4F"   # Ours - 강조용 진한 살구+레드 (단독 대비 확보)
]

font_size = 22
style = {
    "bar": {
        "edgecolor": "white",
        "linewidth": 2.5
    },
    "tick": {
        "fontsize": 18
    },
    "label": {
        "fontsize": font_size,
        "labelpad":20,
        # "weight": "bold"
    },
    "title": {
        "fontsize": font_size,
        "weight": "bold",
        "pad": 10
    },
    "legend": {
        "fontsize": font_size,
        # "loc": "upper center",
        # "bbox_to_anchor": (0.5, 1.2),
        # "ncol": len(method_list)
    },
    "spine": {
        "color": "gray",
        "alpha": 0.5,
        "linestyle": "-",
        "linewidth": 2
    },
    "grid": {
        "color": "gray",
        "linestyle": "--",
        "linewidth": 2,
        "alpha": 0.5
    },
    "text": {
        "fontsize": 14,
        "weight": "bold"
    },
}


def load_metrics(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    numeric_cols: List[str] = [
        "arrival_time", "first_scheduled_time", "finished_time",
        "time_to_first_token", "slo_threshold", "slo_violations",
        "stall_duration", "decode_length", "end_to_end_time", "decode_time", "time_per_output_token"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("time_between_tokens", "stall_times", "stall_durations"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
    return df

def compute_metrics(df: pd.DataFrame) -> dict:
    num_requests = len(df)
    if num_requests == 0:
        return {"TBT attainment": 0, "TPOT": 0, "E2E throughput": 0, "# Violations": 0}

    mean_e2e = df["end_to_end_time"].mean()
    tpot = df["time_per_output_token"].mean()

    tbt_series = (df["decode_length"] - df["slo_violations"]) / df["decode_length"]
    tbt = tbt_series.mean()

    return {
        "TBT attainment": tbt,
        "TPOT": tpot,
        "E2E throughput": mean_e2e,
        "# Violations": df["slo_violations"].sum(),
    }

# 데이터 수집
csv_path_list: Optional[dict] = None
data = {metric: {trace: {} for trace in trace_list} for metric in metric_list}

for method in method_list:
    for trace in trace_list:
        try:
            if csv_path_list and (method, trace) in csv_path_list:
                path = csv_path_list[(method, trace)]
            else:
                path = get_csv_path(EXP, method, trace)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            df = load_metrics(path)
            result = compute_metrics(df)
            for metric in metric_list:
                data[metric][trace][method] = result[metric]
            
            print(f"✅ Success to load {path}")

        except Exception as e:
            print(f"[경고] {method} - {trace}: {e}")
            for metric in metric_list:

                if metric == "E2E throughput" :
                    data[metric][trace][method] = 50 
                elif metric == "# Violations" :
                    data[metric][trace][method] = 0 
                else :
                    data[metric][trace][method]= 0.02

# 시각화 준비
fig, axes = plt.subplots(
    len(metric_list), 
    len(trace_list), 
    figsize=(4*len(trace_list), 4*len(metric_list)), 
    squeeze=False,
)

# metric별 최대값 계산
metric_max_values = {
    metric: max([
        max(data[metric][trace].values(), default=0)
        for trace in trace_list
    ]) for metric in metric_list
}

for i, metric in enumerate(metric_list):
    for j, (trace, trace_label) in enumerate(zip(trace_list, trace_labels)):
        ax = axes[i][j]
        trace_data = data[metric][trace]
        values = [trace_data.get(method, 0) for method in method_list]

        ax.bar(
            method_labels, values, 
            color=colors[:len(method_list)],
            **style["bar"]
        )

        if i==0:
            ax.set_title(f"{trace_label}", **style["title"])

        
        # ax.set_ylim(0, metric_max_values[metric] * 1.15 if metric_max_values[metric] > 0 else 1)
        ax.tick_params(axis='y', labelsize=style["tick"]["fontsize"])

        ax.tick_params(axis='x', which='both', length=0)
        ax.tick_params(axis='y', which='both', length=0)

        ax.yaxis.grid(True, **style["grid"])
        # ax.xaxis.grid(True, **style["grid"])
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_edgecolor(style["spine"]["color"])
            spine.set_alpha(style["spine"]["alpha"])
            spine.set_linewidth(style["spine"]["linewidth"])

        # if i != len(metric_list) - 1:
        ax.set_xticklabels([])
        # else:
        #     ax.set_xticklabels(method_labels, **style["tick"])

        # y축 레이블과 눈금은 첫 번째 열만 표시
        if j == 0:
            ax.set_ylabel(metric_list_lables[i], **style["label"])
        # else:
            # ax.set_yticklabels([])

# yㅊ축 레이블 정렬
fig.align_ylabels(axes[:, 0])

# subplot 간격 조절
plt.subplots_adjust(
    left=0.05,
    right=0.99,
    top=0.93,
    bottom=0.07,
    wspace=0.2,
    hspace=0.1
)

static_legend_handles = [
    mpatches.Patch(color=colors[i], label=static_methods_labels[i])
    for i in range(len(static_methods_labels))
]
dynamic_legend_handles = [
    mpatches.Patch(color=colors[i + len(static_methods_labels)], label=dynamic_methods_labels[i])
    for i in range(len(dynamic_methods_labels))
]

# 범례 추가 (두 줄)
fig.legend(
    handles=static_legend_handles,
    # loc= "upper left",
    bbox_to_anchor=(0.33, 1.01),
    ncol= len(static_methods_labels),
    **style["legend"]
)

fig.legend(
    handles=dynamic_legend_handles,
    # loc= "upper right",
    bbox_to_anchor=(0.66, 1.01),
    ncol= len(dynamic_methods_labels),
    **style["legend"]
)

# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("multi_metric_trace_barplot.jpg", format='jpg', bbox_inches="tight")
# plt.savefig("multi_metric_trace_barplot.pdf", format='pdf', bbox_inches="tight")
