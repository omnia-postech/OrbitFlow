import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ast

# ───────────────────────────────────────────────
# 1. percentile→(Pxx/SLO) 리스트 추출 함수
def extract_percentile_ratio_lists(df: pd.DataFrame, percentiles: list[int]) -> list[list[float]]:
    """각 퍼센타일에 대해 Pxx/SLO 임계치 비율 리스트 반환."""
    lists = [[] for _ in percentiles]
    if {"time_between_tokens", "slo_threshold"}.issubset(df.columns):
        # system-wise SLO
        if df["slo_threshold"].nunique() == 1:
            all_tbt = []
            for tb in df["time_between_tokens"]:
                all_tbt.extend(tb if isinstance(tb, (list, tuple, np.ndarray)) else [tb])
            thr = float(df["slo_threshold"].iloc[0])
            for i, pct in enumerate(percentiles):
                px = np.percentile(all_tbt, pct)
                lists[i].append(px / thr)
        else:
            # per-request SLO
            for tb, thr in zip(df["time_between_tokens"], df["slo_threshold"]):
                if not isinstance(tb, (list, tuple, np.ndarray)) or thr <= 0:
                    continue
                thr_val = float(np.mean(thr)) if isinstance(thr, (list, tuple, np.ndarray)) else float(thr)
                for i, pct in enumerate(percentiles):
                    px = np.percentile(tb, pct)
                    lists[i].append(px / thr_val)
    return lists

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

# ───────────────────────────────────────────────
# 3. 플롯 설정
METHODS       = ["Flexgen", "DeepSpeed", "SelectN", "NoPrefetch", "Ours"]
METHOD_LABELS = ["Flexgen","DeepSpeed","Placeholder(SelectN)","No Prefetch","Ours"]
COLORS        = ["#84C8F4", "#C59FDB", "#7CD6A4", "#63D0C2", "#E05A4F"]
MARKERS       = ["o", "s", "^", "d", "*"]

TRACE_LIST  = [
    "test_fit_static_0",
    "test_shortshort_enough",
    "test_shortlong_less",
    "test_shortlong_enough"
]
PERCENTILES  = [90, 95, 99]
x_positions  = range(len(PERCENTILES))

fig, axes = plt.subplots(1, 4, figsize=(35, 6), sharey=True)
plt.subplots_adjust(
    left=0.05, right=0.99, top=0.93, bottom=0.07,
    wspace=0.05, hspace=0.1
)

for ax, trace in zip(axes, TRACE_LIST):
    for method, label, color, marker in zip(METHODS, METHOD_LABELS, COLORS, MARKERS):
        path = Path(f"/home/heelim/vllm/outputs/benchmark/exp/{method}/{trace}/output.csv")
        df = load_metrics(path)
        ratio_lists = extract_percentile_ratio_lists(df, PERCENTILES)

        ax.plot(
            x_positions,
            ratio_lists,
            label=label,
            color=color,
            marker=marker,
            linewidth=3,
            markersize=10
        )
    ax.set_title(trace, fontsize=35)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"p{p}" for p in PERCENTILES], fontsize=35)
    # ax.set_xlabel("Percentile", fontsize=35)
    ax.grid(alpha=0.3)

# y축 tick 간격 0.5로 설정, 글자 크기 30
max_ylim = axes[0].get_ylim()[1]
y_ticks = np.arange(0, max_ylim + 0.5, 0.5)
axes[0].set_yticks(y_ticks)
for ax in axes:
    ax.tick_params(axis='y', labelsize=35)

# 공통 y축 레이블 & 범례
axes[0].set_ylabel("SLO Scale", fontsize=35)
fig.legend(METHOD_LABELS, loc='upper center', 
           bbox_to_anchor=(0.52, 1.25),
           ncol=len(METHOD_LABELS),
           fontsize=35, frameon=False)

# 폴더 생성 및 저장

plt.savefig("figures/p_slo.jpg", format='jpg', bbox_inches="tight")
