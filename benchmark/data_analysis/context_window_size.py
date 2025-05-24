import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ast

# ───────────────────────────────────────────────
# (기존 load_metrics, compute_throughput, load_metrics_for_slo_scales 함수는 그대로 사용)

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
# 스타일/데이터 정의
TRACE       = "test_fit_static_0"
METHODS     = ["Flexgen","DeepSpeed","SelectN","NoPrefetch","Ours"]
LABELS      = ["Flexgen","DeepSpeed","Placeholder(SelectN)","No Prefetch","Ours"]
COLORS      = ["#84C8F4","#C59FDB","#7CD6A4","#63D0C2","#E05A4F"]
MARKERS     = ['o','s','^','d','*']
PERCENTILES  = [90, 95, 99]
Context_window  = [8, 32, 64, 128]
Context_window_labels = [f"{str(s)}k" for s in Context_window]  # x축 표기용 문자열
TICK_FONT   = 18
LABEL_FONT  = 18
LINE_KW     = dict(linewidth=3, markersize=10)

# ───────────────────────────────────────────────
# 단일 플롯
fig, ax = plt.subplots(figsize=(8, 6))

for m,method in enumerate(METHODS):
    path = Path(f"/home/heelim/vllm/outputs/benchmark/exp/{method}/{TRACE}/output.csv")
    df = load_metrics(path)
    ratio_lists = extract_percentile_ratio_lists(df, PERCENTILES)
    ax.plot(
        range(len(Context_window_labels)), 
        ratio_lists,
        color=COLORS[m], marker=MARKERS[m],
        label=LABELS[m],
        **LINE_KW
    )

# x축: SLO scale (4→1)
ax.set_xticks(range(len(Context_window_labels)))
ax.set_xticklabels(Context_window_labels, fontsize=TICK_FONT)
# ax.set_xlim(SLO_SCALES[0], SLO_SCALES[-1])
ax.tick_params(axis='y', labelsize=TICK_FONT, length=0)
ax.tick_params(axis='x', length=0)

# 레이블 / 타이틀
ax.set_xlabel("Context Window Size", fontsize=LABEL_FONT, labelpad=8)
ax.set_ylabel(f"SLO Attainment", fontsize=LABEL_FONT, labelpad=8)
# ax.set_title(f"{TRACE}  —  {METRIC} Throughput", fontsize=LABEL_FONT, pad=12)

# 90% 기준선 (optional)
# ax.axhline(0.9 * ax.get_ylim()[1], color="gray", linestyle="--", linewidth=1, alpha=0.6)

# 그리드/스파인
ax.grid(True, linestyle="--", alpha=0.3)
for spine in ax.spines.values():
    spine.set_edgecolor("gray")
    spine.set_linewidth(1.5)
    spine.set_alpha(0.5)

# 범례
ax.legend(loc="upper center", 
          ncol=len(METHODS), 
          fontsize=LABEL_FONT, 
          frameon=False,
          bbox_to_anchor=(0.5,1.2), 
          )

# 저장
plt.savefig("figures/conntext_window.jpg", format='jpg', bbox_inches="tight")
# plt.savefig("figures/tbt_total_compare.pdf", format='pdf', bbox_inches="tight")