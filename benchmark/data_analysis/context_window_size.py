import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ast

# ───────────────────────────────────────────────
# CSV 로드 및 synthetic fallback
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
        tbt = [np.random.uniform(0, 2.5, np.random.randint(5, 20)) for _ in range(N)]
        return pd.DataFrame({
            "time_between_tokens": tbt,
            "slo_threshold": np.ones(N, dtype=float)
        })

# ───────────────────────────────────────────────
# 스타일/데이터 정의
TRACE       = "test_fit_static_0"
METHODS     = ["Flexgen", "DeepSpeed", "SelectN", "NoPrefetch", "Ours"]
LABELS      = ["Flexgen", "DeepSpeed", "Placeholder(SelectN)", "No Prefetch", "Ours"]
COLORS      = ["#84C8F4", "#C59FDB", "#7CD6A4", "#63D0C2", "#E05A4F"]
MARKERS     = ['o', 's', '^', 'd', '*']
Context_window = [8, 32, 64, 128]
Context_window_labels = [f"{str(s)}k" for s in Context_window]  # x축 표기용 문자열
TICK_FONT   = 18
LABEL_FONT  = 18
LINE_KW     = dict(linewidth=3, markersize=10)

# ───────────────────────────────────────────────
# 단일 플롯
fig, ax = plt.subplots(figsize=(8, 6))

for m, method in enumerate(METHODS):
    y_values = []
    for context_size in Context_window:
        path = Path(f"/home/heelim/vllm/outputs/benchmark/exp/{method}/{TRACE}/{context_size}/output.csv")
        df = load_metrics(path)
        # slo_threshold 평균값 사용
        slo_mean = pd.to_numeric(df["slo_threshold"], errors="coerce").mean()
        y_values.append(slo_mean if not np.isnan(slo_mean) else 0.0)

    ax.plot(
        range(len(Context_window_labels)),
        y_values,
        color=COLORS[m], marker=MARKERS[m],
        label=LABELS[m],
        **LINE_KW
    )

# x축: Context Window Size
ax.set_xticks(range(len(Context_window_labels)))
ax.set_xticklabels(Context_window_labels, fontsize=TICK_FONT)
ax.tick_params(axis='y', labelsize=TICK_FONT, length=0)
ax.tick_params(axis='x', length=0)

# 레이블 / 타이틀
ax.set_xlabel("Context Window Size", fontsize=LABEL_FONT, labelpad=8)
ax.set_ylabel("SLO Threshold", fontsize=LABEL_FONT, labelpad=8)

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
          bbox_to_anchor=(0.5, 1.2),
          )

# 저장
plt.savefig("figures/context_window_slo_threshold.jpg", format='jpg', bbox_inches="tight")
# plt.savefig("figures/context_window_slo_threshold.pdf", format='pdf', bbox_inches="tight")
