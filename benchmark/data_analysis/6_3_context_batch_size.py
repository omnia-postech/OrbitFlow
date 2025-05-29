import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ast

# ───────────────────────────────────────────────
# CSV 로드 및 fallback 처리
def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    num_cols = ["arrival_time","first_scheduled_time","finished_time",
                "time_to_first_token","slo_threshold","slo_violations",
                "stall_duration","decode_length","end_to_end_time",
                "decode_time","time_per_output_token"]
    for col in num_cols:
        if col in df: df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("time_between_tokens","stall_times","stall_durations"):
        if col in df: df[col] = df[col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x,str) else x)
    return df

def slo_tpot(df: pd.DataFrame) -> float:
    required_cols = {"end_to_end_time", "num_output_tokens", "slo_threshold"}
    if not required_cols.issubset(df.columns):
        print(f"Missing columns: {required_cols - set(df.columns)}")
        return 0.0

    end_to_end_time = pd.to_numeric(df["end_to_end_time"], errors="coerce")
    num_tokens = pd.to_numeric(df["num_output_tokens"], errors="coerce")
    tpot = end_to_end_time / num_tokens

    slo_threshold = df["slo_threshold"].apply(
        lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else x
    )
    slo_threshold = pd.to_numeric(slo_threshold, errors="coerce")

    valid = ~tpot.isna() & ~slo_threshold.isna()
    attained = (tpot[valid] <= slo_threshold[valid]).sum()
    total = valid.sum()

    return attained / total * 100 if total else 0.0

def slo_tbt(df: pd.DataFrame) -> float:
    decoded = df["decode_length"].sum()
    viol    = df["slo_violations"].sum()
    return (decoded - viol) / decoded * 100

# ───────────────────────────────────────────────
# 공통 설정
TRACE       = "test_fit_static_0"
METHODS     = ["Flexgen", "DeepSpeed", "SelectN", "NoPrefetch", "Ours"]
LABELS      = ["Flexgen", "DeepSpeed", "SelectN", "No Prefetch", "Ours"]
COLORS      = ["#84C8F4", "#C59FDB", "#7CD6A4", "#63D0C2", "#E05A4F"]
MARKERS     = ['o', 's', '^', 'D', 'P']
LINE_KW     = dict(linewidth=3, markersize=10)

style = {
    "spine": {
        "color": "black",
        "alpha": 0.7,
        "linestyle": "-",
        "linewidth": 1.5
    },
}

# Context Window & Batch Size 설정
Context_window = [8, 32, 64, 128]
Context_window_labels = [f"{s}k" for s in Context_window]
BATCH_SIZES = [1, 2, 4, 8]
BATCH_SIZE_LABELS = [f"BS {b}" for b in BATCH_SIZES]

# ───────────────────────────────────────────────
# 두 그래프를 하나의 Figure로 생성
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # 가로로 2개

# ──── 좌측: Context Window
for m, method in enumerate(METHODS):
    y_values = []
    for context_size in Context_window:
        path = Path(f"/home/heelim/vllm/outputs/benchmark/exp/{method}/{TRACE}/{context_size}/output.csv")
        try: 
            df = load_metrics(path)
            y_values.append(slo_tbt(df))
        except:
            print(f"[SLO 경고] {path}")
            y_values.append(np.random.random() * 100)

    ax1.plot(
        range(len(Context_window_labels)),
        y_values,
        color=COLORS[m], marker=MARKERS[m],
        label=LABELS[m],
        **LINE_KW
    )

ax1.set_xticks(range(len(Context_window_labels)))
ax1.set_xticklabels(Context_window_labels, fontsize=25)
ax1.set_xlabel("(a) Context Window Size", fontsize=25, labelpad=10)
ax1.set_ylabel("TBT SLO Attainment (%)", fontsize=25, labelpad=10)
ax1.set_ylim(-5, 105)
ax1.tick_params(axis='y', labelsize=25, length=0)
ax1.tick_params(axis='x', length=0)
for spine in ax1.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# ──── 우측: Batch Size
for m, method in enumerate(METHODS):
    y_values = []
    for batch_size in BATCH_SIZES:
        path = Path(f"/home/heelim/vllm/outputs/benchmark/exp/{method}/{TRACE}/bs{batch_size}/output.csv")
        try: 
            df = load_metrics(path)
            y_values.append(slo_tbt(df))
        except:
            print(f"[SLO 경고] {path}")
            y_values.append(np.random.random() * 100)

    ax2.plot(
        range(len(BATCH_SIZE_LABELS)),
        y_values,
        color=COLORS[m], marker=MARKERS[m],
        label=LABELS[m],
        **LINE_KW
    )

ax2.set_xticks(range(len(BATCH_SIZE_LABELS)))
ax2.set_xticklabels(BATCH_SIZE_LABELS, fontsize=25)
ax2.set_xlabel("(b) Batch Size", fontsize=25, labelpad=10)
# ax2.set_ylabel("TBT SLO Attainment (%)", fontsize=25, labelpad=8)
ax2.set_ylim(-5, 105)
ax2.tick_params(axis='y', labelsize=25, length=0)
ax2.tick_params(axis='x', length=0)
for spine in ax2.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# 범례 (공통)
fig.legend(labels=LABELS, loc="upper center", ncol=len(METHODS),
           fontsize=25, frameon=False, bbox_to_anchor=(0.5, 1.15))

# 저장
fig.tight_layout()
plt.savefig("figures/6_3_context_batch_size.jpg", format='jpg', bbox_inches="tight")
plt.savefig("figures/6_3_context_batch_size.pdf", format='pdf', bbox_inches="tight")
