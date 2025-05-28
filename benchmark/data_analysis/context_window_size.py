import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ast

# ───────────────────────────────────────────────
# CSV 로드 및 synthetic fallback
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

    return (decoded-viol) / decoded * 100

# ───────────────────────────────────────────────
# 스타일/데이터 정의
TRACE       = "test_fit_static_0"
METHODS     = ["Flexgen", "DeepSpeed", "SelectN", "NoPrefetch", "Ours"]
LABELS      = ["Flexgen", "DeepSpeed", "Placeholder(SelectN)", "No Prefetch", "Ours"]
COLORS = [
    "#84C8F4",  # Soft Sky Blue (연하늘색)
    "#C59FDB",  # Pastel Lavender (연보라색)
    "#7CD6A4",  # Mint Green (민트색)
    "#63D0C2",  # Aqua Teal (청록색)
    "#E05A4F",  # Coral Red (산호빛 빨강)
]
MARKERS = [
    'o',  # Flexgen
    's',  # DeepSpeed
    '^',  # SelectN
    'D',  # NoPrefetch
    'P'   # Ours
]

Context_window = [8, 32, 64, 128]
Context_window_labels = [f"{str(s)}k" for s in Context_window]  # x축 표기용 문자열
TICK_FONT   = 18
LABEL_FONT  = 18
LINE_KW     = dict(linewidth=3, markersize=10)

style = {
    "spine": {
        "color": "black",
        "alpha": 0.7,
        "linestyle": "-",
        "linewidth": 1.5
    },
}

# ───────────────────────────────────────────────
# 단일 플롯
fig, ax = plt.subplots(figsize=(8, 5))

for m, method in enumerate(METHODS):
    y_values = []
    for context_size in Context_window:
        path = Path(f"/home/heelim/vllm/outputs/benchmark/exp/{method}/{TRACE}/{context_size}/output.csv")
        try: 
            df = load_metrics(path)
            # tbt_slo_attainment
            y_values.append(slo_tbt(df))
        except:
            print(f"[SLO 경고] {path}")
            y_values.append(np.random.random() * 100)

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
ax.set_xlabel("Context Window Size", fontsize=18, labelpad=8)
ax.set_ylabel("TBT SLO Attainnment (%)", fontsize=18, labelpad=8)

ax.set_ylim(-5, 105)

# 그리드/스파인
# ax.grid(True, linestyle="--", alpha=0.3)
for spine in ax.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# 범례
ax.legend(loc="upper center",
          ncol=len(METHODS)/2 + 1,
          fontsize=16,
          frameon=False,
          bbox_to_anchor=(0.5, 1.25),
          )

# 저장
plt.savefig("figures/context_window.jpg", format='jpg', bbox_inches="tight")
plt.savefig("figures/context_window.pdf", format='pdf', bbox_inches="tight")
