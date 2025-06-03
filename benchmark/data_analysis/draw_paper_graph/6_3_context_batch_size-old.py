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
TRACE       = "both_dyn_veryhigh"
METHODS     = ["Flexgen", "SelectN", "Ours"]
LABELS      = ["Flexgen", "SelectN", "Ours"]
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

Context_window = [8, 32, 64, 128]
Context_window_labels = [f"{s}k" for s in Context_window]
BATCH_SIZES = [2, 4, 8]
BATCH_SIZE_LABELS = [f"BS {b}" for b in BATCH_SIZES]
SLO = [3.5, 1.5]

# # ───────────────────────────────────────────────
# # 좌측: Context Window 그래프
# fig, ax1 = plt.subplots(figsize=(8, 6))

# for m, method in enumerate(METHODS):
#     y_values = []
#     for context_size in Context_window:
#         path = Path(f"/home/heelim/vllm/outputs/benchmark/paper_main_exp/{method}/{TRACE}/{context_size}/output.csv")
#         try: 
#             df = load_metrics(path)
#             y_values.append(slo_tbt(df))
#         except:
#             print(f"[SLO 경고] {path}")
#             y_values.append(np.random.random() * 100)

#     ax1.plot(
#         range(len(Context_window_labels)),
#         y_values,
#         color=COLORS[m], marker=MARKERS[m],
#         label=LABELS[m],
#         **LINE_KW
#     )

# ax1.set_xticks(range(len(Context_window_labels)))
# ax1.set_xticklabels(Context_window_labels, fontsize=20)
# ax1.set_xlabel("Context Window Size", fontsize=20, labelpad=10)
# ax1.set_ylabel("TBT SLO Attainment (%)", fontsize=20, labelpad=10)
# ax1.set_ylim(-5, 105)
# ax1.tick_params(axis='y', labelsize=20, length=0)
# ax1.tick_params(axis='x', length=0)
# for spine in ax1.spines.values():
#     spine.set_edgecolor(style["spine"]["color"])
#     spine.set_alpha(style["spine"]["alpha"])
#     spine.set_linewidth(style["spine"]["linewidth"])

# ax1.legend(loc="upper center", ncol=len(METHODS), fontsize=16, bbox_to_anchor=(0.5, 1.15), frameon=False)
# fig.tight_layout()
# plt.savefig("figures/6_3_context_line.jpg", format='jpg', bbox_inches="tight")
# plt.savefig("figures/6_3_context_line.pdf", format='pdf', bbox_inches="tight")


# ───────────────────────────────────────────────
# 우측: Batch Size Grouped Bar Chart (SLO 기준 그룹화)
fig_bar, ax_bar = plt.subplots(figsize=(10, 6))

bar_width = 0.13
x_labels = []
x_pos_base = np.arange(len(SLO) * len(BATCH_SIZES))
x_offset = np.linspace(-bar_width * len(METHODS)/2, bar_width * len(METHODS)/2, len(METHODS))

for m, method in enumerate(METHODS):
    y_values = []
    for slo in SLO:
        for batch_size in BATCH_SIZES:
            path = Path(f"/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo{slo}/{method}/{TRACE}_bs{batch_size}/outputs.csv")
            try: 
                df = load_metrics(path)
                y_values.append(slo_tbt(df))
            except:
                print(f"[SLO 경고] {path}")
                y_values.append(np.random.random() * 100)

    x_positions = x_pos_base + x_offset[m]
    ax_bar.bar(
        x_positions,
        y_values,
        width=bar_width,
        color=COLORS[m],
        label=LABELS[m]
    )

# X축 레이블 구성
for slo in SLO:
    for bs in BATCH_SIZES:
        x_labels.append(f"{slo} / {bs}")

ax_bar.set_xticks(x_pos_base)
ax_bar.set_xticklabels(x_labels, fontsize=18, rotation=30)
ax_bar.set_ylabel("TBT SLO Attainment (%)", fontsize=20)
ax_bar.set_xlabel("SLO / Batch Size", fontsize=20)
ax_bar.set_ylim(-5, 105)
ax_bar.tick_params(axis='y', labelsize=18)

for spine in ax_bar.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

ax_bar.legend(loc="upper center", ncol=len(METHODS), fontsize=16, bbox_to_anchor=(0.5, 1.15), frameon=False)
fig_bar.tight_layout()
plt.savefig("figures/6_3_context_batch_size.jpg", format='jpg', bbox_inches="tight")
plt.savefig("figures/6_3_context_batch_size.pdf", format='pdf', bbox_inches="tight")