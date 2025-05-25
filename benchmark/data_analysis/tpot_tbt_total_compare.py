import pandas as pd, matplotlib.pyplot as plt, numpy as np, ast
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ───────────────────────────────────────────────
# 1. 설정 -------------------------------------------------------
trace_list  = ["test_fit_static_0", "test_shortshort_enough",
               "test_shortlong_less"]
trace_labels = ["(a) Trace 1", "(b) Trace 2",
                "(c) Trace 3"]

method_list   = ["Flexgen","DeepSpeed","SelectN","NoPrefetch","Ours"]
method_labels = ["Flexgen","DeepSpeed","Placeholder(SelectN)",
                 "No Prefetch","Ours"]

metric_list   = ["Low","Mid","High"]
metric_labels = ["Low","Mid","High"]           # 왼쪽 y-라벨

slo_scales    = [4, 3, 2, 1]      # 내림차순!
slo_labels    = [str(s) for s in slo_scales]   # x축 표기용 문자열

colors = [
    "#84C8F4",  # Soft Sky Blue (연하늘색)
    "#C59FDB",  # Pastel Lavender (연보라색)
    "#7CD6A4",  # Mint Green (민트색)
    "#63D0C2",  # Aqua Teal (청록색)
    "#E05A4F",  # Coral Red (산호빛 빨강)
]
markers = [
    'o',  # Flexgen
    's',  # DeepSpeed
    '^',  # SelectN
    'D',  # NoPrefetch
    'P'   # Ours
]

font_size = 35
style = {
    "line":   {"linewidth":4,"markersize":15},
    "tick":   {"fontsize":30},
    "label":  {"fontsize":font_size,"labelpad":5},
    "legend": {"fontsize":font_size},
    "spine": {
        "color": "black",
        "alpha": 0.7,
        "linestyle": "-",
        "linewidth": 2
    },
}

# ───────────────────────────────────────────────
# 2. CSV → DataFrame 유틸리티 --------------------
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
# 3. Figure & GridSpec ---------------------------
N_TRACE, N_METRIC = len(trace_list), len(metric_list)

fig = plt.figure(figsize=(55, 5.5 * N_METRIC))
gs  = GridSpec(
    nrows=N_METRIC, ncols=N_TRACE*2 + 1,
    width_ratios=[2]*N_TRACE + [0.5] + [2]*N_TRACE,
    wspace=0.1, hspace=0.3
)

axes = [[None]*(N_TRACE*2) for _ in range(N_METRIC)]
for i in range(N_METRIC):
    for j in range(N_TRACE):
        axes[i][2*j]   = fig.add_subplot(gs[i, j])               # TPOT
        axes[i][2*j+1] = fig.add_subplot(gs[i, j+N_TRACE+1])     # TBT

# ───────────────────────────────────────────────
# 4. Plot 루프 ----------------------------------
for r, metric in enumerate(metric_list):        # row
    for c, trace in enumerate(trace_list):      # column pair

        ax_L = axes[r][2*c]       # TPOT
        ax_R = axes[r][2*c + 1]   # TBT

        for m, (method, m_label) in enumerate(zip(method_list, method_labels)):
            # ---- TPOT ----
            yL = []
            for idx, sc in enumerate(slo_scales):
                path = Path(f"/home/heelim/vllm/outputs/benchmark/exp/{method}/{trace}_{metric}/{sc}/output.csv")
                try:
                    df = load_metrics(path)
                    yL.append(slo_tpot(df))
                except:
                    print(f"[SLO 경고] {path}")
                    yL.append(10*(idx+1)*(m+1))
            ax_L.plot(slo_scales, yL, **style["line"],
                       marker=markers[m], color=colors[m], label=m_label)

            # 90% 교차점
            for p in range(len(slo_scales)-1):
                y1,y2 = yL[p], yL[p+1]
                if (y1-90)*(y2-90) <= 0 and y1 != y2:
                    x1,x2 = slo_scales[p], slo_scales[p+1]
                    x_cross = x1 + (90-y1)*(x2-x1)/(y2-y1)
                    ax_L.vlines(x_cross, 0, 90, color=colors[m],
                                ls="--", lw=style["line"]["linewidth"], alpha=.8)

            # ---- TBT ----
            yR = []
            for idx, sc in enumerate(slo_scales):
                try:
                    df = load_metrics(Path(f"/exp/{method}/{trace}_{metric}/{sc}/output.csv"))
                    yR.append(slo_tbt(df))
                except:
                    yR.append(10*(idx+1)*(m+1))
            ax_R.plot(slo_scales, yR, **style["line"],
                       marker=markers[m], color=colors[m])

            for p in range(len(slo_scales)-1):
                y1,y2 = yR[p], yR[p+1]
                if (y1-90)*(y2-90) <= 0 and y1 != y2:
                    x1,x2 = slo_scales[p], slo_scales[p+1]
                    x_cross = x1 + (90-y1)*(x2-x1)/(y2-y1)
                    ax_R.vlines(x_cross, 0, 90, color=colors[m],
                                ls="--", lw=style["line"]["linewidth"], alpha=.8)

        # ── 축 공통 서식 ──────────────────────────
        for ax in (ax_L, ax_R):
            ax.set_xticks(slo_scales)  # 원래 순서 그대로
            ax.set_xticklabels([str(s) for s in reversed(slo_scales)], 
                               fontsize=style["tick"]["fontsize"])  # 라벨만 뒤집기
            # ax.set_xlim(slo_scales[-1], slo_scales[0])  # 오른쪽=1.0, 왼쪽=1.5
            ax.set_ylim(-5, 105)
            ax.axhline(90, color="gray", ls="--", lw=style["line"]["linewidth"])
            ax.tick_params(axis='both', labelsize=style["tick"]["fontsize"], length=0)

        # ── y-tick 설정 ─────────────────────────────
        # TPOT 그래프들: 첫 번째 열만 y-tick 표시 (왼쪽)
        # if c == 0:
        #     ax_L.set_yticks([0, 50, 100])
        #     ax_L.set_yticklabels(['0', '50', '100'])
        # else:
        #     ax_L.set_yticks([])
        #     ax_L.set_yticklabels([])
        
        # TBT 그래프들: 마지막 열만 y-tick 표시 (오른쪽)
        if c == N_TRACE - 1:
            ax_L.set_yticks([0, 50, 100])
            ax_L.set_yticklabels(['0', '50', '100'])
            ax_L.yaxis.tick_right()
            ax_L.yaxis.set_label_position("right")
            ax_R.set_yticks([0, 50, 100])
            ax_R.set_yticklabels(['0', '50', '100'])
            ax_R.yaxis.tick_right()
            ax_R.yaxis.set_label_position("right")
        else:
            ax_R.set_yticks([])
            ax_R.set_yticklabels([])
            ax_L.set_yticks([])
            ax_L.set_yticklabels([])

        # ── row-label 왼쪽 첫 패널 ───────────────
        if c == 0:
            ax_L.set_ylabel(metric_labels[r], **style["label"])

        # ── SLO attainment(%) 오른쪽 y-라벨 (Trace 4만) ─
        if c == N_TRACE - 1:
            # ax_L.set_ylabel("SLO attainment (%)", **style["label"])
            # ax_L.yaxis.set_label_position("right")
            ax_R.set_ylabel("SLO attainment (%)", 
                            fontsize=30,
                            labelpad = style["label"]["labelpad"], 
                            rotation=270,
                            )
            ax_R.yaxis.set_label_position("right")
            ax_R.yaxis.set_label_coords(1.28, 0.5)

        if r == len(metric_list) - 1:
            ax_R.set_xlabel(f"SLO Scale", fontsize=style["label"]["fontsize"], labelpad=10)
            ax_L.set_xlabel(f"SLO Scale", fontsize=style["label"]["fontsize"], labelpad=10)
        else:
            ax_R.set_xlabel("")
            ax_L.set_xlabel("")

        # ── Trace 라벨 (아랫쪽) ────────────────────
        if r == N_METRIC - 1:
            for ax, idx in [(ax_L, c), (ax_R, c)]:
                ax.text(0.5, -0.45, trace_labels[idx],
                        transform=ax.transAxes,
                        ha='center', va='top',
                        fontsize=style["label"]["fontsize"])

for rax in axes:
    for ax in rax:
        for spine in ax.spines.values():
            spine.set_edgecolor(style["spine"]["color"])
            spine.set_alpha(style["spine"]["alpha"])
            spine.set_linewidth(style["spine"]["linewidth"])


# ───────────────────────────────────────────────
# 5. 중앙 세로선, 범례, 공통 텍스트 ---------------
fig.add_artist(Line2D([0.52, 0.52], [-0.07, 0.9],
                      transform=fig.transFigure, color="black", lw=2))

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', 
        #    bbox_to_anchor=(0.52, 1.0),
           ncol=len(method_list),
           fontsize=43, frameon=False)

fig.text(0.1, 0.5, "Memory Pressure", va='center',
         rotation='vertical', fontsize=font_size)

fig.text(0.31, -0.065, "TPOT", ha='center',
         fontsize=43, weight='bold')
fig.text(0.715, -0.065, "TBT", ha='center',
         fontsize=43, weight='bold')


# ───────────────────────────────────────────────
# 6. 저장 ---------------------------------------
plt.savefig("figures/tpot_tbt_combined.jpg", format='jpg', bbox_inches="tight")
plt.savefig("figures/tpot_tbt_combined.pdf", format='pdf', bbox_inches="tight")