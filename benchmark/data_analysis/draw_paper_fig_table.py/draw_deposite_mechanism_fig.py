import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np          
import ast
import sys
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 

import matplotlib.patches as mpatches

def load_metrics(csv_path: str | Path) -> pd.DataFrame:
    """
    Read the log CSV and coerce list-encoded columns to Python lists.
    """
    df = pd.read_csv(csv_path)

    # Convert obvious numerics (ignore missing cols gracefully)
    numeric_cols: List[str] = [
        "arrival_time", "first_scheduled_time", "finished_time",
        "time_to_first_token", "slo_threshold", "slo_violations",
        "stall_duration", "decode_length", "end_to_end_time", "decode_time",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Turn list-as-string fields into actual lists
    for col in ("time_between_tokens", "stall_times", "stall_durations"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
    return df


csv1 = Path("/home/heelim/vllm/outputs/benchmark/figure_4_2_token_deposit/outputs.csv")
csv2 = Path("/home/heelim/vllm/outputs/benchmark/figure_4_2_token_deposit/outputs.csv")

df = load_metrics(csv1)
csv_path = csv1
df2 = load_metrics(csv2)
csv_path2 = csv2  

# ── central font-size definitions ──────────────────────────────────
FS_LABEL = 23      # axis-label text
FS_LAGEND = 18
FS_TICK  = 18      # axis-tick text
FS_NOTE  = 18      # in-figure annotations

"""
Latency-fit view (top) + token-deposit simulation (bottom).

Coloured regions in latency panel
---------------------------------
• green   – fit is below / on SLO (within budget)
• orange  – fit exceeds SLO **but** deposit > 0 (masked stall)
• red     – fit exceeds SLO **and** deposit == 0 (visible stall)
"""
import numpy as np
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────────────────

req_cols = (
    "arrival_time", "finished_time",
    "time_to_first_token", "time_between_tokens",
    "slo_threshold",
)
missing = [c for c in req_cols if c not in df.columns]
if missing:
    raise KeyError(f"Missing column(s): {missing}")

t0      = df["arrival_time"].min()           # reference zero (s)
palette = list(plt.cm.tab10.colors)

# -------- figure with three stacked axes -------------------------- #
fig, axes = plt.subplots(
    3, 2, figsize=(11, 7),
    gridspec_kw={"height_ratios": [1, 1, 1]},
)

plt.subplots_adjust(left=0.07, right=0.99,
                    top=0.88, bottom=0.12,
                    wspace=0.03, hspace=0.12)

ax_lat,   ax_lat2   = axes[0]
ax_dep,   ax_dep2   = axes[1]
ax_perc,  ax_perc2  = axes[2]
ax_dep.sharex(ax_lat)        # share x for the top two only

ymax_lat, ymax_dep, ymax_perc = 0.0, 0, 0.0

# ------------------------------------------------------------------ #
for idx, (_, row) in enumerate(df.iterrows()):
    rid        = row["request_id"]
    color      = palette[idx % 10]

    arrival    = row["arrival_time"]
    ttf        = row["time_to_first_token"]
    tbt_list   = row["time_between_tokens"]
    solver_ts  = eval(row["solver_time"])
    slo_thr    = row["slo_threshold"]               # s / token

    # ---- gather latency samples ---------------------------------- #
    xs_req, ys_req = [], []
    first_tok_wall = arrival + ttf
    xs_req.append(first_tok_wall - t0)
    ys_req.append(0.0)                              # ms proxy

    if isinstance(tbt_list, (list, tuple)):
        cum = first_tok_wall
        cutoff = len(tbt_list)-480
        counter = 0
        for st, dt in zip(solver_ts, tbt_list):
            counter += 1
            if counter > cutoff:
                break
            step_time = (dt - st if dt - st >= 0.035 else dt)
            cum += step_time
            xs_req.append(cum - t0)
            ys_req.append(step_time * 1000)         # s → ms
    print(len(xs_req), len(ys_req))

    if len(xs_req) < 2:
        continue
    
    ys_req = [y + 0.3 for y in ys_req]
    # ---- linear fit --------------------------------------------- #
    m, b = np.polyfit(xs_req, ys_req, 1)
    x_min, x_max = xs_req[0], xs_req[-1]
    xs_fit = np.linspace(x_min, x_max, 300)
    xs_rel  = xs_fit
    ys_fit = m * xs_fit + b
    ax_lat.plot(xs_fit, ys_fit, color=color, lw=1.5,
                label=rid if df.shape[0] <= 10 else None)
    

    for ax in (ax_dep, ax_perc):
        ax.axvline(x_min, color="black", lw=1.5, ls="dashed", alpha=0.8)
        ax.axvline(x_max, color="black", lw=1.5, ls="dashed", alpha=0.8)

    # ---- token-deposit simulation (step curve) ------------------- #
    tok_times = [x_min]                # first token at x_min (fit origin)
    t_cur = x_min
    while True:
        lat_ms = m * t_cur + b
        lat_s  = max(lat_ms / 1000.0, 1e-9)
        t_cur += lat_s
        if t_cur > x_max:
            break
        tok_times.append(t_cur)
    deposit_curve_x, deposit_curve_y = [], []

    def rec(t, d):
        deposit_curve_x.append(t)
        deposit_curve_y.append(d)
    green_cnt  = 0   # latency ≤ SLO
    orange_cnt = 0   # latency > SLO  and deposit>0   (masked)
    red_cnt    = 0   # latency > SLO  and deposit==0  (visible)

    deposit = 0
    emit_t  = tok_times[0] + slo_thr
    rec(tok_times[0] - 1e-6, 0)

    ti = 0
    while ti < len(tok_times) or deposit > 0:
        next_tok  = tok_times[ti] if ti < len(tok_times) else np.inf
        next_emit = emit_t if deposit > 0 else np.inf
        deposit_before = deposit
        lat_ms  = m * next_tok + b
        if lat_ms <= slo_thr * 1000:
            green_cnt += 1
        elif deposit_before > 0:
            orange_cnt += 1
        else:
            red_cnt += 1

        if next_tok <= next_emit:
            deposit += 1
            rec(next_tok, deposit_curve_y[-1])
            rec(next_tok, deposit)
            ti += 1
            if deposit == 1:
                emit_t = next_tok + slo_thr
        else:
            deposit = max(deposit - 1, 0)
            rec(next_emit, deposit_curve_y[-1])
            rec(next_emit, deposit)
            emit_t += slo_thr

    ax_dep.step(deposit_curve_x[:-50], deposit_curve_y[:-50],
                where="post", color=color, lw=0.5)
    dep_vals = np.interp(xs_rel, deposit_curve_x, deposit_curve_y)

    # ----------- find first deposit depletion time ---------------- #
    x_depleted = np.inf
    for j in range(100, len(deposit_curve_y)):
        if deposit_curve_y[j - 1] > 0 and deposit_curve_y[j] == 0:
            x_depleted = deposit_curve_x[j]
            break

    # ---- colour regions in latency panel ------------------------ #
    y_slo = slo_thr * 1000
    ax_lat.hlines(y=y_slo, xmin=x_min, xmax=x_max,
                    colors="black", lw=1.5, ls="-.")
    ax_lat.text(
        x_min + 1, y_slo + 2, "",#f"SLO = {int(slo_thr*1000)} ms",
        color="black", fontsize=FS_NOTE, ha="left", va="bottom"
    )

    if np.isfinite(x_depleted) and x_min <= x_depleted <= x_max:
        # depletion marker
        # for ax in (ax_lat, ax_dep, ax_perc):
        #     ax.axvline(x_depleted, color="red", lw=2.0, ls=":", alpha=0.8)
        ax_perc.text(
            x_depleted + 3, 75, "",#f"400 Perceived \nViolations",
            color="red", fontsize=FS_NOTE, ha="left", va="bottom"
        )
        ax_lat.text(
            x_depleted, 75,"", #f"1700 Real \nViolations",
            color="red", fontsize=FS_NOTE, ha="center", va="bottom"
        )
    if m != 0:
        x_cross = (y_slo - b) / m
        if x_min <= x_cross <= x_max:
            # for axv in (ax_lat, ax_dep):
            #     axv.axvline(x_cross, color="orange",
            #                 lw=2.0, ls="--", alpha=0.6)
            # ax_perc.axvline(x_cross - x_min, color="orange",
            #                 lw=2.0, ls="--", alpha=0.6)
            # ax_dep.text(
            #     x_cross + 1, 10, "1300 Masked \nViolations",
            #     color="red", fontsize=FS_NOTE, ha="left", va="bottom"
            # )
            ax_perc.text(
                x_cross + 3, 75, "",#f"{orange_cnt} Masked \nViolations",
                color="red", fontsize=FS_NOTE, ha="left", va="bottom")
    dep_vals   = np.interp(xs_fit, deposit_curve_x, deposit_curve_y)
    mask_above = ys_fit > y_slo
    mask_orange = mask_above & (xs_fit < x_depleted)
    mask_red    = mask_above & (xs_fit >= x_depleted)
    mask_green  = ~mask_above

    ax_dep.fill_between(xs_fit, 0, dep_vals,
                        where=(mask_green & (dep_vals > 0)),
                        step="post", color="green",  alpha=0.15)
    ax_dep.fill_between(xs_fit, 0, dep_vals,
                        where=(mask_orange & (dep_vals > 0)),
                        step="post", color="red", alpha=0.25)
    ax_lat.fill_between(xs_fit, ys_fit, y_slo,
                        where=mask_green, interpolate=True,
                        color="green", alpha=0.15)
    ax_lat.fill_between(xs_fit, ys_fit, y_slo,
                        where=mask_orange, interpolate=True,
                        color="red", alpha=0.25)
    ax_lat.fill_between(xs_fit, ys_fit, y_slo,
                        where=mask_red, interpolate=True,
                        color="red", alpha=0.25)
    ax_perc.fill_between(xs_fit, ys_fit, y_slo,
                            where=mask_red, interpolate=True,
                            color="red", alpha=0.25)

    # ---- perceived-latency curve (bottom) ----------------------- #
    ys_perc = np.where(xs_fit < x_depleted, y_slo, ys_fit)
    xs_rel  = xs_fit               # first token at 0
    ax_perc.plot(xs_rel, ys_perc, color="black", lw=1.5)

    # track maxima for axis scaling
    ymax_lat  = max(ymax_lat,  ys_fit.max())
    ymax_dep  = max(ymax_dep,  max(deposit_curve_y))
    ymax_perc = max(ymax_perc, ys_perc.max())

# ----------- cosmetics -------------------------------------------- #
ax_lat.set_ylabel("Real\nLat. (ms)", fontsize=FS_LABEL)
ax_lat.set_ylim(50, 85)
ax_lat.set_xticks([])

ax_dep.set_ylabel("Deposited\nTokens", fontsize=FS_LABEL)
ax_dep.set_ylim(0, 45)
ax_dep.set_xticks([])

# ax_perc.set_xlabel("Time (s)", fontsize=FS_LABEL)
ax_perc.set_ylabel("User\nLat. (ms)", fontsize=FS_LABEL)
ax_perc.set_ylim(50, 85)
ax_perc.set_xticks([])

for ax in (ax_lat, ax_dep, ax_perc):
    # ax.set_xticks([0, 40, 80])
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xticks([])


ylim = ax_lat.get_ylim()
y_range = ylim[1] - ylim[0]

# 텍스트가 들어갈 영역 계산 (각 subplot의 중간 높이 기준)
text_y = ylim[0] + y_range * 0.9
text_gap = y_range * 0.1  # 텍스트 위아래 여백

# x 위치와 텍스트 라벨 정의
text_info = [
    (x_min, "S1"),
    (x_max, "E1")
]

# 각 위치에 대해 점선과 텍스트 그리기
for x_pos, label in text_info:
    # 텍스트 영역 아래쪽 점선
    ax_lat.axvline(x_pos, ymin=0, ymax=(text_y - text_gap - ylim[0]) / y_range,
                   color="black", lw=1.5, ls="dashed", alpha=0.8)
    
    # 텍스트 영역 위쪽 점선
    ax_lat.axvline(x_pos, ymin=(text_y + text_gap - ylim[0]) / y_range, ymax=1,
                   color="black", lw=1.5, ls="dashed", alpha=0.8)
    
    # 텍스트 추가 (흰색 배경과 함께)
    ax_lat.text(x_pos, text_y, label, ha='center', va='center',
                fontsize=FS_NOTE, color='black',
                )

# fig.tight_layout()

# out_path_png = csv_path.with_name(f"{csv_path.stem}_fig_4_2.png")
# fig.savefig(out_path_png, dpi=150, bbox_inches="tight")
# out_path_pdf = csv_path.with_name(f"{csv_path.stem}_fig_4_2.pdf")
# fig.savefig(out_path_pdf, dpi=150, bbox_inches="tight")
# plt.close(fig)
print(f"green_cnt: {green_cnt}, orange_cnt: {orange_cnt}, red_cnt: {red_cnt}")
# return the PDF path (change if you prefer PNG)
# return out_path_png

"""
Latency-fit view (top) + token-deposit simulation (bottom).

Coloured regions in latency panel
---------------------------------
• green   – fit is below / on SLO (within budget)
• orange  – fit exceeds SLO **but** deposit > 0 (masked stall)
• red     – fit exceeds SLO **and** deposit == 0 (visible stall)
"""
import numpy as np
import matplotlib.pyplot as plt

# ── central font-size definitions ──────────────────────────────────
# ───────────────────────────────────────────────────────────────────

ax_lat =   ax_lat2
ax_dep =   ax_dep2
ax_perc =  ax_perc2
df = df2
csv_path = csv_path2
req_cols = (
    "arrival_time", "finished_time",
    "time_to_first_token", "time_between_tokens",
    "slo_threshold",
)
missing = [c for c in req_cols if c not in df.columns]
if missing:
    raise KeyError(f"Missing column(s): {missing}")

t0      = df["arrival_time"].min()           # reference zero (s)
palette = list(plt.cm.tab10.colors)

# -------- figure with three stacked axes -------------------------- #
# fig, (ax_lat, ax_dep, ax_perc) = plt.subplots(
#     3, 1, figsize=(6, 8),
#     gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 0.3},
# )
ax_dep.sharex(ax_lat)        # share x for the top two only

ymax_lat, ymax_dep, ymax_perc = 0.0, 0, 0.0

# ------------------------------------------------------------------ #

rid1        = df.at[0, "request_id"]
rid2        = "request_1"
color      = palette[0]

arrival    = df.at[0, "arrival_time"]
ttf        = df.at[0, "time_to_first_token"]
tbt_list   = df.at[0, "time_between_tokens"]
solver_ts  = eval(df.at[0, "solver_time"])
slo_thr    = df.at[0, "slo_threshold"]               # s / token

# ---- gather latency samples ---------------------------------- #
xs_req, ys_req = [], []
first_tok_wall = arrival + ttf
xs_req.append(first_tok_wall - t0)
ys_req.append(0.0)                              # ms proxy

if isinstance(tbt_list, (list, tuple)):
    cum = first_tok_wall
    cutoff = len(tbt_list)-480
    counter = 0
    for st, dt in zip(solver_ts, tbt_list):
        counter += 1
        if counter > cutoff:
            break
        step_time = (dt - st if dt - st >= 0.035 else dt) #- 0.003
        cum += step_time
        xs_req.append(cum - t0)
        ys_req.append(step_time * 1000)         # s → ms
print(len(xs_req), len(ys_req))
# ys_req = [y + 0.3 for y in ys_req]
x1_min, x1_max = xs_req[0], xs_req[-1]
x2_min, x2_max = xs_req[99], xs_req[159]

for ax in (ax_dep, ax_perc):
    ax.axvline(x1_min, color="black", lw=1.5, ls="dashed", alpha=0.8)
    ax.axvline(x1_max, color="black", lw=1.5, ls="dashed", alpha=0.8)
    ax.axvline(x2_min, color="black", lw=1.5, ls="dashed", alpha=0.8)
    ax.axvline(x2_max, color="black", lw=1.5, ls="dashed", alpha=0.8)
    

# ---- linear fit --------------------------------------------- #
offset = 700
m1, b1 = np.polyfit(xs_req, ys_req, 1)
m2, b2 = np.polyfit(xs_req[99:160], ys_req[99+offset:160+offset], 1)

xs_fit1 = np.linspace(x1_min, x2_min, 160)
ys_fit1 = m1 * xs_fit1 + b1
xs_rel1  = xs_fit1
ax_lat.plot(xs_fit1, ys_fit1, color=color, lw=1.5,
        label =rid1)

xs_fit2 = np.linspace(x2_min, x2_max, 160)
ys_fit2 = m2 * xs_fit2 + b2
xs_rel2 = xs_fit2
ax_lat.plot(xs_fit2, ys_fit2, color=color, lw=1.5,
        label =rid2)


xs_fit3 = np.linspace(x2_max, x1_max, 160)
ys_fit3 = m1 * xs_fit3 + b1
xs_rel3 = xs_fit2
ax_lat.plot(xs_fit3, ys_fit3, color=color, lw=1.5,
        label =rid2)

# ---- token-deposit simulation (step curve) ------------------- #
tok_times = [x1_min]
t_cur = x1_min

while True:
    # Determine which model to use based on current time
    if x2_min <= t_cur <= x2_max:
        # Use second row's model
        lat_ms = m2 * t_cur + b2
    else:
        # Use first row's model
        lat_ms = m1 * t_cur + b1
    
    lat_s = max(lat_ms / 1000.0, 1e-9)
    tok_times.append(t_cur)
    t_cur += lat_s
    
    if t_cur > x1_max:
        break

deposit_curve_x, deposit_curve_y = [], []

def rec(t, d):
    deposit_curve_x.append(t)
    deposit_curve_y.append(d)

green_cnt  = 0   # latency ≤ SLO
orange_cnt = 0   # latency > SLO  and deposit>0   (masked)
red_cnt    = 0   # latency > SLO  and deposit==0  (visible)

deposit = 0
emit_t  = tok_times[0] + slo_thr
rec(tok_times[0] - 1e-6, 0)

ti = 0
while ti < len(tok_times) or deposit > 0:
    next_tok  = tok_times[ti] if ti < len(tok_times) else np.inf
    next_emit = emit_t if deposit > 0 else np.inf
    deposit_before = deposit

    token_cnt = 1
    if x2_min <= next_tok <= x2_max:
        # Use second row's model
        lat_ms = m2 * next_tok + b2
        token_cnt=2
    else:
        # Use first row's model
        lat_ms = m1 * next_tok + b1

    if lat_ms <= slo_thr * 1000:
        green_cnt += token_cnt
    elif deposit_before > 0:
        orange_cnt += token_cnt
    else:
        red_cnt += token_cnt

    if next_tok <= next_emit:
        deposit += token_cnt
        rec(next_tok, deposit_curve_y[-1])
        rec(next_tok, deposit)
        ti += 1
        if deposit == 1:
            emit_t = next_tok + slo_thr
    else:
        deposit = max(deposit - token_cnt, 0)
        rec(next_emit, deposit_curve_y[-1])
        rec(next_emit, deposit)
        emit_t += slo_thr

ax_dep.step(deposit_curve_x, deposit_curve_y,
            where="post", color=color, lw=0.5)
dep_vals = np.interp(xs_rel1, deposit_curve_x, deposit_curve_y)

# ----------- find first deposit depletion time ---------------- #
x_depleteds = []
for j in range(100, len(deposit_curve_y)):
    if deposit_curve_y[j - 1] > 0 and deposit_curve_y[j] == 0:
        if len(x_depleteds) == 0 or abs(x_depleteds[-1] - deposit_curve_x[j]) > 30 :
            x_depleteds.append(deposit_curve_x[j])

print(x_depleteds)

# ---- colour regions in latency panel ------------------------ #
y_slo = slo_thr * 1000
ax_lat.hlines(y=y_slo, xmin=x1_min, xmax=x1_max,
                colors="black", lw=1.5, ls="-.")
ax_lat.text(
    x1_min + 1, y_slo + 2,  "",#f"SLO = {int(slo_thr*1000)} ms",
    color="black", fontsize=FS_NOTE, ha="left", va="bottom"
)

for i, x_depleted in enumerate(x_depleteds):
    # if i == 0: continue
    if np.isfinite(x_depleted) and x1_min <= x_depleted <= x1_max:
        # depletion marker
        # for ax in (ax_lat, ax_dep, ax_perc):
        #     ax.axvline(x_depleted, color="red", lw=2.0, ls=":", alpha=0.8)
        
        if i == len(x_depleteds) - 1:
            ax_perc.text(
                x_depleted + 3, 75, "",#f"{red_cnt} Perceived \nViolations",
                color="red", fontsize=FS_NOTE, ha="left", va="bottom"
            )
            ax_lat.text(
                x_depleted, 75, "",#f"{orange_cnt + red_cnt} Real \nViolations",
                color="red", fontsize=FS_NOTE, ha="center", va="bottom"
            )

if m1 != 0:
    x_cross = (y_slo - b1) / m1
    if x1_min <= x_cross <= x1_max:
        # for axv in (ax_lat, ax_dep):
        #     axv.axvline(x_cross, color="orange",
        #                 lw=2.0, ls="--", alpha=0.6)
        # ax_perc.axvline(x_cross - x1_min, color="orange",
        #                 lw=2.0, ls="--", alpha=0.6)
        # ax_dep.text(
        #     x_cross + 1, 10, "1300 Masked \nViolations",
        #     color="red", fontsize=FS_NOTE, ha="left", va="bottom"
        # )
        ax_perc.text(
            x_cross + 3, 75,"",# f"{orange_cnt} Masked \nViolations",
            color="red", fontsize=FS_NOTE, ha="left", va="bottom")

# if m2 != 0:
#     x_cross = (y_slo - b2) / m2
#     if x2_min <= x_cross <= x2_max:
# for axv in (ax_lat, ax_dep):
#     axv.axvline(x2_min, color="orange",
#                 lw=2.0, ls="--", alpha=0.6)
# ax_perc.axvline(x2_min, color="orange",
#                 lw=2.0, ls="--", alpha=0.6)


dep_vals   = np.interp(xs_fit1, deposit_curve_x, deposit_curve_y)
mask_above = ys_fit1 > y_slo
mask_orange = mask_above & (xs_fit1 < x_depleted)
mask_red    = mask_above & (xs_fit1 >= x_depleted)
mask_green  = ~mask_above

ax_dep.fill_between(xs_fit1, 0, dep_vals,
                    where=(mask_green & (dep_vals > 0)),
                    step="post", color="green",  alpha=0.15)
ax_dep.fill_between(xs_fit1, 0, dep_vals,
                    where=(mask_orange & (dep_vals > 0)),
                    step="post", color="red", alpha=0.25)
ax_lat.fill_between(xs_fit1, ys_fit1, y_slo,
                    where=mask_green, interpolate=True,
                    color="green", alpha=0.15)
ax_lat.fill_between(xs_fit1, ys_fit1, y_slo,
                    where=mask_orange, interpolate=True,
                    color="red", alpha=0.25)
ax_lat.fill_between(xs_fit1, ys_fit1, y_slo,
                    where=mask_red, interpolate=True,
                    color="red", alpha=0.25)
ax_perc.fill_between(xs_fit1, ys_fit1, y_slo,
                        where=mask_red, interpolate=True,
                        color="red", alpha=0.25)

# ---- perceived-latency curve (bottom) ----------------------- #
ys_perc = np.where(xs_fit1 < x_depleted, y_slo, ys_fit1)
xs_rel  = xs_fit1               # first token at 0
ax_perc.plot(xs_rel, ys_perc, color="black", lw=1.5)

# track maxima for axis scaling
ymax_lat  = max(ymax_lat,  ys_fit1.max())
ymax_dep  = max(ymax_dep,  max(deposit_curve_y))
ymax_perc = max(ymax_perc, ys_perc.max())




# -------------------
dep_vals   = np.interp(xs_fit2, deposit_curve_x, deposit_curve_y)
mask_above = ys_fit2 > y_slo
mask_orange = mask_above & (xs_fit2 < x_depleteds[0])
print(x_depleted)
mask_red    = mask_above & (xs_fit2 >= x_depleteds[0])
mask_green  = ~mask_above

ax_dep.fill_between(xs_fit2, 0, dep_vals,
                    where=(mask_green & (dep_vals > 0)),
                    step="post", color="green",  alpha=0.15)
ax_dep.fill_between(xs_fit2, 0, dep_vals,
                    where=(mask_orange & (dep_vals > 0)),
                    step="post", color="red", alpha=0.25)
ax_lat.fill_between(xs_fit2, ys_fit2, y_slo,
                    where=mask_green, interpolate=True,
                    color="green", alpha=0.15)
ax_lat.fill_between(xs_fit2, ys_fit2, y_slo,
                    where=mask_orange, interpolate=True,
                    color="red", alpha=0.25)
ax_lat.fill_between(xs_fit2, ys_fit2, y_slo,
                    where=mask_red, interpolate=True,
                    color="red", alpha=0.25)
ax_perc.fill_between(xs_fit2, ys_fit2, y_slo,
                        where=mask_red, interpolate=True,
                        color="red", alpha=0.25)

# ---- perceived-latency curve (bottom) ----------------------- #
ys_perc = np.where(xs_fit2 < x_depleteds[0], y_slo, ys_fit2)
xs_rel  = xs_fit2               # first token at 0
ax_perc.plot(xs_rel, ys_perc, color="black", lw=1.5)

# track maxima for axis scaling
ymax_lat  = max(ymax_lat,  ys_fit2.max())
ymax_dep  = max(ymax_dep,  max(deposit_curve_y))
ymax_perc = max(ymax_perc, ys_perc.max())


# -------------------
dep_vals   = np.interp(xs_fit3, deposit_curve_x, deposit_curve_y)
mask_above = ys_fit3 > y_slo
mask_orange = mask_above & (xs_fit3 < x_depleted)
mask_red    = mask_above & (xs_fit3 >= x_depleted)
mask_green  = ~mask_above

ax_dep.fill_between(xs_fit3, 0, dep_vals,
                    where=(mask_green & (dep_vals > 0)),
                    step="post", color="green",  alpha=0.15)
ax_dep.fill_between(xs_fit3, 0, dep_vals,
                    where=(mask_orange & (dep_vals > 0)),
                    step="post", color="red", alpha=0.25)
ax_lat.fill_between(xs_fit3, ys_fit3, y_slo,
                    where=mask_green, interpolate=True,
                    color="green", alpha=0.15)
ax_lat.fill_between(xs_fit3, ys_fit3, y_slo,
                    where=mask_orange, interpolate=True,
                    color="red", alpha=0.25)
ax_lat.fill_between(xs_fit3, ys_fit3, y_slo,
                    where=mask_red, interpolate=True,
                    color="red", alpha=0.25)
ax_perc.fill_between(xs_fit3, ys_fit3, y_slo,
                        where=mask_red, interpolate=True,
                        color="red", alpha=0.25)

# ---- perceived-latency curve (bottom) ----------------------- #
ys_perc = np.where(xs_fit3 < x_depleted, y_slo, ys_fit3)
xs_rel  = xs_fit3               # first token at 0
ax_perc.plot(xs_rel, ys_perc, color="black", lw=1.5)

# track maxima for axis scaling
ymax_lat  = max(ymax_lat,  ys_fit3.max())
ymax_dep  = max(ymax_dep,  max(deposit_curve_y))
ymax_perc = max(ymax_perc, ys_perc.max())



# ----------- cosmetics -------------------------------------------- #
# ax_lat.set_ylabel("Real\nLatency (ms)", fontsize=FS_LABEL)
ax_lat.set_ylim(50, 85)
ax_lat.set_yticks([])

# ax_dep.set_ylabel("Deposited\nTokens", fontsize=FS_LABEL)
ax_dep.set_ylim(0, 45)
ax_dep.set_yticks([])

# ax_perc.set_xlabel("Time (s)", fontsize=FS_LABEL)
# ax_perc.set_ylabel("User Perceived\nLatency (ms)", fontsize=FS_LABEL)
ax_perc.set_ylim(50, 85)
ax_perc.set_yticks([])
    

# apply tick-label size to every axis
for ax in (ax_lat, ax_dep, ax_perc):
    # ax.set_xticks([0, 40, 80])
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xticks([])

ylim = ax_lat.get_ylim()
y_range = ylim[1] - ylim[0]

# 텍스트가 들어갈 영역 계산 (각 subplot의 중간 높이 기준)
text_y = ylim[0] + y_range * 0.9
text_y2 = ylim[0] + y_range * 0.7
text_gap = y_range * 0.1  # 텍스트 위아래 여백

# x 위치와 텍스트 y 위치 정의
x_positions = [
    (x1_min, text_y),
    (x1_max, text_y),
    (x2_min, text_y2),
    (x2_max, text_y)
]

# x 위치, 텍스트 y 위치, 텍스트 라벨 정의
text_info = [
    (x1_min, text_y, "S1"),
    (x2_min, text_y2, "S2"),
    (x1_max, text_y, "E1"),
    (x2_max, text_y, "E2")
]

# 각 위치에 대해 점선과 텍스트 그리기
for x_pos, text_y_pos, label in text_info:
    # 텍스트 영역 아래쪽 점선
    ax_lat.axvline(x_pos, ymin=0, ymax=(text_y_pos - text_gap - ylim[0]) / y_range,
                   color="black", lw=1.5, ls="dashed", alpha=0.8)
    
    # 텍스트 영역 위쪽 점선
    ax_lat.axvline(x_pos, ymin=(text_y_pos + text_gap - ylim[0]) / y_range, ymax=1,
                   color="black", lw=1.5, ls="dashed", alpha=0.8)
    
    # 텍스트 추가 (흰색 배경과 함께)
    ax_lat.text(x_pos, text_y_pos, label, ha='center', va='center', 
                fontsize=FS_NOTE, color='black',)
    

print(f"green_cnt: {green_cnt}, orange_cnt: {orange_cnt}, red_cnt: {red_cnt}")
# return the PDF path (change if you prefer PNG)
# Time 글자 밑에 가로선 2개 추가 (fig.savefig 호출 직전에 추가)
# figure 좌표계를 사용해서 전체 그림 기준으로 선을 그립니다
fig_width = fig.get_figwidth()
fig_height = fig.get_figheight()

# 가로선 그리기
# figure 좌표계에서 선의 y 위치 (Time 글자 아래쪽)
line_y1 = 0.1 

left_start = 0.085
left_end = 0.508
# fig.add_artist(plt.Line2D([left_start, left_end], [line_y1, line_y1], 
#                          transform=fig.transFigure, color='black', lw=1.0))

right_start = 0.555  
right_end = 0.97     
# fig.add_artist(plt.Line2D([right_start, right_end], [line_y1, line_y1], 
                        #  transform=fig.transFigure, color='black', lw=1.0))


# # time stamp 그리기
# # 가로선 양쪽 끝에 tick mark와 레이블 추가
# tick_height = 0.01  # tick mark 높이

# tick_up = line_y1 + tick_height / 2
# tick_down = line_y1 - tick_height / 2

# # tick 레이블 추가 (가로선 아래쪽에)
# label_y = line_y1 - tick_height * 2

# # 왼쪽 subplot의 x축 범위
# xlim_up = ax_perc.get_xlim()[1]
# x_left_80 = left_start + (left_end - left_start) * float(80 / xlim_up)

# # 왼쪽 subplot의 tick marks
# fig.add_artist(plt.Line2D([left_start, left_start], [tick_up, tick_down], 
#                          transform=fig.transFigure, color='black', lw=1.0))
# # fig.add_artist(plt.Line2D([x_left_80, x_left_80], [tick_up, tick_down], 
# #                          transform=fig.transFigure, color='black', lw=1.0))
# fig.add_artist(plt.Line2D([left_end, left_end], [tick_up, tick_down], 
#                          transform=fig.transFigure, color='black', lw=1.0))

# # fig.text(left_start, label_y, '0', ha='center', va='top', 
# #          transform=fig.transFigure, fontsize=FS_TICK)
# fig.text(left_start, label_y, 't1', ha='center', va='top', 
#          transform=fig.transFigure, fontsize=FS_TICK)
# # fig.text(x_left_80, label_y, '80', ha='center', va='top', 
# #          transform=fig.transFigure, fontsize=FS_TICK)
# fig.text(left_end, label_y, 't2', ha='center', va='top', 
#          transform=fig.transFigure, fontsize=FS_TICK)



# # 오른쪽 subplot의 tick marks
# # 왼쪽 끝 tick
# fig.add_artist(plt.Line2D([right_start, right_start], [tick_up, tick_down], 
#                          transform=fig.transFigure, color='black', lw=1.0))

# second_come = right_start+0.025
# fig.add_artist(plt.Line2D([second_come, second_come], [tick_up, tick_down], 
#                          transform=fig.transFigure, color='black', lw=1.0))

# second_out = right_start+0.06
# fig.add_artist(plt.Line2D([second_out, second_out], [tick_up, tick_down], 
#                          transform=fig.transFigure, color='black', lw=1.0))

# # 오른쪽 끝 tick
# fig.add_artist(plt.Line2D([right_end, right_end], [tick_up, tick_down], 
#                          transform=fig.transFigure, color='black', lw=1.0))

# # 오른쪽 subplot의 x축 범위  
# xlim_up = ax_perc2.get_xlim()[1]

# # 오른쪽 subplot 레이블들  
# fig.text(right_start, label_y, 't1', ha='center', va='top', 
#          transform=fig.transFigure, fontsize=FS_TICK)
# fig.text(second_come, label_y, 't2', ha='center', va='top', 
#          transform=fig.transFigure, fontsize=FS_TICK)
# fig.text(second_out, label_y, 't3', ha='center', va='top', 
#          transform=fig.transFigure, fontsize=FS_TICK)
# fig.text(right_end, label_y, 't4', ha='center', va='top', 
#          transform=fig.transFigure, fontsize=FS_TICK)

fig.text(0.28, 0.08, '(a) Single Request', ha='center', va='top', 
         transform=fig.transFigure, fontsize=FS_LABEL)

fig.text(0.75, 0.08, '(b) Two Requests', ha='center', va='top', 
         transform=fig.transFigure, fontsize=FS_LABEL)


# legend

green_patch  = mpatches.Patch(color="green", alpha=0.15, label="Within SLO")
orange_patch = mpatches.Patch(color="red", alpha=0.25, label="Masked Violations")
red_patch    = mpatches.Patch(color="red",  alpha=0.25)# ,  label="Perceived Violations")
dash_line    = Line2D([0], [0], color="black", linestyle="-.", label="SLO Threshold")  # 추가

# --- Figure 전체 위쪽에 범례를 표시 --- #
# loc="upper center"로 하면 그림 상단 중앙에 붙습니다.
# ncol=3 로 한 줄에 3개 아이템을 나란히 배치합니다.
fig.legend(
    # handles=[green_patch, orange_patch, red_patch, dash_line],
    handles=[green_patch, orange_patch, dash_line],
    loc="upper center",
    ncol=4,
    frameon=False,       # 테두리 없이 표시
    fontsize=FS_LAGEND,
    bbox_to_anchor=(0.5, 0.98),  # 그림 위쪽(약간 여백을 두고) 배치
    columnspacing=0.7,
)

fig.savefig(f"/home/xinyuema/vllm/benchmark/data_analysis/draw_paper_fig_table.py/figures/deposit_combined2.png", dpi=150, bbox_inches="tight")
fig.savefig(f"/home/xinyuema/vllm/benchmark/data_analysis/draw_paper_fig_table.py/figures/deposit_combined2.pdf", dpi=1200, bbox_inches="tight")
