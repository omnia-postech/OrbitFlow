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

# ── central font-size definitions ──────────────────────────────────
FS_LABEL = 18      # axis-label text
FS_TICK  = 15      # axis-tick text
FS_NOTE  = 15      # in-figure annotations

def figure_4_2(df: pd.DataFrame, csv_path: Path) -> Path:
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
    fig, (ax_lat, ax_dep, ax_perc) = plt.subplots(
        3, 1, figsize=(6, 8),
        gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 0.25},
    )
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
            for st, dt in zip(solver_ts, tbt_list):
                step_time = (dt - st if dt - st >= 0.035 else dt)
                cum += step_time
                xs_req.append(cum - t0)
                ys_req.append(step_time * 1000)         # s → ms

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

        ax_dep.step(deposit_curve_x, deposit_curve_y,
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
                      colors="black", lw=1.2, ls="--")
        ax_lat.text(
            x_min + 1, y_slo + 2, f"SLO = {int(slo_thr*1000)} ms",
            color="black", fontsize=FS_NOTE, ha="left", va="bottom"
        )

        if np.isfinite(x_depleted) and x_min <= x_depleted <= x_max:
            # depletion marker
            for ax in (ax_lat, ax_dep, ax_perc):
                ax.axvline(x_depleted, color="red", lw=1.0, ls=":", alpha=0.8)
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
                for axv in (ax_lat, ax_dep):
                    axv.axvline(x_cross, color="orange",
                                lw=1.0, ls="--", alpha=0.6)
                ax_perc.axvline(x_cross - x_min, color="orange",
                                lw=1.0, ls="--", alpha=0.6)
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
                            step="post", color="orange", alpha=0.25)
        ax_lat.fill_between(xs_fit, ys_fit, y_slo,
                            where=mask_green, interpolate=True,
                            color="green", alpha=0.15)
        ax_lat.fill_between(xs_fit, ys_fit, y_slo,
                            where=mask_orange, interpolate=True,
                            color="orange", alpha=0.25)
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
    ax_lat.set_ylabel("Real\nLatency (ms)", fontsize=FS_LABEL)
    ax_lat.set_ylim(50, 90)

    ax_dep.set_ylabel("Deposited\nTokens", fontsize=FS_LABEL)
    ax_dep.set_ylim(0, 38)

    ax_perc.set_xlabel("Time (s)", fontsize=FS_LABEL)
    ax_perc.set_ylabel("User Perceived\nLatency (ms)", fontsize=FS_LABEL)
    ax_perc.set_ylim(50, 90)

    # apply tick-label size to every axis
    for ax in (ax_lat, ax_dep, ax_perc):
        ax.set_xticks([0, 40, 80])
        ax.tick_params(labelsize=FS_TICK)

    fig.tight_layout()

    out_path_png = csv_path.with_name(f"{csv_path.stem}_fig_4_2.png")
    fig.savefig(out_path_png, dpi=150, bbox_inches="tight")
    out_path_pdf = csv_path.with_name(f"{csv_path.stem}_fig_4_2.pdf")
    fig.savefig(out_path_pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"green_cnt: {green_cnt}, orange_cnt: {orange_cnt}, red_cnt: {red_cnt}")
    # return the PDF path (change if you prefer PNG)
    return out_path_png


def make_second(df: pd.DataFrame, csv_path: Path) -> Path:
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
    fig, (ax_lat, ax_dep, ax_perc) = plt.subplots(
        3, 1, figsize=(6, 8),
        gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 0.3},
    )
    ax_dep.sharex(ax_lat)        # share x for the top two only

    ymax_lat, ymax_dep, ymax_perc = 0.0, 0, 0.0

    # ------------------------------------------------------------------ #

    rid1        = df.at[0, "request_id"]
    rid2        = df.at[1, "request_id"]
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
        for st, dt in zip(solver_ts, tbt_list):
            step_time = (dt - st if dt - st >= 0.035 else dt) - 0.003
            cum += step_time
            xs_req.append(cum - t0)
            ys_req.append(step_time * 1000)         # s → ms

    # ys_req = [y + 0.3 for y in ys_req]
    x1_min, x1_max = xs_req[0], xs_req[-1]
    x2_min, x2_max = xs_req[99], xs_req[199]

    # ---- linear fit --------------------------------------------- #
    m1, b1 = np.polyfit(xs_req, ys_req, 1)
    m2, b2 = np.polyfit(xs_req[99:200], ys_req[99:200], 1)

    xs_fit1 = np.linspace(x1_min, x2_min, 150)
    ys_fit1 = m1 * xs_fit1 + b1
    xs_rel1  = xs_fit1
    ax_lat.plot(xs_fit1, ys_fit1, color=color, lw=1.5,
            label =rid1)

    xs_fit2 = np.linspace(x2_min, x2_max, 150)
    ys_fit2 = m2 * xs_fit2 + b2
    xs_rel2 = xs_fit2
    ax_lat.plot(xs_fit2, ys_fit2, color=color, lw=1.5,
            label =rid2)

    
    xs_fit3 = np.linspace(x2_max, x1_max, 150)
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
                where="post", color=color, lw=0.1)
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
                    colors="black", lw=1.2, ls="--")
    ax_lat.text(
        x1_min + 1, y_slo + 2, f"SLO = {int(slo_thr*1000)} ms",
        color="black", fontsize=FS_NOTE, ha="left", va="bottom"
    )
    
    for i, x_depleted in enumerate(x_depleteds):
        # if i == 0: continue
        if np.isfinite(x_depleted) and x1_min <= x_depleted <= x1_max:
            # depletion marker
            for ax in (ax_lat, ax_dep, ax_perc):
                ax.axvline(x_depleted, color="red", lw=1.0, ls=":", alpha=0.8)
            
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
            for axv in (ax_lat, ax_dep):
                axv.axvline(x_cross, color="orange",
                            lw=1.0, ls="--", alpha=0.6)
            ax_perc.axvline(x_cross - x1_min, color="orange",
                            lw=1.0, ls="--", alpha=0.6)
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
    for axv in (ax_lat, ax_dep):
        axv.axvline(x2_min, color="orange",
                    lw=1.0, ls="--", alpha=0.6)
    ax_perc.axvline(x2_min, color="orange",
                    lw=1.0, ls="--", alpha=0.6)


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
                        step="post", color="orange", alpha=0.25)
    ax_lat.fill_between(xs_fit1, ys_fit1, y_slo,
                        where=mask_green, interpolate=True,
                        color="green", alpha=0.15)
    ax_lat.fill_between(xs_fit1, ys_fit1, y_slo,
                        where=mask_orange, interpolate=True,
                        color="orange", alpha=0.25)
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
                        step="post", color="orange", alpha=0.25)
    ax_lat.fill_between(xs_fit2, ys_fit2, y_slo,
                        where=mask_green, interpolate=True,
                        color="green", alpha=0.15)
    ax_lat.fill_between(xs_fit2, ys_fit2, y_slo,
                        where=mask_orange, interpolate=True,
                        color="orange", alpha=0.25)
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
                        step="post", color="orange", alpha=0.25)
    ax_lat.fill_between(xs_fit3, ys_fit3, y_slo,
                        where=mask_green, interpolate=True,
                        color="green", alpha=0.15)
    ax_lat.fill_between(xs_fit3, ys_fit3, y_slo,
                        where=mask_orange, interpolate=True,
                        color="orange", alpha=0.25)
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
    ax_lat.set_ylim(50, 90)
    ax_lat.set_yticks([])

    # ax_dep.set_ylabel("Deposited\nTokens", fontsize=FS_LABEL)
    ax_dep.set_ylim(0, 38)
    ax_dep.set_yticks([])

    ax_perc.set_xlabel("Time (s)", fontsize=FS_LABEL)
    # ax_perc.set_ylabel("User Perceived\nLatency (ms)", fontsize=FS_LABEL)
    ax_perc.set_ylim(50, 90)
    ax_perc.set_yticks([])
       

    # apply tick-label size to every axis
    for ax in (ax_lat, ax_dep, ax_perc):
        ax.set_xticks([0, 40, 80])
        ax.tick_params(labelsize=FS_TICK)

    fig.tight_layout()

    out_path_png = csv_path.with_name(f"{csv_path.stem}_fig_4_2.png")
    fig.savefig(out_path_png, dpi=150, bbox_inches="tight")
    out_path_pdf = csv_path.with_name(f"{csv_path.stem}_fig_4_2.pdf")
    fig.savefig(out_path_pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"green_cnt: {green_cnt}, orange_cnt: {orange_cnt}, red_cnt: {red_cnt}")
    # return the PDF path (change if you prefer PNG)
    return out_path_png



"""
1) 먼저 figure_4_2(df, csv_path)와 figure_4_2_2(df, csv_path)를 호출해서
두 개의 PNG 파일을 생성합니다.
2) 그다음, 두 PNG 이미지를 불러와서 하나의 캔버스에 좌우로 붙여서
하나의 그림으로 저장하며, PNG와 PDF 둘 다 출력합니다.
"""
# --- 1. 두 함수를 순서대로 호출해, 각각 PNG를 생성시킨다 --- #
csv1 = Path("/home/heelim/vllm/outputs/benchmark/figure_4_2_token_deposit/outputs.csv")
csv2 = Path("/home/heelim/vllm/outputs/benchmark/figure_4_2_token_deposit_2/outputs.csv")

png1 = figure_4_2(load_metrics(csv1), csv1)     # ex. "..._fig_4_2.png"
png2 = make_second(load_metrics(csv2), csv2)   # ex. "..._fig_4_2_2.png"

# --- 2. 저장된 두 이미지를 불러온다 --- #
img1 = mpimg.imread(png1)
img2 = mpimg.imread(png2)

import matplotlib.patches as mpatches

# --- 3. 새로운 캔버스를 만들고, 1행 2열로 배치 --- #
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(8, 6),   # 넓이를 두 배로 잡아 이미지를 나란히 배치
    gridspec_kw={"wspace": 0.0, "hspace": 0, "width_ratios": [1, 0.832]}
)

ax1.imshow(img1)
ax1.axis("off")

ax2.imshow(img2)
ax2.axis("off")

# ▼ 이 아래 부분을 추가하세요 ▼

# --- 범례용 패치(patches) 생성 --- #
green_patch  = mpatches.Patch(color="green", alpha=0.15, label="Within SLO")
orange_patch = mpatches.Patch(color="orange", alpha=0.25, label="Masked Violations")
red_patch    = mpatches.Patch(color="red",  alpha=0.25,  label="Perceived Violations")

# --- Figure 전체 위쪽에 범례를 표시 --- #
# loc="upper center"로 하면 그림 상단 중앙에 붙습니다.
# ncol=3 로 한 줄에 3개 아이템을 나란히 배치합니다.
fig.legend(
    handles=[green_patch, orange_patch, red_patch],
    loc="upper center",
    ncol=3,
    frameon=False,       # 테두리 없이 표시
    fontsize=10,
    bbox_to_anchor=(0.5, 0.9)  # 그림 위쪽(약간 여백을 두고) 배치
)

# ▼ 범례 추가 끝 ▼

out = Path("/home/heelim/vllm/benchmark/data_analysis/figures/")

# --- 4. 하나로 합친 그림을 PNG로 저장 --- #
fig.savefig(f"{out}/deposit_combined.png", dpi=150, bbox_inches="tight")

# --- 5. 동일한 canvas를 PDF로도 저장 --- #
fig.savefig(f"{out}/deposit_combined.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)
