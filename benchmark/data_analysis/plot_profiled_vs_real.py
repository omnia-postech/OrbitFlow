from __future__ import annotations
import numpy as np          
import ast
import sys
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
# --- at top ---
from dataclasses import dataclass
import statsmodels.api as sm          # after you’ve installed it

import numpy as np
from typing import Tuple

import numpy as np
from typing import Tuple

# ─────────────────────────────────────────────────────────────────────────────
# helper: upper convex hull with optional "keep-only-top-X%" pre-filter
# ─────────────────────────────────────────────────────────────────────────────
def upper_hull(x: np.ndarray,
               y: np.ndarray,
               top_frac: float = 1.0   # e.g. 0.05 keeps the upper 5 %
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    x, y     : 1-D ndarrays of equal length (x need not be sorted).
    top_frac : float in (0, 1]; fraction of points (ranked by y) to keep
               *before* computing the upper convex hull.

    Returns
    -------
    hx, hy   : ndarrays of hull points (sorted by x ascending)
    """
    if not (0.0 < top_frac <= 1.0):
        raise ValueError("top_frac must be in (0, 1].")

    # ── 0 · optional percentile filter ───────────────────────────────────
    if top_frac < 1.0:
        thresh = np.quantile(y, 1.0 - top_frac)      # keep ≥ this y
        mask   = y >= thresh
        x, y   = x[mask], y[mask]

    # ── 1 · sort by x ────────────────────────────────────────────────────
    order = np.argsort(x)
    xs, ys = x[order], y[order]

    # ── 2 · monotone chain for upper hull ───────────────────────────────
    hull: list[Tuple[float, float]] = []
    for xi, yi in zip(xs, ys):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (yi - y1) - (y2 - y1) * (xi - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append((xi, yi))

    hx, hy = np.array(hull).T
    return hx, hy



def _quantile_fit(x: np.ndarray,
                  y: np.ndarray,
                  degree: int = 1,
                  tau: float = 0.95):
    """
    Return (coeffs, y_hat, coverage, pseudo_R2) for a τ-quantile polynomial.

    * `coeffs`  : highest-degree first (aₙ … a₀)
    * `coverage`: fraction of points ≤ y_hat   (≈ tau if model is sensible)
    * `pseudo_R2` (McFadden) for reference, not identical to OLS R².
    """
    # Design matrix  [1, x, x², …]
    X = np.column_stack([x**k for k in range(degree, -1, -1)])
    X = sm.add_constant(X[:, 1:]) if degree == 0 else X  # keep shape happy

    mod = sm.QuantReg(y, X)
    res = mod.fit(q=tau)

    # predict
    y_hat = res.predict(X)

    # pseudo-R²: 1 − (L1_loss_model / L1_loss_const)
    l1_model = np.sum(np.abs(y - y_hat))
    l1_null  = np.sum(np.abs(y - np.percentile(y, tau*100)))
    pseudo_r2 = np.nan if l1_null == 0 else 1.0 - l1_model / l1_null

    coverage = np.mean(y <= y_hat)

    return res.params, y_hat, float(coverage), float(pseudo_r2)

@dataclass(frozen=True)
class FitStats:
    slope: float
    intercept: float
    r2: float
    n_points: int
def _linear_fit_stats(x: np.ndarray, y: np.ndarray) -> FitStats:
    """Return slope, intercept, R² and #points."""
    m, b = np.polyfit(x, y, 1)
    y_hat = m * x + b
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return FitStats(m, b, float(r2), len(x))
# ------------------------------------------------------------------ #
# util: R² for an arbitrary (slope, intercept) pair
# ------------------------------------------------------------------ #
def _r2_from_line(x: np.ndarray, y: np.ndarray, m: float, b: float) -> float:
    y_hat = m * x + b
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

def _draw_four_fits(ax, x: np.ndarray, y: np.ndarray,
                    color_lin="red",
                    color_quad="green",
                    color_env="purple",
                    color_qr="orange",
                    tau: float = 0.99):
    """
    Shows:

      • OLS  linear            (solid      red)
      • OLS  quadratic         (dash-dot   green)
      • Upper envelopes (lin + quad)  (dashed  purple / dotted purple)
      • τ-quantile regression  (solid orange)     ⟵ NEW
    """

    # ---------- existing OLS + envelopes (unchanged) ---------------------
    stats_lin = _linear_fit_stats(x, y)
    m, b, r2_lin = stats_lin.slope, stats_lin.intercept, stats_lin.r2

    delta_lin = float(np.max(y - (m * x + b)))
    b_up_lin  = b + delta_lin + 1e-12
    r2_lup    = _r2_from_line(x, y, m, b_up_lin)

    a2, a1, a0 = np.polyfit(x, y, 2)
    y_hat_q = a2 * x**2 + a1 * x + a0
    ss_res_q   = np.sum((y - y_hat_q) ** 2)
    ss_tot_q   = np.sum((y - y.mean()) ** 2)
    r2_quad    = np.nan if ss_tot_q == 0 else 1.0 - ss_res_q / ss_tot_q


    num_outlier = len(y) // 1000
    # ── Quadratic "almost-envelope": ignore the 5 biggest positive residuals ──
    residuals = y - y_hat_q
    pos_res   = residuals[residuals > 0]                   # only those above curve

    if pos_res.size > num_outlier-1:                                   # drop the top-5 outliers
        # np.partition is O(N) and avoids full sort
        kth = np.partition(pos_res, -num_outlier)[-num_outlier]                # 6-th largest element
        delta_q = float(kth)
    else:                                                  # not enough points
        delta_q = float(pos_res.max()) if pos_res.size else 0.0

    a0_up        = a0 + delta_q + 1e-12
    y_hat_qup    = a2 * x**2 + a1 * x + a0_up
    ss_res_qup   = np.sum((y - y_hat_qup) ** 2)
    r2_qup       = np.nan if ss_tot_q == 0 else 1.0 - ss_res_qup / ss_tot_q

    # ---------- NEW: τ-quantile regression (linear, for clarity) ----------
    coeffs_qr, y_hat_qr, cov_qr, pseudo_r2 = _quantile_fit(x, y,
                                                           degree=1,
                                                           tau=tau)
    mq, bq = coeffs_qr      # slope, intercept (since degree=1)



    # ---------- plotting --------------------------------------------------
    x_fit = np.linspace(x.min(), x.max(), 400)

    ax.plot(x_fit, m * x_fit + b,
            lw=1.2, color=color_lin,
            label=(f"OLS-lin   : y={m:.2e}x+{b:.2e}\n"
                   f"            R²={r2_lin:.4f}"))

    # ax.plot(x_fit, a2 * x_fit**2 + a1 * x_fit + a0,
    #         lw=1.2, ls="-.", color=color_quad,
    #         label=(f"OLS-quad  : {a2:.2e}x²+{a1:.2e}x+{a0:.2e}\n"
    #                f"            R²={r2_quad:.4f}"))

    # ax.plot(x_fit, m * x_fit + b_up_lin,
    #         lw=1.2, ls="--", color=color_env,
    #         label=(f"Upper-lin : y={m:.2e}x+{b_up_lin:.2e}\n"
    #                f"            R²={r2_lup:.4f}"))

    ax.plot(x_fit, a2 * x_fit**2 + a1 * x_fit + a0_up,
            lw=1.2, ls=":", color=color_env,
            label=(f"Upper-quad: {a2:.2e}x²+{a1:.2e}x+{a0_up:.2e}\n"
                   f"            R²={r2_qup:.4f}"))
    # y_hat_qup = a2 * x**2 + a1 * x + a0_up-0.005
    # ss_res_qup = np.sum((y - y_hat_qup) ** 2)
    # r2_qup  = np.nan if ss_tot_q == 0 else 1.0 - ss_res_qup / ss_tot_q
    # ax.plot(x_fit, a2 * x_fit**2 + a1 * x_fit + a0_up-0.005,
    #         lw=1.2, ls=":", color=color_env,
    #         label=(f"Upper-quad1: {a2:.2e}x²+{a1:.2e}x+{a0_up-0.05:.2e}\n"
    #                f"            R²={r2_qup:.4f}"))

    ax.plot(x_fit, mq * x_fit + bq,
            lw=1.2, color=color_qr,
            label=(f"{int(tau*100)}ᵗʰ-QR : y={mq:.2e}x+{bq:.2e}\n"
                   f"            pseudo-R²={pseudo_r2:.4f}, "
                   f"cov={cov_qr:.2%}"))

def _draw_three_fits(ax,
                     x: np.ndarray, y: np.ndarray,
                     color_lin="red",
                     color_quad="green",
                     color_up="purple"):
    """
    Plots four curves:
      • OLS linear                (solid red)
      • OLS quadratic             (dash-dot green)
      • Linear upper envelope     (dashed   purple)
      • Quadratic upper envelope  (dotted   purple)
    """

    # ── 1 · OLS linear ──────────────────────────────────────────────────
    stats_lin = _linear_fit_stats(x, y)
    m, b, r2_lin = stats_lin.slope, stats_lin.intercept, stats_lin.r2

    # ── linear upper envelope (lift intercept) ──────────────────────────
    delta_lin  = float(np.max(y - (m * x + b)))
    b_up_lin   = b + delta_lin + 1e-12
    r2_lup     = _r2_from_line(x, y, m, b_up_lin)

    # ── 2 · OLS quadratic ───────────────────────────────────────────────
    a2, a1, a0 = np.polyfit(x, y, 2)            # y = a2·x² + a1·x + a0
    y_hat_quad = a2 * x**2 + a1 * x + a0
    ss_res_q   = np.sum((y - y_hat_quad) ** 2)
    ss_tot_q   = np.sum((y - y.mean()) ** 2)
    r2_quad    = np.nan if ss_tot_q == 0 else 1.0 - ss_res_q / ss_tot_q

    # ── quadratic upper envelope (lift constant) ────────────────────────
    delta_quad = float(np.max(y - y_hat_quad))
    a0_up      = a0 + delta_quad + 1e-12
    y_hat_qup  = a2 * x**2 + a1 * x + a0_up
    ss_res_qup = np.sum((y - y_hat_qup) ** 2)
    r2_qup     = np.nan if ss_tot_q == 0 else 1.0 - ss_res_qup / ss_tot_q

    # ── Plotting ─────────────────────────────────────────────────────────
    x_fit = np.linspace(x.min(), x.max(), 300)

    # linear OLS
    ax.plot(x_fit, m * x_fit + b,
            lw=1.2, color=color_lin,
            label=(f"OLS-lin   : y={m:.2e}x+{b:.2e}\n"
                   f"            R²={r2_lin:.6f}"))

    # quadratic OLS
    ax.plot(x_fit, a2 * x_fit**2 + a1 * x_fit + a0,
            lw=1.2, ls="-.", color=color_quad,
            label=(f"OLS-quad  : y={a2:.2e}x²+{a1:.2e}x+{a0:.2e}\n"
                   f"            R²={r2_quad:.6f}"))

    # linear upper envelope (dashed)
    ax.plot(x_fit, m * x_fit + b_up_lin,
            lw=1.2, ls="--", color=color_up,
            label=(f"Upper-lin : y={m:.2e}x+{b_up_lin:.2e}\n"
                   f"            R²={r2_lup:.6f}"))

    # quadratic upper envelope (dotted)
    ax.plot(x_fit, a2 * x_fit**2 + a1 * x_fit + a0_up,
            lw=1.2, ls=":", color=color_up,
            label=(f"Upper-quad: y={a2:.2e}x²+{a1:.2e}x+{a0_up:.2e}\n"
                   f"            R²={r2_qup:.6f}"))
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

# ──────────────────────────────────────────────────────────────────────────────
# Robust per-segment outlier masking (MAD-based)
# ──────────────────────────────────────────────────────────────────────────────
def _mask_inliers(y: np.ndarray, z_threshold: float = 3.0) -> np.ndarray:
    """Return boolean mask marking in-liers according to MAD test."""
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if mad == 0:          # all identical; keep everything
        return np.ones_like(y, dtype=bool)
    z   = 0.6745 * (y - med) / mad
    return np.abs(z) < z_threshold


# ──────────────────────────────────────────────────────────────────────────────
# Main driver
# ──────────────────────────────────────────────────────────────────────────────
def write_all_tbt_figures(df: pd.DataFrame, csv_path: Path) -> List[Path]:
    """
    Produce 4 per-segment figures (i = 0‥3) and 1 global figure.
    Each plot shows

      • an OLS fit (solid red)  +  R²_OLS
      • an “upper-envelope” line (dashed purple)  +  R²_upper
        ─ same slope, intercept lifted so *all* points lie under/ON it

    Returns
    -------
    List[Path]
        [seg_0, seg_1, seg_2, seg_3, global_all]
    """
    if "time_between_tokens" not in df.columns:
        raise KeyError("Column 'time_between_tokens' missing!")

    prompt_offsets: Dict[int, int] = {0: 10, 1: 1_000, 2: 10_000, 3: 100_000}
    per_seg_xy: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    # ── collect & clean each segment ─────────────────────────────────────────
    for idx, row in df.iterrows():
        if idx not in prompt_offsets:
            continue

        tbt = row["time_between_tokens"]
        if not isinstance(tbt, (list, tuple)):
            continue  # skip malformed rows

        y_raw = np.asarray(tbt, dtype=float)
        x_raw = np.arange(len(y_raw)) + prompt_offsets[idx]

        mask = _mask_inliers(y_raw)              # per-segment cleaning
        per_seg_xy[idx] = (x_raw[mask], y_raw[mask])

    stem, out_paths = csv_path.stem, []

    # ─────────────────────── helper for plotting two lines ───────────────────
    def _draw_two_fits(ax, x, y, color_ols="red", color_up="purple"):
        # OLS
        stats = _linear_fit_stats(x, y)
        m, b, r2_ols = stats.slope, stats.intercept, stats.r2

        # Upper-envelope (same slope, intercept lifted)
        delta = float(np.max(y - (m * x + b)))          # ≥ 0
        b_up  = b + delta + 1e-12                       # epsilon for safety
        r2_up = _r2_from_line(x, y, m, b_up)

        x_fit = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_fit, m * x_fit + b,
                lw=1.2, color=color_ols,
                label=(f"OLS   : y={m:.10e}x+{b:.10e}\n"
                       f"        R²={r2_ols:.6f}"))
        ax.plot(x_fit, m * x_fit + b_up,
                lw=1.2, ls="--", color=color_up,
                label=(f"Upper : y={m:.10e}x+{b_up:.10e}\n"
                       f"        R²={r2_up:.6f}"))
        return

    # ── 4 individual figures ────────────────────────────────────────────────
    for seg_idx, (x, y) in per_seg_xy.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(x, y, s=8, alpha=0.6)

        # _draw_two_fits(ax, x, y)
        # per-segment
        # _draw_three_fits(ax, x, y)
        _draw_four_fits(ax, x, y) 
        ax.set_xlabel("Output-token index")
        ax.set_ylabel("Δt (s)")
        ax.set_title(f"TBT scatter + fits  (segment {seg_idx})")
        ax.grid(True, linewidth=0.3)
        ax.legend(frameon=False, fontsize="small")
        plt.tight_layout()

        p = csv_path.with_name(f"{stem}_tbt_i{seg_idx}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(p)

    # ── global figure ───────────────────────────────────────────────────────
    all_x = np.concatenate([xy[0] for xy in per_seg_xy.values()])
    all_y = np.concatenate([xy[1] for xy in per_seg_xy.values()])

    fig, ax = plt.subplots(figsize=(8, 4))
    for seg_idx, (x, y) in per_seg_xy.items():
        ax.scatter(x, y, s=8, alpha=0.6, label=f"seg {seg_idx}")

    # _draw_two_fits(ax, all_x, all_y)
    # _draw_three_fits(ax, all_x, all_y)
    _draw_four_fits(ax, all_x, all_y) 
    ax.set_xlabel("Output-token index")
    ax.set_ylabel("Δt (s)")
    ax.set_title("TBT scatter + global fits (all segments)")
    ax.grid(True, linewidth=0.3)
    ax.legend(frameon=False, fontsize="small")
    plt.tight_layout()

    p_all = csv_path.with_name(f"{stem}_tbt_all.png")
    fig.savefig(p_all, dpi=150, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(p_all)

    #  NEW: hull figure ─────────────────────────────────────────────────
    # 1. extract upper-hull points
    hull_x, hull_y = upper_hull(all_x, all_y, top_frac=0.5)

    # 2. polynomial fit on the hull itself (degree 2 here)
    a2_h, a1_h, a0_h = np.polyfit(hull_x, hull_y, deg=2)

    # 3. draw
    fig_h, ax_h = plt.subplots(figsize=(8, 4))
    ax_h.scatter(all_x, all_y, s=8, alpha=0.3, color="grey", label="all points")
    ax_h.plot(hull_x, hull_y, lw=1.2, color="purple",
              label="upper hull (poly-line)")

    x_fit_h = np.linspace(all_x.min(), all_x.max(), 400)
    ax_h.plot(x_fit_h, a2_h * x_fit_h**2 + a1_h * x_fit_h + a0_h,
              lw=1.2, color="red",
              label=(f"hull-fit quad: {a2_h:.2e}x²+{a1_h:.2e}x+{a0_h:.2e}"))

    ax_h.set_xlabel("Output-token index")
    ax_h.set_ylabel("Δt (s)")
    ax_h.set_title("TBT scatter, upper hull, and quadratic fit")
    ax_h.grid(True, linewidth=0.3)
    # ax_h.set_ylim(0, 0.5)
    ax_h.legend(frameon=False, fontsize="small")
    plt.tight_layout()

    p_hull = csv_path.with_name(f"{stem}_tbt_all_hull.png")
    fig_h.savefig(p_hull, dpi=150, bbox_inches="tight")
    plt.close(fig_h)
    out_paths.append(p_hull)
    # ░░ END new hull figure ░░──────────────────────────────────────────────

    return out_paths


def main(argv) -> None:

    csv = argv[0]
    print(f"CSV path: {csv}")
    csv_path = Path(csv).expanduser().resolve()
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    df = load_metrics(csv_path)
    out_path1 = write_all_tbt_figures(df, csv_path)
    print(f"Figure written ➜ {out_path1}")

    csv2 = argv[1] if len(argv) > 1 else None 
    if csv2:
        print(f"CSV path: {csv2}")
        csv_path2 = Path(csv2).expanduser().resolve()
        if not csv_path2.exists():
            sys.exit(f"CSV not found: {csv_path2}")

        df2 = load_metrics(csv_path2)
        out_path2 = write_all_tbt_figures(df2, csv_path2)
        print(f"Figure written ➜ {out_path2}")
        
        df = df[['time_between_tokens', "request_id"]]
        df3 = df2[['time_between_tokens', "request_id"]]
        df3['time_between_tokens'][0]= [x-y for x, y in zip(df2['time_between_tokens'][0], df['time_between_tokens'][0])]
        csv3 = "/home/xinyuema/vllm/outputs/benchmark/Profile/NextLayer/profile_trace/outputs_diff.csv"
        csv_path3 = Path(csv3).expanduser().resolve()
        out_path3 = write_all_tbt_figures(df3, csv_path3)
        print(f"Figure written ➜ {out_path3}")
if __name__ == "__main__":
    import sys
    main(sys.argv[1:])