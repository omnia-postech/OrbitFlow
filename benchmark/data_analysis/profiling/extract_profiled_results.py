from __future__ import annotations
"""
Refactored TBT plotting utility
- Cleans imports and removes duplicates
- Adds small utilities (FitStats, PlotConfig)
- Makes quantile regression optional (graceful fallback if statsmodels is missing)
- Factors repeated logic (drawing fits, masking, figure writing)
- Preserves original CLI behavior: positional CSV 1 is required, CSV 2 optional
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import json
# ─────────────────────────────────────────────────────────────────────────────
# Helper: collect global x/y series for all segments (NO plotting)
# ─────────────────────────────────────────────────────────────────────────────
def collect_global_xy(df: pd.DataFrame,
                      exclude_ranges: Sequence[Tuple[int, int]] | None = None) -> Tuple[np.ndarray, np.ndarray]:
    if "time_between_tokens" not in df.columns:
        raise KeyError("Column 'time_between_tokens' missing!")

    prompt_offsets: Dict[int, int] = {0: 10, 1: 5000, 2: 15000, 3: 25000}
    xs_all: List[np.ndarray] = []
    ys_all: List[np.ndarray] = []

    for idx, row in df.iterrows():
        if idx not in prompt_offsets:
            continue
        tbt = row.get("time_between_tokens")
        if not isinstance(tbt, (list, tuple)):
            continue
        y_raw = np.asarray(tbt, dtype=float)
        x_raw = np.arange(len(y_raw)) + prompt_offsets[idx]
        mask = mask_inliers_mad(y_raw) & keep_mask_excluding_ranges(x_raw, exclude_ranges)
        xs_all.append(x_raw[mask])
        ys_all.append(y_raw[mask])

    if not xs_all:
        raise ValueError("No valid segments found to build global series.")

    all_x = np.concatenate(xs_all)
    all_y = np.concatenate(ys_all)
    return all_x, all_y

# ─────────────────────────────────────────────────────────────────────────────
# Helper: choose best of three linear fit/error methods
# ─────────────────────────────────────────────────────────────────────────────
def best_linear_fit_from_three(x: np.ndarray, y: np.ndarray, cfg: PlotConfig) -> Tuple[float, float, float, str]:
    # Method 1: OLS linear
    fs = linear_fit_stats(x, y)
    best_A, best_B, best_score, best_name = fs.slope, fs.intercept, fs.r2, "ols"

    # Method 2: Linear Upper Envelope (same slope, raised intercept)
    delta_lin = float(np.max(y - (fs.slope * x + fs.intercept)))
    b_up_lin = fs.intercept + delta_lin - cfg.env_offset + 1e-12
    r2_lup = r2_from_line(x, y, fs.slope, b_up_lin)
    if (r2_lup is not np.nan) and (np.isnan(best_score) or r2_lup > best_score):
        best_A, best_B, best_score, best_name = fs.slope, b_up_lin, r2_lup, "upper_linear"

    # Method 3: Quantile Regression (linear). If unavailable, falls back to OLS values with NaN score
    mq, bq, _cov, pseudo_r2 = quantile_fit_linear(x, y, cfg.tau)
    # prefer higher score; if best_score is NaN and pseudo_r2 is not, take it
    if np.isnan(best_score):
        if not np.isnan(pseudo_r2):
            best_A, best_B, best_score, best_name = mq, bq, pseudo_r2, "quantile_linear"
    else:
        if not np.isnan(pseudo_r2) and pseudo_r2 > best_score:
            best_A, best_B, best_score, best_name = mq, bq, pseudo_r2, "quantile_linear"

    return float(best_A), float(best_B), float(best_score), best_name
import ast
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional dependency: statsmodels for quantile regression
try:
    import statsmodels.api as sm  # type: ignore
    _HAS_STATSMODELS = True
except Exception:  # pragma: no cover - optional path
    sm = None  # type: ignore
    _HAS_STATSMODELS = False


# ─────────────────────────────────────────────────────────────────────────────
# Data structures & configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FitStats:
    slope: float
    intercept: float
    r2: float
    n_points: int


@dataclass(frozen=True)
class PlotConfig:
    # colors
    color_lin: str = "red"
    color_quad: str = "green"
    color_env: str = "purple"
    color_qr: str = "orange"
    # modeling
    tau: float = 0.99            # quantile for QR
    env_offset: float = 0.0      # manual downward shift for envelopes
    hull_top_frac: float = 0.5   # fraction for upper hull pre-filter


# ─────────────────────────────────────────────────────────────────────────────
# Core math helpers
# ─────────────────────────────────────────────────────────────────────────────
def linear_fit_stats(x: np.ndarray, y: np.ndarray) -> FitStats:
    """Return slope, intercept, R² and #points for OLS linear fit."""
    m, b = np.polyfit(x, y, 1)
    y_hat = m * x + b
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return FitStats(m, b, float(r2), int(len(x)))


def r2_from_line(x: np.ndarray, y: np.ndarray, m: float, b: float) -> float:
    """Compute R² for a given line y=m x + b w.r.t. points (x, y)."""
    y_hat = m * x + b
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot


def upper_hull(x: np.ndarray,
               y: np.ndarray,
               top_frac: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Upper convex hull via monotone chain with optional pre-filter.

    Parameters
    ----------
    x, y     : 1-D ndarrays of equal length (x need not be sorted).
    top_frac : float in (0, 1]; keep top fraction by y before hull.

    Returns
    -------
    hx, hy   : ndarrays of hull points (sorted by x ascending).
    """
    if not (0.0 < top_frac <= 1.0):
        raise ValueError("top_frac must be in (0, 1].")

    if top_frac < 1.0:
        thresh = np.quantile(y, 1.0 - top_frac)
        mask = y >= thresh
        x, y = x[mask], y[mask]

    order = np.argsort(x)
    xs, ys = x[order], y[order]

    hull: List[Tuple[float, float]] = []
    for xi, yi in zip(xs, ys):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            # cross product ≥ 0 → not a strict right turn → pop
            if (x2 - x1) * (yi - y1) - (y2 - y1) * (xi - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append((xi, yi))

    hx, hy = np.array(hull).T
    return hx, hy


def quantile_fit_linear(x: np.ndarray, y: np.ndarray, tau: float) -> Tuple[float, float, float, float]:
    """τ-quantile linear regression using statsmodels if available.

    Returns (slope, intercept, coverage, pseudo_R2). If statsmodels isn't
    available, falls back to NaNs for metrics and the OLS line for coefficients.
    """
    if not _HAS_STATSMODELS:
        # Fallback: use OLS coefficients but mark metrics as NaN
        fs = linear_fit_stats(x, y)
        return fs.slope, fs.intercept, float("nan"), float("nan")

    # Design matrix for degree-1 polynomial: [x, 1]
    X = np.column_stack([x, np.ones_like(x)])
    mod = sm.QuantReg(y, X)
    res = mod.fit(q=tau)

    # unpack slope/intercept; X columns are [x, 1]
    m_q, b_q = float(res.params[0]), float(res.params[1])

    y_hat = res.predict(X)
    l1_model = np.sum(np.abs(y - y_hat))
    l1_null = np.sum(np.abs(y - np.percentile(y, tau * 100)))
    pseudo_r2 = np.nan if l1_null == 0 else 1.0 - l1_model / l1_null
    coverage = float(np.mean(y <= y_hat))

    return m_q, b_q, coverage, float(pseudo_r2)


def mask_inliers_mad(y: np.ndarray, z_threshold: float = 3.0) -> np.ndarray:
    """MAD-based outlier mask (True for inliers)."""
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if mad == 0:
        return np.ones_like(y, dtype=bool)
    z = 0.6745 * (y - med) / mad
    return np.abs(z) < z_threshold


def keep_mask_excluding_ranges(x: np.ndarray, exclude_ranges: Sequence[Tuple[int, int]] | None) -> np.ndarray:
    """Build boolean mask where points in any [lo, hi] are excluded."""
    if not exclude_ranges:
        return np.ones_like(x, dtype=bool)
    mask = np.ones_like(x, dtype=bool)
    for lo, hi in exclude_ranges:
        mask &= ~((x >= lo) & (x <= hi))
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Plotting pieces
# ─────────────────────────────────────────────────────────────────────────────

def draw_fits(ax: plt.Axes,
              x: np.ndarray,
              y: np.ndarray,
              cfg: PlotConfig) -> None:
    """Draw OLS linear, OLS quadratic (upper envelope), and QR(τ) if available."""
    # OLS linear
    stats_lin = linear_fit_stats(x, y)
    m, b, r2_lin = stats_lin.slope, stats_lin.intercept, stats_lin.r2

    # Linear upper envelope (same slope, shift intercept)
    delta_lin = float(np.max(y - (m * x + b)))
    b_up_lin = b + delta_lin - cfg.env_offset + 1e-12
    r2_lup = r2_from_line(x, y, m, b_up_lin)

    # OLS quadratic + upper envelope (by lifting constant term)
    a2, a1, a0 = np.polyfit(x, y, 2)
    y_hat_q = a2 * x ** 2 + a1 * x + a0
    ss_res_q = np.sum((y - y_hat_q) ** 2)
    ss_tot_q = np.sum((y - y.mean()) ** 2)
    r2_quad = np.nan if ss_tot_q == 0 else 1.0 - ss_res_q / ss_tot_q

    num_outlier = max(1, len(y) // 1000)
    residuals = y - y_hat_q
    pos_res = residuals[residuals > 0]
    if pos_res.size > num_outlier - 1:
        kth = np.partition(pos_res, -num_outlier)[-num_outlier]
        delta_q = float(kth)
    else:
        delta_q = float(pos_res.max()) if pos_res.size else 0.0

    a0_up = a0 + delta_q - cfg.env_offset + 1e-12

    # Quantile regression (linear)
    mq, bq, cov_qr, pseudo_r2 = quantile_fit_linear(x, y, tau=cfg.tau)

    # Draw
    x_fit = np.linspace(x.min(), x.max(), 400)

    ax.plot(x_fit, m * x_fit + b,
            lw=1.2, color=cfg.color_lin,
            label=(f"OLS-lin   : y={m:.2e}x+{b:.2e}\n"
                   f"            R²={r2_lin:.4f}"))

    ax.plot(x_fit, a2 * x_fit ** 2 + a1 * x_fit + a0_up,
            lw=1.2, ls=":", color=cfg.color_env,
            label=(f"Upper-quad: {a2:.2e}x²+{a1:.2e}x+{a0_up:.2e}\n"
                   f"            R²={r2_quad:.4f}"))

    # Only show QR label details if statsmodels is present
    if _HAS_STATSMODELS:
        qr_label = (f"{int(cfg.tau * 100)}ᵗʰ-QR : y={mq:.2e}x+{bq:.2e}\n"
                    f"            pseudo-R²={pseudo_r2:.4f}, cov={cov_qr:.2%}")
    else:
        qr_label = (f"{int(cfg.tau * 100)}ᵗʰ-QR : statsmodels not installed\n"
                    f"            (showing OLS line as placeholder)")

    ax.plot(x_fit, mq * x_fit + bq, lw=1.2, color=cfg.color_qr, label=qr_label)


# ─────────────────────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_metrics(csv_path: str | Path) -> pd.DataFrame:
    """Read the log CSV and coerce list-encoded columns to Python lists."""
    df = pd.read_csv(csv_path)

    numeric_cols: List[str] = [
        "arrival_time", "first_scheduled_time", "finished_time",
        "time_to_first_token", "slo_threshold", "slo_violations",
        "stall_duration", "decode_length", "end_to_end_time", "decode_time",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("time_between_tokens", "stall_times", "stall_durations"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main plotting driver
# ─────────────────────────────────────────────────────────────────────────────

def write_all_tbt_figures(
    df: pd.DataFrame,
    csv_path: Path,
    *,
    exclude_ranges: Sequence[Tuple[int, int]] | None = None,
    env_offset: float = 0.0,
) -> List[Path]:
    """
    Produce per-segment figures and a global figure with several fits.

    Returns a list of saved image paths.
    """
    if "time_between_tokens" not in df.columns:
        raise KeyError("Column 'time_between_tokens' missing!")

    cfg = PlotConfig(env_offset=env_offset)

    def _collect_segments(df: pd.DataFrame) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        # Prompt offsets are kept to mimic the original logic
        prompt_offsets: Dict[int, int] = {0: 10, 1: 5000, 2: 15000, 3: 25000}
        per_seg: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        for idx, row in df.iterrows():
            if idx not in prompt_offsets:
                continue
            tbt = row.get("time_between_tokens")
            if not isinstance(tbt, (list, tuple)):
                continue

            y_raw = np.asarray(tbt, dtype=float)
            x_raw = np.arange(len(y_raw)) + prompt_offsets[idx]

            mask = mask_inliers_mad(y_raw) & keep_mask_excluding_ranges(x_raw, exclude_ranges)
            per_seg[idx] = (x_raw[mask], y_raw[mask])
        return per_seg

    per_seg_xy = _collect_segments(df)

    stem = csv_path.stem
    out_paths: List[Path] = []

    # Per-segment plots
    for seg_idx, (x, y) in per_seg_xy.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(x, y, s=8, alpha=0.6)
        draw_fits(ax, x, y, cfg)
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

    # Global plot
    if per_seg_xy:
        all_x = np.concatenate([xy[0] for xy in per_seg_xy.values()])
        all_y = np.concatenate([xy[1] for xy in per_seg_xy.values()])

        fig, ax = plt.subplots(figsize=(8, 4))
        for seg_idx, (x, y) in per_seg_xy.items():
            ax.scatter(x, y, s=8, alpha=0.6, label=f"seg {seg_idx}")

        draw_fits(ax, all_x, all_y, cfg)
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

        # Upper-hull figure
        hull_x, hull_y = upper_hull(all_x, all_y, top_frac=cfg.hull_top_frac)
        a2_h, a1_h, a0_h = np.polyfit(hull_x, hull_y, deg=2)

        fig_h, ax_h = plt.subplots(figsize=(8, 4))
        ax_h.scatter(all_x, all_y, s=8, alpha=0.3, color="grey", label="all points")
        ax_h.plot(hull_x, hull_y, lw=1.2, color=cfg.color_env, label="upper hull (poly-line)")
        x_fit_h = np.linspace(all_x.min(), all_x.max(), 400)
        ax_h.plot(
            x_fit_h,
            a2_h * x_fit_h ** 2 + a1_h * x_fit_h + a0_h,
            lw=1.2,
            color=cfg.color_lin,
            label=(f"hull-fit quad: {a2_h:.2e}x²+{a1_h:.2e}x+{a0_h:.2e}"),
        )
        ax_h.set_xlabel("Output-token index")
        ax_h.set_ylabel("Δt (s)")
        ax_h.set_title("TBT scatter, upper hull, and quadratic fit")
        ax_h.grid(True, linewidth=0.3)
        ax_h.legend(frameon=False, fontsize="small")
        plt.tight_layout()

        p_hull = csv_path.with_name(f"{stem}_tbt_all_hull.png")
        fig_h.savefig(p_hull, dpi=150, bbox_inches="tight")
        plt.close(fig_h)
        out_paths.append(p_hull)

    return out_paths


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: List[str]) -> None:
    # Usage: plot_profiled_vs_real.py <csv1> [csv2] [--out path.json] [--exclude 0:3500 12000:18000]
    if not argv:
        sys.exit("Usage: plot_profiled_vs_real.py <csv1> [csv2] [--out path.json] [--exclude lo:hi ...]")

    # Parse positional CSVs and simple flags
    args: List[str] = []
    exclude_ranges: List[Tuple[int, int]] = []
    out_path: Path | None = None

    i = 0
    csvs: List[str] = []
    while i < len(argv):
        tok = argv[i]
        if tok == "--out" and i + 1 < len(argv):
            out_path = Path(argv[i + 1]).expanduser().resolve()
            i += 2
            continue
        if tok == "--exclude" and i + 1 < len(argv):
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                try:
                    lo_s, hi_s = argv[i].split(":", 1)
                    exclude_ranges.append((int(lo_s), int(hi_s)))
                except Exception:
                    sys.exit(f"Bad --exclude token: {argv[i]} (expected lo:hi)")
                i += 1
            continue
        # positional
        csvs.append(tok)
        i += 1

    if not csvs:
        sys.exit("Need at least one CSV path")

    # Default output path next to csv1
    if out_path is None:
        out_path = Path(csvs[0]).expanduser().resolve().with_name("profiled_results.json")

    # Load CSV1 (NoPrefetch)
    csv_path1 = Path(csvs[0]).expanduser().resolve()
    if not csv_path1.exists():
        sys.exit(f"CSV not found: {csv_path1}")
    df1 = load_metrics(csv_path1)

    cfg = PlotConfig()

    x1, y1 = collect_global_xy(df1, exclude_ranges)
    A1, B1, R21, name1 = best_linear_fit_from_three(x1, y1, cfg)

    results = {
        "NoPrefetch": {
            "linear": {
                "A": A1,
                "B": B1,
                "R2": round(R21, 4)
            },
            "_method": name1
        }
    }

    # Optional CSV2 (Communication/NextLayer)
    if len(csvs) >= 2:
        csv_path2 = Path(csvs[1]).expanduser().resolve()
        if not csv_path2.exists():
            sys.exit(f"CSV not found: {csv_path2}")
        df2 = load_metrics(csv_path2)
        x2, y2 = collect_global_xy(df2, exclude_ranges)
        A2, B2, R22, name2 = best_linear_fit_from_three(x2, y2, cfg)
        results["Communication"] = {
            "linear": {
                "A": A2,
                "B": B2,
                "R2": round(R22, 4)
            },
            "_method": name2
        }

    # Write JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Profile saved ➜ {out_path}")

if __name__ == "__main__":
    main(sys.argv[1:])