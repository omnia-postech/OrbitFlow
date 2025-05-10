#!/usr/bin/env python
"""
metrics_plots.py
================
Utility to visualise vLLM-style inference logs stored as CSV.

Usage
-----
# overview (E2E latency, throughputs, SLO violations)
python metrics_plots.py stats  /path/to/log.csv

# per-request scatter of inter-token times
python metrics_plots.py tbt    /path/to/log.csv
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────
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
# 1. Trace-level overview  →  PNG
# ──────────────────────────────────────────────────────────────────────────────
def plot_stats_overview(df: pd.DataFrame, csv_path: Path) -> Path:
    """
    Build a bar-chart summary and save next to the CSV.
    Returns the figure path.
    """
    # ── Metrics
    mean_e2e = df["end_to_end_time"].mean()

    # Prefill throughput; if you track 'input_length', use it
    prefill_tokens = df.get("input_length", pd.Series(1, index=df.index))
    prefill_thr = (prefill_tokens / df["time_to_first_token"]).mean()

    # Decode throughput
    out_thr = (df["decode_length"] / df["decode_time"]).mean()

    total_viol = df["slo_violations"].sum()
    total_decodes = df["decode_length"].sum() 
    slo_ratio = total_viol / total_decodes * 100 
    labels = [
        "Mean E2E latency (s)",
        "Mean prefill throughput (tok/s)",
        "Mean output throughput (tok/s)",
        "Total SLO violations",
    ]
    values = [mean_e2e, prefill_thr, out_thr, total_viol]

    # ── Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values)
    ax.set_title("Trace-level summary")
    ax.set_ylabel("Value")
    ax.set_xticklabels(labels, rotation=20, ha="right")
    for i, v in enumerate(values):
        if i == 3: # violation 
            msg = f"{v:.1f}\n{(total_viol / total_decodes * 100):.2f} %"
        else: 
            msg = f"{v:.2f}"
        ax.text(i, v, msg, ha="center", va="bottom", fontsize=8)
    plt.tight_layout()

    out_file = csv_path.with_name(f"{csv_path.stem}_overview.png")
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_file


# ──────────────────────────────────────────────────────────────────────────────
# 2. Per-request scatter of Δt between tokens  →  PNG
# ──────────────────────────────────────────────────────────────────────────────
def plot_time_between_tokens(df: pd.DataFrame, csv_path: Path) -> Path:
    """
    Scatter cloud (no lines) of inter-token latencies per request.
    Saves the figure next to the CSV and returns its path.
    """
    if "time_between_tokens" not in df.columns:
        raise KeyError("Column 'time_between_tokens' missing!")

    fig, ax = plt.subplots(figsize=(8, 4))
    for _, row in df.iterrows():
        tbt = row["time_between_tokens"]
        if isinstance(tbt, (list, tuple)):
            x = range(len(tbt))
            ax.scatter(x, tbt, s=8, alpha=0.6)

    ax.set_xlabel("Output-token index")
    ax.set_ylabel("Δt (s)")
    ax.set_title("Inter-token latency per request (scatter)")
    ax.grid(True, linewidth=0.3)
    ax.set_ylim(0, 0.5)
    plt.tight_layout()
    out_file = csv_path.with_name(f"{csv_path.stem}_tbt.png")
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_file

def plot_time_between_tokens_wallclock(df: pd.DataFrame, csv_path: Path) -> Path:
    """
    Scatter-plot per-token latencies on a wall-clock timeline.

    • Each request gets its own color (Tab10 palette, repeats after 10).
    • Horizontal line at 1 / slo_threshold shows the token-latency budget.
    • X-axis: seconds since the first arrival.
    • Y-axis: latency that produced each token.

    Returns the PNG path.
    """
    required = ("arrival_time", "finished_time",
                "time_to_first_token", "time_between_tokens",
                "slo_threshold")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s): {missing}")

    t0      = df["arrival_time"].min()          # reference zero
    palette = list(plt.cm.tab10.colors)         # 10 distinct colors
    xs, ys, cs = [], [], []                     # scatter data

    fig, ax = plt.subplots(figsize=(10, 4))

    for idx, (_, row) in enumerate(df.iterrows()):

        rid       = row["request_id"]
        color      = palette[idx % 10]
        arrival    = row["arrival_time"]
        finished   = row["finished_time"]
        ttf        = row["time_to_first_token"]
        tbt_list   = row["time_between_tokens"]
        slo_thr    = row["slo_threshold"]

        # 1) scatter the first token
        first_tok_wall = arrival + ttf
        xs.append(first_tok_wall - t0)
        ys.append(ttf)
        cs.append(color)

        # 2) scatter subsequent tokens
        if isinstance(tbt_list, (list, tuple)):
            cum = first_tok_wall
            for dt in tbt_list:
                cum += dt
                xs.append(cum - t0)
                ys.append(dt)
                cs.append(color)
        ax.scatter(xs, ys, s=0.01, alpha=0.001, c=[color], label=rid, linewidths=0)
        # 3) draw SLO line (seconds per token = 1 / tokens-per-second)
        if isinstance(slo_thr, (int, float)) and slo_thr > 0:
            x_start, x_end = arrival - t0, finished - t0
            ax.hlines(y=slo_thr,
                      xmin=arrival - t0,
                      xmax=finished - t0,
                      colors=color, linewidth=1.0, alpha=0.8)
            # Text slightly above the line at its midpoint
            x_mid = (x_start + x_end) / 2
            ax.text(x_mid, slo_thr + 0.01+idx*0.02, rid,
                    color=color, fontsize=8, ha="center", va="bottom")
    # --- final plot settings -----------------------------------------------
    ax.scatter(xs, ys, s=6, alpha=0.6, c=cs, linewidths=0)
    ax.set_xlabel("Wall-clock time since start (s)")
    ax.set_ylabel("Per-token latency Δt (s)")
    ax.set_title("Inter-token latency over time (colored by request)")
    ax.grid(True, linewidth=0.3)
    ax.set_ylim(0, 0.5)

    # Place legend outside plot if up to ten entries; otherwise inside upper right
    if df.shape[0] <= 10:
        ax.legend(title="request_id", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
        fig.subplots_adjust(right=0.78)           # make space for legend
    else:
        ax.legend(title="request_id", loc="upper right", fontsize="small")

    plt.tight_layout()
    out_file = csv_path.with_name(f"{csv_path.stem}_tbt_wallclock.png")
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_file
# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _usage() -> None:
    print(
        "Usage:\n"
        "  python metrics_plots.py stats <log.csv>\n"
        "  python metrics_plots.py tbt   <log.csv>"
    )


def main(argv: List[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    if len(argv) != 2 or argv[0] not in {"stats", "tbt", "tbt_wc"}:
        _usage()
        sys.exit(1)

    mode, csv = argv
    csv_path = Path(csv).expanduser().resolve()
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    df = load_metrics(csv_path)
    if mode == "tbt_wc":
        out_path = plot_time_between_tokens_wallclock(df, csv_path)
        print(f"Figure written ➜ {out_path}")
        return
    if mode == "tbt":
        out_path = plot_time_between_tokens(df, csv_path)
        print(f"Figure written ➜ {out_path}")
        return
    elif mode == "stats":
        out_path = plot_stats_overview(df, csv_path)
    else: 
        raise ValueError(f"Unknown mode: {mode}")
    print(f"Figure written ➜ {out_path}")


if __name__ == "__main__":
    main()
