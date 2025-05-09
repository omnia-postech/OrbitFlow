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
        ax.text(i, v, f"{v:,.2f}", ha="center", va="bottom", fontsize=8)
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
    plt.tight_layout()

    out_file = csv_path.with_name(f"{csv_path.stem}_tbt.png")
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
    if len(argv) != 2 or argv[0] not in {"stats", "tbt"}:
        _usage()
        sys.exit(1)

    mode, csv = argv
    csv_path = Path(csv).expanduser().resolve()
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    df = load_metrics(csv_path)
    out_path = (
        plot_stats_overview(df, csv_path)
        if mode == "stats"
        else plot_time_between_tokens(df, csv_path)
    )
    print(f"Figure written ➜ {out_path}")


if __name__ == "__main__":
    main()
