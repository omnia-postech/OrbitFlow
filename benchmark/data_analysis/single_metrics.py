"""plot_multi_trace_slo.py
================================
Compare **SLO‑violation metrics** and high‑level throughput stats across
multiple vLLM trace CSVs.

Charts produced
---------------
* **Token‑level SLO violations** – per‑token budget misses.
* **Request‑level SLO violations** – TPOT > mean(slo_threshold).
* **Grouped stats** – overall throughput, prefill throughput, decode throughput.

All helper functions accept a list of `(Path, label)` tuples plus
`out_dir`, `out_name`, and `title` keyword arguments.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CSV loading helpers
# ──────────────────────────────────────────────────────────────────────────────
_NUMERIC_COLS: List[str] = [
    "decode_length",
    "slo_violations",
    "time_per_output_token",
    "end_to_end_time",
    "time_to_first_token",
    "decode_time",
]
_LIST_COLS: List[str] = ["slo_threshold", "input_length"]


def _load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in _LIST_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# SLO violation extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_token_slo_stats(df: pd.DataFrame) -> tuple[int, int]:
    viol = int(df.get("slo_violations", pd.Series(0)).sum())
    dec = int(df.get("decode_length", pd.Series(0)).sum())
    return viol, dec


def _extract_request_tpot_stats(df: pd.DataFrame) -> tuple[int, int]:
    if {"time_per_output_token", "slo_threshold"}.issubset(df.columns):
        tpot = pd.to_numeric(df["time_per_output_token"], errors="coerce").to_numpy()
        thr = pd.to_numeric(df["slo_threshold"], errors="coerce").to_numpy()

        valid = ~np.isnan(tpot) & ~np.isnan(thr)
        return int(np.sum(tpot[valid] > thr[valid])), int(np.sum(valid))
    return 0, 0

# ──────────────────────────────────────────────────────────────────────────────
# Generic bar‑plot helper
# ──────────────────────────────────────────────────────────────────────────────

def _plot(values: List[float], totals: List[float], labels: List[str], out_png: Path, *, title: str, ylab: str):
    fig, ax = plt.subplots(figsize=(2.5 + 1 * len(values), 5))
    palette = list(plt.cm.tab10.colors)
    colours = (palette * ((len(values) + 9) // 10))[: len(values)]
    bars = ax.bar(labels, values, color=colours)
    ax.set_title(title)
    ax.set_ylabel(ylab)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    for bar, val, tot in zip(bars, values, totals or values):
        txt = f"{val:.2f}" if not tot else f"{val:.2f}-({val / tot * 100:.1f}%)" if ylab.startswith("#") else f"{val:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), txt, ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written ➜ {out_png}")

# ──────────────────────────────────────────────────────────────────────────────
# Public helpers for SLO charts
# ──────────────────────────────────────────────────────────────────────────────

def plot_token_level_slo(csv_tuples: List[Tuple[Path, str]], *, out_dir: Path, out_name: str, title: str):
    vals, tots, labels = [], [], []
    for p, lbl in csv_tuples:
        df = _load_csv(p.expanduser())
        v, t = _extract_token_slo_stats(df)
        vals.append(v); tots.append(t); labels.append(lbl)
    _plot(vals, tots, labels, out_dir / f"{out_name}.png", title=title, ylab="# Violating tokens")


def plot_request_level_slo(csv_tuples: List[Tuple[Path, str]], *, out_dir: Path, out_name: str, title: str):
    vals, tots, labels = [], [], []
    for p, lbl in csv_tuples:
        df = _load_csv(p.expanduser())
        v, t = _extract_request_tpot_stats(df)
        vals.append(v); tots.append(t); labels.append(lbl)
    _plot(vals, tots, labels, out_dir / f"{out_name}_REQ.png", title=title, ylab="# Violating requests")

# ──────────────────────────────────────────────────────────────────────────────
# New: grouped throughput stats chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_stats_overview_multi(csv_tuples: List[Tuple[Path, str]], *, out_dir: Path, out_name: str, title: str = "Trace stats"):
    metrics = [
        "Overall throughput (tok/s)",
        "Prefill throughput (tok/s)",
        "Decode throughput (tok/s)",
    ]
    vals_per_trace, labels = [], []

    for p, lbl in csv_tuples:
        df = _load_csv(p.expanduser())

        # Overall throughput: (Σ input + Σ decode) / max finished_time
        total_decode = df["decode_length"].sum()
        total_input = df.get("input_length", pd.Series(0, index=df.index)).sum()
        total_tokens = total_input + total_decode
        wall_time = df["finished_time"].max()
        overall_thr = total_tokens / wall_time if wall_time else 0.0

        # Prefill throughput – average across requests
        prefill_tokens = df.get("input_length", pd.Series(1, index=df.index))
        prefill_thr = (prefill_tokens / df["time_to_first_token"]).mean()

        # Decode throughput – use remaining wall‑clock time after all prefill stages
        prefill_total_time = df["time_to_first_token"].sum()
        decode_window = wall_time - prefill_total_time
        decode_window = max(decode_window, 1e-9)  # guard against zero / negative
        decode_thr = total_decode / decode_window 
        decode_thr = (df["decode_length"] / df["decode_time"]).mean()

        vals_per_trace.append([overall_thr, prefill_thr, decode_thr])
        labels.append(lbl)

    vals = np.array(vals_per_trace)
    n_traces, n_metrics = vals.shape
    x = np.arange(n_metrics)
    width = 0.8 / n_traces
    fig, ax = plt.subplots(figsize=(4 + 1.2 * n_metrics * n_traces, 5))
    palette = list(plt.cm.tab10.colors)
    colours = (palette * ((n_traces + 9) // 10))[: n_traces]

    for i in range(n_traces):
        bars = ax.bar(x + (i - (n_traces - 1) / 2) * width, vals[i], width, label=labels[i], color=colours[i])
        for bar, val in zip(bars, vals[i]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Throughput (tokens / second)")
    ax.legend()
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{out_name}_stats.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written ➜ {out_png}")
# ──────────────────────────────────────────────────────────────────────────────
# Batch execution block
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path

    FIG_DIR = Path("/home/xinyuema/vllm/outputs/benchmark/TestPPRBased/figures/")
    os.makedirs(FIG_DIR, exist_ok=True)

    def _run_all(traces, tag: str, descr: str):
        plot_token_level_slo(traces, out_dir=FIG_DIR, out_name=f"{tag}-SLO-TBT", title=f"{descr}-SLO-TBT")
        plot_request_level_slo(traces, out_dir=FIG_DIR, out_name=f"{tag}-SLO-TPOT", title=f"{descr}-SLO-TPOT")
        plot_stats_overview_multi(traces, out_dir=FIG_DIR, out_name=f"{tag}", title=f"{descr}-Stats")

    _run_all([
        (Path("/home/xinyuema/vllm/outputs/benchmark/TestPPRBased/Flexgen/PPR158_TPI005/outputs.csv"), "FlexGen"),
        (Path("/home/xinyuema/vllm/outputs/benchmark/TestPPRBased/Ours/PPR158_TPI005/outputs.csv"), "Ours"),
    ], tag="Trace1", descr="Trace1-(PPR158_TPI005)")

    _run_all([
        (Path("/home/xinyuema/vllm/outputs/benchmark/TestPPRBased/Flexgen/PPR250_TPI051/outputs.csv"), "FlexGen"),
        (Path("/home/xinyuema/vllm/outputs/benchmark/TestPPRBased/Ours/PPR250_TPI051/outputs.csv"), "Ours"),
    ], tag="Trace2", descr="Trace2-(PPR250_TPI051)")

    _run_all([
        (Path("/home/xinyuema/vllm/outputs/benchmark/TestPPRBased/Flexgen/PPR394_TPI099/outputs.csv"), "FlexGen"),
        (Path("/home/xinyuema/vllm/outputs/benchmark/TestPPRBased/Ours/PPR394_TPI099/outputs.csv"), "Ours"),
    ], tag="Trace3", descr="Trace3-(PPR394_TPI099)")
