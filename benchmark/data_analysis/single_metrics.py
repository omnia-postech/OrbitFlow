"""plot_multi_trace_slo.py
================================
Visualise SLO *attainment* (percentage) and throughput statistics across
multiple vLLM trace CSVs.
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
# Font‑size macros
# ──────────────────────────────────────────────────────────────────────────────
TITLE_FONTSIZE   = 20
AXIS_FONTSIZE    = 15
TICK_FONTSIZE    = 15
LEGEND_FONTSIZE  = 15
ANNOT_FONTSIZE   =  15

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
    "finished_time",
]
_LIST_COLS: List[str] = ["slo_threshold", "input_length", "time_between_tokens"]


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
# SLO attainment extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_token_slo_attainment(df: pd.DataFrame) -> tuple[int, int]:
    total_decoded = int(df.get("decode_length", pd.Series(0)).sum())
    viol = int(df.get("slo_violations", pd.Series(0)).sum())
    return total_decoded - viol, total_decoded


def _extract_request_tpot_attainment(df: pd.DataFrame) -> tuple[int, int]:
    """Return (requests_meeting_SLO, total_requests) using TPOT criterion.

    For each request, compare its time_per_output_token (TPOT) with the *mean*
    of its slo_threshold list (or scalar). A request attains SLO if
    TPOT <= mean(threshold).
    """
    if {"time_per_output_token", "slo_threshold"}.issubset(df.columns):
        tpot = pd.to_numeric(df["time_per_output_token"], errors="coerce").to_numpy()
        thr_mean = np.array([
            float(np.mean(v)) if isinstance(v, (list, tuple, np.ndarray)) and len(v)
            else pd.to_numeric(v, errors="coerce")
            for v in df["slo_threshold"]
        ])
        valid = ~np.isnan(tpot) & ~np.isnan(thr_mean)
        attain = (tpot[valid] <= thr_mean[valid]).sum()
        return int(attain), int(valid.sum())
    return 0, 0


# ──────────────────────────────────────────────────────────────────────────────
# Latency‑ratio stats helper (generic percentiles)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_percentile_ratio_lists(df: pd.DataFrame, percentiles: List[int]) -> list[list[float]]:
    """For each percentile, return a list of Pxx/SLO ratios (one per request)."""
    lists = [[] for _ in percentiles]
    if {"time_between_tokens", "slo_threshold"}.issubset(df.columns):
        for tbt, thr in zip(df["time_between_tokens"], df["slo_threshold"]):
            if not isinstance(tbt, (list, tuple, np.ndarray)) or not tbt:
                continue
            if isinstance(thr, (list, tuple, np.ndarray)) and len(thr):
                thr_val = float(np.mean(thr))
            else:
                try:
                    thr_val = float(thr)
                except Exception:
                    continue
            if thr_val <= 0:
                continue
            for idx, pct in enumerate(percentiles):
                px = np.percentile(tbt, pct)
                lists[idx].append(px / thr_val)
    return lists


def _extract_percentile_ratio_stats(df: pd.DataFrame, percentiles: List[int]) -> list[float]:
    """Return mean ratio per percentile (kept for compatibility)."""
    lists = _extract_percentile_ratio_lists(df, percentiles)
    return [float(np.mean(lst)) if lst else 0.0 for lst in lists]

# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

def _plot_percent(values: List[float], labels: List[str], out_png: Path, *, title: str, ylab: str):
    fig, ax = plt.subplots(figsize=(3 + 1 * len(values), 5))
    palette = list(plt.cm.tab10.colors)
    colours = (palette * ((len(values) + 9) // 10))[: len(values)]
    
    bars = ax.bar(labels, values, color=colours)

    # styling
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_ylabel(ylab, fontsize=AXIS_FONTSIZE)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=TICK_FONTSIZE)
    ax.set_ylim(0, 120)

    for bar, pct in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct:.1f}%", ha="center", va="bottom", fontsize=ANNOT_FONTSIZE)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written ➜ {out_png}")


def _plot_grouped(vals: np.ndarray, labels: List[str], metrics: List[str], out_png: Path, title: str, *, ylab: str = "Value"):
    n_traces, n_metrics = vals.shape
    x = np.arange(n_metrics)
    width = 0.8 / n_traces
    fig, ax = plt.subplots(figsize=(2 + 1 * n_metrics * n_traces, 5))
    palette = list(plt.cm.tab10.colors)
    colours = (palette * ((n_traces + 9) // 10))[: n_traces]

    for i in range(n_traces):
        bars = ax.bar(x + (i - (n_traces - 1) / 2) * width, vals[i], width, label=labels[i], color=colours[i])
        for bar, val in zip(bars, vals[i]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=ANNOT_FONTSIZE)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20, ha="right", fontsize=TICK_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_ylabel(ylab, fontsize=AXIS_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written ➜ {out_png}")

# ──────────────────────────────────────────────────────────────────────────────
# Percentile latency ratio grouped chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_latency_ratio_multi(
    csv_tuples: List[Tuple[Path, str]],
    percentiles: List[int],
    *,
    out_dir: Path,
    out_name: str,
    title: str = "Latency percentile / SLO",
):
    """Draw box‑plots of Pxx/SLO ratios across traces.

    * `percentiles` – list like `[90,95,99]`.
    * For each percentile we show one box per trace (grouped by percentile).
    * The distribution comes from per‑request ratios.
    """
    percentiles = sorted(percentiles)
    n_p = len(percentiles)
    n_traces = len(csv_tuples)

    # Collect ratio lists
    ratios_per_percentile: list[list[list[float]]] = [[ ] for _ in range(n_p)]
    trace_labels = []
    for trace_idx, (pth, lbl) in enumerate(csv_tuples):
        trace_labels.append(lbl)
        lists = _extract_percentile_ratio_lists(_load_csv(pth), percentiles)
        for p_idx, lst in enumerate(lists):
            ratios_per_percentile[p_idx].append(lst)

    # Build boxplot data & positions
    box_data = []
    positions = []
    xticks = []
    for p_idx in range(n_p):
        base = p_idx * (n_traces + 1)
        xticks.append(base + (n_traces - 1) / 2)
        for t_idx in range(n_traces):
            positions.append(base + t_idx)
            box_data.append(ratios_per_percentile[p_idx][t_idx])

    fig, ax = plt.subplots(figsize=(3 + 1.5 * n_p, 6))
    palette = list(plt.cm.tab10.colors)
    colours = (palette * ((n_traces + 9)//10))[:n_traces]

    bp = ax.boxplot(
        box_data,
        positions=positions,
        patch_artist=True,
        showfliers=False,
        widths=0.6,
        medianprops={"color": "black"},
    )
    for idx, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colours[idx % n_traces])
        patch.set_edgecolor('black')

    ax.set_xticks(xticks)
    ax.set_xticklabels([f"P{p}" for p in percentiles], fontsize=TICK_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_ylabel("Tail Latency / SLO", fontsize=AXIS_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)

    legend_handles = [plt.Line2D([0],[0], color=colours[i], lw=6) for i in range(n_traces)]
    ax.legend(legend_handles, trace_labels, fontsize=LEGEND_FONTSIZE, title="Trace")

    plt.tight_layout()
    out_png = out_dir / f"{out_name}_pXX.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written ➜ {out_png}")

# Backward‑compat wrapper for P95 only

def plot_p95_latency_ratio_multi(csv_tuples: List[Tuple[Path, str]], *, out_dir: Path, out_name: str, title: str = "P95 latency / SLO"):
    plot_latency_ratio_multi(csv_tuples, [95], out_dir=out_dir, out_name=out_name, title=title)

# ──────────────────────────────────────────────────────────────────────────────
# Public plotting APIscsv_tuples: List[Tuple[Path, str]], *, out_dir: Path, out_name: str, title: str = "P95 latency / SLO"):
    metrics = ["min", "mean", "max"]
    vals_per_trace, labels = [], []
    for p, lbl in csv_tuples:
        ratios = _extract_p95_ratio_stats(_load_csv(p))
        vals_per_trace.append(ratios)
        labels.append(lbl)
    vals = np.array(vals_per_trace)
    _plot_grouped(vals, labels, metrics, out_dir / f"{out_name}_p95.png", title=title, ylab="P95 ÷ SLO (ratio)")

# ──────────────────────────────────────────────────────────────────────────────
# Public plotting APIs
# ──────────────────────────────────────────────────────────────────────────────

def plot_token_level_slo(csv_tuples: List[Tuple[Path, str]], *, out_dir: Path, out_name: str, title: str):
    percents, labels = [], []
    for p, lbl in csv_tuples:
        ok, tot = _extract_token_slo_attainment(_load_csv(p))
        percents.append(ok / tot * 100 if tot else 0.0); labels.append(lbl)
    _plot_percent(percents, labels, out_dir / f"{out_name}.png", title=title, ylab="Token SLO attainment (%)")


def plot_request_level_slo(csv_tuples: List[Tuple[Path, str]], *, out_dir: Path, out_name: str, title: str):
    percents, labels = [], []
    for p, lbl in csv_tuples:
        ok, tot = _extract_request_tpot_attainment(_load_csv(p))
        percents.append(ok / tot * 100 if tot else 0.0); labels.append(lbl)
    _plot_percent(percents, labels, out_dir / f"{out_name}_REQ.png", title=title, ylab="Request SLO attainment (%)")


def plot_stats_overview_multi(csv_tuples: List[Tuple[Path, str]], *, out_dir: Path, out_name: str, title: str = "Trace stats"):
    metrics = ["Overall tokens/s"]
    # # , "Prefill tok/s", "Decode tok/s"]
    # metrics = ["Prefill reqs/s", "Decode tok/s"]
    vals_per_trace, labels = [], []

    for p, lbl in csv_tuples:
        df = _load_csv(p)
        total_decode = df["decode_length"].sum()
        total_input = df.get("input_length", pd.Series(0)).sum()
        total_tokens = total_input + total_decode
        wall_time = df["finished_time"].max()
        overall_thr = total_tokens / wall_time if wall_time else 0.0
        prefill_tokens = df.get("input_length", pd.Series(1))
        prefill_thr = (prefill_tokens / df["time_to_first_token"]).mean()
        prefill_total_time = df["time_to_first_token"].sum()
        # --- Solver-time adjustment -----------------------------------------
        solver_total = 0.0
        if "solver_time" in df.columns:
            for st_list in df["solver_time"]:
                if isinstance(st_list, (list, tuple, np.ndarray)):
                    solver_total += sum(t for t in st_list if t != 100)
                else:
                    try:
                        val = float(st_list)
                        if val != 100:
                            solver_total += val
                    except Exception:
                        pass
        print(f"Solver time total: {solver_total}")
        decode_window = max(wall_time - prefill_total_time - solver_total, 1e-9)
        overall_thr = total_tokens / wall_time if wall_time else 0.0
        decode_thr = total_decode / decode_window
        vals_per_trace.append([overall_thr])
        # vals_per_trace.append([prefill_thr, decode_thr])
        labels.append(lbl)

    vals = np.array(vals_per_trace)
    _plot_grouped(vals, labels, metrics, out_dir / f"{out_name}_stats.png", title=title, ylab="Throughput (tokens / second)")

# ──────────────────────────────────────────────────────────────────────────────
# Batch execution block
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path

    FIG_DIR = Path("/home/xinyuema/vllm/outputs/benchmark/TestPPRBased/figures/")
    os.makedirs(FIG_DIR, exist_ok=True)

    def _run_all(traces, tag: str, descr: str):
        plot_token_level_slo(traces, out_dir=FIG_DIR, out_name=f"{tag}-SLO-TBT", title=f"{descr}-Token‑SLO")
        plot_request_level_slo(traces, out_dir=FIG_DIR, out_name=f"{tag}-SLO-TPOT", title=f"{descr}-Request‑SLO")
        plot_stats_overview_multi(traces, out_dir=FIG_DIR, out_name=f"{tag}", title=f"{descr}-Throughput")
        plot_latency_ratio_multi(traces, [90, 95, 99], out_dir=FIG_DIR, out_name=f"{tag}", title=f"{descr}-Tail-TBT")

    # _run_all([
    #     (Path("/home/xinyuema/vllm/outputs/benchmark/TestPPRBased/Flexgen/PPR158_TPI005/outputs.csv"), "FlexGen"),
    #     (Path("/home/xinyuema/vllm/outputs/benchmark/TestPPRBased/Ours/PPR158_TPI005/outputs.csv"), "Ours"),
    # ], tag="Trace1", descr="Trace1-(PPR158_TPI005)")

    # _run_all([
    #     (Path("/home/xinyuema/vllm/outputs/benchmark/TestPPRBased/Flexgen/PPR250_TPI051/outputs.csv"), "FlexGen"),
    #     (Path("/home/xinyuema/vllm/outputs/benchmark/TestPPRBased/Ours/PPR250_TPI051/outputs.csv"), "Ours"),
    # ], tag="Trace2", descr="Trace2-(PPR250_TPI051)")

    # _run_all([
    #     (Path("/home/xinyuema/vllm/outputs/benchmark/TestPPRBased/Flexgen/PPR394_TPI099/outputs.csv"), "FlexGen"),
    #     (Path("/home/xinyuema/vllm/outputs/benchmark/TestPPRBased/Ours/PPR394_TPI099/outputs.csv"), "Ours"),
    # ], tag="Trace3", descr="Trace3-(PPR394_TPI099)")

    FIG_DIR = Path("/home/xinyuema/vllm/outputs/benchmark/TestPPRBased-SLO2_5/figures/")
    os.makedirs(FIG_DIR, exist_ok=True)

    _run_all([
        (Path("/home/xinyuema/vllm/outputs/benchmark/TestPPRBased-SLO2_5/Flexgen/PPR158_TPI005/outputs.csv"), "FlexGen"),
        (Path("/home/xinyuema/vllm/outputs/benchmark/TestPPRBased-SLO2_5/Ours/PPR158_TPI005/outputs.csv"), "Ours"),
        (Path("/home/xinyuema/vllm/outputs/benchmark/Test0521_SLO2_5/DistNSingle/PPR158_TPI005/outputs.csv"), "DistNSingle"),
    ], tag="Trace1", descr="Trace1")

    _run_all([
        (Path("/home/xinyuema/vllm/outputs/benchmark/TestPPRBased-SLO2_5/Flexgen/PPR250_TPI051/outputs.csv"), "FlexGen"),
        (Path("/home/xinyuema/vllm/outputs/benchmark/TestPPRBased-SLO2_5/Ours/PPR250_TPI051/outputs.csv"), "Ours"),
        (Path("/home/xinyuema/vllm/outputs/benchmark/Test0521_SLO2_5/DistNSingle/PPR250_TPI051/outputs.csv"), "DistNSingle"),
    ], tag="Trace2", descr="Trace2")

    _run_all([
        (Path("/home/xinyuema/vllm/outputs/benchmark/TestPPRBased-SLO2_5/Flexgen/PPR394_TPI099/outputs.csv"), "FlexGen"),
        (Path("/home/xinyuema/vllm/outputs/benchmark/TestPPRBased-SLO2_5/Ours/PPR394_TPI099/outputs.csv"), "Ours"),
        (Path("/home/xinyuema/vllm/outputs/benchmark/Test0521_SLO2_5/DistNSingle/PPR394_TPI099/outputs.csv"), "DistNSingle"),
    ], tag="Trace3", descr="Trace3")
