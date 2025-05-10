#!/usr/bin/env python

"""
compare_metrics.py
==================
Compares multiple vLLM-style inference logs stored as CSV files and their corresponding log files.

Usage:
    python compare_metrics.py --out output.png --trace "Experiment A" log1.csv log2.csv ...
"""

import ast
import sys
from pathlib import Path
from typing import List

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def load_metrics(csv_path: str | Path) -> pd.DataFrame:
    """
    Read the log CSV and coerce list-encoded columns to Python lists.
    """
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


def load_preemption_time(log_path: Path) -> float:
    """
    Parse the log file to extract the 'Preemptions time' value.
    """
    if not log_path.exists():
        print(f"[Warning] Log file not found: {log_path}")
        return 0.0

    with log_path.open("r") as f:
        for line in f:
            if "Preemptions" in line and "time" in line:
                try:
                    return float(line.strip().split(":")[-1].strip().split()[0])
                except Exception as e:
                    print(f"[Warning] Failed to parse Preemptions time in {log_path}: {e}")
                    return 0.0
    print(f"[Warning] No Preemptions time found in {log_path}")
    return 0.0


def plot_stats_comparison(output_path: Path, csv_paths: List[Path], title: str) -> Path:
    """
    Generate 5 bar charts comparing:
    - Mean E2E latency
    - Mean prefill throughput
    - Mean output throughput
    - Total SLO violations
    - Preemptions time (from .log files)
    """
    labels = []
    mean_e2e_list = []
    prefill_thr_list = []
    output_thr_list = []
    slo_viol_list = []
    preempt_time_list = []

    for path in csv_paths:
        df = load_metrics(path)
        method_name = path.parts[-3]  # e.g., "Static2"
        labels.append(method_name)

        mean_e2e = df["end_to_end_time"].mean()
        prefill_tokens = df.get("input_length", pd.Series(1, index=df.index))
        prefill_thr = (prefill_tokens / df["time_to_first_token"]).mean()
        output_thr = (df["decode_length"] / df["decode_time"]).mean()
        slo_viol = df["slo_violations"].sum()

        # 대응하는 .log 파일 찾기
        log_path = path.with_suffix(".log")
        preempt_time = load_preemption_time(log_path)

        mean_e2e_list.append(mean_e2e)
        prefill_thr_list.append(prefill_thr)
        output_thr_list.append(output_thr)
        slo_viol_list.append(slo_viol)
        preempt_time_list.append(preempt_time)

    fig, axs = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(title, fontsize=16)

    metrics = [
        ("Mean E2E latency (s)", mean_e2e_list),
        ("Mean prefill throughput (tok/s)", prefill_thr_list),
        ("Mean output throughput (tok/s)", output_thr_list),
        ("Total SLO violations", slo_viol_list),
        ("Preemptions time (s)", preempt_time_list),
    ]

    for ax, (metric_title, values) in zip(axs.flat, metrics):
        ax.bar(labels, values)
        ax.set_title(metric_title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:,.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave space for suptitle
    fig.delaxes(axs[2, 1])  # 제거할 여분의 subplot
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare vLLM metrics across multiple CSVs.")
    parser.add_argument("--out", required=True, help="Path to save the output comparison image.")
    parser.add_argument("--trace", required=True, help="Title to show on the comparison figure.")
    parser.add_argument("csvs", nargs="+", help="CSV file paths to compare.")

    args = parser.parse_args(argv)

    out_path = Path(args.out).expanduser().resolve()
    title = args.trace
    csv_paths = [Path(p).expanduser().resolve() for p in args.csvs]

    for p in csv_paths:
        if not p.exists():
            sys.exit(f"CSV not found: {p}")

    out_path = plot_stats_comparison(out_path, csv_paths, title)
    print(f"Comparison figure written ➜ {out_path}")


if __name__ == "__main__":
    main()
