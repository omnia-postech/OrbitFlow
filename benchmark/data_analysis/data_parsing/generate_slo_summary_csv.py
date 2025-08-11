import pandas as pd
import numpy as np
import ast
from pathlib import Path
from typing import List
import sys
import json
import re
import argparse

# ───────────────────────────────────────────────
def load_output_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    num_cols = [
        "arrival_time", "first_scheduled_time", "finished_time",
        "time_to_first_token", "slo_threshold", "slo_violations",
        "stall_duration", "decode_length", "end_to_end_time",
        "decode_time", "time_per_output_token"
    ]
    for col in num_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("time_between_tokens", "stall_times", "stall_durations"):
        if col in df:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

# ───────────────────────────────────────────────
def _extract_request_tpot_attainment(output_df, slo_df, total_decode, penalty_tbt=1000.0):
    failed_reqs = set(slo_df.loc[slo_df.get("failed", False), "request_id"]) if {"failed", "request_id"}.issubset(slo_df.columns) else set()

    tpot_list = []
    req_ids = slo_df["request_id"] if "request_id" in slo_df.columns else [f"request_{i}" for i in range(len(slo_df))]
    for rid in req_ids:
        if rid in failed_reqs:
            tpot_list.append(penalty_tbt)
        else:
            vals = output_df.loc[output_df["request_id"] == rid, "time_between_tokens"]
            tpot_list.append(np.mean(vals.iloc[0]) if len(vals) else penalty_tbt)

    tpot = np.asarray(tpot_list, dtype=float)
    thr = pd.to_numeric(output_df["slo_threshold"], errors="coerce").to_numpy().mean()
    valid_mask = ~np.isnan(tpot)
    attain_cnt = (tpot[valid_mask] <= thr).sum()
    return (attain_cnt / valid_mask.sum() * 100) if valid_mask.sum() else 0.0

def _extract_token_slo_attainment_with_TD(output_df, slo_df, total_decoded):
    viol = int(slo_df.get("slo_violation_with_TD", pd.Series(0)).sum())
    return ((total_decoded - viol) / total_decoded * 100) if total_decoded else 0.0

def _extract_token_slo_attainment_no_TD(output_df, slo_df, total_decoded):
    viol = int(slo_df.get("slo_violation_no_TD", pd.Series(0)).sum())
    return ((total_decoded - viol) / total_decoded * 100) if total_decoded else 0.0

def compute_throughput(df):
    total_input = df.get("input_length", pd.Series(0)).sum()
    total_decode = df["decode_length"].sum()
    total_tokens = total_input + total_decode
    wall_time = df["finished_time"].max()
    return total_tokens / wall_time if wall_time and total_tokens else 0.0

def _extract_percentile_ratio_lists(df, percentiles: List[int], total_decode: int):
    lists: List[List[float]] = [[] for _ in percentiles]
    if {"time_between_tokens", "slo_threshold", "expected_output_length"}.issubset(df.columns):
        thr_vals = pd.to_numeric(df["slo_threshold"], errors="coerce").to_numpy()
        flat = []
        for tbt_list, exp_len in zip(df["time_between_tokens"], df["expected_output_length"]):
            if not tbt_list or exp_len is None:
                continue
            flat.extend(tbt_list)
        sorted_flat = sorted(flat)
        thr = thr_vals[0]
        for idx, pct in enumerate(percentiles):
            pos = int(np.floor(pct / 100 * total_decode))
            px = sorted_flat[pos] if 0 <= pos < len(sorted_flat) else np.nan
            lists[idx] = [px / thr if px != 100 else np.nan] if px and not np.isnan(px) else [np.nan]
    return lists

def get_req_per_sec(output_df):
    last_arrival_time = output_df["arrival_time"].iloc[-1]
    return last_arrival_time

def extract_metrics(input_base_path, total_decode):
    output_df = load_output_metrics(Path(input_base_path, "outputs.csv"))
    slo_df = pd.read_csv(Path(input_base_path, "slo_violationv2.csv"))

    tpot_slo = _extract_request_tpot_attainment(output_df, slo_df, total_decode)
    tbt_slo_with_TD = _extract_token_slo_attainment_with_TD(output_df, slo_df, total_decode)
    tbt_slo_no_TD = _extract_token_slo_attainment_no_TD(output_df, slo_df, total_decode)
    throughput = compute_throughput(output_df)
    slo_thr = pd.to_numeric(output_df["slo_threshold"], errors="coerce").to_numpy().mean()
    pct_lists = _extract_percentile_ratio_lists(output_df, [90, 95, 99], total_decode)
    p90_ratio, p95_ratio, p99_ratio = [np.mean(lst) if lst else 0.0 for lst in pct_lists]
    last_arrival_time = get_req_per_sec(output_df)

    return tpot_slo, tbt_slo_with_TD, tbt_slo_no_TD, throughput, slo_thr, p90_ratio, p95_ratio, p99_ratio, last_arrival_time

# ───────────────────────────────────────────────
def make_summerize(input_paths: list, reference_root: Path):
    pattern = re.compile(r"^.*lambda(?P<rate>\d+(?:\.\d+)?)x_cv(?P<cv_num>\d+)$")
    results = []
    for path in input_paths:
        slo = path.parts[-3].replace('slo', '')
        method = path.parts[-2]

        name = path.name
        rate, cv_num = None, None
        match = pattern.match(name)
        if match:
            rate = float(match.group("rate"))
            cv_num = int(match.group("cv_num"))

        ref_json = reference_root / f"{name}.json"
        if not ref_json.exists():
            print(f"⚠️ reference JSON missing: {ref_json}")
            continue

        with open(ref_json, "r") as f:
            reference = json.load(f)
        ref_reqs = reference.get("requests", {})

        total_decode = sum(req["output_length"] for req in ref_reqs.values())

        metrics = extract_metrics(path, total_decode)
        tpot_slo, tbt_slo_with_TD, tbt_slo_no_TD, throughput, slo_thr, p90_ratio, p95_ratio, p99_ratio, last_arrival_time = metrics

        results.append({
            'slo': slo,
            'method': method,
            'arrival_rate': rate,
            'cv_num': cv_num,
            'name': name,
            'tpot_attainment': tpot_slo,
            'tbt_attainment_with_TD': tbt_slo_with_TD,
            'tbt_attainment_no_TD': tbt_slo_no_TD,
            'throughput_tokens_per_sec': throughput,
            'slo_threshold_mean': slo_thr,
            'p90_ratio': p90_ratio,
            'p95_ratio': p95_ratio,
            'p99_ratio': p99_ratio,
            "last_arrival_time": last_arrival_time,
            "req_per_sec": len(ref_reqs) / last_arrival_time
        })

    if results:
        df_result = pd.DataFrame(results)
        output_csv = input_paths[0].parent / "arrival_summerizev2.csv"
        df_result.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"✅ Aggregated CSV saved to: {output_csv}\n")

# ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize benchmark outputs into a single CSV.")
    parser.add_argument("input_base_paths", nargs="+", type=Path, help="List of method directories (e.g., .../slo1/Ours .../slo2/Flexgen)")
    parser.add_argument("--reference-root", type=Path, required=True, help="Path to reference trace directory (JSON)")
    args = parser.parse_args()

    for method_dir in args.input_base_paths:
        input_paths = []
        for trace_dir in sorted(method_dir.iterdir(), key=lambda p: p.name):
            if not trace_dir.is_dir():
                continue
            outputs_csv = trace_dir / 'outputs.csv'
            slo_violation = trace_dir / 'slo_violationv2.csv'
            if outputs_csv.exists() and slo_violation.exists():
                input_paths.append(trace_dir)

        if input_paths:
            make_summerize(input_paths, args.reference_root)
