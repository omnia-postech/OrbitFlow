#!/usr/bin/env python3
# batch_slo_violation.py

import numpy as np
import pandas as pd
import ast, json
from pathlib import Path
import argparse

np.set_printoptions(threshold=np.inf)

Fix = True
to_see = 10000

# ───────────────────────────────────────────────
def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    num_cols = ["arrival_time", "first_scheduled_time", "finished_time",
                "time_to_first_token", "slo_threshold", "slo_violations",
                "stall_duration", "decode_length", "end_to_end_time",
                "decode_time", "time_per_output_token"]
    for c in num_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("time_between_tokens", "stall_times", "stall_durations", "solver_time"):
        if c in df:
            df[c] = df[c].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

def compute_slo_violation(req_id, times, solver, slo_thr):
    times_np = np.asarray(times, dtype=float)
    solver_np = np.asarray(solver, dtype=float)
    real_tbt = times_np - solver_np

    valid = real_tbt >= 0
    exceptions = int((~valid).sum())
    valid_tbt = times_np[valid]
    if valid_tbt.size == 0:
        return 0, exceptions

    deposit = 0
    prev_out = 0
    prev_gen = 0
    violations = 0

    for tbt in valid_tbt:
        if prev_gen + tbt < prev_out + slo_thr:
            deposit += 1
            prev_gen += tbt
        elif prev_gen + tbt == prev_out + slo_thr:
            prev_gen += tbt
            prev_out += slo_thr
        else:
            while deposit > 0 and prev_out + slo_thr < prev_gen + tbt:
                deposit -= 1
                prev_out += slo_thr
            if prev_out + slo_thr < prev_gen + tbt:
                violations += 1
                prev_gen += tbt
                prev_out = prev_gen
            else:
                deposit += 1
                prev_gen += tbt
    return violations, exceptions

# ───────────────────────────────────────────────
def process_experiment(exp_dir: Path, reference_root: Path):
    sim_path = exp_dir / "outputs.csv"
    if not sim_path.exists():
        print(f"outputs.csv not found: {exp_dir}")
        return

    ref_json = reference_root / f"{exp_dir.name}.json"
    if not ref_json.exists():
        print(f"reference JSON missing: {ref_json}")
        return

    with open(ref_json, "r") as f:
        reference = json.load(f)
    ref_reqs = reference.get("requests", {})

    df = load_metrics(sim_path)
    slo_thr = float(df["slo_threshold"].mean())
    results = []

    total_req_len = len(ref_reqs)
    if Fix:
        total_req_len = max(int(str(rid).split("_")[-1]) for rid in df["request_id"].unique()) + 1
        print(f"Fixing total_req_len: (from {len(ref_reqs)})")

    for id in range(total_req_len):
        req_id = f"request_{id}"
        ref_len = ref_reqs.get(req_id, {}).get("output_length", 0)
        try:
            req_row = df[df["request_id"] == req_id].iloc[0]
        except:
            results.append({
                "request_id": req_id,
                "slo_violation_with_TD": ref_len - 1,
                "slo_violation_no_TD": ref_len - 1,
                "reference_len": ref_len - 1,
                "sys_violation": None,
                "decode_length": None,
                "exceptions": True,
                "failed": True,
            })
            continue

        vio, exc = compute_slo_violation(
            id,
            req_row["time_between_tokens"],
            req_row["solver_time"],
            slo_thr
        )

        dl = req_row["decode_length"]
        vio += max(ref_len - dl - 1, 0)
        failed = (ref_len - 1 != dl)

        no_td = [tbt for tbt in req_row["time_between_tokens"] if tbt > slo_thr]

        results.append({
            "request_id": req_id,
            "slo_violation_with_TD": vio,
            "slo_violation_no_TD": len(no_td),
            "reference_len": ref_len - 1,
            "sys_violation": req_row["slo_violations"],
            "decode_length": len(req_row["time_between_tokens"]),
            "exceptions": exc,
            "failed": failed,
        })

    out_csv = exp_dir / "slo_violationv2.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✔  {exp_dir} → slo_violationv2.csv (SLO={slo_thr:.3f})")

# ───────────────────────────────────────────────
def to_exp_path(arg: str) -> Path:
    return Path(arg).expanduser()

def main():
    parser = argparse.ArgumentParser(
        description="Compute SLO violations for experiment directories.")
    parser.add_argument("exp_dirs", nargs="*",
        help="Paths to experiment directories containing outputs.csv")
    parser.add_argument("--missing", action="store_true",
        help="Only process directories that lack slo_violationv2.csv")
    parser.add_argument("--reference-root", type=Path, required=True,
        help="Path to reference JSON trace directory")
    args = parser.parse_args()

    if args.exp_dirs:
        for arg in args.exp_dirs:
            exp_path = to_exp_path(arg)
            out_csv = exp_path / "slo_violationv2.csv"
            if args.missing and out_csv.exists():
                print(f"Skipping (already exists): {out_csv}")
                continue
            process_experiment(exp_path, args.reference_root)
    else:
        print("❗ No experiment directories provided. Please specify --reference-root and paths to experiments.")

if __name__ == "__main__":
    main()
