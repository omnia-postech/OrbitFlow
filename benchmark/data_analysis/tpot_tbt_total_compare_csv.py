import pandas as pd, numpy as np, ast
from pathlib import Path

trace_list  = ["both_static", "batch_dyn", "token_dyn"]
metric_list = ["low", "mid", "high", "veryhigh"]
slo_scales  = [5.5, 4.5, 3.5, 2.5, 1.5, 1, 0.5]
method_list = ["NoPrefetch", "Flexgen", "SelectN", "DistNSingle", "Ours"]

def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    num_cols = ["arrival_time","first_scheduled_time","finished_time",
                "time_to_first_token","slo_threshold","slo_violations",
                "stall_duration","decode_length","end_to_end_time",
                "decode_time","time_per_output_token"]
    for col in num_cols:
        if col in df: df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("time_between_tokens","stall_times","stall_durations"):
        if col in df: df[col] = df[col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x,str) else x)
    return df

def _extract_request_tpot_attainment(df: pd.DataFrame) -> float:
    tpot = np.array([np.mean(tbt) for tbt in df["time_between_tokens"]])
    thr = pd.to_numeric(df["slo_threshold"], errors="coerce").to_numpy().mean()
    valid = ~np.isnan(tpot)
    attain = (tpot[valid] <= thr).sum()
    return (attain / valid.sum() * 100 if valid.sum() else 0.0), attain, valid.sum(), thr

def _extract_token_slo_attainment(df: pd.DataFrame) -> float:
    total_decoded = int(df.get("decode_length", pd.Series(0)).sum())
    viol = int(df.get("slo_violations", pd.Series(0)).sum())
    return ((total_decoded - viol) / total_decoded * 100 if total_decoded else 0.0), (total_decoded - viol), total_decoded

results = []

for metric in metric_list:
    for trace in trace_list:
        for sc in slo_scales:
            for method in method_list:
                path = Path(f"/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo{sc}/{method}/{trace}_{metric}/outputs.csv")
                try:
                    df = load_metrics(path)
                    tpot, attain, valid, thr= _extract_request_tpot_attainment(df) 
                    tbt, son, mom  = _extract_token_slo_attainment(df)
                    results.append({
                        "metric": metric,
                        "trace": trace,
                        "slo_scale": sc,
                        "method": method,
                        "thr": thr,
                        "TPOT_attain": tpot,
                        "attain": attain,
                        "valid": valid,
                        "TBT_attain": tbt,
                        "tbt_attain": son,
                        "tbt_valid": mom,
                    })
                except Exception as e:
                    error_value = -1 if isinstance(e, FileNotFoundError) else -2
                    results.append({
                        "slo_scale": sc,
                        "trace": trace,
                        "method": method,
                        "metric": metric,
                        "thr": error_value,
                        "TPOT_attain": error_value,
                        "attain": error_value,
                        "valid": error_value,
                        "TBT_attain": error_value,
                        "tbt_attain": error_value,
                        "tbt_valid": error_value,
                    })


df_result = pd.DataFrame(results)
df_result.to_csv("aggregated_slo_metrics.csv", index=False)
