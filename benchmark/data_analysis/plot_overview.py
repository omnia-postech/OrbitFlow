
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==== Argument Parsing ====
parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str, required=True, help='Root directory containing data subdirectories.')
parser.add_argument('--x-labels', nargs='+', required=True, help='Names of each x category (subdirectory names).')
parser.add_argument('--trace', type=str, required=True, help='trace name')
parser.add_argument('--metric', type=str, required=True, help='Column name in CSV to average and plot.')
parser.add_argument('--xlabel', type=str, required=True, help='X-axis label')
parser.add_argument('--ylabel', type=str, required=True, help='Y-axis label')
parser.add_argument('--title', type=str, required=True, help='Plot title')
parser.add_argument('--output', type=str, default='graph/bar_plot.jpg', help='Output path for the plot')
args = parser.parse_args()

# ==== 스타일 설정 ====
style = {
    "bar": {
        "edgecolor": "white",
        "linewidth": 2.5
    },
    "tick": {
        "fontsize": 18,
        "rotation": 45
    },
    "label": {
        "fontsize": 24,
        "weight": "bold",
    },
    "title": {
        "fontsize": 24,
        "weight": "bold",
        "pad": 10
    },
    "text": {
        "fontsize": 14,
        "weight": "bold"
    },
    "spine": {
        "color": "gray",
        "alpha": 0.5,
        "linewidth": 2
    },
    "grid": {
        "color": "gray",
        "linestyle": "-",
        "linewidth": 2,
        "alpha": 0.5
    }
}

import re
import argparse

def parse_log_tail(log_text: str):
    stats = {}
    patterns = {
        "overall_runtime": r"Overall runtime\s*:\s*([\d.]+) s",
        "prefill_tokens": r"Prefill time\s*:\s*(\d+)\s+tokens",
        "prefill_time": r"Prefill time\s*:\s*\d+\s+tokens over ([\d.]+) s",
        "prefill_throughput": r"Prefill time.*\(([\d.]+) t/s\)",
        "decode_tokens": r"Decode\s+time\s*:\s*(\d+)\s+tokens",
        "decode_time": r"Decode\s+time\s*:\s*\d+\s+tokens over ([\d.]+) s",
        "decode_throughput": r"Decode\s+time.*\(([\d.]+) t/s\)",
        "preemptions_time": r"Preemptions\s+time\s*:\s*([\d.]+)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, log_text)
        if match:
            value = float(match.group(1)) if "." in match.group(1) else int(match.group(1))
            stats[key] = value

    return stats
    
def get_result(metric, log_dict, df):
    if metric == "e2e_throughput":
        return log_dict["decode_tokens"] / float(log_dict["overall_runtime"])
    elif metric == "slo_attainment_tbt": 
        return (df["decode_length"].sum() - df["slo_violations"].sum()) / df["decode_length"].sum()
    elif metric == "slo_attainment_tpot": 
        return ((df["time_per_output_token"] - df["slo_threshold"]) < 0 ).mean()
    elif metric == "slo_99":
        result = []
        for idx,row in df.iterrows():
            num_list = eval(row["time_between_tokens"])
            row["tbt_sorted"] = sorted(num_list, reverse=True)
            i99 = int(len(row) * 0.99)
        df["99p"] = result
            result.append(row["tbt_sorted"][i99])
        return (df["99p"]).mean()
    elif metric == "slo_95":
        for idx,row in df.iterrows():
            row["tbt_sorted"] = sorted(row["time_between_tokens"], ascending=True)
        df["95p"][idx] = df["tbt_sorted"].iloc[int(len(df) * 0.95)]
        return (df["95p"]).mean()
    elif metric == "slo_90":
        for idx,row in df.iterrows():
            row["tbt_sorted"] = row.sorted(row["tbt_sorted"], ascending=True)
        df["90p"][idx] = df["tbt_sorted"].iloc[int(len(df) * 0.90)]
        return (df["90p"]).mean()
    elif metric in df.columns:
        return df[metric].mean()
    else:
        print(f"[WARNING] Column '{args.metric}' not found in {csv_path}")
        return np.nan

# ==== 데이터 로딩 ====
x_labels = args.x_labels

averages = []
for label in x_labels:
    csv_path = os.path.join(args.input_dir, label + "/" + args.trace, "outputs.csv")
    log_path = os.path.join(args.input_dir, label + "/" + args.trace, "outputs.log")
    log_dict = dict()

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log_text = f.read()
        log_dict = parse_log_tail(log_text)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        averages.append(get_result(args.metric, log_dict, df))
    else:
        print(f"[WARNING] File not found: {csv_path}")
        averages.append(np.nan)

    
    
    

# ==== 바 차트 그리기 ====
x = np.arange(len(x_labels))
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x, averages, color="#82c6a5", **style["bar"])

for i, val in enumerate(averages):
    if not np.isnan(val):
        ax.text(x[i], val * 1.01, f"{val:.1f}", ha="center", va="bottom", **style["text"])

# ==== draw slo_threshold line if metric is slo_9* ====
if args.metric.startswith("slo_9"):
    # Load the first CSV to get the threshold
    first_csv = os.path.join(
        args.input_dir, x_labels[0], args.trace, "outputs.csv"
    )
    if os.path.exists(first_csv):
        df0 = pd.read_csv(first_csv)
        if "slo_threshold" in df0.columns:
            slo_thresh = df0["slo_threshold"].iloc[0]
            # ensure the line is within the y‐range
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(min(ymin, slo_thresh*0.9), max(ymax, slo_thresh*1.1))
            ax.axhline(
                slo_thresh,
                color="red",
                linestyle="--",
                linewidth=2,
                label="SLO Threshold"
            )
            ax.legend(fontsize=style["text"]["fontsize"])

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=style["tick"]["fontsize"], rotation=45)
ax.set_xlabel(args.xlabel, **style["label"])
ax.set_ylabel(args.ylabel, **style["label"])
ax.set_title(args.title, **style["title"])
ax.yaxis.grid(True, **style["grid"])
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

ax.tick_params(axis='x', which='both', length=0)
ax.tick_params(axis='y', which='both', length=0)

os.makedirs(os.path.dirname(args.output), exist_ok=True)


plt.savefig(args.output, format='pdf', bbox_inches="tight")
