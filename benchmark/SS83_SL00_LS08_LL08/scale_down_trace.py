#!/usr/bin/env python3
# scale_trace.py  usage:  python scale_trace.py orig.json scaled.json
import json, math, pathlib, sys
FACTOR = 10          # 1/10 로 축소

def rnd(x):  # 최소 1, 4자리 반올림
    return max(1, int(round(x / FACTOR)))

def main(src, dst):
    with open(src, encoding="utf-8") as f:
        j = json.load(f)

    # 1) top-level
    j["num_gpu_blocks_override"] = rnd(j["num_gpu_blocks_override"])
    j["peak_batch_blocks"]       = None          # 나중에 다시 계산

    # 2) per-request
    for r in j["requests"].values():
        r["input_length"]  = rnd(r["input_length"])
        r["output_length"] = rnd(r["output_length"])
        r["arrival_time"]  = rnd(r["arrival_time"])
        r["sched_time"]    = rnd(r["sched_time"])
        r["wait_time"]     = rnd(r["wait_time"])

    # 3) recalc peak_batch_blocks
    blk_size = 16
    bs = j["batch_size"]
    totals = sorted((r["input_length"] + r["output_length"]
                     for r in j["requests"].values()), reverse=True)
    top_sum = sum(totals[:bs])
    j["peak_batch_blocks"] = math.ceil(top_sum / blk_size)

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(j, f, indent=2)
    print(f"saved ➜ {dst}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python scale_trace.py <orig.json> <scaled.json>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
