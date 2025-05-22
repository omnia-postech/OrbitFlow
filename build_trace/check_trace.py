from typing import List, Dict, Tuple, Optional, Callable, Any
import json
import math
import numpy as np

from pathlib import Path


BLOCK_SIZE_TOK = 16        # tokens per KV block  (vLLM default)

def _tok_to_blk(toks: int, block_size: int = BLOCK_SIZE_TOK) -> int:
    return math.ceil(toks / block_size)

def memory_pressure(
    trace_path: str,
    *,
    plot_PPR: bool = False,
    plot_curve: bool  = True,   # part-1
    shade_lower: bool = True,   # part-2
    shade_upper: bool = True,   # part-3
    plot_guides: bool = True,   # part-4
    plot_cross1: bool  = True,   # part-5
    plot_cross2: bool  = True,   # part-5
) -> Dict[str, float]:
    """
    Analyse a trace and optionally plot per-step KV-cache pressure.

    Each graphic layer can be toggled via keyword switches documented
    in the function signature above.
    """
    # ---------- 0. Load trace ----------
    with open(trace_path, "r") as f:
        jj = json.load(f)
    gpu_blocks: int = jj["num_gpu_blocks_override"]

    # ---------- 1. Parse requests ----------
    reqs, last_step = [], 0
    for r in jj["requests"].values():
        arr, out, inp = map(int, (r["arrival_time"],
                                  r["output_length"],
                                  r["input_length"]))
        end = arr + out
        last_step = max(last_step, end)
        reqs.append({"arr": arr, "end": end, "inp": inp, "out": out})
    if not reqs:
        raise ValueError("Trace has no requests")

    # ---------- 2. Sweep timeline ----------
    peak_blocks = peak_step = shortfall_sum = ge2_steps = 0
    pressure_curve, over_steps = [], 0
    for s in range(last_step + 1):
        live_blocks = 0
        for r in reqs:
            if r["arr"] <= s <= r["end"]:
                decoded = min(s - r["arr"], r["out"])
                live_blocks += _tok_to_blk(r["inp"] + decoded)
        pressure_curve.append(live_blocks / gpu_blocks)
        if live_blocks > peak_blocks:
            peak_blocks, peak_step = live_blocks, s
        if live_blocks > gpu_blocks:
            shortfall_sum += live_blocks - gpu_blocks
            over_steps += 1
        if live_blocks >= 2 * gpu_blocks:
            ge2_steps += 1

    # ---------- 3. Aggregate stats ----------
    steps_total = last_step + 1
    ppr        = peak_blocks / gpu_blocks
    tpi        = over_steps / steps_total
    ov_frac    = shortfall_sum / (gpu_blocks * steps_total)
    ge2_frac   = ge2_steps / steps_total
    out = {
        "PPR": ppr, "TPI": tpi, "OV_block_step": shortfall_sum,
        "OV_frac": ov_frac, "GE2_frac": ge2_frac,
        "peak_blocks": peak_blocks, "peak_step": peak_step,
        "gpu_blocks": gpu_blocks, "total_steps": steps_total,
    }

    # ---------- 4. Optional plot ----------
    if plot_PPR:
        import matplotlib.pyplot as plt

        x = np.arange(steps_total)
        y = np.asarray(pressure_curve)

        fig, ax = plt.subplots(figsize=(8, 3))

        # —A— overload shading ----------------------------------------
        if shade_lower or shade_upper:
            over = y > 1.0
            segments, in_seg = [], False
            for i, flag in enumerate(over):
                if flag and not in_seg:
                    seg_start, in_seg = i, True
                elif not flag and in_seg:
                    segments.append((seg_start, i)); in_seg = False
            if in_seg:
                segments.append((seg_start, steps_total))

            for s, e in segments:
                xs, ys = x[s:e], y[s:e]
                if shade_lower:
                    ax.fill_between(xs, 0.0, 1.0, color="C0", alpha=0.25)
                if shade_upper:
                    ax.fill_between(xs, 1.0, ys, color="C1", alpha=0.25)

        # —B— main curve & capacity line ------------------------------
        if plot_curve:
            ax.plot(x, y, linewidth=0.8)
            ax.axhline(1.0, linestyle="--", linewidth=0.8, label="capacity")

        # —C— extra horizontal guides ---------------------------------
        valid_offload_layers = {16, 10, 8, 6, 5, 4, 3, 2, 1, 0}
        total_layers = 32
        lines = [total_layers / (total_layers - ol) for ol in valid_offload_layers]
        if plot_guides:
            for thr in lines:
                ax.axhline(thr, linestyle="--", linewidth=0.8)

        # —— crossings ------------------------------------------------
        if plot_cross1 or plot_cross2:
            thresholds = [1.0] + lines
            type1_pts, type2_pts = [], []
            for i in range(1, steps_total):
                prev, cur = y[i-1], y[i]
                crossed_thr = [thr for thr in thresholds
                               if (prev-thr)*(cur-thr) < 0
                               or (prev == thr) ^ (cur == thr)]
                if len(crossed_thr) == 1:
                    type1_pts.append((i, crossed_thr[0]))
                elif len(crossed_thr) > 1:
                    type2_pts.append((i, max(crossed_thr)))

            out["type1_count"] = len(type1_pts)
            out["type2_count"] = len(type2_pts)

            if plot_cross1 and type1_pts:            # draw Type-1 only if enabled
                ax.scatter(*zip(*type1_pts),
                           marker="o", s=20, label="Type 1")

            if plot_cross2 and type2_pts:            # draw Type-2 only if enabled
                ax.scatter(*zip(*type2_pts),
                           marker="s", s=30, label="Type 2")

        # —E— cosmetics & save ----------------------------------------
        ax.set_xlabel("decode step")
        ax.set_ylabel("live_blocks / gpu_blocks")
        ax.set_title(f"MPR={ppr:.2f}  OTF={tpi:.2f}  OVF={ov_frac:.2f},TJ={len(type1_pts)}, BJ={len(type2_pts)}")
        ax.legend(loc="upper right")
        plt.tight_layout()

        png_path = Path(trace_path).with_suffix(".png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        out["plot_path"] = str(png_path)

    return out

if __name__ == "__main__":
    import glob

    base_dir = "/home/sychoy/vllm/trace_pool/static_8k_pressure/bimodal/"
    trace_paths = glob.glob(f"{base_dir}/trace_*.json")

    for path in trace_paths:  # 정렬은 선택 사항
        metrics = memory_pressure(
            trace_path=path,
            plot_PPR=True
        )
        print(path)
        print(metrics)
    

    base_dir = "/home/sychoy/vllm/trace_pool/static_8k_pressure/uniform/"
    trace_paths = glob.glob(f"{base_dir}/trace_*.json")

    for path in sorted(trace_paths):  # 정렬은 선택 사항
        metrics = memory_pressure(
            trace_path=path,
            plot_PPR=True
        )
        print(path)
        print(metrics)