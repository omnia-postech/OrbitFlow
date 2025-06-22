#!/usr/bin/env python3
"""
Parse a log produced by the solver and

1. count all “Optimal solution (or best found)” blocks
2. keep only blocks that contain ≥ 2 data rows
3. split those blocks into
      – uniform-offload blocks   (every data row has the same offload value)
      – non-uniform-offload ones
4. write the two groups to `uniform_blocks.log` and `non_uniform_blocks.log`
"""

import re
from pathlib import Path
from statistics import median  
# LOG_PATH          = Path("/home/xinyuema/vllm/vllm/worker/distn/token_dyn_mid_outputs.log")          # input
# LOG_PATH          = Path("/home/xinyuema/vllm/outputs/benchmark/Test_0618_New_solver_latModelModified/Ours/both_dyn_low/outputs.log")          # input
LOG_PATH          = Path("/home/xinyuema/vllm/outputs/benchmark/Test_0618_New_solver_latModelModified/Ours/token_dyn_low/outputs.log")          # input
# LOG_PATH          = Path("/home/xinyuema/vllm/outputs/benchmark/Test_0618_New_solver/Ours/both_dyn_low/outputs.log")          # input
# LOG_PATH          = Path("/home/xinyuema/vllm/vllm/worker/distn/both_dyn_low_outputs.log")          # input
UNIFORM_OUT_PATH  = Path("uniform_blocks.log")      # outputs
NONUNIFORM_OUT_PATH = Path("non_uniform_blocks.log")
COMBINED_OUT_PATH = Path("combined_blocks.log")  
SINGLE_OUT_PATH = Path("single_row_blocks.log")  # blocks with only one data row

# ---------------------------------------------------------------------
# helpers --------------------------------------------------------------
# ---------------------------------------------------------------------
data_row      = re.compile(r"^\s*\d+\s*\|")        # an id | … table row
decode_window = re.compile(r"decode_window size\s*:\s*([\d.]+)")

def extract_decode_window(block: list[str]) -> float | None:
    """Return the decode-window size in this block, or None if not found."""
    for ln in block:
        if (m := decode_window.search(ln)):
            return float(m.group(1))
    return None

# ---------------------------------------------------------------------
# 1. Pull out every block that begins with the marker line -------------
# ---------------------------------------------------------------------
block_start   = re.compile(r"^--- Optimal solution \(or best found\) ---")
decode_window = re.compile(r"decode_window size\s*:\s*([\d.]+)")

blocks, current = [], []
in_block = False
with LOG_PATH.open() as f:
    for line in f:
        if block_start.match(line):
            # we’re starting a new block
            if current:               # save the previous one (shouldn’t happen
                blocks.append(current)  # unless decode_window was missing)
            current = [line.rstrip("\n")]
            in_block = True
            continue

        if in_block:
            current.append(line.rstrip("\n"))

            # ---- end the block right after the decode-window line ----
            if decode_window.search(line):
                blocks.append(current)
                current, in_block = [], False   # reset for next block
# push the very last block if file ended without a trailing blank line
if current:
    blocks.append(current)
all_blocks = []
for blk in blocks: 
    all_blocks.append("\n".join(blk))
print(f"Total blocks found: {len(blocks)}")

# ---------------------------------------------------------------------
# 2. keep only blocks with ≥ 2 data rows (UNCHANGED) ------------------
# ---------------------------------------------------------------------
multirow_blocks  = []
single_row_blocks = []

for blk in blocks:
    if sum(1 for ln in blk if data_row.match(ln)) > 1:
        multirow_blocks.append(blk)
    else:
        single_row_blocks.append(blk)
print(f"Blocks with ≥ 2 data rows: {len(multirow_blocks)}")

# ---------------------------------------------------------------------
# 3. classify + capture decode_window  (patched) ----------------------
# ---------------------------------------------------------------------
uniform_blocks, nonuniform_blocks = [], []
uniform_windows, nonuniform_windows = [], []
single_blocks = []
for blk in single_row_blocks:
    single_blocks.append("\n".join(blk))
    
for blk in multirow_blocks:
    # grab every “offload” value in the table
    offload_vals = [
        int(m.group(1))
        for ln in blk
        if (m := re.match(r"^\s*\d+\s*\|\s*(\d+)\s*\|", ln))
    ]
    dw = extract_decode_window(blk)             # may be None

    # --------- uniform vs non-uniform split ----------
    target_blocks   = uniform_blocks   if len(set(offload_vals)) == 1 else nonuniform_blocks
    target_windows  = uniform_windows  if len(set(offload_vals)) == 1 else nonuniform_windows

    target_blocks.append("\n".join(blk))
    if dw is not None:                          # <- filter out missing values
        target_windows.append(dw)

# ---------------------------------------------------------------------
# stats printout (patched) --------------------------------------------
# ---------------------------------------------------------------------
def print_stats(label, blocks, windows):
    print(f"  • {label:21} blocks: {len(blocks)}")
    if windows:
        print(f"       median decode_window: {median(windows):.1f}")
        print(f"       max decode_window: {max(windows):.1f}")
    else:
        print(f"       median decode_window: n/a")

print_stats("uniform-offload",   uniform_blocks,   uniform_windows)
print_stats("non-uniform-offload", nonuniform_blocks, nonuniform_windows)

# ---------------------------------------------------------------------
# 4. Write log files ---------------------------------------------------
# ---------------------------------------------------------------------
UNIFORM_OUT_PATH.write_text("\n\n".join(uniform_blocks) + "\n")
NONUNIFORM_OUT_PATH.write_text("\n\n".join(nonuniform_blocks) + "\n")
COMBINED_OUT_PATH.write_text("\n\n".join(all_blocks) + "\n")
SINGLE_OUT_PATH.write_text("\n\n".join(single_blocks) + "\n")

print(f"Wrote {UNIFORM_OUT_PATH}  and  {NONUNIFORM_OUT_PATH} and {COMBINED_OUT_PATH} and {SINGLE_OUT_PATH}")
