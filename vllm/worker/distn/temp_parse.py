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

LOG_PATH          = Path("/home/xinyuema/vllm/vllm/worker/distn/token_dyn_veryhigh_outputs.log")          # input
UNIFORM_OUT_PATH  = Path("uniform_blocks.log")      # outputs
NONUNIFORM_OUT_PATH = Path("non_uniform_blocks.log")

# ---------------------------------------------------------------------
# 1. Pull out every block that begins with the marker line -------------
# ---------------------------------------------------------------------
block_start = re.compile(r"^--- Optimal solution \(or best found\) ---")

blocks = []
current = []

with LOG_PATH.open() as f:
    in_block = False
    for line in f:
        if block_start.match(line):
            # starting a new block
            if current:
                blocks.append(current)
            current = [line.rstrip("\n")]
            in_block = True
            continue

        # End of block is either a blank-only line or end of file
        if in_block and line.strip() == "":
            blocks.append(current)
            current = []
            in_block = False
            continue

        if in_block:
            current.append(line.rstrip("\n"))

# push the very last block if file ended without a trailing blank line
if current:
    blocks.append(current)

print(f"Total blocks found: {len(blocks)}")

# ---------------------------------------------------------------------
# 2. Keep only blocks with ≥ 2 data rows -------------------------------
# ---------------------------------------------------------------------
data_row = re.compile(r"^\s*\d+\s*\|")   # row begins with an integer “id | …”

multirow_blocks = [b for b in blocks if sum(1 for ln in b if data_row.match(ln)) > 1]
print(f"Blocks with ≥ 2 data rows: {len(multirow_blocks)}")

# ---------------------------------------------------------------------
# 3. Split into uniform vs non-uniform offload -------------------------
# ---------------------------------------------------------------------
uniform_blocks = []
nonuniform_blocks = []

for blk in multirow_blocks:
    offload_vals = [
        int(m.group(1))
        for ln in blk
        if (m := re.match(r"^\s*\d+\s*\|\s*(\d+)\s*\|", ln))
    ]
    if len(set(offload_vals)) == 1:          # uniform
        uniform_blocks.append("\n".join(blk))
    else:
        nonuniform_blocks.append("\n".join(blk))

print(f"  • uniform-offload   blocks: {len(uniform_blocks)}")
print(f"  • non-uniform-offload blocks: {len(nonuniform_blocks)}")

# ---------------------------------------------------------------------
# 4. Write log files ---------------------------------------------------
# ---------------------------------------------------------------------
UNIFORM_OUT_PATH.write_text("\n\n".join(uniform_blocks) + "\n")
NONUNIFORM_OUT_PATH.write_text("\n\n".join(nonuniform_blocks) + "\n")

print(f"Wrote {UNIFORM_OUT_PATH}  and  {NONUNIFORM_OUT_PATH}")
