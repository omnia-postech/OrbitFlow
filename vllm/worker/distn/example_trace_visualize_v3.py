import sys, re, collections, itertools, math
import pandas as pd, matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker
from collections import defaultdict
import numpy as np

# ──────────────────────────────
# Constants & Style Settings
# ──────────────────────────────
LAYER_NUM = 32
PAUSE_DIST = 0    
MAX_OFFLOAD = LAYER_NUM // 2
MIN_LEN = 1

# Font and Style Configuration
FONT_SIZE = 18               # General font size
TICK_FONT_SIZE = 18          # Tick label font size
LEGEND_FONT_SIZE = 18        # Legend text size
LABEL_FONT_SIZE = 20         # Axis label size
TITLE_FONT_SIZE = 18         # (if used)
LINE_WIDTH = 3             # Line thickness
LINE_WIDTH_BOUND = 2
LEGEND_COL_SPACING = 0.9     # Spacing between legend columns
LEGEND_NUM_COLS = 6          # Number of legend columns

# Graph size and layout (moved here from later part)
fig, (ax_mem, ax_dist) = plt.subplots(
    nrows=2, ncols=1, figsize=(11, 5), sharex=True,
    gridspec_kw=dict(height_ratios=[1, 1], hspace=0.1)
)

############################################################################
# Utility: merge pauses that are shorter than MIN_LEN into neighbours
############################################################################
def squash_short_pauses(segments: dict[str, list[tuple[int,int,int|None]]],
                        min_len: int = 3) -> None:
    for rid, segs in segments.items():
        merged: list[tuple[int,int,int|None]] = []
        i = 0
        while i < len(segs):
            s, e, d = segs[i]
            if (e - s + 1 < min_len):
                prev_live = merged[-1] if merged else None
                next_live = segs[i + 1] if i + 1 < len(segs) else None
                if prev_live and next_live and prev_live[2] == next_live[2]:
                    _, _, d_live = prev_live
                    merged[-1] = (prev_live[0], next_live[1], d_live)
                    i += 2
                elif prev_live:
                    merged[-1] = (prev_live[0], e, prev_live[2])
                    i += 1
                elif next_live:
                    merged.append((s, next_live[1], next_live[2]))
                    i += 2
                else:
                    merged.append((s, e, 0))
                    i += 1
            else:
                merged.append((s, e, d))
                i += 1
        segments[rid] = merged

##########################################################################
# 1. read log
##########################################################################
if len(sys.argv) != 3:
    print("Usage: python visualise_dist_mem.py  /path/to/temp.log /path/to/output.png")
    sys.exit(1)

log_path = sys.argv[1]
output_path = sys.argv[2]

step_re    = re.compile(r"STEP\s+(\d+)")
dist_re    = re.compile(r"dist=\{([^\}]*)\}")
mem_re     = re.compile(r"mem\s+(\d+)/")
pause_re   = re.compile(r"Paused\s+(\S+)\s.*step\s+(\d+)")
resume_re  = re.compile(r"(?:Resumed|Admitted)\s+(\S+).*step\s+(\d+)")
finish_re  = re.compile(r"FINISHED\s+(\S+).*step\s+(\d+)")
pattern_req = re.compile(r"(request_\d+):\d+t/(\d+)b")

def parse_dist_blob(blob: str):
    blob = blob.strip()
    if not blob:
        return []
    pairs = [p.strip() for p in blob.split(",")]
    for p in pairs:
        rid, d = [x.strip() for x in p.split(":")]
        yield rid, int(float(d))   

def add_seg(rid, s, e, d):
    segments[rid].append((s, e, d))
    if (e - s + 1) < MIN_LEN:
        skip_steps.update(range(s, e + 1))

segments = collections.defaultdict(list)
open_seg = {}
mem_rows = []
mem_req_rows = defaultdict(list)
skip_steps = set()

pause_open: dict[str, int] = {}
finished: set[str] = set()
last_step = None

with open(log_path, encoding="utf-8") as f:
    for ln in f:
        m_step = step_re.search(ln)
        if m_step:
            step = int(m_step.group(1))
            last_step = step
            m_mem = mem_re.search(ln)
            if m_mem:
                used_blocks = int(m_mem.group(1))
                mem_rows.append((step, used_blocks))
            else:
                mem_rows.append((step, math.nan))

            for m_req in re.finditer(r"(request_\d+):\d+t/(\d+)b", ln):
                rid, bytes_used = m_req.group(1), int(m_req.group(2))
                blocks_used = bytes_used
                mem_req_rows[rid].append((step, blocks_used))

            blob = dist_re.search(ln)
            if blob:
                for rid, dist in parse_dist_blob(blob.group(1)):
                    if dist == -1: dist = 32
                    if rid in finished:
                        continue
                    if rid in pause_open:
                        p_start = pause_open.pop(rid)
                        add_seg(rid, p_start, step - 1, PAUSE_DIST)
                    if rid in open_seg:
                        s, d_prev = open_seg[rid]
                        if d_prev != dist:
                            add_seg(rid, s, step - 1, d_prev)
                            open_seg[rid] = (step, dist)
                    else:
                        open_seg[rid] = (step, dist)

        m_pause = pause_re.search(ln)
        if m_pause:
            rid, st = m_pause.group(1), int(m_pause.group(2))
            if rid in open_seg:
                s, d = open_seg.pop(rid)
                add_seg(rid, s, st - 1, d)
            pause_open[rid] = st

        if (m_fin := finish_re.search(ln)):
            rid = m_fin.group(1)
            step = int(m_fin.group(2))
            if rid in open_seg:
                s, d = open_seg.pop(rid)
                add_seg(rid, s, step, d)
            finished.add(rid)

for rid, (s, d) in open_seg.items():
    add_seg(rid, s, step, d)
for rid, p_start in pause_open.items():
    add_seg(rid, p_start, step, PAUSE_DIST)

if last_step is not None:
    for rid, (start, dist) in open_seg.items():
        add_seg(rid, start, last_step, dist)

squash_short_pauses(segments, min_len=3)

##########################################################################
# 3. Color Map
##########################################################################
all_dist = sorted({d for segs in segments.values() for _,__,d in segs})
# palette = cm.get_cmap("tab10", len(all_dist))
# dist2col = {d: palette(i) for i, d in enumerate(all_dist)}

colors = ["#82C0FF", "#FFAE96", "#9FD7C6", "#FBD8B6", "#BCA1D3", "#FBE1A4"]
if len(colors) < len(segments):
    colors = colors * (len(segments) // len(colors) + 1)

step_colors = {f"request_{i}": colors[i] for i in range(len(colors))}

##########################################################################
# 4. Plot
##########################################################################
skip_steps = set()
for segs in segments.values():
    for s, e, d in segs:
        if (e - s + 1) < MIN_LEN:
            skip_steps.update(range(s, e + 1))

new_mem_rows = []
last_good = None
for step, used in mem_rows:
    if step in skip_steps:
        if last_good is not None:
            new_mem_rows.append((step, last_good))
    else:
        new_mem_rows.append((step, used))
        last_good = used
mem_rows = new_mem_rows

def extract_number(label):
    match = re.search(r"request_(\d+)", label)
    return int(match.group(1)) if match else float('inf')

for idx, (rid, segs) in enumerate(segments.items()):
    base_y = 0
    colour = step_colors.get(rid, 'k')
    prev_d = None
    xs, ys = [], []

    for j, (s, e, d) in enumerate(segs):  
        if (e - s + 1) < MIN_LEN:
            continue
        if d in (0, None):
            if prev_d is not None:
                y_pause = prev_d + base_y
            prev_d = prev_d
            continue

        xs.extend([s, e + 1])
        ys.extend([d + base_y] * 2)
        prev_d = d

        is_last = (j == len(segs) - 1)
        nxt_is_pause = (not is_last and segs[segs.index((s, e, d))+1][2] in (0, None))
        if is_last or nxt_is_pause:
            ax_dist.step(xs, ys, where="post", lw=LINE_WIDTH, color=colour, label=rid, zorder=extract_number(rid)+1)
            ax_dist.scatter(xs[0], ys[0], s=35, c=colour, zorder=extract_number(rid)+1)
            ax_dist.scatter(xs[-1], ys[-1], s=35, facecolors='none', edgecolors=colour, zorder=extract_number(rid)+1)
            xs, ys = [], []

dist_line = ax_dist.axhline(y=32, color='black', lw=LINE_WIDTH_BOUND, ls="--", label="Boundary (32)", alpha=0.7, zorder=0)
ax_dist.yaxis.set_label_coords(-0.05, 0.55)
ax_dist.set_xlabel("Step", fontsize=LABEL_FONT_SIZE)
ax_dist.set_yticks([0, 8, 16, 24, 32], labels=["0", "8", "16", "24", "32"])
ax_dist.invert_yaxis()

ax_dist.set_ylabel("Offload Dist.", fontsize=LABEL_FONT_SIZE)
ax_dist.yaxis.set_label_coords(-0.065, 0.5)  # (x, y) 좌표

df_mem_req = pd.DataFrame(index=pd.Index(sorted({s for r in mem_req_rows.values() for s, _ in r}), name="step"))
for rid, records in mem_req_rows.items():
    df = pd.DataFrame(records, columns=["step", rid]).set_index("step")
    df_mem_req[rid] = df.reindex(df_mem_req.index).fillna(0)

df_mem_req = df_mem_req[sorted(df_mem_req.columns, key=lambda rid: int(rid.split("_")[1]))]
bottom = np.zeros(len(df_mem_req))
x = df_mem_req.index.to_numpy()

for idx, rid in enumerate(df_mem_req.columns):
    y = df_mem_req[rid].to_numpy() / 1000
    ax_mem.fill_between(x, bottom, bottom + y, label=rid, alpha=1, color=colors[idx], lw=0)
    bottom += y

df_sys = pd.DataFrame(mem_rows, columns=["step", "used"]).dropna()

def format_k(y, pos):
    return f'{y:.0f}k' if y != 0 else '0'
ax_mem.set_yticks([0, 5, 10, 15])
ax_mem.yaxis.set_major_formatter(ticker.FuncFormatter(format_k))
mem_bound_line = ax_mem.axhline(y=16, color='black', lw=LINE_WIDTH_BOUND, ls='-.', zorder=1, alpha = 0.7)
ax_mem.set_ylabel("KV Blocks", fontsize=LABEL_FONT_SIZE)

mem_handles, mem_labels = ax_mem.get_legend_handles_labels()
mem_pairs = [(int(lbl.split('_')[1]), h) for lbl, h in zip(mem_labels, mem_handles)]
mem_pairs.sort(key=lambda x: x[0])
sorted_labels = [f"Req.{idx}" for idx, _ in mem_pairs]
sorted_handles = [h for _, h in mem_pairs]

fig.legend(
    sorted_handles, sorted_labels,
    loc="upper center", ncol=LEGEND_NUM_COLS,
    fontsize=LEGEND_FONT_SIZE,
    columnspacing=LEGEND_COL_SPACING,
    handletextpad = 0.4,
    bbox_to_anchor=(0.5, 1), frameon=False,
)

ax_mem.legend(
    [mem_bound_line], ["Available Memory"],
    loc="lower left", fontsize=LEGEND_FONT_SIZE, frameon=False,
    bbox_to_anchor=(0.55, 0),
)

ax_dist.legend(
    [dist_line], ["No Offload"],
    loc="lower left", fontsize=LEGEND_FONT_SIZE, frameon=False,
    bbox_to_anchor=(0.6, 0),
)

for ax in [ax_dist, ax_mem]:
    ax.tick_params(axis='x', which='both', length=0, labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis='y', which='both', length=0, labelsize=TICK_FONT_SIZE)

fig.tight_layout()
fig.savefig(output_path, bbox_inches="tight")
print(f"Saved figure to {output_path}")
fig.savefig("solver_example.pdf", bbox_inches="tight")
plt.close(fig)
