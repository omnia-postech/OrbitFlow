# visualise_dist_mem.py  ── run:  python visualise_dist_mem.py  /path/to/temp.log
import sys, re, collections, itertools, math
import pandas as pd, matplotlib.pyplot as plt
from matplotlib import cm
LAYER_NUM = 32
PAUSE_DIST = 0    
MAX_OFFLOAD = LAYER_NUM // 2          # 32-layer model ⇒ 16 is the upper-bound
                                      #  ❱  adjust here if you ever change LAYER_NUM
MIN_LEN=5
from collections import defaultdict
############################################################################
# Utility: merge pauses that are shorter than MIN_LEN into neighbours
############################################################################
def squash_short_pauses(segments: dict[str, list[tuple[int,int,int|None]]],
                        min_len: int = 3) -> None:
    """
    In-place edit of the segments dict.

      segments[rid]  ==  list of (start, end, dist_or_None)
      • pause  :=  dist is 0 or None
      • live   :=  dist is a real distance (int)

    Rule
    ----
    If a *pause* has length < min_len **and** it has a live segment on
    both sides with the **same distance**, then:
        live A … pause … live B   →   live A…B   (pause disappears)
    Otherwise – e.g. only one neighbour, or the neighbours have different
    d – we simply stretch the *preceding* live segment across the pause.

    After processing, every RID’s list is still chronologically ordered.
    """
    for rid, segs in segments.items():
        merged: list[tuple[int,int,int|None]] = []

        i = 0
        while i < len(segs):
            s, e, d = segs[i]

            # ── candidate pause to squash? ────────────────────────────────
            if (e - s + 1 < min_len):
                prev_live = merged[-1] if merged else None
                next_live = segs[i + 1] if i + 1 < len(segs) else None

                if prev_live and next_live and prev_live[2] == next_live[2]:
                    # case A: same-d live on both sides  → fuse A + pause + B
                    _, _, d_live = prev_live
                    merged[-1] = (prev_live[0], next_live[1], d_live)
                    i += 2                                      # skip next_live
                elif prev_live:
                    # case B: only previous live → extend it forward
                    merged[-1] = (prev_live[0], e, prev_live[2])
                    i += 1
                elif next_live:
                    # case C: only next live → extend it backward
                    merged.append((s, next_live[1], next_live[2]))
                    i += 2
                else:
                    # orphan pause at ends – keep but mark as proper pause
                    merged.append((s, e, 0))
                    i += 1
            else:
                # normal segment, just copy
                merged.append((s, e, d))
                i += 1

        segments[rid] = merged
  
##########################################################################
# 1. read log
##########################################################################
if len(sys.argv) != 2:
    print("Usage: python visualise_dist_mem.py  /path/to/temp.log")
    sys.exit(1)

log_path = sys.argv[1]

step_re    = re.compile(r"STEP\s+(\d+)")
dist_re    = re.compile(r"dist=\{([^\}]*)\}")
mem_re     = re.compile(r"mem\s+(\d+)/")
pause_re   = re.compile(r"Paused\s+(\S+)\s.*step\s+(\d+)")
resume_re  = re.compile(r"(?:Resumed|Admitted)\s+(\S+).*step\s+(\d+)")
finish_re  = re.compile(r"FINISHED\s+(\S+).*step\s+(\d+)")
pattern_req = re.compile(r"(request_\d+):\d+t/(\d+)b")

# helpers ----------------------------------------------------------------
def parse_dist_blob(blob: str):
    blob = blob.strip()
    if not blob:
        return []
    pairs = [p.strip() for p in blob.split(",")]
    for p in pairs:
        rid, d = [x.strip() for x in p.split(":")]
        yield rid, int(float(d))   # blob may contain “-1” or “31” etc.
def add_seg(rid, s, e, d):
    # helper used everywhere instead of writing segments[rid].append(...)
    segments[rid].append((s, e, d))
    if (e - s + 1) < MIN_LEN:          # nano-segment → remember its steps
        skip_steps.update(range(s, e + 1))
# data holders -----------------------------------------------------------
segments   = collections.defaultdict(list)   # rid -> [(s,e,d)]
open_seg   = {}                              # rid -> (start, dist)
mem_rows   = []                              # (step, used)
mem_req_rows = defaultdict(list)   # per-request: {request_id: [(step, used_blocks)]}

skip_steps = set()      #  ←  NEW : steps that fall in segments < MIN_LEN
##########################################################################
# 2. walk log once
##########################################################################
last_step = None                        # <─ keep track of the most-recent STEP
pause_open: dict[str, int] = {}
finished: set[str] = set()          #  ← NEW
with open(log_path, encoding="utf-8") as f:
    for ln in f:
        # ───────────────── STEP line ───────────────────────────────────
        m_step = step_re.search(ln)
        if m_step:
            step = int(m_step.group(1))

            # ── memory (fig-2) ─────────────────────────────────────────
            m_mem = mem_re.search(ln)
            if m_mem:
                used_blocks = int(m_mem.group(1))
                mem_rows.append((step, used_blocks))
            else:
                mem_rows.append((step, math.nan))

            # ── request-wise memory ───────────────────────────────────
            for m_req in re.finditer(r"(request_\d+):\d+t/(\d+)b", ln):
                rid, bytes_used = m_req.group(1), int(m_req.group(2))
                # blocks_used = bytes_used // 16      # convert to blocks (assuming 16-token block size)
                blocks_used = bytes_used
                mem_req_rows[rid].append((step, blocks_used))
                # ── distance blob ──────────────────────────────────────────
                blob = dist_re.search(ln)
                if blob:
                    for rid, dist in parse_dist_blob(blob.group(1)):
                        if dist == -1: 
                            dist = 32
                        if rid in finished:                 # <── NEW guard
                            continue
                        # close an outstanding PAUSE if this is the first time
                        # the request reappears after a “Paused …” log
                        if rid in pause_open:
                            p_start = pause_open.pop(rid)
                            add_seg(rid, p_start, step - 1, PAUSE_DIST)
                        # handle normal open/close of live segments
                        if rid in open_seg:
                            s, d_prev = open_seg[rid]
                            if d_prev != dist:                # distance changed
                                add_seg(rid, s, step - 1, d_prev)
                                open_seg[rid] = (step, dist)
                        else:
                            open_seg[rid] = (step, dist)

        # ───────────────── pause line ──────────────────────────────────
        m_pause = pause_re.search(ln)
        if m_pause:
            rid, st = m_pause.group(1), int(m_pause.group(2))
            if rid in open_seg:                            # close live slice
                s, d = open_seg.pop(rid)
                add_seg(rid, s, st - 1, d)
            pause_open[rid] = st                           # remember start

        # ───────────────── resume/admit line – nothing to do here ──────
        # handled implicitly when the next STEP blob appears

        # ───────────────── finish line ─────────────────────────────────
        if (m_fin := finish_re.search(ln)):
            rid  = m_fin.group(1)
            step = int(m_fin.group(2))

            if rid in open_seg:                     # close running seg
                s, d = open_seg.pop(rid)
                add_seg(rid, s,step, d)

            finished.add(rid)                       # <── remember it's done

# ── tidy up any still-open slices at EOF ────────────────────────────────
for rid, (s, d) in open_seg.items():
    add_seg(rid, s,step, d)
for rid, p_start in pause_open.items():
    add_seg(rid, p_start,step, PAUSE_DIST)
if last_step is not None:               # guard against empty logs
    for rid, (start, dist) in open_seg.items():
        add_seg(rid, start,last_step, dist)

squash_short_pauses(segments, min_len=3)
##########################################################################
# 3. colour map  (distance -> colour)  -----------------------------------
all_dist = sorted({d for segs in segments.values() for _,__,d in segs})
palette  = cm.get_cmap("tab10", len(all_dist))
dist2col = {d: palette(i) for i, d in enumerate(all_dist)}

colors = ["#4DA6FF",  # Sky Blue
 "#FF8C69",  # Coral Orange
 "#76C7AE",  # Pastel Mint
 "#FFB3BA",  # Pastel Pink
 "#9F79C1",  # Lavender Purple
 "#F9D57E"]  # Soft Yellow-Orange


##########################################################################
      
MIN_LEN   = 5                # ignore segments shorter than this
PAUSE_VAL = 32                # y-value we use for pauses
v_off     = 0.1              # vertical offset step between request

# sychoy
step_colors = {f"request_{i}": colors[i]
               for i in range(len(colors))}               # extend as needed

##########################################################################
# 4-A. build a set of steps we want to “mask”
skip_steps: set[int] = set()
for segs in segments.values():
    for s, e, d in segs:
        if (e - s + 1) < MIN_LEN:          # the very same condition you used
            skip_steps.update(range(s, e + 1))

# 4-B. walk mem_rows once and carry-forward the last good value
new_mem_rows = []
last_good = None
for step, used in mem_rows:
    if step in skip_steps:
        if last_good is not None:          # silent carry-forward
            new_mem_rows.append((step, last_good))
    else:
        new_mem_rows.append((step, used))
        last_good = used                   # update tracker

# replace the old list
mem_rows = new_mem_rows
# fig, (ax_dist, ax_mem) = plt.subplots(
fig, (ax_mem, ax_dist) = plt.subplots(
    nrows=2, ncols=1, figsize=(11, 7), sharex=True,
    # gridspec_kw=dict(height_ratios=[2, 1], hspace=0.25)
    gridspec_kw=dict(height_ratios=[1, 2], hspace=0.25)
)

# ───────────────────────── stepped view  (ax_dist) ──────────────────────
MIN_LEN   = 5                 # ignore super-short segments
PAUSE_VAL = 32                # y-value for pauses
v_off     = 0.12              # vertical offset between requests

# sychoy
for idx, (rid, segs) in enumerate(segments.items()):
    base_y  = -idx * v_off          # invert Y (top req has y≈0)
    colour  = step_colors.get(rid, 'k')
    prev_d  = None
    xs, ys  = [], []

    for j, (s, e, d) in enumerate(segs):  
        if (e - s + 1) < MIN_LEN:
            continue

        if d in (0, None):                          # ----- PAUSE -----
            if prev_d is not None:
                y_pause = prev_d + base_y
                ax_dist.step([s, e + 1], [y_pause, y_pause],
                             where="post", color=colour, ls="--", lw=1.0, )
            prev_d = prev_d          # unchanged
            continue

        # ----- LIVE SEGMENT -----
        xs.extend([s, e + 1])
        ys.extend([d + base_y]*2)
        prev_d = d                                        # update

        # flush if next is pause or last segment
        is_last       = (j == len(segs) - 1)
        nxt_is_pause = (not is_last and segs[segs.index((s, e, d))+1][2] in (0, None))
        if is_last or nxt_is_pause:
            ax_dist.step(xs, ys, where="post", lw=1.7, color=colour, label=rid)
            ax_dist.scatter(xs[0],  ys[0],  s=28, c=colour)               # start
            ax_dist.scatter(xs[-1], ys[-1], s=28, facecolors='none',
                            edgecolors=colour)                            # end
            xs, ys = [], []

# cosmetics
ax_dist.set_ylabel("Prefetch distance\n(offset per request)")
# ax_dist.set_title("Prefetch distance per request — stepped view\n"
                #   "(● start, ○ end, dashed = paused)")
handles, labels = ax_dist.get_legend_handles_labels()
uniq = dict(zip(labels, handles))
ax_dist.legend(uniq.values(), 
               uniq.keys(), 
               ncol=3, 
               loc="lower right",
               fontsize="small")
ax_dist.invert_yaxis()

# # ───────────────────────── memory-usage (ax_mem) ───────────────────────
# df_mem = pd.DataFrame(mem_rows, columns=["step", "used"])
# # ax_mem.plot(df_mem["step"], df_mem["used"], color="tab:blue", label="System Total")
# # ax_mem.fill_between(df_mem["step"], df_mem["used"], color="green", alpha=0.3)

# # Plot each request's memory
# for req_id, entries in mem_req_rows.items():
#     df_req = pd.DataFrame(entries, columns=["step", "used"])
#     ax_mem.plot(df_req["step"], df_req["used"], label=req_id, lw=1.2, alpha=0.8)

# ax_mem.set_xlabel("Step")
# ax_mem.set_ylabel("KV blocks in use")
# ax_mem.set_title("System-wide and Per-request KV Memory Usage")
# ax_mem.legend(loc="upper right", fontsize="x-small")

# Build individual series for each request with explicit step-to-memory mapping
import numpy as np
all_steps = sorted({step for rows in mem_req_rows.values() for step, _ in rows})
all_steps_idx = pd.Index(all_steps, name="step")

df_mem_req = pd.DataFrame(index=all_steps_idx)

for rid, records in mem_req_rows.items():
    df = pd.DataFrame(records, columns=["step", rid]).set_index("step")
    df_mem_req[rid] = df.reindex(all_steps_idx).fillna(0)

# Sort columns numerically by request ID
def rid_key(rid: str):
    return int(rid.split("_")[1])

df_mem_req = df_mem_req[sorted(df_mem_req.columns, key=rid_key)]

# ─────────────── plot memory stack only during lifetime ────────────────
bottom = np.zeros(len(df_mem_req))
x = df_mem_req.index.to_numpy()

for idx, rid in enumerate(df_mem_req.columns):
    y = df_mem_req[rid].to_numpy()
    ax_mem.fill_between(x, bottom, bottom + y, label=rid, alpha=0.5, color=colors[idx],)
    bottom += y

# Overlay system total for comparison
df_sys = pd.DataFrame(mem_rows, columns=["step", "used"]).dropna()
# ax_mem.plot(df_sys["step"], df_sys["used"], color="black", lw=1.0, label="System Total")

ax_mem.set_xlabel("Step")
ax_mem.set_ylabel("KV blocks in use")
ax_mem.set_title("Per-request Memory Usage Breakdown")
# ax_mem.legend(fontsize="x-small", loc="upper left", ncol=2)


# ───────────────────────── save / show ──────────────────────────────────
fig.tight_layout()
fig.savefig("fig_combined.png")   #     ← NEW combined figure
fig.savefig("solver_example.pdf")   #     ← NEW combined figure
plt.close(fig)

