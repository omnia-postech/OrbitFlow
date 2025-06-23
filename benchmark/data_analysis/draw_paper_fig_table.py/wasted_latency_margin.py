import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FuncFormatter

# ------------------------------------------------------------------
# GLOBAL FONT-SIZE “MACROS”
# ------------------------------------------------------------------
AXIS_FONT   = 22     # axis labels & legend
TICK_FONT   = 22     # tick labels
TEXT_FONT   = 24     # annotation text
ARROW_LW    = 4    # arrow line-width
ARROW_MS    = 22     # arrow mutation-scale (head size)
# ------------------------------------------------------------------

style = {
    "line":   {"linewidth": 4, "markersize": 10},
    "bar":    {"edgecolor": "black", "linewidth": 2, "width": 0.6, "alpha": 0.8},
    "tick":   {"fontsize": TICK_FONT},
    "label":  {"fontsize": AXIS_FONT, "labelpad": 10},
    "title":  {"fontsize": AXIS_FONT + 2, "pad": 15},
    "legend": {"fontsize": AXIS_FONT, "loc": "upper left"},
    "spine":  {"color": "black", "alpha": 0.9, "linestyle": "-", "linewidth": 2.5},
    "grid":   {"color": "gray", "linestyle": "--", "linewidth": 3, "alpha": 0.7},
    "text":   {"fontsize": TEXT_FONT},
    "arrow":  {"linewidth": ARROW_LW, "mutation_scale": ARROW_MS},   # NEW
}


# --- Parameters ---------------------------------------------------
slo_thres = 0.052
slo_thres_ms = slo_thres * 1_000           # milliseconds

x_values = np.arange(2 * 1024, 16 * 1024 + 1)
x_k      = x_values / 1024

compute = 1.23e-6 * x_values + 0.04
communicate = 2.0e-5 * x_values + 0.04      # not plotted but kept for context

tbt_ms  = compute * 1_000
tpot_ms = np.cumsum(compute) / np.arange(1, len(compute) + 1) * 1_000
mask    = tbt_ms <= slo_thres_ms

idx_int          = np.argmax(compute >= slo_thres)
x_int_k          = x_values[idx_int] / 1024
y_int_ms         = tbt_ms[idx_int]

# --- Plot ---------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(9, 7))

# SLO line
ax1.axhline(y=slo_thres_ms, color='gray', linestyle='--', linewidth=2,
            label=f'SLO = {slo_thres_ms:.0f} ms')

# TPOT & TBT
ax1.plot(x_k, tpot_ms, label='TPOT', **style["line"])
ax1.plot(x_k, tbt_ms,  label='TBT',  color='orange', **style["line"])

# Green region
ax1.fill_between(x_k, tbt_ms, slo_thres_ms, where=mask,
                 color='green', alpha=0.2)

# Intercept marker
ax1.scatter(x_int_k, y_int_ms, marker='^', s=180, color='red',
            label='First TBT SLO violation', zorder=6)

# Arrow & label
base_x   = 4.0
idx_a    = np.argmin(np.abs(x_k - base_x))
head_x   = base_x + 0.6
head_y   = tpot_ms[idx_a] + 4.5
tail_x   = head_x + 3.4
tail_y   = head_y - 1.9

# Arrow & annotation (uses style["arrow"]) -------------------------
ax1.annotate(
    '',
    xy=(head_x, head_y),              # head
    xytext=(tail_x, tail_y),          # tail
    arrowprops=dict(
        arrowstyle='<-',
        color='black',
        **style["arrow"]              # linewidth & mutation_scale from macro
    )
)
ax1.text(
    head_x + 1.7*3.4, head_y + 1.7*(-2),
    'Latency margin\nabsorbed by TPOT',
    ha='center', va='center', fontsize=TEXT_FONT
)

# Axis labels & ticks
ax1.set_xlabel('Sequence length', fontsize=AXIS_FONT)
ax1.set_ylabel('Decode latency (ms)', fontsize=AXIS_FONT)
ax1.set_xticks([2, 4, 8, 16])
ax1.set_ylim(40,60)
ax1.set_yticks(np.arange(40, 61, 5))
ax1.set_xticklabels(['2k', '4k', '8k', '16k'], fontsize=TICK_FONT)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}'))
ax1.tick_params(axis='y', labelsize=TICK_FONT)

# Legend & layout
ax1.legend(prop={'size': AXIS_FONT}, frameon=False)   # no bounding box
ax1.set_title('')   # no title
plt.tight_layout()
plt.savefig('figures/wasted_latency_margin.png')
plt.savefig('figures/wasted_latency_margin.pdf')
