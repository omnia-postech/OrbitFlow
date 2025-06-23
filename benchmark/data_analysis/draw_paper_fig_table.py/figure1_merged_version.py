import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FuncFormatter

# ---------------- Shared style (take from first file) ----------------
style = {
    "line":   {"linewidth": 4, "markersize": 10},
    "bar":    {"edgecolor": "black", "linewidth": 2, "width": 0.6, "alpha": 0.8},
    "tick":   {"fontsize": 26},
    "label":  {"fontsize": 26, "labelpad": 10},
    "title":  {"fontsize": 30, "pad": 15},
    "legend": {"fontsize": 22, "loc": "upper left"},
    "spine":  {"color": "black", "alpha": 0.9, "linestyle": "-", "linewidth": 2.5},
    "grid":   {"color": "gray", "linestyle": "--", "linewidth": 3, "alpha": 0.7},
    "text":   {"fontsize": 26},
    "arrow":  {"linewidth": 2.5, "mutation_scale": 22},
}

colors = {
    'compute': '#3CC58F',
    'communicate': '#FF8C69'
}

# ------------- Create 1×3 layout ------------------------------------
fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(19, 6))

# ====================================================================
# (a) KV‑cache size bar plot (copied from first file) -----------------
x_labels = ['1k', '16k', '128k', '1M']
x_pos = [0, 1, 2, 3]
y_data = [3, 8, 40, 320]
y_mapped = [val if val <= 50 else 50 + (val - 230) * 50/100 for val in y_data]

bars = ax_a.bar(x_pos, y_mapped,
                color='#FF8C69',
                edgecolor=style["bar"]["edgecolor"],
                linewidth=style["bar"]["linewidth"],
                width=style["bar"]["width"],
                alpha=style["bar"]["alpha"])

ax_a.set_title('Model: LLaMA3-70B', **style["title"])
ax_a.set_xlabel('Sequence Length', **style["label"])
ax_a.set_ylabel('KVCache Size (GB)', **style["label"])
ax_a.set_xticks(x_pos)
ax_a.set_xticklabels(x_labels, **style["tick"])
ax_a.set_ylim(0, 110)
ax_a.set_yticks([])
ax_a.text(-0.35, 55, '50', ha='left', va='center', **style["text"])
ax_a.text(-0.35, 102, '400', ha='left', va='center', **style["text"])
ax_a.axhline(y=50, **style["grid"])
ax_a.grid(False)
actual_vals = [1, 5, 40, 320]
for i, (bar, val) in enumerate(zip(bars, actual_vals)):
    if i == 3:
        ax_a.text(bar.get_x()+bar.get_width()/2-0.01, bar.get_height()+1, '320',
                  ha='center', va='bottom', **style["text"])
for spine in ax_a.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])
ax_a.tick_params(axis='x', which='both', length=0)
ax_a.tick_params(axis='y', which='both', length=0)

# ====================================================================
# (b) Compute vs. communicate time plot ------------------------------
context_lengths = ['2k', '4k', '8k', '16k']
x_idx = np.arange(len(context_lengths))
x_vals = [int(c[:-1]) * 1024 for c in context_lengths]
compute = [1.23e-6 * v + 4.00e-2 for v in x_vals]
commun  = [2.0e-5 * v + 4.0e-2 for v in x_vals]

ax_b.plot(x_idx, compute, marker='o', color=colors['compute'], label='Comp. Time', **style["line"])
ax_b.plot(x_idx, commun,  marker='s', color=colors['communicate'], label='Comm. Time', **style["line"])

for i in [0, -1]:
    x_p   = x_idx[i]
    y_cpt = compute[i]
    y_com = commun[i]
    y_mid = (y_cpt + y_com) / 2
    ratio = y_com / y_cpt
    ratio_text = f'{ratio:.2f}x'
    ax_b.add_patch(FancyArrowPatch((x_p, y_com), (x_p, y_cpt),
                    arrowstyle='<->', color='black', **style["arrow"]))
    text_x = x_p + (0.4 if i == 0 else -0.35)
    ax_b.text(text_x, y_mid, ratio_text, ha='center', va='center', fontsize=27)

ax_b.set_xticks(x_idx)
ax_b.set_xticklabels(context_lengths, **style["tick"])
ax_b.tick_params(axis='y', labelsize=style["tick"]["fontsize"])
ax_b.set_xlabel('Sequence length', **style["label"])
ax_b.set_ylabel('Latency (s)', **style["label"])
ax_b.set_yticks([0.1,0.200,0.300])
ax_b.set_ylim(-0.0,0.45)  # set y-limits to match the original plot

ax_b.set_title('Comm. and Comp. time', **style["title"])
for spine in ax_b.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])
ax_b.tick_params(axis='x', which='both', length=0)
ax_b.tick_params(axis='y', which='both', length=0)
ax_b.legend(**style["legend"], frameon=False)

# ====================================================================
# (c) Latency margin plot -------------------------------------------
#   TPOT vs TBT with SLO
slo_thres = 0.052
slo_thres_ms = slo_thres * 1_000
x_val_tokens = np.arange(2*1024, 16*1024 + 1)
x_k = x_val_tokens/1024
compute = 1.23e-6 * x_val_tokens + 0.04
tbt_ms  = compute * 1000
tpot_ms = np.cumsum(compute)/np.arange(1, len(compute)+1)*1000
mask = tbt_ms <= slo_thres_ms
idx_int = np.argmax(compute >= slo_thres)
x_int_k = x_val_tokens[idx_int]/1024
y_int_ms = tbt_ms[idx_int]


# SLO
ax_c.axhline(y=slo_thres_ms, color='gray', linestyle='--', linewidth=2,
             label=f'SLO = {slo_thres_ms:.0f} ms')

# fill
ax_c.fill_between(x_k, tbt_ms, slo_thres_ms, where=mask, color='green', alpha=0.2)

# triangle marker
ax_c.scatter(x_int_k, y_int_ms, marker='^', s=180, color='red',
             zorder=6)
ax_c.text(x_int_k + 1,         # ↗ adjust horizontal offset to taste
          y_int_ms ,          # ↑ vertical offset (ms)
          'First TBT SLO\nviolation',
          ha='left', va='center',
          fontsize=style["text"]["fontsize"]*0.8)
# label='First TBT \nSLO violation', 
# lines
ax_c.plot(x_k, tpot_ms, label='TPOT', **style["line"])
ax_c.plot(x_k, tbt_ms,  label='TBT', color='orange', **style["line"])
ax_c.set_title('Latency Margin', **style["title"])
# arrow & label
base_x = 4.0
idx_arrow = np.argmin(np.abs(x_k - base_x))
head_x, head_y = base_x + 0.6, tpot_ms[idx_arrow] + 4.5
tail_x, tail_y = head_x + 3.4, head_y - 1.9
ax_c.annotate('', xy=(head_x, head_y), xytext=(tail_x, tail_y),
              arrowprops=dict(arrowstyle='<-', color='black', **style["arrow"]))
ax_c.text(head_x + 2*3.4, head_y + 1.7*(-2.2),
          'Latency margin\nabsorbed by TPOT',
          ha='center', va='center', fontsize=style["text"]["fontsize"]*0.8)

ax_c.set_xlabel('Sequence length', **style["label"])
ax_c.set_ylabel('Latency (ms)', **style["label"])
ax_c.set_xticks([2,8,16])
ax_c.set_xticklabels(['2k','8k','16k'], **style["tick"])
ax_c.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}'))
ax_c.tick_params(axis='y', labelsize=style["tick"]["fontsize"])
ax_c.tick_params(axis='x', which='both', length=0)
ax_c.tick_params(axis='y', which='both', length=0)
ax_c.legend(prop={'size': style["legend"]["fontsize"]}, frameon=False,ncol=1,
            columnspacing=0.6,      # ⇐ adjust this value to control the gap
            # handletextpad=0.3       # optional: tighten icon-to-text spacing
    )

for spine in ax_c.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# ====================================================================
# labels (a)(b)(c)
fig.text(0.15, -0.04, '(a)', ha='center', va='center', fontsize=style["title"]["fontsize"])
fig.text(0.5, -0.04, '(b)', ha='center', va='center', fontsize=style["title"]["fontsize"])
fig.text(0.85, -0.04, '(c)', ha='center', va='center', fontsize=style["title"]["fontsize"])

plt.tight_layout()

plt.savefig('figures/merged_graphs.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/merged_graphs.pdf', dpi=1200, bbox_inches='tight')
plt.show()
