import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

# style
style = {
    "bar": {
        "edgecolor": "white",
        "linewidth": 3,
        "alpha": 0.5
    },
    "tick": {
        "fontsize": 18,
        "weight": "bold",
    },
    "label": {
        "fontsize": 24,
        "weight": "bold"
    },
    "title": {
        "fontsize": 24,
        "weight": "bold",
        "pad": 10
    },
    "legend": {
        "fontsize": 20,
        "loc": "upper center",
        "bbox_to_anchor": (0.5, 1.2),
        "ncol": 2,
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

colors = {
    'ttft': '#D81B60',
    'tpot': '#55A868',
}

hatches = {
    'ttft': '/',
    'tpot': '\\',
}


# ==== data ====
context_lengths = [128, 256, 512, 1024, 2048, 4096, 16384]
# x_ticks = [-1e6, -1e5, -1e4, -1e3, -1e2, -1e1, 0, 1e1, 1e2]
x_tick_labels = [r'$10^6$', r'$10^5$', r'$10^4$', r'$10^3$', r'$10^2$', r'$10^1$', '0', r'$10^1$', r'$10^2$']
           
ttft_latency = np.array([98, 99, 101, 1002, 10020, 19000, 978000])     # 왼쪽 (음수로 그릴 예정)
tpot_latency = np.array([40, 50, 55, 70, 80, 85, 90])     # 오른쪽

fig, ax = plt.subplots(figsize=(10, 6))

bar_height = 0.5
y_pos = np.arange(len(context_lengths))


# ==== 막대 그리기 ====
ttft_latency_log = np.log10(ttft_latency)
tpot_latency_log = np.log10(tpot_latency)
ax.barh(y_pos, -ttft_latency_log,
        color=colors['ttft'], hatch=hatches['ttft'], **style["bar"],
        label='TTFT')
ax.barh(y_pos, tpot_latency_log, 
        color=colors['tpot'], hatch=hatches['tpot'], **style["bar"],
        label='TPOT')

# ==== 눈금 및 축 설정 ====
ax.set_yticks(y_pos)
ax.set_yticklabels(context_lengths, **style["tick"])
ax.set_ylabel('Context Lengths', **style["label"])


# ax.set_xticks(x_ticks)
ax.set_xlim(-np.log10(1e6), np.log10(1e2))  # x축 범위 설정
ax.set_xticklabels(x_tick_labels, **style["tick"])
ax.set_xlabel('Latency (Log Scale, ms)', fontsize=style["tick"]["fontsize"])

# ==== 눈금 제거 ====
ax.tick_params(axis='x', which='both', length=0)
ax.tick_params(axis='y', which='both', length=0)

# ==== 그리드 및 테두리 ====
ax.xaxis.grid(True, **style["grid"])
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# ==== 범례 ====
ax.legend(**style["legend"])

# ==== 레이아웃 및 저장 ====
plt.savefig('graph/horizantal_opposite_bar.jpg', format='jpg', bbox_inches="tight")
# plt.savefig('graph/horizantal_opposite_bar.pdf', format='pdf',bbox_inches="tight")
