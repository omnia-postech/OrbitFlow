import matplotlib.pyplot as plt
import numpy as np

# style
style = {
    "line": {
        "linewidth": 3, 
        "markersize": 15
    },
    "tick": {
        "fontsize": 18
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
        "loc": "upper left"
    },
    "spine": {
        "color": "gray",
        "alpha": 0.5,
        "linestyle": "-",
        "linewidth": 2
    },
    "grid": {
        "color": "gray",
        "linestyle": "--",
        "linewidth": 2,
        "alpha": 0.5
    }
}

colors = {
    'vllm': '#a2e1bd',
    'layerkv': '#55cbcd',
}

marker = {
    'vllm': 'o',
    'layerkv': 's',
}

linestyle = {
    'vllm': '-',
    'layerkv': '--',
}

# ==== sample data ====
vllm = np.array([100, 120, 130, 150, 11000, 19000, 28000])
layerkv = np.array([90, 95, 100, 105, 110, 130, 300])

x_label = np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
x = np.arange(len(x_label))

fig, ax = plt.subplots(figsize=(12, 6))

# ==== draw line ====
ax.plot(x_label, vllm, **style["line"],
        marker=marker['vllm'], linestyle=linestyle['vllm'], color=colors['vllm'], 
        label="vLLM")
ax.plot(x_label, layerkv, **style["line"],
        marker=marker['layerkv'], linestyle=linestyle['layerkv'], color=colors['layerkv'], 
        label="LayerKV")

# ==== draw tick and title ====
ax.set_ylim(-1000, 30000)
ax.set_yticks(np.arange(0, 30000, 10000))
ax.tick_params(axis='y', labelsize=style["tick"]["fontsize"])
ax.set_ylabel("P99 TTFT (ms)", **style["label"])

# ax.set_xticks(x_label)
# ax.set_xlim(x_label[0], x_label[-1])
ax.tick_params(axis='x', labelsize=style["tick"]["fontsize"])

ax.set_title("Llama2-7B", **style["title"])

# ==== grid and spine ====
ax.yaxis.grid(True, **style["grid"])
# ax.xaxis.grid(True, **style["grid"])
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# ==== 범례 ====
ax.legend(**style["legend"])

plt.savefig('graph/line_graph_with_point.jpg', format='jpg', bbox_inches="tight")
# plt.savefig('graph/line_graph_with_point.pdf', format='pdf', bbox_inches="tight")