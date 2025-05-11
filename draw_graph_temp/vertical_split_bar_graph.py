import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

# ==== 스타일 설정 묶음 ====
style = {
    "bar": {
        "edgecolor": "white",
        "linewidth": 2.5
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
        "pad" :10
    },
    "legend": {
        "fontsize": 18,
        "loc": "upper center",
        "bbox_to_anchor": (0.5, 1.2),
        "ncol": 2,
    },
    "text": {
        "fontsize": 24,
        "weight": "bold",
        "ha":'center', 
        "va":'top'
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
    'static': '#91b6e1',
    'dynamic': '#f4a261',
}

hatches = {
    'static': '/',
    'dynamic': '\\',
}

# ==== data ====
labels = ['GPU', 'CPU', 'Mem', 'I/O', 'Total', 'W/o GPU', 'Total2']
labels_name = ['GPU', 'CPU', 'Mem', 'I/O', 'Total', 'W/o GPU', 'Total']
bottom_texts = ["TX2", "Xavier"] 

Static = np.array([0.302, 0.302, 0.302, 0.302, 0.302, 3.10, 3.89])
Dynamic = np.array([6.65, 1.186, 1.073, 0.18, 9.75 , 4.21, 11.21])

bar_width = 0.5

fig, ax = plt.subplots(figsize=(15,5))
for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(1.5)

# ==== draw bars ====
x = np.arange(len(labels))
ax.bar(x, Static, width=bar_width, 
       color=colors["static"], hatch=hatches["static"], label='Static', **style["bar"])
ax.bar(x, Dynamic, bottom=Static, width=bar_width, 
       color=colors["dynamic"], hatch=hatches["dynamic"], label='Dynamic', **style["bar"])

# ==== draw vertical line ====
split_index = 5  # Between 5th and 6th bar (0-indexed)
split_x = (x[split_index - 1] + x[split_index]) / 2 
ax.axvline(x=split_x, **style["grid"])


# ==== draw lower text ====
left_center_x = np.mean([x[0], x[4]])    # 'GPU'~'Total'
right_center_x = np.mean([x[5], x[6]])   # 'W/o GPU'~'Total2'

ax.text(left_center_x, -2, bottom_texts[0], **style["text"])
ax.text(right_center_x, -2, bottom_texts[1], **style["text"])


# ==== draw tick label and title ====
ax.set_xticks(x)
ax.set_xticklabels(labels_name, fontsize=style["tick"]["fontsize"])
ax.tick_params(axis='y', labelsize=style["tick"]["fontsize"])
ax.set_ylabel('Power (W)', **style["label"])

# ==== gird and out line ====
ax.set_ylim(top=16)
# 간격 조정
ax.set_yticks(np.arange(0, 16 + 1, 3))
ax.yaxis.grid(True, **style["grid"])
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# ==== delete tick ====
ax.tick_params(axis='x', which='both', length=0)
ax.tick_params(axis='y', which='both', length=0)

# ==== legend ====
ax.legend(**style["legend"])

plt.savefig('graph/vertical_split_bar_graph.jpg', format='jpg', bbox_inches="tight")
# plt.savefig('CarM100_Energy_Consumption.pdf', format='pdf', bbox_inches="tight")