import matplotlib.pyplot as plt
import numpy as np

style = {
    "bar": {
        "edgecolor": "white",
        "linewidth": 3
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
        "loc": "lower right"
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
    'Train': '#f4a261',
    'Profile': '#91b6e1',
}

hatches = {
    'Train': '//',
    'Profile': 'o',
}

# data
Train_ = [735, 726, 555.55, 672.21]
Profile_ = [81.88,  356.45, 6329, 28113]

names = ['Miro', 'Conf\n+\nSample', 'Conf', 'None']
num_groups = len(names)
y = np.arange(num_groups)

fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.5
Max_value = 10000

# ==== 막대 그리기 ====
ax.barh(names, Train_, color=colors['Train'], hatch=hatches['Train'], **style["bar"],
        label='Train time')
ax.barh(names, Profile_, left=Train_, color=colors['Profile'], hatch=hatches['Profile'], **style["bar"],
        label='Profile time')

# ==== 텍스트 표시 ====
for i in range(num_groups):
    totals = Train_[i] + Profile_[i]
    if totals > Max_value:
        text = f"{totals:.1f} sec"
        ax.text(Max_value * 0.85, i, text,
                ha='left', va='center', **style["text"])

# ==== 축 및 타이틀 설정 ====
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=style["tick"]["fontsize"])

ax.tick_params(axis='x', labelsize=style["tick"]["fontsize"])
ax.set_xlabel('Time (sec)', **style["label"])

ax.set_title('Title', **style["title"])

# ==== 눈금 제거 ====
ax.tick_params(axis='x', which='both', length=0)
ax.tick_params(axis='y', which='both', length=0)

# ==== 그리드 및 테두리 ====
ax.set_xlim(right=10000)
ax.yaxis.grid(True, **style["grid"])
ax.xaxis.grid(True, **style["grid"])
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# ==== 범례 ====
ax.legend(**style["legend"])

plt.savefig('graph/horizantal_split_bar_graph.jpg', format='jpg', bbox_inches="tight")
# plt.savefig('graph/horizantal_split_bar_graph.pdf', format='pdf',bbox_inches="tight")
