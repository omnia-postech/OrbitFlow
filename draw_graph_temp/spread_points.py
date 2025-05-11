import matplotlib.pyplot as plt

style = {
    "tick": {
        "fontsize": 18
    },
    "label": {
        "fontsize": 24,
        "weight": "bold"
    },
    "spine": {
        "color": "gray",
        "alpha": 0.5,
        "linewidth": 2
    }
}

# data
x = [0, 7_000, 10_000]
y = [60, 35, 10]

markers = ['D', 'v', 'p']
colors = ['crimson', 'purple', 'indigo']
sizes = [400, 500, 500]

fig, ax = plt.subplots(figsize=(8, 4))

# draw scatter plot
for i in range(len(x)):
    ax.scatter(x[i], y[i], marker=markers[i], s=sizes[i], color=colors[i], edgecolor='black', linewidth=1.5)

# 축 설정
ax.set_xlim(-1000, 11000)
ax.set_ylim(0, 100)

ax.set_xticks([0, 5000, 10000])
ax.set_xticklabels(['0K', '5K', '10K'], fontsize=style["tick"]["fontsize"])
ax.set_yticks([0, 50, 100])
ax.set_yticklabels([0, 50, 100], fontsize=style["tick"]["fontsize"])

ax.set_ylabel("Split\nCIFAR-100", **style["label"])

# ==== 눈금 제거 ====
ax.tick_params(axis='x', length=0)
ax.tick_params(axis='y', length=0)

# ==== 그리드 및 테두리 ====
for spine in ax.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])


plt.savefig('graph/split_cifar100_scatter.jpg', format='jpg', bbox_inches='tight')
# plt.savefig('graph/split_cifar100_scatter.pdf', format='pdf', bbox_inches='tight')

