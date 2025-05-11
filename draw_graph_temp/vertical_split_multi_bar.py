import matplotlib.pyplot as plt
import numpy as np

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
        "loc": "upper left",
        # "linewidth": 2.5
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
    'linear': '#82c6a5',
    'flash': '#f7965c',
    'masking': '#8da6d8',
    'sparse': '#e7a2d3'
}

hatches = {
    'linear': '//',
    'flash': '.',
    'masking': '\\',
    'sparse': '*'
}

# ==== 데이터 ====
context_lengths = ['8', '16', '32', '64', '128']
num_groups = len(context_lengths)
x = np.arange(num_groups)

linear = [20, 20, 20, 20, 20]
flash_attention = [30, 65, 143.1, 283.6, 565.3]
masking = [20, 25, 35, 40, 50]
sparse_attention = [10, 13, 14, 14, 16]

bar_width = 0.3

Max_value = 100

fig, ax = plt.subplots(figsize=(12, 6))

# ==== 막대 그리기 ====
offsets = [-1, 0, 1]
for idx, offset in enumerate(offsets):
    x_offset = x + offset * bar_width

    # Base: linear
    ax.bar(x_offset, linear, width=bar_width,
           color=colors['linear'], hatch=hatches['linear'], **style["bar"], label='Linear' if idx == 0 else "") 

    # 추가 구성 요소
    if idx == 0:  # Flash Attention
        ax.bar(x_offset, flash_attention, bottom=linear, width=bar_width, 
               color=colors['flash'], hatch=hatches['flash'], label='Flash attention', **style["bar"])
    elif idx == 1:  # Mask Refresh
        ax.bar(x_offset, masking, bottom=linear, width=bar_width, 
               color=colors['masking'], hatch=hatches['masking'], label='Masking', **style["bar"])
        bottom2 = np.array(linear) + np.array(masking)
        ax.bar(x_offset, sparse_attention, bottom=bottom2, width=bar_width, 
               color=colors['sparse'], hatch=hatches['sparse'], label='Sparse Attention', **style["bar"])
    elif idx == 2:  # Mask Cached
        ax.bar(x_offset, sparse_attention, bottom=linear, width=bar_width, 
               color=colors['sparse'], hatch=hatches['sparse'], **style["bar"])

# ==== 텍스트 표시 ====
for i in range(num_groups):
    xpos = [x[i] + o * bar_width for o in offsets]
    totals = [
        linear[i] + flash_attention[i],
        linear[i] + masking[i] + sparse_attention[i],
        linear[i] + sparse_attention[i]
    ]
    for j in range(3):
        if totals[j] > Max_value:
            ax.text(xpos[j], Max_value*0.92, f"{totals[j]:.1f} μs",
                    ha='center', va='bottom', **style["text"])

# ==== 라벨 추가 ====
final_idx = 4  # '128'
xpos_final = [x[final_idx] + o * bar_width for o in offsets]
labels_final = ['Flash Attention', 'HiP (Mask Refresh)', 'HiP (Mask Cached)']
for i in range(3):
    ax.text(xpos_final[i], 5, labels_final[i],
            ha='center', va='bottom',
            rotation=90,
            fontsize=18)

# ==== 축 및 타이틀 설정 ====
ax.set_xticks(x)
ax.set_xticklabels(context_lengths, fontsize=style["tick"]["fontsize"])
ax.tick_params(axis='y', labelsize=style["tick"]["fontsize"])

ax.set_xlabel('Context length (x1024 tokens)', **style["label"])
ax.set_ylabel('Latency (μs)', **style["label"])
ax.set_title('Latency Breakdown in Single Layer Decoding', **style["title"])

# ==== 그리드 및 테두리 ====
ax.set_ylim(top=100)
ax.yaxis.grid(True, **style["grid"])
ax.xaxis.grid(True, **style["grid"])
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# ==== 눈금 제거 ====
ax.tick_params(axis='x', which='both', length=0)
ax.tick_params(axis='y', which='both', length=0)

# ==== 범례 ====
ax.legend(**style["legend"])

# ==== 저장 ====
plt.savefig("graph/vertical_split_multi_bar.jpg", format='jpg', bbox_inches="tight")
