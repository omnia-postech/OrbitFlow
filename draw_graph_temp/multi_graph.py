import matplotlib.pyplot as plt
import numpy as np

# style
style = {
    "bar": {
        "edgecolor": "white",
        "linewidth": 2.5
    },
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
    },
    "text": {
        "fontsize": 14,
        "weight": "bold"
    },
}

fig, axes = plt.subplots(2, 2, figsize=(30,15))

(ax1, ax2), (ax3, ax4) = axes

# 첫번째 그래프
colors1 = {
    'ttft': '#D81B60',
    'tpot': '#55A868',
}

hatches1 = {
    'ttft': '/',
    'tpot': '\\',
}

# ==== data ====
context_lengths = [128, 256, 512, 1024, 2048, 4096, 16384]
# x_ticks = [-1e6, -1e5, -1e4, -1e3, -1e2, -1e1, 0, 1e1, 1e2]
x_tick_labels = [r'$10^6$', r'$10^5$', r'$10^4$', r'$10^3$', r'$10^2$', r'$10^1$', '0', r'$10^1$', r'$10^2$']
           
ttft_latency = np.array([98, 99, 101, 1002, 10020, 19000, 978000])     # 왼쪽 (음수로 그릴 예정)
tpot_latency = np.array([40, 50, 55, 70, 80, 85, 90])     # 오른쪽

bar_height = 0.5
y_pos = np.arange(len(context_lengths))


# ==== 막대 그리기 ====
ttft_latency_log = np.log10(ttft_latency)
tpot_latency_log = np.log10(tpot_latency)
ax1.barh(y_pos, -ttft_latency_log,
        color=colors1['ttft'], hatch=hatches1['ttft'], **style["bar"],
        label='TTFT')
ax1.barh(y_pos, tpot_latency_log, 
        color=colors1['tpot'], hatch=hatches1['tpot'], **style["bar"],
        label='TPOT')

# ==== 눈금 및 축 설정 ====
ax1.set_yticks(y_pos)
ax1.set_yticklabels(context_lengths, **style["tick"])
ax1.set_ylabel('Context Lengths', **style["label"])

# ax1.set_xticks(x_ticks)
ax1.set_xlim(-np.log10(1e6), np.log10(1e2))  # x축 범위 설정
ax1.set_xticklabels(x_tick_labels, **style["tick"])
ax1.set_xlabel('Latency (Log Scale, ms)', fontsize=style["tick"]["fontsize"])

# ==== 눈금 제거 ====
ax1.tick_params(axis='x', which='both', length=0)
ax1.tick_params(axis='y', which='both', length=0)

# ==== 그리드 및 테두리 ====
ax1.xaxis.grid(True, **style["grid"])
ax1.set_axisbelow(True)

for spine in ax1.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# ==== 범례 ====
ax1.legend(**style["legend"])


# 두번째 그래프
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

# ==== draw line ====
ax2.plot(x_label, vllm, **style["line"],
        marker=marker['vllm'], linestyle=linestyle['vllm'], color=colors['vllm'], 
        label="vLLM")
ax2.plot(x_label, layerkv, **style["line"],
        marker=marker['layerkv'], linestyle=linestyle['layerkv'], color=colors['layerkv'], 
        label="LayerKV")

# ==== draw tick and title ====
ax2.set_ylim(-1000, 30000)
ax2.set_yticks(np.arange(0, 30000, 10000))
ax2.tick_params(axis='y', labelsize=style["tick"]["fontsize"])
ax2.set_ylabel("P99 TTFT (ms)", **style["label"])

ax2.tick_params(axis='x', labelsize=style["tick"]["fontsize"])

ax2.set_title("Llama2-7B", **style["title"])

# ==== grid and spine ====
ax2.yaxis.grid(True, **style["grid"])
ax2.set_axisbelow(True)

for spine in ax2.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# ==== 범례 ====
ax2.legend(**style["legend"])


# 3번째 그래프
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
x_tick_labels = [r'$10^6$', r'$10^5$', r'$10^4$', r'$10^3$', r'$10^2$', r'$10^1$', '0', r'$10^1$', r'$10^2$']
           
ttft_latency = np.array([98, 99, 101, 1002, 10020, 19000, 978000])     # 왼쪽 (음수로 그릴 예정)
tpot_latency = np.array([40, 50, 55, 70, 80, 85, 90])     # 오른쪽

bar_height = 0.5
y_pos = np.arange(len(context_lengths))


# ==== 막대 그리기 ====
ttft_latency_log = np.log10(ttft_latency)
tpot_latency_log = np.log10(tpot_latency)
ax3.barh(y_pos, -ttft_latency_log,
        color=colors['ttft'], hatch=hatches['ttft'], **style["bar"],
        label='TTFT')
ax3.barh(y_pos, tpot_latency_log, 
        color=colors['tpot'], hatch=hatches['tpot'], **style["bar"],
        label='TPOT')

# ==== 눈금 및 축 설정 ====
ax3.set_yticks(y_pos)
ax3.set_yticklabels(context_lengths, **style["tick"])
ax3.set_ylabel('Context Lengths', **style["label"])

ax3.set_xlim(-np.log10(1e6), np.log10(1e2))  # x축 범위 설정
ax3.set_xticklabels(x_tick_labels, **style["tick"])
ax3.set_xlabel('Latency (Log Scale, ms)', fontsize=style["tick"]["fontsize"])

# ==== 눈금 제거 ====
ax3.tick_params(axis='x', which='both', length=0)
ax3.tick_params(axis='y', which='both', length=0)

# ==== 그리드 및 테두리 ====
ax3.xaxis.grid(True, **style["grid"])
ax3.set_axisbelow(True)

for spine in ax3.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# ==== 범례 ====
ax3.legend(**style["legend"])

# 4번째 그래프

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

# ==== 막대 그리기 ====
offsets = [-1, 0, 1]
for idx, offset in enumerate(offsets):
    x_offset = x + offset * bar_width

    # Base: linear
    ax4.bar(x_offset, linear, width=bar_width,
           color=colors['linear'], hatch=hatches['linear'], **style["bar"], label='Linear' if idx == 0 else "") 

    # 추가 구성 요소
    if idx == 0:  # Flash Attention
        ax4.bar(x_offset, flash_attention, bottom=linear, width=bar_width, 
               color=colors['flash'], hatch=hatches['flash'], label='Flash attention', **style["bar"])
    elif idx == 1:  # Mask Refresh
        ax4.bar(x_offset, masking, bottom=linear, width=bar_width, 
               color=colors['masking'], hatch=hatches['masking'], label='Masking', **style["bar"])
        bottom2 = np.array(linear) + np.array(masking)
        ax4.bar(x_offset, sparse_attention, bottom=bottom2, width=bar_width, 
               color=colors['sparse'], hatch=hatches['sparse'], label='Sparse Attention', **style["bar"])
    elif idx == 2:  # Mask Cached
        ax4.bar(x_offset, sparse_attention, bottom=linear, width=bar_width, 
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
            ax4.text(xpos[j], Max_value*0.92, f"{totals[j]:.1f} μs",
                    ha='center', va='bottom', **style["text"])

# ==== 라벨 추가 ====
final_idx = 4  # '128'
xpos_final = [x[final_idx] + o * bar_width for o in offsets]
labels_final = ['Flash Attention', 'HiP (Mask Refresh)', 'HiP (Mask Cached)']
for i in range(3):
    ax4.text(xpos_final[i], 5, labels_final[i],
            ha='center', va='bottom',
            rotation=90,
            fontsize=18)

# ==== 축 및 타이틀 설정 ====
ax4.set_xticks(x)
ax4.set_xticklabels(context_lengths, fontsize=style["tick"]["fontsize"])
ax4.tick_params(axis='y', labelsize=style["tick"]["fontsize"])

ax4.set_xlabel('Context length (x1024 tokens)', **style["label"])
ax4.set_ylabel('Latency (μs)', **style["label"])
ax4.set_title('Latency Breakdown in Single Layer Decoding', **style["title"])

# ==== 그리드 및 테두리 ====
ax4.set_ylim(top=100)
ax4.yaxis.grid(True, **style["grid"])
ax4.xaxis.grid(True, **style["grid"])
ax4.set_axisbelow(True)

for spine in ax4.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# ==== 눈금 제거 ====
ax4.tick_params(axis='x', which='both', length=0)
ax4.tick_params(axis='y', which='both', length=0)

# ==== 범례 ====
ax4.legend(**style["legend"])

plt.savefig('graph/multi.jpg', format='jpg', bbox_inches="tight")