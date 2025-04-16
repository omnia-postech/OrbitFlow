import numpy as np
import matplotlib.pyplot as plt

# Recomp ratio
recomp_ratios = [0, 25, 50, 75]  # (단위: %)

# Data transfer time 
data_transfer = np.array([10636, 7974, 5308, 2666]) / 1000

# GPU 연산 시간들
qkv_proj = np.array([81.375, 5854, 11598, 17537]) / 1000
rotary_embedding = np.array([3.391, 536.32, 1031, 1555]) / 1000
flash_attention = np.array([421.792, 416, 418, 418]) / 1000
output_proj = np.array([55.072, 52, 53, 53.344]) / 1000

cpu_write = np.full(len(recomp_ratios), 2.4 / 1000)

print(len(qkv_proj), len(rotary_embedding), len(flash_attention), len(output_proj), len(cpu_write))

# x축 위치 설정
x = np.arange(len(recomp_ratios))
width = 0.35  # 막대 폭

fig, ax = plt.subplots(figsize=(10, 6))

# (1) Data Transfer Bar (왼쪽에 위치하도록 offset)
bar_data_transfer = ax.bar(
    x - width/2, 
    data_transfer, 
    width, 
    label='Data transfer',
    color='#1f77b4'
)

# GPU 연산 시간 누적을 위한 bottom 계산
qkv_bottom = qkv_proj
rotary_bottom = [qkv_proj[i] + rotary_embedding[i] for i in range(len(qkv_proj))]
flash_bottom = [qkv_proj[i] + rotary_embedding[i] + flash_attention[i] for i in range(len(qkv_proj))]
output_bottom = [qkv_proj[i] + rotary_embedding[i] + flash_attention[i] + output_proj[i] for i in range(len(qkv_proj))]
# qkv_bottom = qkv_proj
# rotary_bottom = qkv_proj + rotary_embedding
# flash_bottom = qkv_proj + rotary_embedding + flash_attention
# output_bottom = qkv_proj + rotary_embedding + flash_attention + output_proj
# cpu_bottom = qkv_proj + rotary_embedding + flash_attention + output_proj + cpu_write

# (2) GPU 연산 시간(누적 막대: qkv_proj, rotary embedding, flash-attention, output proj)
bar_qkv = ax.bar(
    x + width/2,
    qkv_proj,
    width,
    label='QKV Proj',
    color='#ff7f0e'
)
bar_rotary = ax.bar(
    x + width/2,
    rotary_embedding,
    width,
    bottom=qkv_bottom,
    label='Rotary Embedding',
    color='#2ca02c'
)
bar_flash = ax.bar(
    x + width/2,
    flash_attention,
    width,
    bottom=rotary_bottom,
    label='Flash-Attention',
    color='#d62728'
)
bar_output = ax.bar(
    x + width/2,
    output_proj,
    width,
    bottom=flash_bottom,
    label='Output Proj',
    color='#9467bd'
)

bar_cpu_write = ax.bar(
    x + width/2,
    cpu_write,
    width,
    bottom=output_bottom,
    label='CPU Write',
    color='#8c564b'
)

# 각 바에 데이터 값 추가
def add_values_to_bars(bars, values):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + bar.get_y(),
            f'{value:.2f}',
            ha='center',
            va='center',
            fontsize=8
        )

# 데이터 값 추가
# add_values_to_bars(bar_data_transfer, data_transfer)
# add_values_to_bars(bar_qkv, qkv_proj)
# add_values_to_bars(bar_rotary, rotary_embedding)
# add_values_to_bars(bar_flash, flash_attention)
# add_values_to_bars(bar_output, output_proj)
# add_values_to_bars(bar_cpu_write, cpu_write)

# x축 눈금 및 레이블 설정
ax.set_xticks(x)
ax.set_xticklabels([f'{r}%' for r in recomp_ratios])
ax.set_xlabel('Recomputation ratio (%)')
ax.set_ylabel('Time (ms)')
# ax.set_title('')

# 범례(legend)
ax.legend()

# 레이아웃 조정
plt.tight_layout()
plt.savefig("recomp.png")
