import numpy as np
import matplotlib.pyplot as plt

# Recomp ratio
recomp_ratios = [0, 25, 50, 75]  # (단위: %)

# Data transfer time
data_transfer = [10636, 7974, 5308, 2666]

# GPU 연산 시간들
qkv_proj = [81.375, 5854, 11598, 17537]
rotary_embedding = [3.391, 536.32, 1031, 1555]
flash_attention = [421.792, 416, 418, 418]
output_proj = [55.072, 52, 53, 53.344]

# x축 위치 설정
x = np.arange(len(recomp_ratios))
width = 0.35  # 막대 폭

fig, ax = plt.subplots(figsize=(10, 6))

# (1) Data Transfer Bar (왼쪽에 위치하도록 offset)
bar_data_transfer = ax.bar(
    x - width/2, 
    data_transfer, 
    width, 
    label='Data Transfer',
    color='#1f77b4'
)

# GPU 연산 시간 누적을 위한 bottom 계산
qkv_bottom = qkv_proj
rotary_bottom = [qkv_proj[i] + rotary_embedding[i] for i in range(len(qkv_proj))]
flash_bottom = [qkv_proj[i] + rotary_embedding[i] + flash_attention[i] for i in range(len(qkv_proj))]

# (2) GPU 연산 시간(누적 막대: qkv_proj, rotary embedding, flash-attention, output proj)
bar_qkv = ax.bar(
    x + width/2,
    qkv_proj,
    width,
    label='qkv_projection',
    color='#ff7f0e'
)
bar_rotary = ax.bar(
    x + width/2,
    rotary_embedding,
    width,
    bottom=qkv_bottom,
    label='rotary_embedding',
    color='#2ca02c'
)
bar_flash = ax.bar(
    x + width/2,
    flash_attention,
    width,
    bottom=rotary_bottom,
    label='flash_attention',
    color='#d62728'
)
bar_output = ax.bar(
    x + width/2,
    output_proj,
    width,
    bottom=flash_bottom,
    label='output_projection',
    color='#9467bd'
)

# x축 눈금 및 레이블 설정
ax.set_xticks(x)
ax.set_xticklabels([f'{r}%' for r in recomp_ratios])
ax.set_xlabel('Recomp ratio (%)')
ax.set_ylabel('Time (ms)')
# ax.set_title('')

# 범례(legend)
ax.legend()

# 레이아웃 조정
plt.tight_layout()
plt.savefig("recomp.png")
