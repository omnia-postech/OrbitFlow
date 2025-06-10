import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# 스타일 설정
style = {
    "line":   {
        "linewidth":4,
        "markersize":10
        },
    "bar": {
        "edgecolor": "black",
        "linewidth": 2,
        "width": 0.6,
        "alpha": 0.8
    },
    "tick": {
        "fontsize": 28
    },
    "label": {
        "fontsize": 28,
        "labelpad": 10
    },
    "title": {
        "fontsize": 30,
        "pad": 15
    },
    "legend": {
        "fontsize": 25,
        "loc": "upper left"
    },
    "spine": {
        "color": "black",
        "alpha": 0.9,
        "linestyle": "-",
        "linewidth": 2.5
    },
    "grid": {
        "color": "gray",
        "linestyle": "--",
        "linewidth": 3,
        "alpha": 0.7
    },
    "text": {
        "fontsize": 28,
    },
}

colors = {
    'compute': '#3CC58F',
    'communicate': '#FF8C69'
}

# 1x2 subplot 생성
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

# ===== 왼쪽 그래프: KV Cache =====
# 데이터 설정
x_labels = ['1k', '16k', '128k', '1M']
x_positions = [0, 1, 2, 3]
y_data = [3, 8, 40, 320]

# 축 절단을 위한 특별한 y값 매핑
y_mapped = []
for val in y_data:
    if val <= 50:
        y_mapped.append(val)
    else:  # 320인 경우
        y_mapped.append(50 + (val - 230) * 50/100)

# 막대 그래프 생성 (주황색)
bars = ax1.bar(x_positions, y_mapped, 
            color='#FF8C69',
            edgecolor=style["bar"]["edgecolor"],
            linewidth=style["bar"]["linewidth"],
            width=style["bar"]["width"],
            alpha=style["bar"]["alpha"])

# 제목과 축 라벨 설정
ax1.set_title('Model: LLaMA3-70B', **style["title"])
ax1.set_xlabel('Sequence Length', **style["label"])
ax1.set_ylabel('KVCache Size (GB)', **style["label"])

# x축 설정
ax1.set_xticks(x_positions)
ax1.set_xticklabels(x_labels, **style["tick"])

# y축 설정 - 절단된 축
ax1.set_ylim(0, 110)
ax1.set_yticks([])

ax1.text(-0.35, 55, '50', ha='left', va='center', **style["text"])
ax1.text(-0.35, 102, '400', ha='left', va='center', **style["text"])

# 50GB 기준선 (점선)
ax1.axhline(y=50, **style["grid"])

# 그리드 제거
ax1.grid(False)

# 막대 위에 실제 값 표시
actual_values = [1, 5, 40, 320]
for i, (bar, value) in enumerate(zip(bars, actual_values)):
    if i == 3:  # 마지막 막대 (320)
        ax1.text(bar.get_x() + bar.get_width()/2 -0.01, bar.get_height() + 1, 
               '320', ha='center', va='bottom', 
               **style["text"])

for spine in ax1.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# 눈금 제거
ax1.tick_params(axis='x', which='both', length=0)
ax1.tick_params(axis='y', which='both', length=0)

fig.text(0.255, -0.05, '(a)', ha='center', va='center', fontsize = style["title"]["fontsize"])

# ===== 오른쪽 그래프: Communication and Computation =====
# 데이터
context_length_int = [2, 4, 8, 16]
context_lengths = ['2k', '4k', '8k', '16k']

num_groups = len(context_lengths)
x = np.arange(num_groups)

# x 값은 context_lengths에 1024를 곱한 값
x_values = [length * 1024 for length in context_length_int]

# compute = 1.23e-6 * x + 3.54e-2
compute = [1.23e-6 * x_val + 4.00e-2 for x_val in x_values]

# communicate = 3.98e-5 * x + 4.99e-2  
communicate = [2.0e-5 * x_val + 4.0e-2 for x_val in x_values]

# 비율 계산 (첫 번째와 마지막 지점에서)
ratio_first = communicate[0] / compute[0]
ratio_last = communicate[-1] / compute[-1]

markers = ['o','s','^']
# 막대 그리기

ax2.plot(x, compute, #width=bar_width, 
            **style["line"],
        marker=markers[0], 
        color=colors['compute'],
        label='Comp. Time')
ax2.plot(x, communicate, #width=bar_width, 
            **style["line"],
        marker=markers[1], 
        color=colors['communicate'],
        label='Comm. Time')

for i in [0, -1]:  # 첫 번째와 마지막 지점
    x_pos = x[i]
    y_compute = compute[i]
    y_communicate = communicate[i]
    
    # 두 점의 중간 지점
    y_mid = (y_compute + y_communicate) / 2
    
    # 비율 계산 및 텍스트 크기 추정
    ratio = communicate[i] / compute[i]
    ratio_text = f'{ratio:.2f}x'
    
    # 텍스트 높이 추정 (데이터 좌표계에서)
    y_range = y_communicate - y_compute
    text_gap = y_range * 0.1  # 텍스트 공간을 위한 간격
    
    # # 위쪽 화살표 (communicate 쪽에서 중간까지)
    # arrow_top = FancyArrowPatch((x_pos, y_communicate), 
    #                            (x_pos, y_mid + text_gap/2),
    #                            arrowstyle='<-', 
    #                            mutation_scale=15,
    #                            color='black',
    #                            linewidth=1.5)
    # ax2.add_patch(arrow_top)
    
    # # 아래쪽 화살표 (중간에서 compute 쪽까지)
    # arrow_bottom = FancyArrowPatch((x_pos, y_mid - text_gap/2), 
    #                               (x_pos, y_compute),
    #                               arrowstyle='->', 
    #                               mutation_scale=15,
    #                               color='black',
    #                               linewidth=1.5)
    # ax2.add_patch(arrow_bottom)

    ax2.add_patch(
        FancyArrowPatch((x_pos, y_communicate), 
                    (x_pos, y_compute),
                    arrowstyle='<->', 
                    mutation_scale=15,
                    color='black',
                    linewidth=2.5)
    )
    
    # 비율 텍스트 추가 (배경 없이)
    if i == 0:
        x_pos += 0.4
    else:
        x_pos -= 0.35

    ax2.text(x_pos, y_mid, ratio_text, 
            ha='center', va='center', 
            fontsize = 27)
    
    
# 축 및 타이틀 설정
ax2.set_xticks(x)
ax2.set_xticklabels(context_lengths, **style["tick"])
ax2.tick_params(axis='y', labelsize=style["tick"]["fontsize"])

ax2.set_xlabel('Sequence length', **style["label"])
ax2.set_ylabel('Latency (s)', **style["label"])
ax2.set_title('Com. and Comp. time', **style["title"])

# 테두리 설정
for spine in ax2.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# 눈금 제거
ax2.tick_params(axis='x', which='both', length=0)
ax2.tick_params(axis='y', which='both', length=0)

# 범례
ax2.legend(**style["legend"])


fig.text(0.775, -0.05, '(b)', ha='center', va='center', fontsize = style["title"]["fontsize"])


# 레이아웃 조정
plt.tight_layout()

# 저장
plt.savefig("figures/combined_graphs.jpg", format='jpg', bbox_inches="tight")
plt.savefig("figures/combined_graphs.pdf", format='pdf', bbox_inches="tight")
