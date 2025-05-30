import numpy as np
import matplotlib.pyplot as plt

# 원래 데이터
x = np.array([0, 1, 2])
y = np.array([10, 1000, 820713377])
labels = [
    "Uniform\nDistance Driven",
    "Request-Wise\nDistance Driven",
    "Both"
]

# 스타일 정의
style = {
    "bar": {
        "edgecolor": "black",
        "linewidth": 1.5,
        "alpha": 0.8
    },
    "spine": {
        "color": "black",
        "alpha": 0.7,
        "linewidth": 1.5
    }
}

# Figure 생성
fig, ax = plt.subplots(figsize=(8, 5))

# Bar 차트 그리기
ax.bar(x, y, color="#FAC07D", **style["bar"])

# x축 설정
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=17)
ax.tick_params(axis='x', length=0)

# y축: 로그₂ 스케일 유지
ax.set_yscale('log', base=2)
ax.tick_params(axis='y', labelsize=18, length=0)
ax.set_ylabel("Number of Cases", fontsize=18, labelpad=8)

# 스파인 스타일
for spine in ax.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# 저장
plt.savefig("figures/search_space.jpg", bbox_inches="tight")
plt.savefig("figures/search_space.pdf", bbox_inches="tight")
