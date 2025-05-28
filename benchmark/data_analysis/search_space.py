import numpy as np
import matplotlib.pyplot as plt

# 원래 데이터
x_orig    = np.array([0, 1, 2])
y_values  = np.array([10, 1000, 820713377])
x_labels  = ["Uniform\nDistance Driven",
             "Request-Wise\nDistance Driven",
             "Both"]

# ───────────────────────────────────────────────
# log₂ 스케일로 변환 후 보간
y_log2        = np.log2(y_values)
x_interp      = np.linspace(0, 2, 200)
y_interp_log2 = np.interp(x_interp, x_orig, y_log2)
y_interp      = 2 ** y_interp_log2   # 다시 원래 단위로

# ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

# 곡선 그리기
ax.plot(x_interp, y_interp, color="#C59FDB", linewidth=3)

# 원래 데이터 포인트 표시
ax.scatter(x_orig, y_values, color="#C59FDB", marker='o', s=80)

# x축
ax.set_xticks(x_orig)
ax.set_xticklabels(x_labels, fontsize=18)
ax.tick_params(axis='x', length=0)

# y축: 로그 스케일(2의 거듭제곱 눈금)
ax.set_yscale('log', base=2)
ax.tick_params(axis='y', labelsize=18, length=0)
ax.set_ylabel("Number of Cases", fontsize=18, labelpad=8)

# (원하면 y축 눈금을 2ⁿ 형태로 고정)
# ticks = [2**i for i in range(0, 30, 5)]  # 예: 1,32,1024,...
# ax.set_yticks(ticks)
# ax.set_yticklabels([f"$2^{i}$" for i in range(0,30,5)], fontsize=16)

# 스파인 스타일
style = {"spine": {"color": "black", "alpha": .7, "linewidth": 1.5}}
for spine in ax.spines.values():
    spine.set_edgecolor(style["spine"]["color"])
    spine.set_alpha(style["spine"]["alpha"])
    spine.set_linewidth(style["spine"]["linewidth"])

# 저장
plt.savefig("figures/search_space_log2.jpg", bbox_inches="tight")
plt.savefig("figures/search_space_log2.pdf", bbox_inches="tight")

