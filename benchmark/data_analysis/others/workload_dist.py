import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.ticker as ticker

# ── JSON에서 데이터 불러오기 ──
input_lengths = []
output_lengths = []
BASE_DIR = Path("/home/heelim/vllm/build_trace/traces/sy_requests")

json_paths = [
    p for p in BASE_DIR.rglob("*.json")
    if p.is_file() and not p.name.endswith(".metrics.json")
]

for jp in json_paths:
    try:
        with jp.open("r") as f:
            data = json.load(f)
        reqs = data.get("requests", {})
        input_lengths.extend(r["input_length"] for r in reqs.values())
        output_lengths.extend(r["output_length"] for r in reqs.values())
    except Exception as e:
        print(f"[WARN] {jp}: {e}")

# 평균 계산 및 'k' 단위 포맷
avg_in = np.mean(input_lengths) / 1000
avg_out = np.mean(output_lengths) / 1000
label_in = f"Input    (avg={avg_in:.1f}k)"
label_out = f"Output (avg={avg_out:.1f}k)"

# bins 정의
max_tok = max(max(input_lengths), max(output_lengths))
bins = np.linspace(0, max_tok, 43)

# ── 단일 subplot 생성 ──
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

# 히스토그램 겹쳐 그리기
ax.hist(input_lengths, bins=bins, density=True,
        color="#4DA6FF", alpha=0.6, label=label_in)
ax.hist(output_lengths, bins=bins, density=True,
        color="#FF8C69", alpha=0.6, label=label_out)

# x축: 0 → "0", 그 외는 K단위
fmt = ticker.FuncFormatter(lambda x, pos: "0" if x == 0 else f"{int(x / 1000)}K")
ax.xaxis.set_major_formatter(fmt)

# y축: 과학적 표기
ax.ticklabel_format(style='sci', axis='y', scilimits=(-5, -5), useOffset=False)
ax.set_yticks([0, 5e-05, 10e-5, 15e-5])
ax.tick_params(axis='both', labelsize=20, length=0)

# 축 범위
ax.set_xlim(-3000, max_tok + 3000)

# 축 라벨
ax.set_xlabel("# Tokens", fontsize=27)
ax.set_ylabel("Density", fontsize=27)

# y축 지수 라벨 폰트 크기
txt = ax.yaxis.get_offset_text()
txt.set_fontsize(25)

# 범례
ax.legend(fontsize=21, frameon=False, handletextpad=0.5, loc="upper right",
          bbox_to_anchor=(1.05, 1.0))

# 저장
Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/workload_dist.jpg", bbox_inches="tight", dpi=300)
plt.savefig("figures/workload_dist.pdf", bbox_inches="tight")

print("✅ 그래프 저장 완료: figures/workload_dist.jpg")
