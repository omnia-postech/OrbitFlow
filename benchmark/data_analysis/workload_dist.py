import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── JSON에서 데이터 불러오기 ──
# with open("/home/heelim/vllm/benchmark/selected_traces/lambda1.0x_cv1.json", "r") as f:
#     data = json.load(f)



input_lengths = []
output_lengths = []


BASE_DIR = Path("/home/heelim/vllm/build_trace/traces/all_traces_v7_request_52_for_both_dyn")

# bim50 하위의 bim50*.json 파일만, metrics.json 은 제외
json_paths = [
    p for p in BASE_DIR.rglob("*/bim50/bim50*.json")
    if p.is_file() and not p.name.endswith(".metrics.json")
]

for jp in json_paths:
    try:
        print(jp)

        with jp.open("r") as f:
            data = json.load(f)
        # 요청별 input_length, output_length 추출
            
        print(f"open {jp}")
        requests = data.get("requests", {})
        input_lengths.extend(req["input_length"] for req in requests.values())
        output_lengths.extend(req["output_length"] for req in requests.values())
    except Exception as e:
        print(f"[WARN] {jp} 읽는 중 에러: {e}")
        continue

# print(input_lengths)

# 평균 계산
avg_in  = np.mean(input_lengths)
avg_out = np.mean(output_lengths)

# ── 히스토그램 그리기 ──
fig, ax = plt.subplots(figsize=(8, 6))

bins = np.linspace(0, max(max(input_lengths), max(output_lengths)), 50)

print(max(input_lengths))
print(max(output_lengths))

ax.hist(input_lengths,  bins=bins, density=True,
        color="#FF8C69", alpha=0.6,
        label=f"Input (avg={avg_in:.1f})")

ax.hist(output_lengths, bins=bins, density=True,
        color="#4DA6FF", alpha=0.6,
        label=f"Output (avg={avg_out:.1f})")

# 과학적 표기 (y축)
ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0), )
ax.yaxis.get_offset_text().set_fontsize(18)
ax.tick_params(axis='both',
    labelsize=20,
    length=0)

# 레이블 및 범위 설정
ax.set_xlabel("# Tokens", fontsize=20)
ax.set_ylabel("Density", fontsize=20)
ax.set_xlim(-200, bins.max() + 200)

# 범례
ax.legend(fontsize=20)

plt.savefig("figures/workload_dist.jpg", bbox_inches="tight", dpi=300)
plt.savefig("figures/workload_dist.pdf", bbox_inches="tight")
