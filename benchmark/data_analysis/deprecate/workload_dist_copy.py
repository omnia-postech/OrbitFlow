import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter

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
        input_lengths.extend(r["input_length"]  for r in reqs.values())
        output_lengths.extend(r["output_length"] for r in reqs.values())
    except Exception as e:
        print(f"[WARN] {jp}: {e}")

# 평균 계산 및 'k' 단위 포맷
avg_in  = np.mean(input_lengths)  / 1000
avg_out = np.mean(output_lengths) / 1000
label_in  = f"Input (avg={avg_in:.1f}k)"
label_out = f"Output (avg={avg_out:.1f}k)"

# bins 정의
max_tok = max(max(input_lengths), max(output_lengths))
bins = np.linspace(0, max_tok, 43)

# ── 1×2 subplot 생성 ──
fig, (ax_in, ax_out,) = plt.subplots(
    2, 1, 
    # figsize=(12, 4), 
    figsize=(6, 8), 
    sharey=True,
    gridspec_kw={"wspace": 0.15, 
    "hspace": 0.25
    }
)

# 히스토그램
ax_in.hist(input_lengths,  bins=bins, density=True,
            color="#4DA6FF", alpha=0.6, label=label_in)

ax_out.hist(output_lengths, bins=bins, density=True,
            color="#FF8C69", alpha=0.6, label=label_out)

# x축 포맷터: 0→'0', 그 외→'{x/1000}K'
fmt = ticker.FuncFormatter(lambda x, pos: "0" if x == 0 else f"{int(x/1000)}K")
for ax in (ax_out, ax_in):
    ax.xaxis.set_major_formatter(fmt)
    # ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0), )

    # 1) 과학적 표기 켜기 (y축에 10^n 형태 지수 표시)
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useOffset=False)
    ax.ticklabel_format(style='sci',
                    axis='y',
                    scilimits=(-5, -5),
                    useOffset=False,)
                    # useMathText=True)

    # 2) 눈금 레이블만 정수로
    # ax.yaxis.set_major_formatter(
    #     ticker.FuncFormatter(lambda y, pos: f"{int(y)}")
    # )
    ax.set_yticks([0, 5e-05, 10e-5, 15e-5])           # y축은 0,1,2만
    ax.tick_params(axis='both', labelsize=20, length=0)
    ax.legend(fontsize=21, frameon=False, handletextpad=0.5,loc="upper right", 
              bbox_to_anchor=(1.05, 1.0),
              )
    ax.set_xlim(-3000, max_tok + 3000)

# 축 레이블
    
    # 그 아래, ax_out에도 y축 눈금 라벨이 나오도록
ax_out.yaxis.set_visible(True)

ax_out.tick_params(labelright=False, labelleft=True)
ax_out.set_xlabel("# Tokens", fontsize=22)
# ax_in .set_xlabel("# Tokens", fontsize=22)
ax_in.set_ylabel("Density", fontsize=22)
ax_out.set_ylabel("Density", fontsize=22)

txt = ax_in.yaxis.get_offset_text()
txt.set_fontsize(20)   # 원하는 크기로
txt = ax_out.yaxis.get_offset_text()
txt.set_fontsize(20)

plt.savefig("figures/workload_dist.jpg", bbox_inches="tight", dpi=300)
plt.savefig("figures/workload_dist.pdf", bbox_inches="tight")
