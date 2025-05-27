# trace_jump_hist.py
# ──────────────────────────────────────────────────────────
# * TJ   = type-1 (token-jump)  count
# * BJ   = type-2 (batch-jump)  count
#   ─►  x-축 = jump 개수(정수) , y-축 = 그 개수를 가진 trace 수
# ──────────────────────────────────────────────────────────
import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# ① metrics 파일이 모여 있는 “최상위” 폴더를 맞춰 주세요
root = Path("all_traces_v6")

metrics = list(root.rglob("*.metrics.json"))
if not metrics:
    raise FileNotFoundError(f"*.metrics.json not found under {root}")

tj, bj = [], []
for mf in metrics:
    try:
        with open(mf, encoding="utf-8") as fp:
            m = json.load(fp)
        tj.append(int(m.get("type1_count", 0)))
        bj.append(int(m.get("type2_count", 0)))
    except Exception:
        continue

# ② TJ 히스토그램 ------------------------------------------------------------
plt.figure(figsize=(7,4))
bins = np.arange(0, max(tj)+2) - 0.5          # 정수 bin (0,1,2,…)
plt.hist(tj, bins=bins, edgecolor="black")
plt.xlabel("TJ_count per trace")
plt.ylabel("Number of traces")
plt.title(f"TJ distribution  (N={len(tj)})")
plt.tight_layout()
plt.savefig("trace_analysis/tj_hist.png", dpi=150)
plt.close()

# ③ BJ 히스토그램 ------------------------------------------------------------
plt.figure(figsize=(7,4))
bins = np.arange(0, max(bj)+2) - 0.5
plt.hist(bj, bins=bins, edgecolor="black", color="tab:orange")
plt.xlabel("BJ_count per trace")
plt.ylabel("Number of traces")
plt.title(f"BJ distribution  (N={len(bj)})")
plt.tight_layout()
plt.savefig("trace_analysis/bj_hist.png", dpi=150)
plt.close()

print("[saved] tj_hist.png, bj_hist.png")
