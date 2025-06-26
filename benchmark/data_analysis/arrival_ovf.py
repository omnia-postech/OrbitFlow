import json
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


style = {
    "line": {"linewidth": 4, "markersize": 15},
}

root = Path("/home/heelim/vllm/build_trace/traces/requests_types_32k/all_traces_arrival_rate_CV")

# root 바로 아래에 있는 디렉토리만
immediate_subdirs = [p for p in root.iterdir() if p.is_dir()]

for d in immediate_subdirs:
    # ◆ arrival rates는 리스트로
    # arrival_list = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    arrival_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    ovf_list     = []

    for rate in arrival_list:
        
        json_path = f"{d}/lognormal_lambda{rate}x_cv1/lognormal_lambda{rate}x_cv1.metrics.json"
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"[WARN] 파일 없음, 건너뜁니다: {json_path}")
            ovf_list.append(np.nan)
            continue

        # "OV_frac" 키가 없으면 0.0
        OV_frac = data.get("OV_frac", 0.0)
        # 혹시 문자열 등이라면 float으로 변환 시도
        if not isinstance(OV_frac, (int, float)):
            try:
                OV_frac = float(OV_frac)
            except:
                print(f"[WARN] OV_frac 변환 실패: {OV_frac!r}")
                OV_frac = np.nan

        ovf_list.append(OV_frac)

    # 길이 검증
    if len(ovf_list) != len(arrival_list):
        raise RuntimeError("ovf_list와 arrival_list 길이가 일치하지 않습니다.")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        arrival_list,
        ovf_list,
        marker="o",
        color="#FF8C69",
        label="OV Fraction",
        **style["line"]
    )

    ax.set_xlabel("Arrival Rate", fontsize=14)
    ax.set_ylabel("OV Fraction", fontsize=14)
    ax.set_xticks(arrival_list)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{d}/arrival_ovf.jpg", bbox_inches="tight", dpi=300)

