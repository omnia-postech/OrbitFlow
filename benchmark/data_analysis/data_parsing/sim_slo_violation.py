import numpy as np
import pandas as pd
import ast
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

def load_metrics(path: Path) -> pd.DataFrame:
    """CSV에서 주요 메트릭과 리스트 컬럼을 로드"""
    df = pd.read_csv(path)
    num_cols = [
        "arrival_time", "first_scheduled_time", "finished_time",
        "time_to_first_token", "slo_threshold", "slo_violations",
        "stall_duration", "decode_length", "end_to_end_time",
        "decode_time", "time_per_output_token"
    ]
    for col in num_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("time_between_tokens", "stall_times", "stall_durations", "solver_time"):
        if col in df:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

def compute_slo_violation(
    id, times_list: list, solver_list: list, slo_thr: float
):
    # print(f"request_{id}")
    """
    real_tbt < 0인 경우 exception으로 카운트, 나머지는 gen_times 누적.
    out_times: gen_times[0]부터 gen_times[-1] + slo_thr까지 slo_thr 간격 생성.
    생성(gen)과 출력(out) 이벤트를 하나의 리스트로 합쳐 정렬하여 처리.
    """
    # real TBT 계산 및 예외 처리
    times_np = np.array(times_list, dtype=float)
    solver_np = np.array(solver_list, dtype=float)
    real_tbt = times_np - solver_np
    valid_mask = real_tbt >= 0
    exceptions = int((~valid_mask).sum())
    valid_times = times_np[valid_mask]
    if valid_times.size == 0:
        return 0, exceptions

    # 누적 생성 시각
    gen_times = np.cumsum(valid_times)
    start_time = gen_times[0]
    end_time = gen_times[-1] + slo_thr
    num_out = int(np.floor((end_time - start_time) / slo_thr)) + 1
    out_times = start_time + slo_thr * np.arange(num_out, dtype=float)

    # 이벤트 리스트 병합 & 정렬 (동시 시간일 땐 gen 먼저)
    events = [(t, 'gen') for t in gen_times] + [(t, 'out') for t in out_times]
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'gen' else 1))

    # deposit/violation 계산
    deposit = 0
    violations = 0
    duplicate_flag = False
    for time, typ in events:
        if typ == 'gen':
            deposit += 1
            duplicate_flag = False
            # if id == 8:
            #     print(f"Gen token {time}")
        else:  # 'out'
            if deposit > 0:
                deposit -= 1
                duplicate_flag = False
                # if id == 8:
                #     print(f"Out token {time}")
            else:
                if not duplicate_flag:
                    violations += 1
                    duplicate_flag = True
                    # if id == 8:
                    #     print(f"🚨 vioations: {time}")
    # if id == 8:
    #     print(f"🚨 vioations: {violations}")
    return violations, exceptions

# 작업 경로 설정
base_paths = [
"/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/both_static_low",
"/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/both_static_mid",
"/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/both_static_high",
"/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/both_static_veryhigh",

"/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/batch_dyn_low",
"/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/batch_dyn_mid",
"/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/batch_dyn_high",
"/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/batch_dyn_veryhigh",

"/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/token_dyn_low",
"/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/token_dyn_mid",
"/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/token_dyn_high",
"/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/token_dyn_veryhigh",

"/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/both_dyn_low",
"/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/both_dyn_mid",
"/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/both_dyn_high",
"/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/NoPrefetch/both_dyn_veryhigh",
]

for base in base_paths:
    base_path = Path(base)
    sim_path = base_path / "outputs.csv"
    output_path = base_path / "slo_violation.csv"

    if not sim_path.exists():
        logging.warning(f"Missing file: {sim_path}")
        continue

    df = load_metrics(sim_path)
    slo_thr = float(df["slo_threshold"].mean())

    results = []
    for req_id, (t_list, s_list) in enumerate(zip(df["time_between_tokens"], df["solver_time"])):
        violations, exceptions = compute_slo_violation(req_id, t_list, s_list, slo_thr)
        results.append({
            "request_id": f"request_{req_id}",
            "slo_violation": violations,
            "exceptions": exceptions
        })

    pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved results to {output_path} mean SLO threshold {slo_thr:.3f}")
