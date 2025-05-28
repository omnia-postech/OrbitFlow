import pandas as pd, matplotlib.pyplot as plt, numpy as np, ast
import pandas as pd
from pathlib import Path

base_paths = [
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/SelectN/both_static_low",
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/SelectN/both_static_mid",
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/SelectN/both_static_high",
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/SelectN/both_static_veryhigh",

    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/SelectN/batch_dyn_low",
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/SelectN/batch_dyn_mid",
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/SelectN/batch_dyn_high",
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo5.5/SelectN/batch_dyn_veryhigh",

    "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/token_dyn_low",
    "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/token_dyn_mid",
    "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/token_dyn_high",
    "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/token_dyn_veryhigh",
]
# test_path = "/home/sychoy/vllm/outputs/benchmark/paper_main_exp/slo2.5/Flexgen/both_static_mid"

for base_path in base_paths : 
    sim_path = Path(base_path, "outputs.csv")
    output_path = Path(base_path, "slo_violation.csv")

    def load_metrics(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        num_cols = ["arrival_time","first_scheduled_time","finished_time",
                    "time_to_first_token","slo_threshold","slo_violations",
                    "stall_duration","decode_length","end_to_end_time",
                    "decode_time","time_per_output_token"]
        for col in num_cols:
            if col in df: df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("time_between_tokens","stall_times","stall_durations", "solver_time"):
            if col in df: df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x,str) else x)
        return df


    df = load_metrics(sim_path)
    slo_thr = df["slo_threshold"].mean()
    print(f"slo_threshold: {slo_thr}")

    slo_violation_results = []

    exception_total = 0
    for id, (req, solvers) in enumerate(zip(df["time_between_tokens"], df["solver_time"])):
        exception_num = 0
        slo_violation = 0
        deposit = 0

        time_list = set()

        gen_time = dict()
        cur_time = 0
        # real tbt 구하기
        for i, (time, solver_time) in enumerate(zip(req, solvers)):
            time = float(time)
            solver_time = float(solver_time)

            real_tbt = time-solver_time
            # real_tbt = time
            if (real_tbt) < 0 :
                # raise Exception(f"value: {real_tbt}, {time}, {solver_time}")
                exception_num += 1
                continue
                
            cur_time += time
            time_list.add(cur_time)
            gen_time[cur_time] = True

        out_time = dict()
        # output time 넣기
        for i in range(len(req)-exception_num):
            cur_time = (i+1) * slo_thr
            time_list.add(cur_time)
            out_time[cur_time] = True
        
        time_list = sorted(time_list)

        duplicate = False
        for time in time_list:
            # token gen 인지 확인
            if time in gen_time:
                deposit += 1

                # print(f"Add token in {time} and deposit {deposit}")
                duplicate = False
            
            # token output 인지 확인
            # deposit에서 토큰을 꺼냄 / 없다면 violation 증가
            if time in out_time:
                if (deposit > 0):
                    # print(f"Out deposit token in {time}")
                    deposit -= 1
                else :
                    # token gen이 길어져 output time이 여러번 와도 1번만 count 됨
                    if (duplicate is False):
                        slo_violation += 1
                        duplicate = True
                        print(f"violation : {time}")

        slo_violation_results.append({
            "request_id": f"request_{id}",
            "slo_violation": slo_violation,
            "exceptions": exception_num
        })

        exception_total += exception_num

        # print()

    violation_df = pd.DataFrame(slo_violation_results)
    violation_df.to_csv(output_path, index=False)
    # print("Saved to slo_violation_per_request.csv")
            
    print(f"exception total : {exception_total}")


