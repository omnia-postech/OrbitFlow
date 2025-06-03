import re
from pathlib import Path
import math
import pandas as pd
import numpy as np
import ast
from pathlib import Path
from typing import List
import sys
import ast, json, logging

def count_miqcp_and_optimal(file_path: Path) -> tuple[int, int]:
    """
    outputs.log에서
    1) "Solving non-convex MIQCP" 등장 횟수를 count_solve_nonconvex 변수에 저장
    2) "--- Optimal solution (or best found) ---" 등장 횟수를 count_optimal_solution 변수에 저장
    그리고 두 값을 튜플로 반환합니다.
    """
    count_solve_nonconvex = 0
    count_optimal_solution = 0

    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "Solving non-convex MIQCP" in line:
                    count_solve_nonconvex += 1
                if "--- Optimal solution (or best found) ---" in line:
                    count_optimal_solution += 1
    except FileNotFoundError:
        print(f"[Error] 파일을 찾을 수 없습니다: {file_path}")

    return count_solve_nonconvex, count_optimal_solution


def get_last_step_from_log(file_path: Path) -> int | None:
    """
    vllm_msg.log에서 마지막으로 등장하는 'Step <숫자>' 부분의 숫자를 반환합니다.
    해당 패턴을 찾지 못하면 None을 반환합니다.
    """
    step_pattern = re.compile(r"Step\s+(\d+)")
    last_step = None

    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # 예시 라인 예) "CRITICAL 2025-05-31T17:10:05 test_distN.py:406] Step 102247, step_tokens = {...}"
                match = step_pattern.search(line)
                if match:
                    last_step = int(match.group(1))
    except FileNotFoundError:
        print(f"[Error] 파일을 찾을 수 없습니다: {file_path}")
    return last_step


def make_summerize(input_paths: list):
    results = []

    OUTPUT_CSV = input_paths[0].parent / "solver_summerize.csv"
    for path in input_paths:

        slo = path.parts[-3].replace('slo', '')
        method = path.parts[-2]
        trace_metric = path.name  # 'both_static_low'
        trace, metric = trace_metric.rsplit('_', 1)

        output_log = Path(path, "outputs.log")
        vllm_log = Path(path, "vllm_msg.log")

        total_solver_cnt, solver_out_cnt = count_miqcp_and_optimal(output_log)
        fall_back_cnt = abs(total_solver_cnt - solver_out_cnt)

        decode_step_cnt = get_last_step_from_log(vllm_log)

        results.append({
            'slo': slo,
            'method': method,
            'trace': trace,
            'metric': metric,
            'solver_cnt': total_solver_cnt,
            'fall_back_cnt': fall_back_cnt,
            'total_decode_cnt': decode_step_cnt
        })

    df_result = pd.DataFrame(results)
    df_result.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    for i in input_paths:
        print(f"{i}")
    print(f"Aggregated CSV saved to {OUTPUT_CSV}")
    print()


def main():

    BASE_DIR = Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp")
    

    input_base_paths = [
        # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1/Ours")
    ]

    if len(input_base_paths) > 0:
        
        for ours_dir in input_base_paths:
            if (ours_dir.name != "Ours"):
                print(f"Skip {ours_dir}")

            input_paths = []    

            # 각 trace_metric 디렉터리
            for trace_dir in sorted(ours_dir.iterdir(), key=lambda p: p.name):
                if not trace_dir.is_dir():
                    continue

                output_log = trace_dir / "outputs.log"
                vllm_log = trace_dir / "vllm_msg.log"
                
                if output_log.exists() and vllm_log.exists():
                    input_paths.append(trace_dir)
            
            make_summerize(input_paths)
        
        return 0

    # ──────────────────────────────────────────────────────
    # input_paths: slo 디렉터리 하위의 method별 trace_metric 디렉터리 필터링
        
    print("Run All Start")
    print()
    for slo_dir in BASE_DIR.glob('slo*'):
        if not slo_dir.is_dir():
            continue
        for method_dir in slo_dir.iterdir():
            if not method_dir.is_dir():
                continue

            if (method_dir.name != "Ours"):
                # print(f"Skip {method_dir}")
                continue

            input_paths = []    

            # 각 trace_metric 디렉터리
            for trace_dir in sorted(method_dir.iterdir(), key=lambda p: p.name):
                if not trace_dir.is_dir():
                    continue
                
                output_log = trace_dir / "outputs.log"
                vllm_log = trace_dir / "vllm_msg.log"
                
                if output_log.exists() and vllm_log.exists():
                    input_paths.append(trace_dir)
            
            if (len(input_paths) > 0):
                make_summerize(input_paths)
        

            # ──────────────────────────────────────────────────────
            # 추출 함수가 정의된 모듈 import
            # from your_module import extract_metrics

            # ──────────────────────────────────────────────────────
            # 데이터 수집 및 저장


if __name__ == "__main__":
    main()