import os
import pandas as pd
import numpy as np
import ast

# 설정값들 정의
old_sc = 2.5         # 원본 slo 폴더 이름 (예: slo5)
new_sc_list = [0.5, 1.5, 3.5, 4.5, 5.5]        # 새로 만들 slo 폴더 이름 (예: slo10)
base_path = "/home/heelim/vllm/outputs/benchmark/paper_main_exp"
# test_path = "/home/sychoy/vllm/outputs/benchmark/paper_main_exp"

metrics = ["low","mid","high", "veryhigh"]
traces = ["both_static", "batch_dyn"]#, "token_dyn"]
# methods = ["Flexgen"]
# methods = ["NoPrefetch"]
methods = ["SelectN"]


def update_slo_values(new_sc_list, method, trace, metric):
    # 기존 파일 경로
    old_path = os.path.join(base_path, f"slo{old_sc}", method, f"{trace}_{metric}", "outputs.csv")
    
    for new_sc in new_sc_list:

        try:
        # 새로운 파일 경로
            new_dir = os.path.join(base_path, f"slo{new_sc}", method, f"{trace}_{metric}")
            new_path = os.path.join(new_dir, "outputs.csv")

            # 파일 존재 확인
            if not os.path.isfile(old_path):
                print(f"[경고] 파일 없음: {old_path}")
                continue

            # CSV 읽기
            df = pd.read_csv(old_path)

            origin_threshold = float(df['slo_threshold'].iloc[0])        
            
            new_threshold = (origin_threshold / old_sc) * new_sc
            df['slo_threshold'] = new_threshold

            def count_violations(tbt_str):
                tbt_list = ast.literal_eval(tbt_str) if isinstance(tbt_str, str) else tbt_str
                return sum(t > new_threshold for t in tbt_list)

            # slo_violation 계산
            df['slo_violations'] = df['time_between_tokens'].apply(count_violations)

            # 디렉토리 생성
            os.makedirs(new_dir, exist_ok=True)

            # 저장
            df.to_csv(new_path, index=False)
            print(f"{origin_threshold} -> {new_threshold}")
            print(f"[완료] 저장됨: {new_path}")
        except Exception as e:
            print(f"❌ Error: {old_path}")
            print(e)

# 전체 반복 실행
for method in methods:
    for trace in traces:
        for metric in metrics:
            update_slo_values(
                new_sc_list=new_sc_list,
                method=method,
                trace=trace,
                metric=metric,
            )
