import numpy as np
import pandas as pd
import ast, json, logging
from pathlib import Path
import argparse

def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    num_cols = ["arrival_time", "finished_time",
                "time_to_first_token", "slo_threshold"]
    for c in num_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("time_between_tokens", "solver_time"):
        if c in df:
            df[c] = df[c].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
    return df


origin_path = Path("/home/heelim/vllm/outputs/benchmark/figure_4_2_token_deposit/outputs.csv")
origin_df = load_metrics(origin_path)

# 1. origin_df를 deep copy로 new_df 생성 (origin_df는 row 1개만 있음)
new_df = origin_df.copy(deep=True)

# 2. new_df에 첫번째 row의 복사본을 두번째 row로 추가
new_df = pd.concat([new_df, origin_df.copy(deep=True)], ignore_index=True)

# 첫번째 row에 접근하기 위한 인덱스
idx0 = 0
idx1 = 1  # 두번째 row 인덱스

# 3. origin_df 첫번째 row의 time_between_tokens의 860번부터 1060번(포함)까지 추출
orig_tbt = origin_df.at[idx0, "time_between_tokens"]
# Python slicing은 end-exclusive이므로 1060 포함하려면 1061까지 슬라이스
running_tbt = orig_tbt[860:1061]  # 길이 = 1061 - 860 = 201

# 4. new_df 첫번째 row의 time_between_tokens 수정:
#    index 99부터 99+i 위치에 running_tbt[i*2] 값을 할당 (i=0..99)
new_tbt_0 = new_df.at[idx0, "time_between_tokens"].copy()  # 기존 리스트 복사
for i in range(100):
    new_tbt_0[99 + i] = running_tbt[i * 2]
new_df.at[idx0, "time_between_tokens"] = new_tbt_0

# 5. 변경된 time_between_tokens 값만큼 첫번째 row의 finished_time에 더하기
#    -> 변화된 구간은 99..198 인덱스에서 running_tbt[i*2] 값으로 대체된 부분
added_time = sum(running_tbt[i * 2] for i in range(100))
new_df.at[idx0, "finished_time"] = new_df.at[idx0, "finished_time"] + added_time

# 6. new_df 두번째 row의 time_between_tokens을
#    running_tbt[i*2] (i=0..99) 값으로 길이 100 리스트로 변경
new_tbt_1 = [running_tbt[i * 2] for i in range(100)]
new_df.at[idx1, "time_between_tokens"] = new_tbt_1

# 7. new_df 두번째 row의 arrival_time을
#    (첫번째 row의 time_to_first_token) + sum(first_row.time_between_tokens[0:99]) 으로 수정
first_row_ttf = new_df.at[idx0, "time_to_first_token"]
sum_first_0_98 = sum(new_tbt_0[0:99])  # 인덱스 0부터 98까지
new_df.at[idx1, "arrival_time"] = first_row_ttf + sum_first_0_98

# 8. new_df 두번째 row의 time_to_first_token을 첫번째 row의 time_between_tokens[99]로 변경
new_df.at[idx1, "time_to_first_token"] = new_tbt_0[99]

# 결과 확인
print("=== Modified DataFrame ===")
print(new_df)

# 필요 시 CSV로 저장
out_path = Path("/home/heelim/vllm/outputs/benchmark/figure_4_2_token_deposit/modified_outputs.csv")
new_df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"Modified DataFrame saved to: {out_path}")