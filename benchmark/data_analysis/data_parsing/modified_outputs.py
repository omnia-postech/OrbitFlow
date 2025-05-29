import pandas as pd
import json
from pathlib import Path

# ───────────────────────────────────────────────
# 1. 경로 설정
# ROOT_DIR = Path('/home/heelim/vllm/outputs/benchmark/paper_main_exp')
# SELECTED_TRACES_DIR = Path('/home/heelim/vllm/benchmark/selected_traces')

ROOT_DIR = Path('/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/batch_dyn_veryhigh/')
SELECTED_TRACES_DIR = Path('/home/heelim/vllm/benchmark/selected_traces')

# ───────────────────────────────────────────────
# 2. 모든 outputs.csv 파일 찾기 및 처리
for outputs_path in ROOT_DIR.rglob('outputs.csv'):
    trace = outputs_path.parent.name  # e.g., "both_static"

    # 대응 JSON 파일 경로
    trace_json_path = SELECTED_TRACES_DIR / f'{trace}.json'
    if not trace_json_path.exists():
        print(f"Warning: JSON for trace '{trace}' not found at {trace_json_path}")
        continue

    # CSV 로드
    df = pd.read_csv(outputs_path, dtype={"request_id": str})

    # JSON 로드 및 매핑 생성
    with open(trace_json_path, 'r', encoding='utf-8') as f:
        trace_data = json.load(f)

    # JSON 구조: {"requests": {"request_0": {"output_length": 24}, …}}
    # req_map: "request_0" → 24
    req_map = {
        req_id: info.get('output_length', None)
        for req_id, info in trace_data.get('requests', {}).items()
    }

    # JSON에서 output_length를 매핑
    df['json_output_length'] = df['request_id'].map(req_map).astype(float)

    # ───────────────────────────────────────────────
    # 1) finish_reason의 현재 dtype 확인
    print(df['finish_reason'].dtype)
    # 만약 'category' 라면…

    # 2) 문자열(object)로 변환
    df['finish_reason'] = df['finish_reason'].astype(str)

    # 3) 혹시 공백 문제가 있을 수도 있으니 strip
    df['finish_reason'] = df['finish_reason'].str.strip()

    # 4) mask 다시 계산
    mask = df['finish_reason'] == 'length_capped'
    mismatch = mask & (df['decode_length'] != df['json_output_length']-1)
    print("수정 대상 개수:", mismatch.sum())

    # 5) 할당
    df.loc[mismatch, 'finish_reason'] = 'not_enough_length_capped'

    for _, row in df.iterrows():
        print(
            f"request_id={row['request_id']}\t"
            f"decode_length(CSV)={row['decode_length']}\t"
            f"output_length(JSON)={row['json_output_length']}\t"
            f"finish_reason={row['finish_reason']}"
        )


    # ───────────────────────────────────────────────
    # 4. 결과 저장
    modified_path = outputs_path.parent / 'modified_outputs.csv'
    # 임시 컬럼 제거
    df.drop(columns=['json_output_length'], inplace=True)
    df.to_csv(modified_path, index=False, encoding='utf-8-sig')

    print(f"Processed: {outputs_path} → {modified_path}")
