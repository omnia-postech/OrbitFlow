#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모든 .json 파일(단, *.metric.json 제외)을 열어
  1) batch_size 개 요청 중 input+output 토큰 합이 가장 큰 조합을 찾고
  2) (토큰 합 / 16) 을 올림한 값을 'peak_batch_blocks' 로 기록
  3) 파일을 덮어쓴다.
"""

import json, math, pathlib, sys
from collections import OrderedDict

BLOCK_SIZE_TOK = 16          # KV-블록 당 토큰 수 (vLLM 기본값)

def compute_peak_blocks(data: dict) -> int:
    """trace JSON 하나에서 peak batch 블록 수 계산"""
    batch_size = int(data.get("batch_size", 1))
    reqs = data.get("requests", {})
    totals = sorted(
        (r["input_length"] + r["output_length"] for r in reqs.values()),
        reverse=True,
    )
    top_sum = sum(totals[:batch_size])   # 가장 큰 batch_size 개만 합산
    return math.ceil(top_sum / BLOCK_SIZE_TOK)

def insert_after_vocab(orig: dict, peak_val: int) -> OrderedDict:
    """'vocab' 바로 뒤에 'peak_batch_blocks' 삽입해 새 OrderedDict 반환"""
    new = OrderedDict()
    for k, v in orig.items():
        new[k] = v
        if k == "vocab":
            new["peak_batch_blocks"] = peak_val
    if "vocab" not in orig:          # 안전장치: 없으면 맨 끝에 추가
        new["peak_batch_blocks"] = peak_val
    return new

def process_file(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    peak_blocks = compute_peak_blocks(data)
    data_out = insert_after_vocab(data, peak_blocks)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data_out, f, indent=2, ensure_ascii=False)

    print(f"{path.name:40}  -> peak_batch_blocks = {peak_blocks}")

def main():
    here = pathlib.Path(".")
    # 1) 모든 *.json
    json_files = here.glob("*.json")

    # 2) *.metric.json, *.metrics.json 은 제외
    SKIP_SUFFIXES = (".metric.json", ".metrics.json")
    targets = [p for p in json_files if not p.name.endswith(SKIP_SUFFIXES)]

    if not targets:
        print("변환할 .json 파일이 없습니다.")
        return

    for jp in sorted(targets):
        try:
            process_file(jp)
        except Exception as e:
            print(f"[ERROR] {jp.name}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
