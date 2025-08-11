import re
import ast
import csv
import sys
import os
from collections import defaultdict

def parse_log_file(log_file_path, output_csv_path):
    # 로그 줄을 파싱하기 위한 정규 표현식
    timestamp_pattern = r'^\w+\s+(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+\S+:\d+\]\s*(.*)'
    # distance_pattern = r'distance:\s*(\{.*?\})'  # 사용 안함
    forecast_pattern = (
        r"cache_engine\.py:1445\]\s*GPU-blk forecast:\s*"
        r"now=\d+\s+\+alloc=\d+\s+[−\-]free=\d+\s+⇒\s*after=(\d+)\s+⇒\s*total=(\d+)"
    )

    step_tokens_pattern = r"test_distN\.py:418\] Step \d+, step_tokens\s*=\s*(\{.*?\})"

    # 로그 파일 읽기
    try:
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"오류: 입력 파일 {log_file_path}이(가) 존재하지 않습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"로그 파일 읽기 중 오류 발생: {e}")
        sys.exit(1)

    is_next_layer = "NextLayer" in log_file_path

    # CSV 데이터 준비
    csv_data = []

    # 로그 줄 단위로 순회
    if is_next_layer:
        for line in lines:
            timestamp_match = re.match(timestamp_pattern, line)
            if not timestamp_match:
                continue
            message = timestamp_match.group(2)

            forecast_match = re.search(forecast_pattern, message)
            forecast_match = re.search(forecast_pattern, line)
            if forecast_match:
                after_val = int(forecast_match.group(1))
                total_val = int(forecast_match.group(2))

                if after_val > total_val:
                    continue
                csv_data.append({
                    'total': total_val,
                    'after': after_val,
                })
    else:
        i = 0
        while i < len(lines) - 1:
            line = lines[i]
            forecast_match = re.search(forecast_pattern, line)
            if forecast_match:
                total_val = int(forecast_match.group(1))
                used_val = None

                if is_next_layer:
                    step_line = lines[i + 1]
                    step_match = re.search(step_tokens_pattern, step_line)
                    if step_match:
                        try:
                            step_tokens = ast.literal_eval(step_match.group(1))
                            token_sum = sum(step_tokens.values())
                            used_val = math.ceil(token_sum / 16)
                        except Exception as e:
                            print(f"⚠️ step_tokens 파싱 오류: {e}")
                            used_val = None

                if used_val is not None and used_val <= total_val:
                    csv_data.append({
                        'total': total_val,
                        'after': used_val
                    })

                i += 2  # 다음 forecast 블록으로
            else:
                i += 1


    print(f"총 {len(csv_data)}개의 로그 항목이 파싱되었습니다.")
    # CSV 파일 작성
    try:
        with open(output_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['total_num', 'used_num']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_data:
                writer.writerow({
                    'total_num': row['total'],
                    'used_num': row['after']
                })
        print(f"CSV 파일이 {output_csv_path}에 생성되었습니다.")
    except Exception as e:
        print(f"CSV 파일 작성 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("사용법: python parse_log_to_csv.py <입력_로그_파일_디렉토리>")
        sys.exit(1)

    log_file_path = sys.argv[1] + "/vllm_msg.log"
    output_csv_path = sys.argv[1] + "/mem_util_output.csv"

    if not os.path.exists(log_file_path):
        print(f"오류: 입력 파일 {log_file_path}이(가) 존재하지 않습니다.")
        sys.exit(1)

    parse_log_file(log_file_path, output_csv_path)
