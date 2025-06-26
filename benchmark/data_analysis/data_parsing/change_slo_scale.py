import os
import argparse
import ast
import pandas as pd

def update_slo_values(old_path: str, new_path: str, old_sc: float, new_sc: float):
    try:
        df = pd.read_csv(old_path)
        origin_threshold = float(df['slo_threshold'].iloc[0])
        new_threshold    = (origin_threshold / old_sc) * new_sc
        df['slo_threshold'] = new_threshold

        def count_violations(tbt_str):
            tbt_list = ast.literal_eval(tbt_str) if isinstance(tbt_str, str) else tbt_str
            return sum(t > new_threshold for t in tbt_list)

        df['slo_violations'] = df['time_between_tokens'].apply(count_violations)

        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        df.to_csv(new_path, index=False)
        print(f"{origin_threshold} -> {new_threshold}")
        print(f"[완료] 저장됨: {new_path}")
    except Exception as e:
        print(f"[ERROR] {old_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="SLO threshold을 다른 slo scale로 복사해서 저장합니다."
    )
    parser.add_argument("--old-sc",      type=float, required=True,
                        help="원본 slo scale (예: 2.5)")
    parser.add_argument("--new-sc-list", type=float, nargs="+", required=True,
                        help="새 slo scale 목록 (예: 1.5 1.0)")
    parser.add_argument("--base-path",   type=str,   required=True,
                        help="루트 디렉토리 (e.g. paper_main_exp_32k)")
    parser.add_argument("--is-arrival",  action="store_true",
                        help="arrival 모드로 실행하려면 지정")
    parser.add_argument("--arrival-rate-list", type=float, nargs="+",
                        help="(arrival 모드일 때 필수) arrival rate 목록")
    parser.add_argument("--cv-list",     type=int, nargs="+",
                        help="(arrival 모드일 때 필수) cv 목록")
    parser.add_argument("--metrics",     type=str,   nargs="+",
                        help="(arrival 모드가 아닐 때 필수) metrics 목록")
    parser.add_argument("--traces",      type=str,   nargs="+",
                        help="(arrival 모드가 아닐 때 필수) traces 목록")
    parser.add_argument("--methods",     type=str,   nargs="+", required=True,
                        help="처리할 method 목록")
    parser.add_argument("--arrival-tpl",  type=str, required=True,
                        default="32k_lambda{rate}x_cv{cv}",
                        help="arrival 모드에서 사용하는 서브디렉토리 템플릿. "
                             "{rate}와 {cv}가 반드시 포함되어야 합니다.")

    args = parser.parse_args()

    # 모드별 필수 인자 체크
    if args.is_arrival:
        if not args.arrival_rate_list or not args.cv_list:
            parser.error("--is-arrival 모드일 때는 --arrival-rate-list 와 --cv-list 가 필요합니다.")
    else:
        if not args.metrics or not args.traces:
            parser.error("arrival 모드가 아닐 때는 --metrics 와 --traces 가 필요합니다.")

    old_sc      = args.old_sc
    new_sc_list = args.new_sc_list
    base_path   = args.base_path
    methods     = args.methods

    if args.is_arrival:
        # arrival 모드
        for method in methods:
            for rate in args.arrival_rate_list:
                for cv in args.cv_list:
                    # 템플릿에 rate, cv를 넣어서 디렉토리 이름 생성
                    dir_name = args.arrival_tpl.format(rate=rate, cv=cv)
                    old_path = os.path.join(
                        base_path, f"slo{old_sc}", method,
                        dir_name, "outputs.csv"
                    )
                    if not os.path.isfile(old_path):
                        print(f"[WARN] 파일 없음: {old_path}")
                        continue
                    for new_sc in new_sc_list:
                        new_dir_name = args.arrival_tpl.format(rate=rate, cv=cv)
                        new_path = os.path.join(
                            base_path, f"slo{new_sc}", method,
                            new_dir_name, "outputs.csv"
                        )
                        update_slo_values(old_path, new_path, old_sc, new_sc)
    else:
        # non-arrival 모드
        for method in methods:
            for trace in args.traces:
                for metric in args.metrics:
                    old_path = os.path.join(
                        base_path, f"slo{old_sc}", method,
                        f"{trace}_{metric}", "outputs.csv"
                    )
                    if not os.path.isfile(old_path):
                        print(f"[WARN] 파일 없음: {old_path}")
                        continue
                    for new_sc in new_sc_list:
                        new_path = os.path.join(
                            base_path, f"slo{new_sc}", method,
                            f"{trace}_{metric}", "outputs.csv"
                        )
                        update_slo_values(old_path, new_path, old_sc, new_sc)

if __name__ == "__main__":
    main()