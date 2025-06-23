#!/usr/bin/env python3
# batch_slo_violation.py
import numpy as np
import pandas as pd
import ast, json, logging
from pathlib import Path
import argparse

# ───────────────────────────────────────────────
# 0. 전역 경로 & 로깅 설정
# ───────────────────────────────────────────────
ROOT_DIR = Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp")
REFERENCE_ROOT = Path("/home/heelim/vllm/benchmark/selected_traces")

Fix = True

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),                                   # 콘솔
        logging.FileHandler(ROOT_DIR / "slo_violation_batch.log",  # 파일
                            mode="a", encoding="utf-8")
    ],
)
log = logging.getLogger(__name__)

# ───────────────────────────────────────────────
# 1. 기존 함수들 (수정 없음)
# ───────────────────────────────────────────────
def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    num_cols = ["arrival_time", "first_scheduled_time", "finished_time",
                "time_to_first_token", "slo_threshold", "slo_violations",
                "stall_duration", "decode_length", "end_to_end_time",
                "decode_time", "time_per_output_token"]
    for c in num_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("time_between_tokens", "stall_times",
              "stall_durations", "solver_time"):
        if c in df:
            df[c] = df[c].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
    return df

to_see = 10000

def compute_slo_violation(req_id, times, solver, slo_thr):
    times_np   = np.asarray(times,  dtype=float)
    solver_np  = np.asarray(solver, dtype=float)
    real_tbt   = times_np - solver_np

    valid      = real_tbt >= 0
    exceptions = int((~valid).sum())
    valid_tbt  = times_np[valid]
    if valid_tbt.size == 0:
        return 0, exceptions
    
    deposit = 0
    prev_out = 0
    prev_gen = 0

    violations = 0

    for tbt in valid_tbt:
        # slo 보다 먼저 token 생성
        if prev_gen + tbt < prev_out + slo_thr:
            deposit += 1
            prev_gen += tbt
            if req_id == to_see:
                print(f"generate at {prev_gen} , deposit : {deposit} tbt : {tbt}")
        # 생성 시 바로 나감
        elif prev_gen + tbt == prev_out + slo_thr:
            prev_gen += tbt
            prev_out += slo_thr
            if req_id == to_see: 
                print(f"generate at {prev_gen} , out at {prev_out} , deposit : {deposit} tbt : {tbt}")
        # deposit에 있는 것 쓰거나, violation 
        else:
            # 다음 생성까지 deposit에 있는 것을 계속 꺼내써야함
            while(deposit > 0 and prev_out + slo_thr < prev_gen + tbt):
                # deposit에 있는 것 꺼내쓰기
                deposit -= 1
                prev_out += slo_thr
                if req_id == to_see: 
                    print(f"use deposite at {prev_out} deposit : {deposit} tbt : {tbt}")
            
            # 새 토큰 생성 전에 token 나가야 하면, violation 발생 후 생성 직후 out
            if (prev_out + slo_thr < prev_gen + tbt):
                violations += 1
                prev_gen += tbt
                prev_out = prev_gen
                if req_id == to_see:
                    print(f"violation at {prev_out} tbt : {tbt}")
            # 토큰 생성
            else :
                deposit += 1
                prev_gen += tbt
                if req_id == to_see:
                    print(f"generate at {prev_gen} , deposit : {deposit} tbt : {tbt}")

    return violations, exceptions

# ───────────────────────────────────────────────
# 2. 개별 실험 폴더 처리
# ───────────────────────────────────────────────
def process_experiment(exp_dir: Path):
    sim_path = exp_dir / "outputs.csv"
    if not sim_path.exists():
        log.warning(f"outputs.csv not found: {exp_dir}")
        return

    ref_json = REFERENCE_ROOT / f"{exp_dir.name}.json"
    if not ref_json.exists():
        log.warning(f"reference JSON missing: {ref_json}")
        return

    with open(ref_json, "r") as f:
        reference = json.load(f)
    ref_reqs = reference.get("requests", {})

    df      = load_metrics(sim_path)
    slo_thr = float(df["slo_threshold"].mean())
    results = []

    total_req_len = len(ref_reqs)
    
    if Fix:
        total_req_len = max(int(str(rid).split("_")[-1]) for rid in df["request_id"].unique()) + 1
        log.warning(f"Fixing total_req_len: (from {len(ref_reqs)})")

    for id in range(total_req_len):
        ref_len = ref_reqs[f"request_{id}"]["output_length"]
        try:
            req_row = df[df["request_id"] == f"request_{id}"].iloc[0]
        except:
            # print(f"not {id}")
            results.append(
                {
                    "request_id": f"request_{id}",
                    "slo_violation": ref_len-1,
                    "exceptions":   True, 
                    "failed": True,
                }
            ) 
            continue
        
        vio, exc = compute_slo_violation(
            id, 
            req_row["time_between_tokens"], 
            req_row["solver_time"], 
            slo_thr
        )

        dl = req_row["decode_length"]

        # print(f"{req_id} {vio}")
        ref_len  = reference["requests"][f"request_{id}"]["output_length"]
        vio     += max(ref_len - dl - 1, 0)       # 미생성 토큰 보정
        # print(f"{req_id} {vio}")
        failed = False
        if ref_len -1 != dl:
            failed = True   

        # slo_thr 보다 작은 tbt 값만 모아서 no_TD 리스트로
        no_td = [
            tbt for tbt in req_row["time_between_tokens"]
            if tbt >= slo_thr
        ]

        results.append(
            {"request_id": f"request_{id}",
             "slo_violation_with_TD": vio,
             "slo_violation_no_TD": len(no_td),
             "exceptions":   exc, 
             "failed": failed,
             }
        )        

    out_csv = exp_dir / "slo_violationv2.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8-sig")
    log.info(f"✔  {exp_dir.relative_to(ROOT_DIR)} → slo_violationv2.csv "
             f"(SLO={slo_thr:.3f})")

# ───────────────────────────────────────────────
# 3. 루트 전체 순회
# 1) 둘 다 빠졌던 실험만 재계산
# python batch_slo_violation.py --missing

# 2) NoPrefetch/both_static_high 만 다시 계산
# python batch_slo_violation.py NoPrefetch/both_static_high

# 3) 절대경로 두 개를 한 번에
# python batch_slo_violation.py \
#   /home/heelim/.../OursMinusPause/batch_dyn_low \
#   /home/heelim/.../Flexgen/token_dyn_mid

# ───────────────────────────────────────────────
def to_exp_path(arg: str) -> Path:
    """CLI 인자를 Path 로 변환 (절대·상대 경로 모두 지원)"""
    p = Path(arg).expanduser()
    return p if p.is_absolute() else ROOT_DIR / p

def main():
    parser = argparse.ArgumentParser(
        description="Compute SLO violations (all, missing, or selected).")
    parser.add_argument("exp_dirs", nargs="*",
        help="Experiment directories (absolute or relative to ROOT_DIR). "
             "If omitted, run on all experiments.")
    parser.add_argument("--missing", action="store_true",
        help="Only run on experiments that lack slo_violationv2.csv")
    args = parser.parse_args()

    # ① 특정 디렉터리 지정
    if args.exp_dirs:
        for arg in args.exp_dirs:
            process_experiment(to_exp_path(arg))
        return

    # ② --missing 플래그
    if args.missing:
        scan = [d for d in ROOT_DIR.rglob("outputs.csv")
                if not (d.parent / "slo_violationv2.csv").exists()]
    else:
        # ③ 기본: 전체 순회
        scan = ROOT_DIR.rglob("outputs.csv")

    skip_paths = [
        # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/both_dyn_mid"),
        # Path("/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NoPrefetch/token_dyn_low"),
        # 추가하고 싶은 경로들...
    ]

    for csv_path in sorted(scan):
        if csv_path.parent in skip_paths:
            print(f"Skipping {csv_path}")
            continue  # 이 경로는 넘김
        try:
            process_experiment(csv_path.parent)
        except Exception as e:
            log.error(f"{csv_path} ")

if __name__ == "__main__":
    main()
