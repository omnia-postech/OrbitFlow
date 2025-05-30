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


def compute_slo_violation(req_id, times, solver, slo_thr):
    times_np   = np.asarray(times,  dtype=float)
    solver_np  = np.asarray(solver, dtype=float)
    real_tbt   = times_np - solver_np

    valid      = real_tbt >= 0
    exceptions = int((~valid).sum())
    valid_tbt  = times_np[valid]
    if valid_tbt.size == 0:
        return 0, exceptions

    gen_times  = np.cumsum(valid_tbt)
    start      = gen_times[0]
    end        = gen_times[-1] + slo_thr
    out_times  = start + slo_thr * np.arange(
                    int(np.floor((end - start) / slo_thr)) + 1, dtype=float)

    events = [(t, "gen") for t in gen_times] + \
             [(t, "out") for t in out_times]
    events.sort(key=lambda x: (x[0], 0 if x[1] == "gen" else 1))

    deposit, violations, dup = 0, 0, False
    for _t, typ in events:
        if typ == "gen":
            deposit += 1
            dup = False
        else:  # out
            if deposit > 0:
                deposit -= 1
                dup = False
            elif not dup:
                violations += 1
                dup = True
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

    for req_id, (tbt, solver, dl) in enumerate(
        zip(df["time_between_tokens"],
            df["solver_time"],
            df["decode_length"])
    ):
        try:
            ref_len = ref_reqs[f"request_{req_id}"]["output_length"]
        except KeyError:
            OVERVIEW_PNG = "outputs_overview.png"
            # 아직 시뮬레이션이 끝나지 않은 상태로 간주
            if not (exp_dir / OVERVIEW_PNG).exists():
                log.info(f"⏳  {exp_dir.relative_to(ROOT_DIR)} "
                         f"still running – skipped")
                return          # ★ 실험 전체를 건너뜀
            # 개요 PNG가 있는데도 KeyError면 진짜 오류
            log.error(f"Inconsistent reference JSON for {exp_dir} "
                      f"(missing request_{req_id})")
            return
        
        
        vio, exc = compute_slo_violation(req_id, tbt, solver, slo_thr)
        ref_len  = reference["requests"][f"request_{req_id}"]["output_length"]
        vio     += max(ref_len - dl - 1, 0)       # 미생성 토큰 보정
        failed = False
        if ref_len -1 != dl:
            failed = True   
        results.append(
            {"request_id": f"request_{req_id}",
             "slo_violation": vio,
             "exceptions":   exc, 
             "failed": failed,
             }
        )        

    out_csv = exp_dir / "slo_violation.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8-sig")
    log.info(f"✔  {exp_dir.relative_to(ROOT_DIR)} → slo_violation.csv "
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
        help="Only run on experiments that lack slo_violation.csv")
    args = parser.parse_args()

    # ① 특정 디렉터리 지정
    if args.exp_dirs:
        for arg in args.exp_dirs:
            process_experiment(to_exp_path(arg))
        return

    # ② --missing 플래그
    if args.missing:
        scan = [d for d in ROOT_DIR.rglob("outputs.csv")
                if not (d.parent / "slo_violation.csv").exists()]
    else:
        # ③ 기본: 전체 순회
        scan = ROOT_DIR.rglob("outputs.csv")

    for csv_path in sorted(scan):
        process_experiment(csv_path.parent)

if __name__ == "__main__":
    main()
