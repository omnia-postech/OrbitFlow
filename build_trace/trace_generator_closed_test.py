# make_case_traces.py
# ------------------------------------------------------------
#  케이스 2·3·4 전용 trace 생성기
# ------------------------------------------------------------
import random, math, json
from typing import List, Dict, Tuple, Optional, Any, Callable
from pathlib import Path
from trace_generator import (
    RequestType, save_request_json, build_arrival_patterns,
    generate_trace, _ovf_for_blocks, memory_pressure_plot,_slug_ap
)

# ─────────────────────────────────────────────────────────────
# 공통 파라미터
# ─────────────────────────────────────────────────────────────
NUM_REQ     = 100          # 요청 개수
BATCH_SIZE  = 4            # 시스템 batch_size 와 동일해야 함
VOCAB       = (200, 30_000)
BASE_DIR    = Path("traces/case234")      # 모든 산출물 루트
# BASE_DIR    = Path("traces/case3_v1")      # 모든 산출물 루트
BLOCK_SIZE_TOK = 16

LEVEL_SPEC = {
        "vlow"     : dict(target_ovf = 0.05, tol = 0.010),  # 매우 여유
        "low"      : dict(target_ovf = 0.12, tol = 0.015),
        "mid_low"  : dict(target_ovf = 0.22, tol = 0.020),
        "mid"      : dict(target_ovf = 0.35, tol = 0.030),
        "mid_high" : dict(target_ovf = 0.55, tol = 0.040),
        "high"     : dict(target_ovf = 0.80, tol = 0.050),  # 상한 유지
    }    

def find_blocks_for_ovf(
    reqs: List[Tuple[int,int,int]],   # (arr, inp, out)    
    target_ovf: float,
    tol: float = 0.01,
    blk_size: int = 16,
    min_blk: int | None = None,
    max_blk_cap: int = 2**13,        # ← 8 192 blocks ≈ 127 k tok
) -> tuple[int, bool, float]:
    # ── 1) 초기 lo, hi ──────────────────────────────────────────
    peak_tok = max(inp + out for _, inp, out in reqs)
    lo = max(min_blk or 1, 1)
    # hi = max(lo, 4096)                        # 4k 블록을 최초 hi 로

    def ovf(b): return _ovf_for_blocks(reqs, b, blk_size)

    cur_lo_ovf = ovf(lo)

    # ── (1) 목표와 이미 근접한 경우 -----------------------------
    if abs(cur_lo_ovf - target_ovf) <= tol:
        return lo, False, cur_lo_ovf

    # ── (2) 트래픽이 너무 가벼워 목표보다 OVF가 작음 -------------
    if cur_lo_ovf < target_ovf:
        return lo, True, cur_lo_ovf  # cannot push OVF up without violating peak token size

    # ── (3) upper bound 찾기 -------------------------------------
    hi = max(lo, 4096)
    while ovf(hi) > target_ovf and hi < max_blk_cap:
        hi *= 2
    hi = min(hi, max_blk_cap)

    hi_ovf = ovf(hi)
    if hi_ovf > target_ovf:  # still high even at cap
        return hi, True, hi_ovf

    # ── (4) 이분 탐색 -------------------------------------------
    while lo < hi:
        mid = (lo + hi) // 2
        cur = ovf(mid)
        if abs(cur - target_ovf) <= tol:
            return mid, False, cur
        if cur > target_ovf:
            lo = mid + 1
        else:
            hi = mid

    final_ovf = ovf(lo)
    return lo, False, final_ovf


# arrival pattern: bimodal(버스티) 하나만 사용해도 충분
def _default_arrival_patterns(req_json):
    return build_arrival_patterns(req_json, scales=[0.5])   # 50% 타임라인 길이

# ─────────────────────────────────────────────────────────────
# 1. RequestType 카테고리 정의 헬퍼
# ─────────────────────────────────────────────────────────────
# ------------------------------------------------------------
#  Token-Static  (Case-2 · Case-4) : 짧은 출력
#   • 입력 길이  → short / mid / long 세 단계 모두 포함
#   • 출력 길이  → 8 ~ 256 token  (KV 증가 무시할 만큼 ‘정적’)
# ------------------------------------------------------------
def _mk_short_output_types(tag_prefix: str,
                           in_bins  = (256, 512, 1024, 2048, 4096, 8192, 16384), # short, mid, long
                           out_low  = 4,
                           out_high = 32):
    """
    • 입력 길이는 bin 리스트에서 선택  → short/mid/long 다양성 확보 (+- 10% 범위) 
    • 출력 길이는 8–256 토큰 → KV 증가 거의 없음  ⇒ token-static
    """
    rts = []
    for inp in in_bins:
        lo_in = max(16, int(inp * 0.9)//16*16)
        hi_in = max(16, int(inp * 1.1)//16*16)
        name  = f"{tag_prefix}_IN{lo_in}-{hi_in}_OUT{out_low}-{out_high}"
        rts.append(RequestType(name,
                               lo_in, hi_in,
                               out_low, out_high,
                               sampling_method="uniform"))
    return rts                          # 3 개

# ------------------------------------------------------------
#  Token-Dynamic  (Case-3) : 긴 출력
#   • 입력 길이  → short / mid (디코딩 증가 효과만 보려면 과하게 길 필요 X)
#   • 출력 길이  → 4 096 ~ 16 384 token  (최대 16 K)
# ------------------------------------------------------------
def _mk_long_output_types(tag_prefix: str,
                          in_bins  = (256, 512, 1024, 2048),   # short/mid
                          out_low  = 256,
                          out_high = 8192):
    """
    • 출력 512–1024 토큰 → decode 동안 KV 크게 팽창  ⇒ token-dynamic
    """
    rts = []
    for inp in in_bins:
        lo_in = max(16, int(inp * 0.9)//16*16)
        hi_in = max(16, int(inp * 1.1)//16*16)
        name  = f"{tag_prefix}_IN{lo_in}-{hi_in}_OUT{out_low}-{out_high}"
        rts.append(RequestType(name,
                               lo_in, hi_in,
                               out_low, out_high,
                               sampling_method="uniform"))
    return rts                          # 2 개

# ─────────────────────────────────────────────────────────────
# 2. 케이스별 trace 빌더
# ─────────────────────────────────────────────────────────────
def build_case2_traces():
    """Case-2  (token-static, batch-dynamic)"""
    # ① Request JSON
    rts = _mk_short_output_types("C2")
    probs = {rt: 1/len(rts) for rt in rts}
    req_json = BASE_DIR / "token_static_batch_dyn.json"
    save_request_json(
        path=str(req_json),
        request_types=probs,
        num_req=NUM_REQ,
        batch_size=BATCH_SIZE,
        vocab=VOCAB,
        static=False            # arrival 다양화 (batch-dynamic)
    )

    # req_data = json.loads(req_json.read_text())

    # ② Trace(s)
    for ap in _default_arrival_patterns(req_json):
        arr_times = ap.generate_arrival_times(NUM_REQ)
        trace = generate_trace(
            req_data = __import__("json").loads(req_json.read_text()),
            arr_obj  = ap,
            arr_times=arr_times,
            max_len  = 32_384,
            static   = False
        )
        
        # (A) req_tuples – OVF 계산용
        req_tuples = [(r.arrival_time, r.input_length, r.output_length)
                      for r in trace.requests.values()]

        # (B) stem – 파일명 공통 접두
        stem_base  = f"C2_{_slug_ap(ap)}"           # 예: C2_bim50
        out_dir    = BASE_DIR / "token_static_batch_dyn"
        out_dir.mkdir(parents=True, exist_ok=True)

    for lvl, cfg in LEVEL_SPEC.items():
        blk, unmet, ovf_actual= find_blocks_for_ovf(
            req_tuples,
            target_ovf = cfg["target_ovf"],
            tol        = cfg["tol"],
            blk_size   = BLOCK_SIZE_TOK
        )

        # 블록 수 적용 & 간단 스케줄 시뮬
        trace.num_gpu_blocks_override = blk
        trace.add_estimate_sched(
            max_model_len       = 32_384,
            num_gpu_blocks= blk,
            block_size    = BLOCK_SIZE_TOK,
            max_parallel  = BATCH_SIZE
        )

        # (D) 저장
        out_json = out_dir / f"{stem_base}_{lvl}.json"
        trace.save_to_json(out_json, skip_token_ids=True)

        # (E) 메모리-프레셔 메트릭 + PNG
        metrics = memory_pressure_plot(str(out_json), plot=True)
        (out_json.with_suffix(".metrics.json")
                ).write_text(json.dumps(metrics, indent=2))

        print(f"[C2] saved {out_json}  (blk={blk}, OVF={ovf_actual:.2f})")

def _make_batch_spaced_arrivals(outputs: list[int],
                                batch: int = 4) -> list[int]:
    """
    outputs : 길이 N 의 output-token 리스트  
    batch   : 시스템 batch_size  
    반환값  : batch_size 묶음마다 동일 time,  
             다음 묶음은 이전 묶음의 max(output) 만큼 뒤로 이동
    """
    arr, t = [], 0
    for i in range(0, len(outputs), batch):
        grp = outputs[i:i+batch]
        arr.extend([t]*len(grp))          # 같은 시간에 B개 arrival
        t += max(grp)                     # 간격 = max output
    return arr


# ─────────────────────────────────────────────────────────────
# 3. Case-3  (token-dynamic, batch-static)
# ─────────────────────────────────────────────────────────────
def build_case3_traces():
    rts   = _mk_long_output_types("C3")
    probs = {rt: 1/len(rts) for rt in rts}

    req_json = BASE_DIR / "token_dynamic_batch_static.json"
    save_request_json(
        path=str(req_json),
        request_types = probs,
        num_req       = NUM_REQ,
        batch_size    = BATCH_SIZE,
        vocab         = VOCAB,
        static        = False          # batch-static
    )
    req_data = json.loads(req_json.read_text())

    for ap in _default_arrival_patterns(req_json):
        # arr_times = ap.generate_arrival_times(NUM_REQ)

        # ① 길이 먼저 샘플링해 둔다
        req_vals = [req_data["requests"][k]
                    for k in sorted(req_data["requests"],
                                    key=lambda s: int(s.split("_")[1]))]
        outputs = [rv["output_length"] for rv in req_vals]

        arr_times = _make_batch_spaced_arrivals(outputs, BATCH_SIZE)

        trace = generate_trace(
            req_data = req_data,
            arr_obj  = ap,
            arr_times= arr_times,
            max_len  = 32_384,
            static   = True
        )
        req_tuples = [(r.arrival_time, r.input_length, r.output_length)
                      for r in trace.requests.values()]

        stem_base  = f"C3_{_slug_ap(ap)}"         # 예: C3_bim50
        out_dir    = BASE_DIR / "token_dynamic_batch_static"
        out_dir.mkdir(parents=True, exist_ok=True)

        for lvl, cfg in LEVEL_SPEC.items():
            blk, _, ovf_actual = find_blocks_for_ovf(
                req_tuples,
                target_ovf = cfg["target_ovf"],
                tol        = cfg["tol"],
                blk_size   = BLOCK_SIZE_TOK
            )
            trace.num_gpu_blocks_override = blk
            trace.add_estimate_sched(
                max_model_len        = 32_384,
                num_gpu_blocks = blk,
                block_size     = BLOCK_SIZE_TOK,
                max_parallel   = BATCH_SIZE
            )

            out_json = out_dir / f"{stem_base}_{lvl}.json"
            trace.save_to_json(out_json, skip_token_ids=True)

            metrics = memory_pressure_plot(str(out_json), plot=True)
            (out_json.with_suffix(".metrics.json")
                    ).write_text(json.dumps(metrics, indent=2))

            print(f"[C3] saved {out_json} (blk={blk}, OVF={ovf_actual:.2f})")


# ─────────────────────────────────────────────────────────────
# 4. Case-4  (token-static, batch-static)
# ─────────────────────────────────────────────────────────────
def build_case4_traces():
    rts   = _mk_short_output_types("C4", in_bins=(256, 1024, 4096))
    probs = {rt: 1/len(rts) for rt in rts}

    req_json = BASE_DIR / "token_static_batch_static.json"
    save_request_json(
        path=str(req_json),
        request_types = probs,
        num_req       = NUM_REQ,
        batch_size    = BATCH_SIZE,
        vocab         = VOCAB,
        static        = True
    )
    req_data = json.loads(req_json.read_text())

    for ap in _default_arrival_patterns(req_json):
        arr_times = ap.generate_arrival_times(NUM_REQ)

        trace = generate_trace(
            req_data = req_data,
            arr_obj  = ap,
            arr_times= arr_times,
            max_len  = 32_384,
            static   = True
        )
        req_tuples = [(r.arrival_time, r.input_length, r.output_length)
                      for r in trace.requests.values()]

        stem_base  = f"C4_{_slug_ap(ap)}"
        out_dir    = BASE_DIR / "token_static_batch_static"
        out_dir.mkdir(parents=True, exist_ok=True)

        for lvl, cfg in LEVEL_SPEC.items():
            blk, _, ovf_actual = find_blocks_for_ovf(
                req_tuples,
                target_ovf = cfg["target_ovf"],
                tol        = cfg["tol"],
                blk_size   = BLOCK_SIZE_TOK
            )
            trace.num_gpu_blocks_override = blk
            trace.add_estimate_sched(
                max_model_len        = 32_384,
                num_gpu_blocks = blk,
                block_size     = BLOCK_SIZE_TOK,
                max_parallel   = BATCH_SIZE
            )

            out_json = out_dir / f"{stem_base}_{lvl}.json"
            trace.save_to_json(out_json, skip_token_ids=True)

            metrics = memory_pressure_plot(str(out_json), plot=True)
            (out_json.with_suffix(".metrics.json")
                    ).write_text(json.dumps(metrics, indent=2))

            print(f"[C4] saved {out_json} (blk={blk}, OVF={ovf_actual:.2f})")

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    # build_case2_traces()
    build_case3_traces()
    # build_case4_traces()
