# analyse_traces.py  ── category-specific percentile rules
import json, re, numpy as np, pandas as pd
from pathlib import Path

# ─────────────────────────── 0. 설정 ────────────────────────────
ROOT = Path("all_traces_v2")        # *.metrics.json 최상위 폴더

REPRESENTATIVE_PER_BUCKET = 1    # 버킷당 대표 trace 수
PPR_MAX = 2.0                  # ← 2 를 넘는 trace 는 분석에서 제외

# 0-1 압력 약어 → 레이블
LVL_MAP = {"vl":"low", "lo":"low", "ml":"mid", "md":"mid", "mh":"high", "hi":"high"}

# 0-2 카테고리별 퍼센타일 규칙
#   • 키 이름 : 카테고리
#   • 값 : dict {TJ_min_pct, TJ_max_pct, BJ_min_pct, BJ_max_pct}
#          ─ min_pct 는 “이 이상”,  max_pct 는 “이 이하” 로 해석
CATEGORY_RULES = {
    # batch_dynamic : BJ 크고 TJ 작을 때
    "batch_dynamic" : dict(TJ_max_pct=100, BJ_min_pct=96),

    # both_dynamic  : 둘 다 큰 trace
    "both_dynamic"  : dict(TJ_min_pct=75, BJ_min_pct=80),

    # both_static : 둘 다 아주 작을 때
    "both_static"   : dict(TJ_max_pct=10, BJ_max_pct=10),

    # token_dynamic : TJ 크고 BJ 작을 때
    "token_dynamic" : dict(TJ_min_pct=65, BJ_max_pct=15),
}

# ─────── 0. 삭제 규칙 정의 ──────────────────────────────────
# NOTE(HONG): trying to delete traces that are not useful
def should_delete(pressure, ovf):
    if pressure == "high" and ovf < 0.01:
        return True
    if pressure == "mid"  and ovf < 0.01:
        return True
    if pressure == "low"  and ovf <= 0.005:
        return True
    return False

def delete_family(stem: Path):
    """
    stem :  *.json 경로(Path) 또는 *.metrics.json 경로
            (확장자를 제외한 공통 prefix 로 처리)
    """
    for ext in (".json", ".metrics.json", ".png"):
        f = stem.with_suffix(ext)
        if f.exists():
            f.unlink()

# ───── 2. metrics 로드 ───────────────────────────────────────
rows = []
for mf in ROOT.rglob("*.metrics.json"):
    try:
        m = json.load(open(mf, encoding="utf-8"))
        tj = int(m.get("type1_count", 0))
        bj = int(m.get("type2_count", 0))
        ovf = float(m.get("OV_frac", 0.0))
        ppr = float(m.get("PPR", 0.0))  # PPR = Pressure-to-Performance Ratio
        
        # ① PPR 2.0 초과 trace → 그냥 건너뜀
        # NOTE(HONG): trace that are over 2.0 PPR takes too much time to run experiments
        if ppr > PPR_MAX:
            continue
        
        tag = re.search(r'_(vl|lo|ml|md|mh|hi)_', mf.name)
        if not tag:
            continue
        pressure = LVL_MAP[tag.group(1)]

        # ① OVF 기준 위반 → 세 파일 삭제 후 skip
        if should_delete(pressure, ovf):
            delete_family(mf.with_suffix(""))   # stem 전달
            continue

        # ② 통과한 trace 기록
        rows.append(dict(
            trace_path=str(mf).replace(".metrics.json", ".json"),
            TJ=tj, BJ=bj, OVF=ovf, PPR=ppr, pressure=pressure
        ))
    except Exception:
        continue

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError(f"No metrics found under {ROOT}")

# ───── 3. 퍼센타일 경계 계산 ──────────────────────────────────
needed = {v for r in CATEGORY_RULES.values() for v in r.values() if isinstance(v,(int,float))}
tj_pct = {p: np.percentile(df["TJ"], p) for p in needed}
bj_pct = {p: np.percentile(df["BJ"], p) for p in needed}

def ok(val, min_p=None, max_p=None, pct_map=None):
    if min_p is not None and val < pct_map[min_p]: return False
    if max_p is not None and val > pct_map[max_p]: return False
    return True

def classify(tj, bj):
    for cat, rule in CATEGORY_RULES.items():
        if ok(tj, rule.get("TJ_min_pct"), rule.get("TJ_max_pct"), tj_pct) and \
           ok(bj, rule.get("BJ_min_pct"), rule.get("BJ_max_pct"), bj_pct):
            return cat
    return "unclassified"

df["category"] = [classify(t, b) for t, b in zip(df["TJ"], df["BJ"])]
df.to_csv("trace_summary.csv", index=False)

# ───── 4. 버킷별 카운트 (unclassified 제외) 저장 ───────────────
count_tbl = (df[df["category"] != "unclassified"]
             .pivot_table(index="category", columns="pressure",
                           values="trace_path", aggfunc="count", fill_value=0))
count_tbl.to_csv("bucket_counts.csv")
print("\nTrace counts per bucket (saved → bucket_counts.csv)")
print(count_tbl)

# ───── 5. unclassified 제외 전체 trace 목록 저장 ───────────────
(df[df["category"] != "unclassified"]
   .sort_values(["category","pressure","trace_path"])
   .to_csv("traces_by_bucket.csv", index=False))

# ───── 6. 대표 trace 선택 & 저장 ───────────────────────────────
eligible = df[df["category"].isin(CATEGORY_RULES.keys())]
rep = (eligible.assign(dev=lambda d: abs(
                            d.OVF - d.groupby(["category","pressure"])
                                      ["OVF"].transform("median")))
               .sort_values("dev")
               .groupby(["category","pressure"])
               .head(REPRESENTATIVE_PER_BUCKET))
rep.to_csv("selected_12_traces.csv", index=False)
print("\n대표 trace 목록은 selected_12_traces.csv 로 저장 완료.")