# =======================================================================
# unified_trace_toolkit.py
#   • request.json   생성  (Request-generator)
#   • trace          생성  (Arrival-generator + Trace builder)
# =======================================================================
import json, random, math, heapq, os, warnings
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Callable
import numpy as np
import math
import matplotlib.pyplot as plt

import numpy as np
try:
    from transformers import AutoTokenizer          # (token-bank 선택 시 사용)
except ImportError:
    AutoTokenizer = None

from collections import OrderedDict
inline = lambda obj: json.dumps(obj, ensure_ascii=False,
                                separators=(", ", ": "))

# ────────────────────────────────────────────────────────────────────
# 0. 공통 상수 & 헬퍼
# ────────────────────────────────────────────────────────────────────
PROFILED_A = 1.0017431830666432e-06
PROFILED_B = 0.049519613282613506

BLOCK_SIZE_TOK = 16        # vLLM KV-block 기본 크기

def _sample_poisson(lmbd: float) -> int:
    """Knuth Poisson 샘플러."""
    L, p, k = math.exp(-lmbd), 1.0, 0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1

def clamp(v, lo, hi): return max(lo, min(v, hi))

# ────────────────────────────────────────────────────────────────────
# 1. ❶ request.json 생성기
# ────────────────────────────────────────────────────────────────────
# 1-1) Request 객체 (arrival 없음)
@dataclass
class RawRequest:
    category: str
    input_length: int
    output_length: int
    token_ids: List[int]
    slo: Optional[float] = None


# 1-2) RequestType  ─ 입력/출력 길이 샘플러
class RequestType:
    """
    Args:
        category_name (str): A descriptive name for this request type.
        min_input_tokens (int): Minimum input token length.
        max_input_tokens (int): Maximum input token length.
        min_output_tokens (int): Minimum output token length.
        max_output_tokens (int): Maximum output token length.
        sampling_method (str): How to sample input/output lengths. One of:
            - "uniform": simple randint within [min, max]
            - "lognormal_approx": approximate lognormal fit based on [min, max]
            - "ratio": for tasks where output is a fraction of input
            - "default": same as uniform or your existing approach
        **kwargs: Optional parameters to control the sampling. For example:
            - mu_in, sigma_in, mu_out, sigma_out for lognormal
            - ratio_min, ratio_max for ratio-based, etc.
    """
    SUPPORTED = {"uniform", "lognormal_approx", "ratio", "default"}

    def __init__(self, category_name: str,
                 min_input_tokens: int, max_input_tokens: int,
                 min_output_tokens: int, max_output_tokens: int,
                 sampling_method: str = "default",
                 dataset_name: Optional[str] = None,
                 **kwargs):
        self.cat = category_name
        self.min_in, self.max_in = min_input_tokens, max_input_tokens
        self.min_out, self.max_out = min_output_tokens, max_output_tokens
        self.method = sampling_method
        self.kw = kwargs
        self.bank = None                     # ShareGPT 토큰 뱅크 (옵션)

        if dataset_name == "ShareGPT":
            self._load_or_build_token_bank(dataset_name)

        if self.method not in self.SUPPORTED:
            raise ValueError(f"Unknown sampling_method={self.method}")

        self.sampler = self._build_sampler()

    # ───────────── 내부: 샘플러 & 토큰뱅크 ─────────────
    def _build_sampler(self) -> Callable[[], Tuple[int, int]]:
        if self.method in {"uniform", "default"}:
            return lambda: (random.randint(self.min_in, self.max_in),
                            random.randint(self.min_out, self.max_out))

        if self.method == "ratio":
            rmin = self.kw.get("ratio_min", .1)
            rmax = self.kw.get("ratio_max", .3)
            return lambda: self._ratio_sampler(rmin, rmax)

        if self.method == "lognormal_approx":
            mu_in, sig_in  = self._approx_logn(self.min_in, self.max_in)
            mu_out, sig_out= self._approx_logn(self.min_out, self.max_out)
            return lambda: self._lognorm_sampler(mu_in, sig_in, mu_out, sig_out)

    @staticmethod
    def _approx_logn(lo, hi, pct=.95):
        med = math.sqrt(lo * hi); mu = math.log(med)
        z = {0.90:1.282,0.95:1.645,0.975:1.96,0.99:2.33}[pct]
        sigma = max((math.log(hi) - mu)/z, 0.5)
        return mu, sigma

    def _lognorm_sampler(self, mu1, s1, mu2, s2):
        while True:
            i = int(random.lognormvariate(mu1, s1))
            if self.min_in <= i <= self.max_in: break
        while True:
            o = int(random.lognormvariate(mu2, s2))
            if self.min_out <= o <= self.max_out: break
        return i, o

    def _ratio_sampler(self, rmin, rmax):
        i = random.randint(self.min_in, self.max_in)
        o = clamp(int(i * random.uniform(rmin, rmax)), self.min_out, self.max_out)
        return i, o

    # ───────────── 토큰뱅크 (ShareGPT) ─────────────
    def _load_or_build_token_bank(self, name: str):
        path = f"/home/sychoy/vllm/samples/{name}.json"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.bank = [json.loads(line) for line in f]
        else:
            print(f"[Info] building token-bank {path}")
            self.bank = self._build_sharegpt_bank(path)

    def _build_sharegpt_bank(self, out_path) -> List[dict]:
        if AutoTokenizer is None:
            raise RuntimeError("transformers 패키지가 필요합니다.")
        tok = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct", use_fast=True)
        src = "/home/sychoy/vllm/downloads/ShareGPT_V3_unfiltered_cleaned_split.json"
        with open(src, encoding="utf-8") as f: raw = json.load(f)
        raw = [d for d in raw if len(d["conversations"]) >= 2]
        pairs = [(d["conversations"][0]["value"],
                  d["conversations"][1]["value"]) for d in raw]

        parsed = []
        for p,c in pairs:
            if p and c:
                ids = tok.encode(p, truncation=True, max_length=2048)
                parsed.append(dict(input_length=len(p),
                                   output_length=len(c),
                                   input_token_ids=ids))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path,"w",encoding="utf-8") as f:
            for item in parsed: f.write(json.dumps(item)+"\n")
        return parsed

    # ───────────── Request 생성 ─────────────
    def generate(self, vocab=(200,30_000), trace_lim=None) -> RawRequest:
        while True:
            inp, out = self.sampler()
            if trace_lim and inp+out > trace_lim: continue
            break

        token_ids = ([random.randint(*vocab) for _ in range(inp)]
                     if self.bank is None else
                     random.choice(self.bank)["input_token_ids"])

        slo = (inp+out)*PROFILED_A + PROFILED_B
        return RawRequest(self.cat, inp, out, token_ids, slo)

def peak_batch_blocks(req_dict: dict, batch_size: int,
                      blk: int = BLOCK_SIZE_TOK) -> int:
    """
    req_dict : {"request_0": {...}, ...}  (input/output 길이를 포함해야 함)
    batch_size : 동시에 GPU에 들어갈 수 있는 request 수
    blk : KV-block 당 토큰 수
    -----------------------------------------------------------
    반환값 : 최악의 배치(길이 상위 batch_size 개)의
            (input+output) 합계를 blk 로 나눈 뒤 올림한 블록 수
    """
    totals = sorted((r["input_length"] + r["output_length"]
                     for r in req_dict.values()),
                    reverse=True)
    top_sum = sum(totals[:batch_size])
    return math.ceil(top_sum / blk)

# 1-3) request.json 저장
def save_request_json(path: str,
                      request_types: Dict[RequestType,float],
                      num_req: int,
                      batch_size: int,
                      vocab=(200,30_000),
                      skip_token_ids=True,
                      static=False):
    # 확률 누적
    cum, srt = 0., sorted(request_types.items(), key=lambda x: x[1], reverse=True)
    cum_probs: List[Tuple[RequestType,float]]=[]
    for rt,p in srt: cum+=p; cum_probs.append((rt,cum))

    def pick_rt() -> RequestType:
        r=random.random()
        for rt,c in cum_probs:
            if r<=c: return rt

    reqs: List[RawRequest]=[]
    if static:                                             # 같은 req 반복
        uniq = math.ceil(num_req / batch_size)
        templates=[pick_rt().generate(vocab) for _ in range(uniq)]
        for t in templates: reqs.extend([t]*batch_size)
        reqs=reqs[:num_req]
    else:
        reqs=[pick_rt().generate(vocab) for _ in range(num_req)]

    peak_blk = peak_batch_blocks(
        {f"request_{i}": dict(input_length=r.input_length,
                              output_length=r.output_length)
         for i, r in enumerate(reqs)},
        batch_size
    )

    # 직렬화 형태 맞추기
    out = OrderedDict([
        ("batch_size", batch_size),
        ("vocab",      inline(vocab)),
        ("peak_batch_blocks", peak_blk),      # ★ 추가 ★
        ("requests",   None),
    ])

    req_lines = []
    for i, r in enumerate(reqs):
        comma = "," if i < len(reqs) - 1 else ""
        payload = dict(
            category     = r.category,
            input_length = r.input_length,
            output_length= r.output_length,
        )
        if not skip_token_ids:
            payload["token_ids"] = r.token_ids
        req_lines.append(
            f'    "request_{i}": {inline(payload)}{comma}'
        )
    out["requests"] = "{\n" + "\n".join(req_lines) + "\n  }"

    # ── write -------------------------------------------------------
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = ["{"]
    for idx, (k, v) in enumerate(out.items()):
        comma = "," if idx < len(out) - 1 else ""
        if k in ("vocab", "requests"):
            lines.append(f'  "{k}": {v}{comma}')
        else:
            lines.append(f'  "{k}": {v}{comma}')
    lines.append("}\n")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {path}")

# ────────────────────────────────────────────────────────────────────
# 2. ❷ Arrival-pattern / Trace 생성기  (기존 코드 그대로)
# ────────────────────────────────────────────────────────────────────
class ArrivalPattern:                           # … 이하 동일 …
    def generate_arrival_times(self, n:int) -> List[int|float]:
        raise NotImplementedError

class DiscreteUniformArrival(ArrivalPattern):
    def __init__(self,max_step:int): self.max_step=max_step
    def generate_arrival_times(self,n:int)->List[int]:
        return [random.randint(0,self.max_step) for _ in range(n)]
    def __str__(self): return f"UniformArrival(max_step={self.max_step})"

class DiscretePeriodicArrival(ArrivalPattern):
    def __init__(self,interval:int): self.interval=interval
    def generate_arrival_times(self,n:int)->List[int]:
        return [i*self.interval for i in range(n)]
    def __str__(self): return f"PeriodicArrival(interval={self.interval})"

class DiscretePoissonArrival(ArrivalPattern):
    def __init__(self,lmbd:float,max_steps:int=100_000):
        self.lmbd,self.mx=lmbd,max_steps
    def generate_arrival_times(self,n:int)->List[int]:
        arr,t,tot=[],0,0
        while tot<n and t<self.mx:
            for _ in range(_sample_poisson(self.lmbd)):
                arr.append(t); tot+=1
                if tot==n: break
            t+=1
        return arr
    def __str__(self):
        return f"PoissonArrival(lambda={self.lmbd},max={self.mx})"

class DiscreteBimodalArrival(ArrivalPattern):
    def __init__(self,l1:float,l2:float,p:float,max_steps:int=100_000):
        self.l1,self.l2,self.p,self.mx=l1,l2,p,max_steps
    def generate_arrival_times(self,n:int)->List[int]:
        arr,t,tot=[],0,0
        while tot<n and t<self.mx:
            lam=self.l1 if random.random()<self.p else self.l2
            for _ in range(_sample_poisson(lam)):
                arr.append(t); tot+=1
                if tot==n: break
            t+=1
        return arr
    def __str__(self):
        return (f"BimodalArrival(l1={self.l1},l2={self.l2},"
                f"p={self.p},max={self.mx})")

class PoissonBurstyArrivalPattern(ArrivalPattern):
    def __init__(self,lb:float,lg:float=0): self.lb,self.lg=lb,lg
    def generate_arrival_times(self,n:int)->List[int]:
        arr,cur,tot=[],0,0
        while tot<n:
            burst=min(_sample_poisson(self.lb),n-tot)
            for _ in range(burst):
                offset=random.randint(int(self.lb*.5),self.lb)
                arr.append(cur+offset)
            tot+=burst
            cur+=int(random.uniform(self.lg*.8,self.lg))
        return sorted(arr)
    def __str__(self):
        return f"PoissonBurstyArrivalPattern(lb={self.lb},lg={self.lg})"
    
'''
Introduce a new arrival pattern to generate inter-arrival times with controlled arrival rate (λ) and CV, 
using a lognormal distribution for flexible CV values (e.g., CV < 1 for regular, CV = 1 for Poisson-like, CV > 1 for bursty arrivals).
'''
class LognormalArrival(ArrivalPattern):
    def __init__(self, lmbd: float, cv: float, max_steps: int = 100_000):
        self.lmbd = lmbd  # Arrival rate (requests per step)
        self.cv = cv      # Coefficient of Variation
        self.max_steps = max_steps
        # Compute lognormal parameters
        self.sigma = math.sqrt(math.log(cv**2 + 1))
        self.mu = math.log(1/lmbd) - (self.sigma**2)/2

    def generate_arrival_times(self, n: int) -> List[float]:
        arr, t, tot = [], 0, 0
        while tot < n and t < self.max_steps:
            inter_arrival = random.lognormvariate(self.mu, self.sigma)
            t += inter_arrival
            arr.append(int(round(t)))
            tot += 1
        return sorted(arr[:n])  # Ensure exactly n requests, sorted

    def __str__(self):
        return f"LognormalArrival(lambda={self.lmbd},cv={self.cv},max={self.max_steps})"

# Trace 구조체 (arrival 포함) — 기존 코드 그대로
@dataclass
class TraceReq:
    category:str; input_length:int; output_length:int
    arrival_time:int|float; token_ids:List[int]
    sched_time:Optional[int|float]=None
    wait_time:Optional[int|float]=None; slo:Optional[int|float]=None

@dataclass
class Trace:
    requests:Dict[int,TraceReq]
    arrival_pattern_name:str
    batch_size:int; vocab:Tuple[int,int]; max_model_len:int
    num_gpu_blocks_override:Optional[int]=None

    def add_estimate_sched(
        self,
        max_model_len: int,
        num_gpu_blocks: int,
        block_size: int = 16,
        max_parallel: int = 4,
    ) -> None:
        """
        Light-weight scheduling simulator that *allows CPU off-loading* when a
        request’s pre-fill tokens exceed current GPU capacity.

        Assumptions
        -----------
        • Only the portion that fits in the remaining GPU capacity is loaded;
        the rest is resident on CPU, so the request can always start.
        • Processing duration = 1 + output_length (same as before).
        • `gpu_tok` below means “tokens actually occupying GPU blocks”.
        • Memory for those `gpu_tok` tokens is released when the request finishes.
        • OVF calculations elsewhere still use full input_length so that the
        “GPU short-fall” fraction is accurately reflected.
        """
        self.num_gpu_blocks_override = num_gpu_blocks
        self.batch_size = max_parallel
        self.max_model_len = max_model_len

        items_sorted = sorted(
            self.requests.items(), key=lambda x: x[1].arrival_time
        )
        tokens_capacity = num_gpu_blocks * block_size    # current free tokens
        running: list[tuple[int, int, str]] = []         # (finish, gpu_tok, key)
        heapq.heapify(running)
        current_time = 0

        # group by identical arrival times
        groups: dict[int | float, list[tuple[str, TraceReq]]] = defaultdict(list)
        for key, req in items_sorted:
            groups[req.arrival_time].append((key, req))

        for arrival_time in sorted(groups):
            # A. free finished tasks up to this arrival_time
            while running and running[0][0] <= arrival_time:
                fin, in_use, _ = heapq.heappop(running)
                tokens_capacity += in_use

            # B. schedule every request arriving at this time
            for req_key, req_obj in groups[arrival_time]:
                req_in = req_obj.input_length          # total prefill tokens

                # make sure we account for any finishes exactly at arrival_time
                while running and running[0][0] <= req_obj.arrival_time:
                    fin, in_use, _ = heapq.heappop(running)
                    tokens_capacity += in_use

                # Wait while either parallelism or memory blocks us
                can_run_now = False
                while not can_run_now:
                    if len(running) >= max_parallel:
                        fin, in_use, _ = heapq.heappop(running)
                        current_time = fin
                        tokens_capacity += in_use
                        continue

                    if req_in <= tokens_capacity:
                        can_run_now = True
                    else:
                        # GPU cap insufficient but off-loading is allowed.
                        # If nothing is running we start immediately
                        # with partial GPU usage (rest on CPU).
                        if not running:
                            can_run_now = True
                        else:
                            fin, in_use, _ = heapq.heappop(running)
                            current_time = fin
                            tokens_capacity += in_use

                start_time = max(req_obj.arrival_time, current_time)
                req_obj.sched_time = start_time
                req_obj.wait_time = start_time - req_obj.arrival_time

                finish_time = start_time + 1 + req_obj.output_length

                gpu_tok = min(req_in, tokens_capacity)   # actual GPU usage
                tokens_capacity -= gpu_tok
                heapq.heappush(running, (finish_time, gpu_tok, req_key))


    def save_to_json(self,path:Path,skip_token_ids=True):
        def as_dict(r:TraceReq):
            d=r.__dict__.copy()
            d.pop("slo", None)
            if skip_token_ids: d.pop("token_ids",None)
            return d
        # (2) ─ OrderedDict으로 키 순서 고정 -------------------------
        peak_blk = peak_batch_blocks(
            {f"dummy_{i}": dict(input_length=r.input_length,
                                output_length=r.output_length)
            for i, r in self.requests.items()},
            self.batch_size
        )

        data = OrderedDict([
            ("batch_size",        self.batch_size),
            ("max_model_len",     self.max_model_len),
            ("num_gpu_blocks_override", self.num_gpu_blocks_override),
            ("arrival_pattern",   self.arrival_pattern_name),
            ("vocab",             inline(self.vocab)),
            ("peak_batch_blocks", peak_blk),      # ★ 추가
            ("requests",          None),               # 뒤에서 교체
        ])

        # (3) ─ requests 서브블록: 한 줄에 하나 ----------------------
        req_items = sorted(self.requests.items())      # id 오름차순
        req_lines = []
        for i, (_, rv) in enumerate(req_items):
            comma = "," if i < len(req_items) - 1 else ""
            req_lines.append(
                f'    "request_{i}": {inline(as_dict(rv))}{comma}'
            )
        req_block = "{\n" + "\n".join(req_lines) + "\n  }"
        data["requests"] = req_block

        # (4) ─ 최종 직렬화 -----------------------------------------
        lines = ["{"]
        for idx, (k, v) in enumerate(data.items()):
            comma = "," if idx < len(data) - 1 else ""
            # 이미 문자열 형태로 만든 항목은 그대로 사용
            if k in ("vocab", "requests"):
                lines.append(f'  "{k}": {v}{comma}')
            else:
                val = f'"{v}"' if isinstance(v, str) else v
                lines.append(f'  "{k}": {val}{comma}')
        lines.append("}\n")

        path.write_text("\n".join(lines), encoding="utf-8")

# ───────────── trace 생성 util ─────────────
def load_json(path:str): return json.loads(Path(path).read_text())

def compress_idle(reqs:List[TraceReq])->List[TraceReq]:
    reqs.sort(key=lambda r:r.arrival_time)
    grouped=defaultdict(list); [grouped[r.arrival_time].append(r) for r in reqs]
    last,out=0,[]
    for t in sorted(grouped):
        batch,max_out=grouped[t],max(r.output_length for r in grouped[t])
        if t>last:
            shift=t-(last+1)
            for r in batch: r.arrival_time-=shift
            t=last+1
        last=t+1+max_out; out.extend(batch)
    return sorted(out,key=lambda r:r.arrival_time)

def compute_gpu_blocks(reqs:Dict[str,Any],max_len:int,blk:int,ranges:List[float]):
    min_b=max(r["input_length"] for r in reqs.values())//blk
    avg_b=(sum(r["input_length"]+r["output_length"] for r in reqs.values())/
           len(reqs))//blk
    max_b=max_len//blk
    vals={round(min(max(int(avg_b+r*(max_b-min_b)),min_b),max_b)) for r in ranges}
    return sorted(vals)

def generate_trace(req_data:Dict[str,Any],arr_obj:ArrivalPattern,
                   arr_times:List[int|float],max_len:int,static:bool)->Trace:
    B, v = req_data["batch_size"],tuple(req_data["vocab"])
    # for i in range(min(1,len(arr_times))): arr_times[i]=0 # The first request arrives at time 0
    # if static:
    #     for i in range(0,len(arr_times),B):
    #         arr_times[i:i+B]=[arr_times[i]]*min(B,len(arr_times)-i)
    if arr_times:  # Shift all times to make first request arrive at 0
        t0 = arr_times[0]
        arr_times = [t - t0 for t in arr_times]
    if static:
        for i in range(0, len(arr_times), B):
            arr_times[i:i+B] = [arr_times[i]] * min(B, len(arr_times) - i)

    reqs={}
    for idx,(key,at) in enumerate(zip(sorted(req_data["requests"],
                            key=lambda k:int(k.split("_")[1])),arr_times)):
        r=req_data["requests"][key]
        tot=r["input_length"]+r["output_length"]
        slo=tot*PROFILED_A+PROFILED_B
        reqs[idx]=TraceReq(r["category"],r["input_length"],r["output_length"],
                           at, r.get("token_ids",[]),slo=slo)
        
    return Trace(requests=reqs, arrival_pattern_name=str(arr_obj), 
                 batch_size=B, vocab=v, max_model_len=max_len)
    # return Trace(requests={i:r for i,r in enumerate(compress_idle(list(reqs.values())))},
    #              arrival_pattern_name=str(arr_obj), batch_size=B,
    #              vocab=v, max_model_len=max_len)

# ───────────────────────────────────────────────────────────
# (★) build_sched_save:  trace 저장 → 메모리-압력 계산 추가
# ───────────────────────────────────────────────────────────
def build_sched_save(
    request_json: str,
    blk_size: int = 16,
    static: bool = False,
    skip_token_ids: bool = True,
    arrival_rate_scales: list[float] = [0.5, 1.0, 2.0, 5.0, 10.0],
    cvs: list[float] = [0.5, 1.0, 2.0]
):
    req_data = load_json(request_json)
    mix_tag = Path(request_json).stem.replace("req_", "")
    arrival_patterns = build_arrival_patterns(request_json, arrival_rate_scales, cvs)
    base_reqs = [(0, r["input_length"], r["output_length"]) for r in req_data["requests"].values()]

    # Calculate worst-case batch blocks
    req_dict = {f"request_{i}": dict(input_length=r[1], output_length=r[2]) 
                for i, r in enumerate(base_reqs)}
    worst_case_blocks = peak_batch_blocks(req_dict, req_data["batch_size"], blk_size)
    gamma = 0.23  # 23 %
    fixed_blocks = max(1, math.ceil(gamma * worst_case_blocks))
    print(f"[capacity] peak={worst_case_blocks}  gamma={gamma}  GPU fixed_blocks={fixed_blocks}")
    # fixed_blocks = 4000  # Fixed GPU block budget (e.g., ~8 GB)

    for ap in arrival_patterns:
        tag_ap = _slug_ap(ap)
        arr_times = ap.generate_arrival_times(len(base_reqs))
        arr_times.sort()
        # T = getattr(ap, "ideal_T", None)
        # if T and arr_times and arr_times[-1] != 0:
        #     s = T / arr_times[-1]
        #     arr_times = [int(round(at * s)) for at in arr_times]

        req_tuples = [(at, inp, out) for at, (_, inp, out) in zip(arr_times, base_reqs)]
        base_dir = Path(request_json).parent / 'all_traces_arrival_rate_CV' / mix_tag / tag_ap
        base_dir.mkdir(parents=True, exist_ok=True)

        trace = generate_trace(req_data, ap, arr_times.copy(), max_len=16_384, static=static)
        trace.num_gpu_blocks_override = fixed_blocks
        trace.add_estimate_sched(16_384, fixed_blocks, blk_size, req_data["batch_size"])

        out_json = base_dir / f"{tag_ap}.json"
        trace.save_to_json(out_json, skip_token_ids)
        print(f"[saved] {out_json} (blk={fixed_blocks})")

        metrics = memory_pressure_plot(trace_path=str(out_json), plot=True, shade=(True, True), crosses=(True, True))
        (base_dir / f"{tag_ap}.metrics.json").write_text(json.dumps(metrics, indent=2))

# ────────────────────────────────────────────────────────
# ❸ Arrival-Pattern 그리드  (100 개 Request 전제)
# ────────────────────────────────────────────────────────
# 기대 도착 ≈ λ × max_steps = 100  →  max_steps ≈ 100 / λ
# def _best_max_steps(lam, margin=0.2, minimum=120):
#     base = math.ceil(100 / lam)            # 100 개 맞추는 최소 길이
#     return max(minimum, int(base * (1 + margin)))  # 20 % 여유

# ──────────────────────────────────────────────────────────
#  output-token 합 × scale  ⇒ 타임라인 길이를 정해 주는 Arrival
# ──────────────────────────────────────────────────────────
class TokenScaledArrival(ArrivalPattern):
    def __init__(self, T: int, scale_tag: str):
        self.T = max(1, int(T))            # 전체 디코드 스텝
        self.tag = scale_tag               # 슬러그용
    def generate_arrival_times(self, n: int) -> List[int]:
        times = np.linspace(0, self.T, n, endpoint=False, dtype=int)
        np.random.shuffle(times)
        return times.tolist()
    def __str__(self):
        return f"TokenScaledArrival(scale={self.tag})"


def _timeline_len(outputs: list[int],
                  scale: float = 0.8,
                  minimum: int = 120) -> int:
    """
    타임라인 길이 T = max(minimum,  scale × Σ output_tokens)
      • scale : 0.0–1.0  (예: 0.8 → 80 %)
    """
    return max(minimum, int(sum(outputs) * scale))

def build_arrival_patterns(
    request_json: str,
    arrival_rate_scales: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0],  # Scales for base λ
    cvs: list[float] = [0.5, 1.0, 2.0]  # CV values
) -> list[ArrivalPattern]:
    data = load_json(request_json)
    outputs = [r["output_length"] for r in data["requests"].values()]
    n_req = len(outputs)
    total_steps = sum(outputs)  # Total output tokens
    T_base = total_steps / n_req  # Average steps per request    
    λ_base = n_req / total_steps  # Base arrival rate (req/step)
    print(f"[Info] Total requests={n_req}, total_steps = {total_steps}, T_base={T_base:.2f} steps/request, λ_base={λ_base:.5f} req/step")    
    # decode_step_ms = 40.0  # Fixed for tagging (ms)
    # λ_base_ms = λ_base / decode_step_ms  # Base rate in req/ms
    # print(f"[Info] λ_base_ms={λ_base_ms:.2f} req/ms")
    patterns = []
    for scale in arrival_rate_scales:
        λ_step = scale * λ_base  # Scaled arrival rate (req/step)                
        for cv in cvs:
            buffer = 1.0 + cv + (λ_base / max(λ_step, 1e-9))
            max_steps = int(math.ceil(n_req / λ_step * buffer))

            tag = f"lambda{scale:.1f}x_cv{int(cv*10):02d}"  # e.g., "lambda2.0x_32rps_cv20"
            pattern = LognormalArrival(lmbd=λ_step, cv=cv, max_steps=max_steps)
            setattr(pattern, "scale_tag", tag)
            setattr(pattern, "scale_factor", scale)
            # setattr(pattern, "base_rate_ms", λ_base_ms)  # Store for _slug_ap
            setattr(pattern, "ideal_T", total_steps)  # Total timeline
            patterns.append(pattern)
    return patterns

# def build_request_types_S_L(short_bins=(128,256,512),
#                             long_bins =(1024,2048,4096,8192),
#                             pct=0.2, max_total=32384):
#     span = lambda c: (max(16,int((c*(1-pct))//16*16)),
#                       max(16,int((c*(1+pct))//16*16)))
#     S, L = short_bins, long_bins
#     combos = []
#     for in_bin, out_bin, tag in [
#         *( (i,o,"SS") for i in S for o in S ),
#         *( (i,o,"SL") for i in S for o in L ),
#         *( (i,o,"LS") for i in L for o in S ),
#         *( (i,o,"LL") for i in L for o in L ),
#     ]:
#         if in_bin + out_bin > max_total: continue
#         li, hi = span(in_bin)
#         lo, ho = span(out_bin)
#         name = f"{tag}_I{li}-{hi}_O{lo}-{ho}"
#         combos.append(RequestType(name, li, hi, lo, ho,
#                                   sampling_method="uniform",
#                                   dataset_name="ShareGPT"))
#     return combos            # 36개 (기본)

def build_request_types_SML(
        short_bins=(64, 128, 256),
        mid_bins  =(512, 1024),
        long_bins =(2048, 4096),
        pct=0.2, max_total=16_384):
    span = lambda c: (max(16, int((c*(1-pct))//16*16)),
                      max(16, int((c*(1+pct))//16*16)))

    S, M, L = short_bins, mid_bins, long_bins
    combos = []
    for tag_src, in_set, out_set in [
        ("S", S, S), ("S", S, M), ("S", S, L),
        ("M", M, S), ("M", M, M), ("M", M, L),
        ("L", L, S), ("L", L, M), ("L", L, L),
    ]:
        for i in in_set:
            for o in out_set:
                if i + o > max_total:          # 모델 context upper-bound
                    continue
                li, hi = span(i)
                lo, ho = span(o)
                name   = f"{tag_src}{'SML'[out_set is M]+('L' if out_set is L else 'S')}" \
                         f"_I{li}-{hi}_O{lo}-{ho}"
                combos.append(RequestType(name, li, hi, lo, ho,
                                          sampling_method="uniform",
                                          dataset_name="None"))
    return combos

# ───────────────────────────────────────────────
# 1. 9-차원 격자 생성  (SS·SM·SL·MS·MM·ML·LS·LM·LL)
# ───────────────────────────────────────────────
def grid_mixtures(step: float = 0.25):
    """
    step 간격 격자에서 Σ=1 을 만족하는 9-tuple 리스트 반환
      m = (pSS, pSM, pSL, pMS, pMM, pML, pLS, pLM, pLL)
    """
    vals = np.arange(0, 1 + 1e-9, step)
    mixes = []
    # 8개 축을 먼저 돌린 뒤 마지막 축은 1-Σ 로 결정
    for a in vals:                     # SS
        for b in vals:                 # SM
            for c in vals:             # SL
                for d in vals:         # MS
                    for e in vals:     # MM
                        for f in vals: # ML
                            for g in vals: # LS
                                for h in vals: # LM
                                    s = a+b+c+d+e+f+g+h
                                    if s > 1:    # 마지막 축이 음수 → skip
                                        continue
                                    i = 1 - s    # pLL
                                    mixes.append((a,b,c,d,e,f,g,h,round(i,10)))
    # 깔끔한 반올림
    return [tuple(round(x, 4) for x in m) for m in mixes]

# ───────────────────────────────────────────────
# 2. 휴리스틱 대표 13-개  (one-hot 9개 + 쌍 3개 + uniform)
# ───────────────────────────────────────────────
def _heuristic_13(mixes):
    one_hot   = [m for m in mixes if m.count(1) == 1]          # 9
    # ‘한 쌍이 0.5-0.5, 나머지 0’ 형태로 간단히 3개만 뽑기
    pair_mix  = []
    seen      = set()
    for m in mixes:
        if sorted(m)[-2:] == [0.5, 0.5]:
            key = tuple(sorted(np.where(np.asarray(m) == 0.5)[0]))
            if key not in seen:
                seen.add(key); pair_mix.append(m)
            if len(pair_mix) == 3:
                break
    uniform   = tuple([round(1/9,4)]*9)
    return one_hot + pair_mix + [uniform]                      # 13개

# ───────────────────────────────────────────────
# 3. K-Means / LHS 는 그대로 9-D 사용 가능
#    (wrapper 만 차원 제약 없음)
# ───────────────────────────────────────────────
def _kmeans_select(mixes, k:int, random_state=0):
    from sklearn.cluster import KMeans
    if len(mixes) <= k: return mixes
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    km.fit(np.asarray(mixes))
    return [tuple(map(float,c)) for c in km.cluster_centers_]

def _lhs_simplex(k:int, random_state=0):
    rng = np.random.default_rng(random_state)
    # LHS in 8-D cube, 마지막 축은 1-Σ
    P = (rng.random((k,8)) + rng.permutation(k)[:,None]) / k
    mixes=[]
    for row in P:
        tail = 1 - row.sum()
        if tail < 0:
            vec = rng.dirichlet([1]*9)
        else:
            vec = np.concatenate([row,[tail]])
            vec /= vec.sum()
        mixes.append(tuple(round(float(x),4) for x in vec))
    return mixes

# ───────────────────────────────────────────────
# 4. mix 선택 wrapper
# ───────────────────────────────────────────────
def select_mixtures(
    mixes:list,
    k:int = 13,
    mode:str = "heuristic",
    random_state:int = 0):
    mode = mode.lower()
    if mode == "heuristic":
        base = _heuristic_13(mixes)
        return base if k <= len(base) else base + mixes[:k-len(base)]
    if mode == "kmeans":
        return _kmeans_select(mixes, k, random_state)
    if mode == "lhs":
        return _lhs_simplex(k, random_state)
    raise ValueError(f"unknown mode {mode}")

# ───────────────────────────────────────────────
# 5. 태그 함수 (9-tuple → 짧은 문자열)
# ───────────────────────────────────────────────
def mix_tag(m):
    labels = ["SS","SM","SL","MS","MM","ML","LS","LM","LL"]
    pct    = [int(x*100+0.5) for x in m]
    return "_".join(f"{lab}{p:02d}" for lab,p in zip(labels,pct))

# ───────────────────────────────────────────────────────────
# helper : ArrivalPattern → slug 문자열
#   e.g.  "DiscretePoissonArrival(lambda_per_step=2)"
#      →  "discretepoisson_l2"
# ───────────────────────────────────────────────────────────
import re

def _slug_ap(ap: ArrivalPattern) -> str:
    """
    ArrivalPattern → 짧은 slug 문자열
      • scale_tag 속성이 있으면  ➜  <패턴코드><scale>
        - TokenScaledArrival  → tsa60
        - Uniform             → uni60   … 등
      • 없으면 기존 규칙 유지
    """
    
    # ── 1) scale_tag 우선 ………………………………………………………………………
    if hasattr(ap, "scale_tag"):
        tag = ap.scale_tag                # "60", "80", …

        if isinstance(ap, TokenScaledArrival):
            return f"tsa{tag}"
        if isinstance(ap, DiscreteUniformArrival):
            return f"uni{tag}"
        if isinstance(ap, DiscretePeriodicArrival):
            return f"per{tag}"
        if isinstance(ap, DiscretePoissonArrival):
            return f"poi{tag}"
        if isinstance(ap, DiscreteBimodalArrival):
            return f"bim{tag}"
        if isinstance(ap, PoissonBurstyArrivalPattern):
            return f"bur{tag}"

    # ── 2) 기존 패턴 (scale_tag 없는 경우) …………………
    if isinstance(ap, DiscreteUniformArrival):         # e.g. uni200
        return f"uni{ap.max_step}"
    if isinstance(ap, DiscretePeriodicArrival):        # e.g. per25
        return f"per{ap.interval}"
    if isinstance(ap, DiscretePoissonArrival):         # e.g. poi1
        return f"poi{str(ap.lmbd).rstrip('0').rstrip('.')}"
    if isinstance(ap, DiscreteBimodalArrival):
        l1 = str(ap.l1).rstrip('0').rstrip('.')
        l2 = str(ap.l2).rstrip('0').rstrip('.')
        p  = str(ap.p ).rstrip('0').rstrip('.')
        return f"bim{l1}-{l2}_p{p}"
    if isinstance(ap, PoissonBurstyArrivalPattern):    # e.g. bur5-20
        return f"bur{ap.lb}-{ap.lg}"
    if isinstance(ap, LognormalArrival):
        scale = getattr(ap, "scale_factor", 1.0)    
        cv = str(ap.cv).rstrip('0').rstrip('.')
        return f"lognormal_lambda{scale:.1f}x_cv{cv}"

    # ── 3) fallback – 클래스 이니셜만 ……………………
    cls = ap.__class__.__name__
    return ''.join(w[0] for w in re.findall(r'[A-Z][^A-Z]*', cls)).lower()


# ────────────────────────────────────────────────
#  Memory-pressure 분석 유틸
#    • PPR, TPI, OV … 계산
#    • plot_PPR=True 이면 PNG 저장
# ────────────────────────────────────────────────
import numpy as np, math, json
def _tok_to_blk(toks: int, blk: int = 16) -> int:
    if isinstance(toks, np.ndarray):
        return np.ceil(toks / blk).astype(int)
    else:                           # 파이썬 스칼라
        return math.ceil(toks / blk)

def memory_pressure_plot(
    trace_path: str,
    *,
    plot: bool        = True,            # 그림 자체를 그릴지 여부
    curve: bool       = True,            # 주 곡선
    shade: Tuple[bool, bool] = (True, True),  # (lower, upper)
    guides: bool      = True,            # 가이드선
    crosses: Tuple[bool, bool] = (True, True),# (TJ, BJ)
) -> Dict[str, float]:
    """
    vLLM trace → 메모리-압력 지표 & (선택) 시각화 PNG
    -------------------------------------------------------------------
    Parameters
    ----------
    trace_path : str
        *.json trace 파일 경로
    plot : bool
        PNG 자체를 만들지 여부.
    curve / shade / guides / crosses : 세부 레이어 토글
        shade     = (아래 0–1, 위 1–overflow)
        crosses   = (Type-1, Type-2)
    -------------------------------------------------------------------
    Returns
    -------
    Dict[str, float]
        PPR, TPI, OVF 등 + 선택적 'plot_path'
    """
    # 0) 로드 ----------------------------------------------------------------
    with open(trace_path, encoding="utf-8") as f:
        j = json.load(f)

    gpu_blocks = j.get("num_gpu_blocks_override") or j.get("num_gpu_blocks")
    if gpu_blocks is None:
        raise KeyError("trace JSON에 'num_gpu_blocks_override' 키가 없습니다.")

    # 1) 요청 파싱 ------------------------------------------------------------
    reqs: List[Tuple[int, int, int, int]] = []   # (arr, end, inpBlk, out)
    last_step = 0
    for r in j["requests"].values():
        arr = int(r["arrival_time"])
        out = int(r["output_length"])
        inp_blk = _tok_to_blk(int(r["input_length"]))
        last_step = max(last_step, arr + out)
        reqs.append((arr, arr + out, inp_blk, out))

    if not reqs:
        raise ValueError("Trace has no requests")

    # 2) 타임라인 스윕 (NumPy) -------------------------------------------------
    steps = np.arange(last_step + 1)
    live_blocks = np.zeros_like(steps, dtype=np.int32)

    for arr, end, inp_blk, out in reqs:
        span = slice(arr, end + 1)
        dec  = np.minimum(steps[span] - arr, out)
        live_blocks[span] += inp_blk + _tok_to_blk(dec)

    y = live_blocks / gpu_blocks                   # pressure ratio

    peak_blocks  = int(live_blocks.max())
    peak_step    = int(live_blocks.argmax())
    over_mask    = live_blocks > gpu_blocks
    ge2_mask     = live_blocks >= 2 * gpu_blocks

    shortfall_sum = int((live_blocks - gpu_blocks).clip(min=0).sum())
    over_steps    = int(over_mask.sum())
    ge2_steps     = int(ge2_mask.sum())
    total_steps   = len(steps)

    # 3) 메트릭 ---------------------------------------------------------------
    out = dict(
        PPR      = peak_blocks / gpu_blocks,
        TPI      = over_steps / total_steps,
        OV_block_step = shortfall_sum,
        OV_frac  = shortfall_sum / (gpu_blocks * total_steps),
        GE2_frac = ge2_steps / total_steps,
        peak_blocks = peak_blocks,
        peak_step   = peak_step,
        gpu_blocks  = gpu_blocks,
        total_steps = total_steps,
    )

    # 4) crossings (TJ/BJ) ----------------------------------------------------
    tj_cnt = bj_cnt = 0
    if any(crosses):
        thr = [1.0] + [32 / (32 - ol) for ol in
                       (16, 10, 8, 6, 5, 4, 3, 2, 1, 0)]
        thr = np.asarray(thr)

        prev = y[:-1, None]
        curr = y[1:,  None]
        hit  = ((prev - thr) * (curr - thr) < 0) | ((prev == thr) ^ (curr == thr))

        hit_cnt = hit.sum(axis=1)
        tj_cnt  = int((hit_cnt == 1).sum())
        bj_cnt  = int((hit_cnt >  1).sum())

    out.update(type1_count=tj_cnt, type2_count=bj_cnt)

    # 5) 그림 ---------------------------------------------------------------
    if plot:
        fig, ax = plt.subplots(figsize=(8, 3))

        # A. shading -----------------------------------------------------
        sh_low, sh_up = shade
        if sh_low or sh_up:
            over_seg, in_seg, seg_start = [], False, None
            for i, flag in enumerate(over_mask):
                if flag and not in_seg:
                    seg_start, in_seg = i, True
                elif not flag and in_seg:
                    over_seg.append((seg_start, i)); in_seg = False
            if in_seg: over_seg.append((seg_start, total_steps))

            for s, e in over_seg:
                xs = steps[s:e]
                ys = y[s:e]
                if sh_low:
                    ax.fill_between(xs, 0.0, 1.0, color="C0", alpha=.25)
                if sh_up:
                    ax.fill_between(xs, 1.0, ys, color="C1", alpha=.25)

        # B. main curve ---------------------------------------------------
        if curve:
            ax.plot(steps, y, lw=.8, label="pressure")
            ax.axhline(1.0, ls="--", lw=.8, label="capacity")

        # C. guides -------------------------------------------------------
        if guides:
            for g in [1.0] + [32 / (32 - ol) for ol in (16,10,8,6,5,4,3,2,1,0)]:
                ax.axhline(g, ls="--", lw=.5, color="grey", alpha=.5)

        # D. crossings ----------------------------------------------------
        cr1, cr2 = crosses
        if (cr1 or cr2) and (tj_cnt or bj_cnt):
            thr = [1.0] + [32 / (32 - ol) for ol in (16,10,8,6,5,4,3,2,1,0)]
            type1_x = [];  type1_y = []
            type2_x = [];  type2_y = []
            for i in range(1, total_steps):
                prev, cur = y[i-1], y[i]
                hit = [t for t in thr
                       if (prev-t)*(cur-t) < 0 or (prev == t) ^ (cur == t)]
                if len(hit) == 1:
                    type1_x.append(i); type1_y.append(hit[0])
                elif len(hit) > 1:
                    type2_x.append(i); type2_y.append(max(hit))
            if cr1 and type1_x:
                ax.scatter(type1_x, type1_y, s=20, marker="o", label="TJ")
            if cr2 and type2_x:
                ax.scatter(type2_x, type2_y, s=30, marker="s", label="BJ")

        # E. final touches -----------------------------------------------
        ax.set_xlabel("decode step")
        ax.set_ylabel("live / gpu")
        ax.set_title(
            f"PPR={out['PPR']:.2f}  OTF={out['TPI']:.2f}  "
            f"OVF={out['OV_frac']:.2f}, TJ={tj_cnt}, BJ={bj_cnt}"
        )
        ax.legend(frameon=False)
        plt.tight_layout()

        png = Path(trace_path).with_suffix(".png")
        fig.savefig(png, dpi=150)
        plt.close(fig)
        out["plot_path"] = str(png)

    return out
# ------------------------------------------------------------
# (A)  주어진 블록 수 → OVF 계산  (trace 생성 이전·가벼운 시뮬)
# ------------------------------------------------------------
def _ovf_for_blocks(reqs: List[Tuple[int,int,int]],
                    blocks: int, blk_size: int = 16) -> float:
    peak = over_sum = steps = 0
    for (arr, inp, out) in reqs:
        end = arr + out
        steps = max(steps, end)

    for t in range(steps + 1):
        live = 0
        for (arr, inp, out) in reqs:
            if arr <= t <= arr+out:
                live += math.ceil((inp + min(t-arr, out)) / blk_size)
        if live > blocks:
            over_sum += live - blocks
    return over_sum / (blocks * (steps + 1))

def find_blocks_for_ovf(
    reqs: List[Tuple[int,int,int]],   # (arr, inp, out)    
    target_ovf: float,
    tol: float = 0.01,
    blk_size: int = 16,
    min_blk: int | None = None,
    max_blk_cap: int = 2**13,        # ← 8 192 blocks ≈ 127 k tok
) -> int:    
    # ── 1) 초기 lo, hi ──────────────────────────────────────────
    peak_tok = max(inp + out for _, inp, out in reqs)
    lo = max(min_blk or 1, 1)
    # hi = max(lo, 4096)                        # 4k 블록을 최초 hi 로

    def ovf(b): return _ovf_for_blocks(reqs, b, blk_size)

    cur_lo_ovf = ovf(lo)

    # ── (1) 목표와 이미 근접한 경우 -----------------------------
    if abs(cur_lo_ovf - target_ovf) <= tol:
        return lo, False

    # ── (2) 트래픽이 너무 가벼워 목표보다 OVF가 작음 -------------
    if cur_lo_ovf < target_ovf:
        return lo, True  # cannot push OVF up without violating peak token size

    # ── (3) upper bound 찾기 -------------------------------------
    hi = max(lo, 4096)
    while ovf(hi) > target_ovf and hi < max_blk_cap:
        hi *= 2
    hi = min(hi, max_blk_cap)

    if ovf(hi) > target_ovf:  # still high even at cap
        return hi, True

    # ── (4) 이분 탐색 -------------------------------------------
    while lo < hi:
        mid = (lo + hi) // 2
        cur = ovf(mid)
        if abs(cur - target_ovf) <= tol:
            return mid, False
        if cur > target_ovf:
            lo = mid + 1
        else:
            hi = mid
    return lo, False

LEVEL_ABBR = {
    "vlow"     : "vl",   # very-low
    "low"      : "lo",
    "mid_low"  : "ml",
    "mid"      : "md",
    "mid_high" : "mh",
    "high"     : "hi",
}

def _ovf_str(val: float) -> str:
    """
    0.25  → ov25
    1.53  → ov1p5
    13.07 → ov13p1   (두 번째 자리 0 이면 삭제)
    """
    if val < 1:
        return f"ov{int(round(val*100))}"
    s = f"{val:.1f}".rstrip("0").rstrip(".")     # 1.50 → 1.5 , 13.0 → 13
    return "ov" + s.replace(".", "p")


# ────────────────────────────────────────────────────────────────────
# 3. 예시 실행
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # NOTE(HONG): request generation with various length combinations
    RT_list = build_request_types_SML()                    # 36개
    
    # ① 모든 격자 혼합 생성 (step=0.25 ⇒ 35개)
    all_mix = grid_mixtures(step=0.33)

    # ② 대표 샘플 선택
    mixes_h9  = select_mixtures(all_mix, k=9,  mode="heuristic")
    mixes_k12 = select_mixtures(all_mix, k=12, mode="kmeans",  random_state=42)
    mixes_l20 = select_mixtures(all_mix, k=20, mode="lhs",     random_state=42)

    print("heuristic 9 :", mixes_h9[:3], "...")
    print("kmeans 12   :", mixes_k12[:3], "...")
    print("lhs 20      :", mixes_l20[:3], "...")

    # Combine all mixes into a single list
    mixes = mixes_h9 + mixes_k12 + mixes_l20
    
    # Print total number of mixes
    print(f"Total number of mixes: {len(mixes)}")
    
    for mix in mixes:                           # mixes_h9 + mixes_k12 + mixes_l20 ...
        probs = {rt: 0 for rt in RT_list}
        block = len(RT_list) // 9               # 9
        for j, p in enumerate(mix):       # j = 0‥8 (SS, SM, …, LL)
            for rt in RT_list[j*block:(j+1)*block]:
                probs[rt] = p / block

        total = sum(probs.values())
        for k in probs:
            probs[k] /= total  

        # ─────────────── 변경된 저장 경로 ───────────────
        tag = mix_tag(mix)                      # 한눈에 보이는 태그
        save_request_json(
            path=f"traces/requests_types_16k/req_{tag}.json",      # traces 폴더에 저장
            request_types=probs,
            num_req=52,
            batch_size=4,
            skip_token_ids=True,
        )

    # ────────────────────────────────────────────────────────
    #  모든 request.json × Arrival-Pattern 조합으로 trace 생성
    #    (앞서 “traces/req_*.json” 으로 저장한 요청 파일들을 대상으로)
    # ────────────────────────────────────────────────────────    

    level_spec = {
        "vlow"     : dict(target_ovf = 0.05, tol = 0.010),  # 매우 여유
        "low"      : dict(target_ovf = 0.12, tol = 0.015),
        "mid_low"  : dict(target_ovf = 0.22, tol = 0.020),
        "mid"      : dict(target_ovf = 0.35, tol = 0.030),
        "mid_high" : dict(target_ovf = 0.55, tol = 0.040),
        "high"     : dict(target_ovf = 0.80, tol = 0.050),  # 상한 유지
    }    

    for req_json in Path("traces/requests_types_16k").glob("req_*.json"):
        build_sched_save(
            request_json=str(req_json),
            blk_size=BLOCK_SIZE_TOK,
            static=False,
            skip_token_ids=True,
            arrival_rate_scales=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],  # req/s
            cvs=[1,2,3,4,5,6,7],  # Coefficient of Variation
        )
    # build_sched_save(
    #         request_json=str("/home/heelim/vllm/build_trace/traces/requests_types_32k/req_SS11_SM11_SL11_MS11_MM11_ML11_LS11_LM11_LL11.json"),
    #         blk_size=BLOCK_SIZE_TOK,
    #         static=False,
    #         skip_token_ids=True,
    #         arrival_rate_scales=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],  # req/s
    #         cvs=[1, 2, 3, 4, 5, 6, 7]
    #     )
