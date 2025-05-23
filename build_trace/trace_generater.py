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
try:
    from transformers import AutoTokenizer          # (token-bank 선택 시 사용)
except ImportError:
    AutoTokenizer = None

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

    # 직렬화 형태 맞추기
    out=dict(batch_size=batch_size,
             vocab=vocab,
             requests={f"request_{i}":dict(
                         category=r.category,
                         input_length=r.input_length,
                         output_length=r.output_length,
                         **({} if skip_token_ids else {"token_ids":r.token_ids})
                       )
                       for i,r in enumerate(reqs)})

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, separators=(",",":")))
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
    def __str__(self): return f"DiscreteUniformArrival(max_step={self.max_step})"

class DiscretePeriodicArrival(ArrivalPattern):
    def __init__(self,interval:int): self.interval=interval
    def generate_arrival_times(self,n:int)->List[int]:
        return [i*self.interval for i in range(n)]
    def __str__(self): return f"DiscretePeriodicArrival(interval={self.interval})"

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
        return f"DiscretePoissonArrival(lambda={self.lmbd},max={self.mx})"

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
        return (f"DiscreteBimodalArrival(l1={self.l1},l2={self.l2},"
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



PATTERN_MAP={c.__name__:c for c in [
    DiscreteUniformArrival, DiscretePeriodicArrival,
    DiscretePoissonArrival, DiscreteBimodalArrival,
    PoissonBurstyArrivalPattern]}

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

    def add_estimate_sched(self,tokens_per_block:int,
                           gpu_blocks:int,max_parallel:int):
        cap=gpu_blocks*tokens_per_block
        running:List[Tuple[int|float,int]]=[]; heapq.heapify(running)
        for r in sorted(self.requests.values(), key=lambda x:x.arrival_time):
            while running and running[0][0]<=r.arrival_time:
                _,m=heapq.heappop(running); cap+=m
            mem=r.input_length; cur=r.arrival_time
            while (len(running)>=max_parallel) or mem>cap:
                fin,mm=heapq.heappop(running); cur=fin; cap+=mm
            r.sched_time, r.wait_time = cur, cur-r.arrival_time
            finish=cur+1+r.output_length
            heapq.heappush(running,(finish,mem)); cap-=mem

    def save_to_json(self,path:Path,skip_token_ids=True):
        def as_dict(r:TraceReq):
            d=r.__dict__.copy()
            if skip_token_ids: d.pop("token_ids",None)
            return d
        data=dict(
            batch_size=self.batch_size,
            max_model_len=self.max_model_len,
            num_gpu_blocks=self.num_gpu_blocks_override,
            arrival_pattern=self.arrival_pattern_name,
            vocab=self.vocab,
            requests={k:as_dict(v) for k,v in self.requests.items()},
        )
        path.write_text(json.dumps(data,separators=(",",":")))
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
    B,v=req_data["batch_size"],tuple(req_data["vocab"])
    for i in range(min(B,len(arr_times))): arr_times[i]=0
    if static:
        for i in range(0,len(arr_times),B):
            arr_times[i:i+B]=[arr_times[i]]*min(B,len(arr_times)-i)

    reqs={}
    for idx,(key,at) in enumerate(zip(sorted(req_data["requests"],
                            key=lambda k:int(k.split("_")[1])),arr_times)):
        r=req_data["requests"][key]; tot=r["input_length"]+r["output_length"]
        slo=tot*PROFILED_A+PROFILED_B
        reqs[idx]=TraceReq(r["category"],r["input_length"],r["output_length"],
                           at,r.get("token_ids",[]),slo=slo)
    return Trace(requests={i:r for i,r in enumerate(compress_idle(list(reqs.values())))},
                 arrival_pattern_name=str(arr_obj), batch_size=B,
                 vocab=v, max_model_len=max_len)

# 상위 함수
def build_sched_save(request_json:str, arrival_patterns:List[ArrivalPattern],
                     max_tokens:int, block_size:int, gpu_ranges:List[float],
                     static=False, skip_token_ids=True):
    req_data=load_json(request_json)
    gpu_blocks=compute_gpu_blocks(req_data["requests"],max_tokens,
                                  block_size,gpu_ranges)
    req_path=Path(request_json)
    for arr_obj in arrival_patterns:
        arr=arr_obj.generate_arrival_times(len(req_data["requests"])); arr.sort()
        out_dir=req_path.parent/arr_obj.__class__.__name__.lower(); out_dir.mkdir(exist_ok=True)
        for blk in gpu_blocks:
            trace=generate_trace(req_data,arr_obj,arr.copy(),max_tokens,static)
            trace.num_gpu_blocks_override=blk
            trace.add_estimate_sched(block_size,blk,req_data["batch_size"])
            out=out_dir/f"trace_{blk}.json"; trace.save_to_json(out,skip_token_ids)
            print(f"[saved] {out}")

# ────────────────────────────────────────────────────────────────────
# 3. 예시 실행
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ➊ request.json 생성
    short_short = RequestType("Short-Short ShareGPT",
                             600,1000, 600,1000,
                             sampling_method="uniform", dataset_name="ShareGPT")
    short_long = RequestType("Short-Long ShareGPT",
                             300,900, 6500,7500,
                             sampling_method="uniform", dataset_name="ShareGPT")
    long_short = RequestType("Long-Short ShareGPT",
                             4500,6500, 1300,1900,
                             sampling_method="uniform", dataset_name="ShareGPT")
    long_long = RequestType("Long-Long ShareGPT",
                             2000,8000, 2000,10000,
                             sampling_method="uniform", dataset_name="ShareGPT")    
    

    probs={short_long:0.6, long_short:0.4}
    req_json_path="request.json"
    save_request_json(req_json_path, probs,
                      num_req=10, batch_size=4,
                      skip_token_ids=True, static=False)

    # ➋ trace 생성
    arrival_patterns=[
        DiscreteUniformArrival(max_step=5_000),
        DiscreteBimodalArrival(lambda1=0.005, lambda2=0.0001, p=0.7, max_steps=5_000),
    ]
    build_sched_save(
        request_json=req_json_path,
        arrival_patterns=arrival_patterns,
        max_tokens=8_000,
        block_size=BLOCK_SIZE_TOK,
        gpu_ranges=[-0.6,-0.3,0,0.3,0.6,1.0],
        static=True,
        skip_token_ids=True,
    )
