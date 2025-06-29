# ──────────────────────────────────────────────────────────────
#  Sim-KV  (minimal skeleton)
# ──────────────────────────────────────────────────────────────
import logging
from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Dict, List, Deque, Optional, Tuple
from collections import deque, OrderedDict
from solver import Request as SolverReq          # ← the solver-side Request
import json, random


log = logging.getLogger("simkv")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
BLOCK_SIZE = 16                     # tokens per KV-block
# ----------------------------------------------------------------------
# Data models
# ----------------------------------------------------------------------

# solver-side Request is *not* used here
@dataclass
class RequestSpec:
    category: str
    input_length: int
    output_length: int
    arrival_time: int
    token_ids: list[int]
    sched_time: Optional[int] = None
    wait_time: Optional[int] = None

@dataclass
class Trace:
    requests: Dict[str, RequestSpec]
    batch_size: int
    request_type_probs: any
    vocab: Tuple[int, int]
    num_gpu_blocks_override: Optional[int] = None
    max_model_len: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None

    @classmethod
    def load_from_json(cls, path: str) -> "Trace":
        with open(path, "r") as f:
            d = json.load(f)

        vmin, vmax = d["vocab"]
        rq = {}
        for rid, r in d["requests"].items():
            if "token_ids" not in r or len(r["token_ids"]) != r["input_length"]:
                r["token_ids"] = [random.randint(vmin, vmax)
                                  for _ in range(r["input_length"])]
            rq[rid] = RequestSpec(**r)

        return cls(
            requests=rq,
            batch_size=d["batch_size"],
            max_model_len=d["max_model_len"],
            num_gpu_blocks_override=d["num_gpu_blocks_override"],
            request_type_probs=d.get("request_type_probs"),
            vocab=tuple(d["vocab"]),
            gpu_memory_utilization=d.get("gpu_memory_utilization"),
        )


@dataclass(frozen=True)
class TraceReq:
    id: str
    prompt_len: int
    output_len: int
    arrival_time: int

@dataclass
class ReqRuntime:
    prompt_len: int
    generated: int = 0        # decode tokens emitted so far
    deposit: int = 0
    def memory(self) -> int:
        return self.prompt_len + self.generated

# ----------------------------------------------------------------------
# Component 0 – DelaySimulator (already implemented elsewhere)
# ----------------------------------------------------------------------
class DelaySimulator:
    # reuse your existing implementation
    ...

# ----------------------------------------------------------------------
# Component 1 – MemoryStat
# ----------------------------------------------------------------------
class MemoryStat:
    def __init__(self):
        self.total_tokens: int = 0
        self.per_req: Dict[str, int] = {}

    def update(self, runtimes: Dict[str, ReqRuntime]):
        self.per_req = {rid: rt.memory() for rid, rt in runtimes.items()}
        self.total_tokens = sum(self.per_req.values())

# ----------------------------------------------------------------------
# Component 2 – SolverInputGenerator
# ----------------------------------------------------------------------
class SolverInputGenerator:
    def __init__(self, trace: Trace):
        self.cap_blocks = trace.num_gpu_blocks_override
        self.cap_tokens = self.cap_blocks * BLOCK_SIZE

        self.pending: Deque[TraceReq] = deque(
            TraceReq(rid,
                     req.input_length,
                     req.output_length,
                     req.arrival_time)
            for rid, req in sorted(trace.requests.items(),
                                   key=lambda kv: kv[1].arrival_time)
        )
        self.active: "OrderedDict[str, TraceReq]" = OrderedDict()
        self.paused: Deque[TraceReq] = deque()

        # runtime counters for generated tokens
        self.generated: Dict[str, int] = {}

    # ----------------------------------------------------------------------
    def _mem_tokens(self) -> int:
        return sum(r.prompt_len + self.generated.get(r.id, 0)
                   for r in self.active.values())

    # ----------------------------------------------------------------------
    def _trace_to_solver(self, tr: TraceReq) -> SolverReq:
        """
        Minimal mapping → solver.Request
        • context_len_in_blocks = ceil(prompt/16)
        • other fields initialised to 0 – you will fill real values later
        """
        return SolverReq(
            id=tr.id,
            context_len_in_blocks=math.ceil(tr.prompt_len / BLOCK_SIZE),
            layer_time=0.0,
            deposit_count=0,
            slo=0.0,
            gpu_layers_on_gpu=0,
        )

    # ----------------------------------------------------------------------
    def step(self, t: int) -> List[SolverReq]:
        """Advance to simulation step *t*, return list[solver.Request]."""

        # 1) admit arrivals that can fit
        while self.pending and self.pending[0].arrival_time <= t:
            cand = self.pending[0]
            if self._mem_tokens() + cand.prompt_len <= self.cap_tokens:
                self.active[cand.id] = self.pending.popleft()
                self.generated[cand.id] = 0
            else:
                break     # stop admitting – memory full

        # 2) FIFO evict while over capacity
        while self._mem_tokens() > self.cap_tokens:
            vid, vict = self.active.popitem(last=True)
            self.paused.append(vict)
            self.generated.pop(vid, None)

        # 3) build solver input
        return [self._trace_to_solver(tr) for tr in self.active.values()]

    # ----------------------------------------------------------------------
    # helper for simulator to advance token counters
    # ----------------------------------------------------------------------
    def incr_generated(self, rid: str, n: int = 1):
        if rid in self.generated:
            self.generated[rid] += n
# ----------------------------------------------------------------------
# Component 3 – ControlPlane
# ----------------------------------------------------------------------
class ControlPlane:
    def __init__(self):
        self.paused: Dict[str, ReqRuntime] = {}

    def handle_infeasible(self, runtimes: Dict[str, ReqRuntime]) -> Optional[str]:
        # pick longest request to pause
        victim = max(runtimes.items(), key=lambda kv: kv[1].memory())[0]
        self.paused[victim] = runtimes.pop(victim)
        log.warning("Paused %s due to infeasible placement.", victim)
        return victim

    def try_resume(self, runtimes: Dict[str, ReqRuntime]):
        # naive: resume everything
        for rid, rt in list(self.paused.items()):
            runtimes[rid] = self.paused.pop(rid)

# ----------------------------------------------------------------------
# Component 4 – StepLogPrinter
# ----------------------------------------------------------------------
class StepLogPrinter:
    def __init__(self):
        self.step = 0
    def log(self, mem: MemoryStat):
        log.info("STEP %4d | mem_tot=%d | %s",
                 self.step, mem.total_tokens, mem.per_req)
        self.step += 1

# ----------------------------------------------------------------------
# Component 5 – Simulator loop
# ----------------------------------------------------------------------
class Simulator:
    def __init__(self, gen: SolverInputGenerator, solver):
        self.gen = gen
        self.solver     = solver
        self.runtimes: Dict[str, ReqRuntime] = {}
        # self.delay_sim  = DelaySimulator(v_tps=1.0)  # customise
        self.mem_stat   = MemoryStat()
        self.ctrl_plane = ControlPlane()
        self.logger     = StepLogPrinter()
        self.decode_left: int = 0
        self.step: int = 0

    # ---- helpers -----------------------------------------------------
    def _admit_arrivals(self):
        """admit requests whose arrival_time ≤ current step"""
        while self.gen.pending and self.gen.pending[0].arrival_time <= self.step:
            req = self.gen.pending.popleft()      # FIFO arrival
            self.runtimes[req.id] = ReqRuntime(prompt_len=req.prompt_len)
            self.gen.generated[req.id] = 0        # initialise decode counter
            log.info("Admitted %s at step %d", req.id, self.step)

    def _call_solver(self):
        inputs = self.gen.build(self.runtimes)
        result = self.solver.solve(inputs)            # returns None ↔ infeasible
        if result is None:
            self.ctrl_plane.handle_infeasible(self.runtimes)
            return self._call_solver()                # retry immediately
        self.decode_left = int(result.window)
        log.info("Solver window=%d", self.decode_left)

    # ---- main loop ---------------------------------------------------
    def run(self):
        while self.gen.pending or self.runtimes:
            self._admit_arrivals()

            if self.decode_left == 0:
                self.ctrl_plane.try_resume(self.runtimes)
                self._call_solver()

            # emit one token per active request
            for rid, rt in self.runtimes.items():
                rt.generated += 1                     # decode token
                # self.delay_sim.on_token(rid, self.step)

            self.decode_left = max(self.decode_left - 1, 0)

            # finish requests whose output_len reached
            for rid in list(self.runtimes):
                # assume we stored output_len somewhere accessible
                pass  # add finish logic here

            # bookkeeping & logging
            self.mem_stat.update(self.runtimes)
            self.logger.log(self.mem_stat)

            self.step += 1

# ----------------------------------------------------------------------
# Usage example
# ----------------------------------------------------------------------
import argparse, json, logging
# ────── run simulation ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("main")

# ────── CLI ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--trace", required=True, help="Path to trace JSON file")
args = parser.parse_args()
# ────── dummy solver (always returns 32-token window) ───────────────────────
class DummySolver:
    def solve(self, reqs):
        class R: window = 32
        return R()
# ────── build generator from trace file ─────────────────────────────────────
trace = Trace.load_from_json(args.trace)
# print(f"trace: {trace}")
gen   = SolverInputGenerator(trace)
sim   = Simulator(gen, DummySolver())
sim.run()