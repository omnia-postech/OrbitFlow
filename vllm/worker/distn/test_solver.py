# ──────────────────────────────────────────────────────────────
#  Sim-KV  (minimal skeleton)
# ──────────────────────────────────────────────────────────────
from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Dict, List, Deque, Optional, Tuple
from collections import deque, OrderedDict
from solver import Request as SolverReq          # ← the solver-side Request
from solver import ProfileBasedEstimator
from solver import ResultList, Solver_v1, Solver_updated
import json, random

BLOCK_SIZE = 16                     # tokens per KV-block
LAYER_NUM = 32          # number of transformer layers (L)
PREFILL_RATE_TPS = 500          # prompt-copy speed (tokens / s)
HALF_OFFLOAD = LAYER_NUM // 2          # assume 50 % layers on CPU at admission

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


@dataclass
class TraceReq:
    id: str
    prompt_len: int
    output_len: int
    arrival_time: int
    offload_num: int = 0 
@dataclass
class ReqRuntime:
    prompt_len: int
    generated: int = 0
    offload_num: int = 0             # NEW – layers residing on *CPU*

    def memory_blocks(self) -> int:
        live_layers = LAYER_NUM - self.offload_num
        return live_layers * math.ceil((self.prompt_len + self.generated) / BLOCK_SIZE)

class DelaySimulator:
    """
    Keeps per-request “token deposit” and releases tokens in real time based on
    the global SLO (seconds / token).  A release that falls short increments
    a violation counter.
    """

    def __init__(self, slo_s_per_token: float):
        self.v_tps = 1.0 / slo_s_per_token          # tokens per second
        self.deposit: Dict[str, int] = defaultdict(int)
        self.last_time: Dict[str, float] = {}       # virtual time, per request
        self.violations: Dict[str, int] = defaultdict(int)
        self._total_violations: int = 0 
        self.now: float = 0.0                       # global virtual clock
    # handy accessor ----------------------------------------------------
    def total_violations(self) -> int:
        """Return the cumulative number of token-level SLO violations."""
        return self._total_violations
    # ----- bookkeeping --------------------------------------------------
    def register(self, rid: str):
        self.deposit[rid]   = 0
        self.last_time[rid] = self.now
        self.violations[rid] = 0
        self._total_violations += 1
    def finish(self, rid: str):
        self.deposit.pop(rid, None)
        self.last_time.pop(rid, None)
        self.violations.pop(rid, None)

    def violation_count(self, rid: str) -> int:
        return self.violations.get(rid, 0)

    # ----- push newly generated token(s) --------------------------------
    def on_token(self, rid: str, n: int = 1):
        self.deposit[rid] += n

    # ----- advance virtual clock & release tokens -----------------------
    def advance_time(self, dt: float, active_rids: List[str]):
        """
        dt : simulated time elapsed this decode step (seconds)
        active_rids : list[str] – only these requests are considered for release
        """
        self.now += dt
        for rid in active_rids:
            if rid not in self.last_time:           # safety
                self.last_time[rid] = self.now - dt

            elapsed = self.now - self.last_time[rid]
            EPS = 1e-9
            should_release = int((elapsed * self.v_tps) + EPS)
            if should_release <= 0:
                continue

            if self.deposit[rid] >= should_release:
                # normal release
                self.deposit[rid] -= should_release
                self.last_time[rid] += should_release / self.v_tps
            else:
                # violation: backlog smaller than allowed release
                self.violations[rid] += 1
                self.deposit[rid] = 0
                self.last_time[rid] = self.now    # reset stopwatch

# ----------------------------------------------------------------------
# Component 1 – MemoryStat
# ----------------------------------------------------------------------
class MemoryStat:
    def __init__(self, cap_blocks: int):
        self.cap_blocks = cap_blocks * LAYER_NUM       # capacity in blocks
        self.total_blocks = 0
        # rid → (tokens, blocks)
        self.per_req: dict[str, tuple[int, int]] = {}

    def update(self, runtimes: dict):
        self.per_req = {}
        for rid, rt in runtimes.items():
            tok = rt.prompt_len + rt.generated
            blk = rt.memory_blocks()
            self.per_req[rid] = (tok, blk)

        self.total_blocks = sum(b for _, b in self.per_req.values())
# ----------------------------------------------------------------------
# Component 2 – SolverInputGenerator
# ----------------------------------------------------------------------
class SolverInputGenerator:
    def __init__(self, trace: Trace, slo: float):
        self.cap_blocks = trace.num_gpu_blocks_override * LAYER_NUM
        self.cap_tokens = self.cap_blocks * BLOCK_SIZE
        self.global_slo = slo
        self.max_batch  = trace.batch_size
        self.all_trace_reqs: dict[str, TraceReq] = {
            rid: TraceReq(rid,
                          req.input_length,
                          req.output_length,
                          req.arrival_time)
            for rid, req in trace.requests.items()
        }
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
        self.idle_jump: int | None = None   # ← new; next step when idle
    # ------------------------------------------------------------------
    def build(self, runtimes: dict, deposits: dict[str, int]) -> list[SolverReq]:
        return [self._trace_to_solver(tr, deposits.get(tr.id, 0))
                for rid, tr in self.active.items()
                if rid in runtimes]
    # ----------------------------------------------------------------------
    def _mem_blocks(self) -> int:
        """Current GPU KV usage (blocks), taking off-loading into account."""
        total = 0
        for tr in self.active.values():
            live_layers = LAYER_NUM - tr.offload_num          # use the *current* plan
            tok = tr.prompt_len + self.generated.get(tr.id, 0)
            total += live_layers * math.ceil(tok / BLOCK_SIZE)
        return total
    def _mem_tokens(self) -> int:
        return sum(r.prompt_len + self.generated.get(r.id, 0)
                   for r in self.active.values())
    def _blocks_for_tokens(self, n_tokens: int) -> int:
        return math.ceil(n_tokens / BLOCK_SIZE)

    def _total_blocks_if_next_token(self,
                                    prompt: int,
                                    generated: int,
                                    offload_num: int = 0) -> int:
        """
        Returns GPU KV-blocks the request will occupy *after* the next token,
        taking KV off-loading into account.

            live_layers = L – offload_num
            current_blocks = live_layers * ceil(tokens / 16)
            if the next token crosses a 16-token boundary, we add live_layers.
        """
        live_layers = LAYER_NUM - offload_num

        tokens_now = prompt + generated
        current_blocks = live_layers * math.ceil(tokens_now / BLOCK_SIZE)

        needs_new_block = (tokens_now % BLOCK_SIZE == 0)
        return current_blocks + (live_layers if needs_new_block else 0)
    # ----------------------------------------------------------------------
    def _trace_to_solver(self, tr: TraceReq, dep: int) -> SolverReq:
        tot_tokens = tr.prompt_len + self.generated.get(tr.id, 0)
        return SolverReq(
            id                   = tr.id,
            context_len_in_blocks= math.ceil(tot_tokens / BLOCK_SIZE) + 2,  # ← FIXED
            layer_time           = 0.0,
            deposit_count        = dep,
            slo                  = self.global_slo,
            gpu_layers_on_gpu    = LAYER_NUM - tr.offload_num,
        )
    # ----------------------------------------------------------------------
    def step(self, t: int, p_step: int) -> List[SolverReq]:
        """
        Admission / eviction policy (v2)

        •  drop  : if  half-offload memory  > global capacity  ⇒ permanently REMOVE
        •  pause : if  half-offload memory ≤ global capacity  but can’t fit *now*
                    ⇒ keep in paused / pending
        •  admit : if  half-offload memory  fits current free blocks  ⇒ add to batch
        """
        paused_this_step = False
        # ── 0. permanently drop arrivals that can never fit ───────────────────
        while self.pending and self.pending[0].arrival_time <= t:
            cand = self.pending[0]
            need_half = self._total_blocks_if_next_token(
                            cand.prompt_len, 0, HALF_OFFLOAD)
            if need_half > self.cap_blocks:
                logging.info("REMOVED %s (needs %d>%d blocks even w/½-offload)",
                            cand.id, need_half, self.cap_blocks)
                self.pending.popleft()
            else:
                break

        # ── 1. pause victims until current usage ≤ cap ─────────────────────–––
        if self._mem_blocks() > self.cap_blocks:
            # rid, tr = self.active.popitem(last=True)
            # self.paused.append(tr)
            # paused_this_step = True       # mark
            logging.info("Memory pressure detected at step %d; triggering solver", p_step)
        # ── 2. resume ONE paused request if it (with ½-offload) fits now ───────
        if not paused_this_step and self.paused:
            if self.paused and len(self.active) < self.max_batch:
                cand = self.paused[0]
                need_half = self._total_blocks_if_next_token(
                                cand.prompt_len,
                                self.generated.get(cand.id, 0),
                                HALF_OFFLOAD)
                if (len(self.active) < self.max_batch
                        and self._mem_blocks() + need_half <= self.cap_blocks):
                    cand.offload_num = HALF_OFFLOAD          # mark 16 layers on-CPU
                    self.paused.popleft()
                    self.active[cand.id] = cand
                    logging.info("Resumed %s at step %d (½-offload)", cand.id, p_step)

        # ── 3. admit all pending arrivals that fit right now (½-offload test) ─
        while self.pending and self.pending[0].arrival_time <= t:
            cand = self.pending[0]
            need_half = self._total_blocks_if_next_token(
                            cand.prompt_len, 0, HALF_OFFLOAD)
            if need_half > self.cap_blocks:
                logging.info("REMOVED %s (needs %d>%d blocks even w/½-offload)",
                            cand.id, need_half, self.cap_blocks)
                self.pending.popleft()
                continue
            if (len(self.active) < self.max_batch
                    # and self._mem_blocks() + need_half <= self.cap_blocks):
                    and  need_half <= self.cap_blocks):
                cand.offload_num = HALF_OFFLOAD          # start in ½-offload mode
                self.pending.popleft()
                self.active[cand.id] = cand
                self.generated[cand.id] = 0
                logging.info("Admitted %s at step %d (½-offload)", cand.id, p_step)
            else:
                break                                   # head arrival still won’t fit

        # # ── 4. next-token overflow guard (uses CURRENT offload) ───────────────
        # extra_needed = sum(
        #     (LAYER_NUM - tr.offload_num)
        #     for tr in self.active.values()
        #     if (tr.prompt_len + self.generated.get(tr.id, 0)) % BLOCK_SIZE == 0
        # )
        # if self._mem_blocks() + extra_needed > self.cap_blocks and self.active:
        #     rid, tr = self.active.popitem(last=True)
        #     self.paused.append(tr)
        #     paused_this_step = True       # mark
        #     logging.info("Paused %s (would exceed capacity on next token) at step %d", rid, p_step)

        # ── 5. idle-gap hint for Simulator.run() ───────────────────────────────
        self.idle_jump = (
            None if (self.active or self.paused or not self.pending)
            else self.pending[0].arrival_time
        )

    # helper for simulator to advance token counters
    # ----------------------------------------------------------------------
    def incr_generated(self, rid: str, n: int = 1):
        if rid in self.generated:
            self.generated[rid] += n
    @property
    def next_arrival_step(self) -> int | None:
        """Return arrival_time of the next pending request, or None."""
        return self.pending[0].arrival_time if self.pending else None
# ----------------------------------------------------------------------
# Component 3 – ControlPlane
# ----------------------------------------------------------------------
class ControlPlane:
    def __init__(self, pause_mode="min"):
        # keep eviction order
        self.mem_paused    : "OrderedDict[str, ReqRuntime]" = OrderedDict()
        self.solver_paused : "OrderedDict[str, ReqRuntime]" = OrderedDict()
        self.pause_mode = pause_mode  # "min" or "max"

    def handle_infeasible(self, runtimes: Dict[str, ReqRuntime], p_step: int) -> None:
        if self.pause_mode == "max":
            victim = max(runtimes.items(), key=lambda kv: kv[1].memory_blocks())[0]
        elif self.pause_mode == "min":
            victim = min(runtimes.items(), key=lambda kv: kv[1].memory_blocks())[0]
        self.solver_paused[victim] = runtimes.pop(victim)
        gen.active.pop(victim, None)          # ← add this line
        gen.generated.pop(victim, None)       # ← and this one
        log.warning("Paused %s due to infeasible placement at step %d", victim, p_step)

    def try_solver_resume(self, runtimes: Dict[str, ReqRuntime], p_step:int) -> bool:
        if not self.solver_paused:
            return False
        rid, rt = self.solver_paused.popitem(last=False)   # now legal
        runtimes[rid] = rt

        # ─── restore generator tables ───────────────────────
        tr = gen.all_trace_reqs[rid]             # or however you index them
        gen.active[rid]     = tr
        gen.generated[rid]  = rt.generated       # keep counters consistent        

        log.info("Re-inserted %s after a completion at step %d", rid, p_step)
        return True
# ----------------------------------------------------------------------
# Component 4 – StepLogPrinter
# ----------------------------------------------------------------------
class StepLogPrinter:
    def __init__(self, capacity_blocks: int, *, log_only_solver_steps=False):
        self.step   = 0
        self.capacity = capacity_blocks
        self.log_only_solver_steps = log_only_solver_steps

    def maybe_log(self, mem, dep_before, dep_after, delay_sim,
                step_latency, solver_log=None, p_step=None):
        # ─── optional suppression ───────────────────────────────────────────
        if self.log_only_solver_steps and not solver_log:
            self.step += 1
            return 
        if p_step: 
            self.step = p_step
        # ─── system-level mem ───────────────────────────────────────────────
        sys_mem = f"{mem.total_blocks}/{self.capacity}b"
        # ─── per-request details (tokens / blocks / deposit / viol)──────────
        parts = []
        for rid, (tok, blk) in mem.per_req.items():
            b = dep_before.get(rid, 0)
            a = dep_after.get(rid, 0)
            delta = a - b
            viol  = delay_sim.violation_count(rid)
            parts.append(
                f"{rid}:{tok}t/{blk}b  {b}->{a} (Δ{delta:+d})  viol={viol}"
            )
        req_str = "; ".join(parts) or "{}"
        # ─── single condensed line ─────────────────────────────────────────
        if solver_log:
            logging.info("STEP %4d | mem %s | lat %.4fs | %s | %s",
                        self.step, sys_mem, step_latency, solver_log, req_str)
        else:
            logging.info("STEP %4d | mem %s | lat %.4fs | %s",
                        self.step, sys_mem, step_latency, req_str)
        self.step += 1
# ----------------------------------------------------------------------
# Component 5 – Simulator loop
# ----------------------------------------------------------------------
class Simulator:
    def __init__(self, gen, solver, estimator, delay_sim,pause_mode="min"):
        self.delay_sim  = delay_sim
        self.gen = gen
        self.solver     = solver
        self.estimator  = estimator
        self.runtimes: Dict[str, ReqRuntime] = {}
        # self.delay_sim  = DelaySimulator(v_tps=1.0)  # customise
        self.mem_stat   = MemoryStat(self.gen.cap_blocks)
        self.ctrl_plane = ControlPlane(pause_mode=pause_mode)
        self.logger = StepLogPrinter(self.gen.cap_blocks,
                             log_only_solver_steps=True)   # <-- set to False for old behaviour

        self.decode_left: int = 0
        self.step: int = 0
        self.p_step: int = 0
        self.last_results = None
    # ---- helpers -----------------------------------------------------
    def _call_solver(self, inputs=None) -> tuple[str | None, bool]:
        if inputs is None:
            inputs = self.gen.build(self.runtimes,
                               {rid: self.delay_sim.deposit.get(rid, 0)
                                for rid in self.gen.active})

        log.info(f"solver input: {inputs}")
        log_str = None                         # ⇦ what we’ll send to StepLogPrinter
        while True:                            # ─── retry-loop ───
            res_list = self.solver.solve(inputs,
                                        gpu_block_capacity=self.gen.cap_blocks)
            if res_list is None:               # infeasible  ➜  pause 1 victim
                self.ctrl_plane.handle_infeasible(self.runtimes, self.p_step)
                inputs = self.gen.build(self.runtimes,
                               {rid: self.delay_sim.deposit.get(rid, 0)
                                for rid in self.gen.active})
                self.blocked = True 
                return self._call_solver(inputs)     # recursive
            break                              # feasible
        self.blocked = False                        # all good again
        self.decode_left = int(res_list[0].window)
        log_str = (f"dist={{{', '.join(f'{r.id}:{r.n}' for r in res_list)}}}  "
                f"t={res_list[0].actual_time:.4f}s  "
                f"win={self.decode_left:.1f}")

        # apply off-load decisions to runtimes
        for r in res_list:
            if r.id in self.runtimes:
                self.runtimes[r.id].offload_num = r.offload_num
                self.gen.active[r.id].offload_num = r.offload_num      # ← NEW
        return log_str, True                  # feasible=True
    @staticmethod
    def _needs_new_block(tokens_now: int) -> bool:
        """True if adding one token crosses a 16-token block boundary."""
        return tokens_now % BLOCK_SIZE == 0
    def _post_solver_guard(self) -> bool:
        """
        Return True iff we had to pause ONE victim because the very next
        token would overflow *after* the solver’s placement.
        """
        extra = sum(
            (LAYER_NUM - tr.offload_num)
            for tr in self.gen.active.values()
            if (tr.prompt_len + self.gen.generated[tr.id]) % BLOCK_SIZE == 0
        )
        if self.gen._mem_blocks() + extra <= self.gen.cap_blocks:
            return False                     # all good

        rid, tr = self.gen.active.popitem(last=True)
        self.runtimes.pop(rid, None)
        # self.gen.generated.pop(rid, None)
        self.gen.paused.append(tr)
        log.info("Paused %s (would exceed capacity on next token)", rid)
        return True

    # ────────────────────────────────────────────────────────────────────
    # main loop  (replace the whole previous body)
    # ────────────────────────────────────────────────────────────────────
    def run(self):
        prev_active: set[str] = set()
        self.blocked = False           # NEW
        while True:
            #--------------- termination test ---------------------------
            if self.p_step == 5022: 
                log.info(self.gen.pending)
                log.info(self.runtimes)
                log.info(self.gen.paused)
                log.info(self.ctrl_plane.solver_paused)
            if not (self.gen.pending or self.runtimes or self.gen.paused or self.ctrl_plane.solver_paused):
                log.info(self.gen.pending)
                log.info(self.runtimes)
                log.info(self.gen.paused)
                log.info(self.ctrl_plane.solver_paused)
                break
            if self.gen.pending and \
               not (self.runtimes or self.gen.active or self.gen.paused):
                # nothing live → jump straight to next arrival
                self.step = self.gen.pending[0].arrival_time

            solver_log = None

            #--------------- generator step (admits / pauses / evicts) --
            self.gen.step(self.step,self.p_step)

            #--------------- newcomers & pre-fill latency ---------------
            newcomers = [rid for rid in self.gen.active
                         if rid not in self.runtimes]
            if newcomers:                                     # pre-fill happening
                pf_tokens   = sum(self.gen.active[r].prompt_len
                                   for r in newcomers)
                pf_latency  = pf_tokens / PREFILL_RATE_TPS    # seconds

                # advance virtual clock for already-running requests
                self.delay_sim.advance_time(pf_latency,
                                            list(self.runtimes.keys()))

                # register newcomers *after* time advance
                for rid in newcomers:
                    tr = self.gen.active[rid]
                    self.runtimes[rid] = ReqRuntime(
                        prompt_len = tr.prompt_len,
                        generated  = self.gen.generated.get(rid, 0),   # ← keep progress
                        offload_num= tr.offload_num
                    )
                    self.delay_sim.register(rid)

                # bookkeeping / logging for a pure pre-fill step
                self.mem_stat.update(self.runtimes)
                self.logger.maybe_log(
                    mem=self.mem_stat,
                    dep_before={},                # nothing changed
                    dep_after={},
                    delay_sim=self.delay_sim,
                    step_latency=pf_latency,
                    solver_log=None,
                    p_step=self.p_step
                )
                self.step += 1
                self.p_step += 1 
                prev_active = set(self.gen.active.keys())
                self.decode_left = 0
                continue                         # skip decode part this loop
            #----------------------------------------------------------------

            #--------------- sync drops ------------------------------------
            for rid in list(self.runtimes):
                if rid not in self.gen.active:            # evicted / finished
                    self.runtimes.pop(rid, None)
                    self.delay_sim.finish(rid)

            #--------------- detect comp-change & maybe run solver ----------
            curr_active = set(self.gen.active.keys())
            comp_changed = curr_active != prev_active

            need_solver = ((self.gen._mem_blocks() > self.gen.cap_blocks)
                           or (not self.blocked and (comp_changed or self.decode_left == 0))
                           )
            if need_solver:
                while True:                          # solver-retry loop
                    solver_log, _ = self._call_solver()
                    # if the next-token guard paused someone -> loop again
                    if self._post_solver_guard():
                        continue
                    break
                prev_active = set(self.gen.active)    # batch after solver               

            prev_active  = set(self.gen.active)

            #--------------- one-token decode emission ---------------------
            before_dep = {r: self.delay_sim.deposit.get(r, 0)
                          for r in self.runtimes}
            for rid, rt in self.runtimes.items():
                rt.generated += 1
                self.gen.incr_generated(rid)
                self.delay_sim.on_token(rid)

            self.decode_left = max(self.decode_left - 1, 0)

            #--------------- retire finished -------------------------------
            finished_in_this_step = False
            for rid, rt in list(self.runtimes.items()):
                if rt.generated >= self.gen.active[rid].output_len:
                    finished_in_this_step = True
                    self.runtimes.pop(rid, None)
                    self.gen.active.pop(rid, None)
                    self.gen.generated.pop(rid, None)
                    log.info("FINISHED %s at step %d", rid, self.p_step)
            # ── if a request finished, try to resume one solver-paused req ──
            if finished_in_this_step:
                if self.ctrl_plane.try_solver_resume(self.runtimes, self.p_step):
                    # we changed batch composition → run solver once
                    solver_log, _ = self._call_solver()
            #--------------- latency, deposits, log ------------------------
            total_tokens_now = sum(rt.prompt_len + rt.generated
                                   for rt in self.runtimes.values())
            step_latency = self.estimator.estimate_by_profiled_results(
                               total_tokens_now, which="NoPrefetch", mode="linear")

            self.delay_sim.advance_time(step_latency, list(self.runtimes.keys()))
            after_dep = {r: self.delay_sim.deposit.get(r, 0) for r in self.runtimes}

            self.mem_stat.update(self.runtimes)
            self.logger.maybe_log(
                mem=self.mem_stat,
                dep_before=before_dep,
                dep_after=after_dep,
                delay_sim=self.delay_sim,
                step_latency=step_latency,
                solver_log=solver_log,
                p_step = self.p_step
            )
            self.step += 1
            self.p_step += 1
        log.info("Simulation finished after %d steps" % self.step)

        # -------- summary metrics --------------------------------------
        total_viols = self.delay_sim.total_violations()
        log.info("TOTAL token-level SLO violations: %d", total_viols)
        log.info(f"Total token-level SLO violations: {total_viols}")
# ----------------------------------------------------------------------
# Usage example
# ----------------------------------------------------------------------
import argparse, json, logging
logging.getLogger("gurobipy").setLevel(logging.WARNING)

# ────── run simulation ──────────────────────────────────────────────────────

# ────── CLI ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--trace", required=True, help="Path to trace JSON file")
parser.add_argument("--log-file", required=True, help="Where to write the run log")
parser.add_argument("--pause-mode", required=True, help="min or max")
args = parser.parse_args()
print(args)
logging.basicConfig(
    filename=args.log_file,
    filemode="w",                  # overwrite each run
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True                     # ← overrides any existing handlers
)
log = logging.getLogger("main")

# ────── dummy solver (always returns 32-token window) ───────────────────────
class DummySolver:
    def solve(self, reqs):
        # return ResultList()
        return 0
# ────── build generator from trace file ─────────────────────────────────────
trace = Trace.load_from_json(args.trace)
profiled_path = "/home/heelim/vllm/benchmark/scripts/profiled_results_A6000.json"
estimator     = ProfileBasedEstimator(profiled_path)

cap_tokens = trace.num_gpu_blocks_override * BLOCK_SIZE
global_slo = estimator.estimate_by_profiled_results(
                 cap_tokens,
                 which="NoPrefetch",
                 mode="linear") *1
delay_sim = DelaySimulator(global_slo)
log.info("Global SLO %.4f s (for %d tokens = %d blocks)",
         global_slo, cap_tokens, trace.num_gpu_blocks_override)

gen = SolverInputGenerator(trace, slo=global_slo)
sim = Simulator(gen, Solver_v1(), estimator, delay_sim,pause_mode=args.pause_mode)
sim.run()
