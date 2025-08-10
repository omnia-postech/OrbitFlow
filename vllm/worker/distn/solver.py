import math
import os
import gurobipy as gp
from gurobipy import GRB
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
from typing     import List, Optional
from tabulate   import tabulate          # pip install tabulate
from pathlib import Path
import json
from vllm.logger import init_logger
logger = init_logger(__name__)

# Use environment variable or default path for profiled results
DEFAULT_PROFILED_PATH = os.environ.get(
    "PROFILED_RESULTS_PATH",
    "/home/xinyuema/vllm/benchmark/scripts/profiled_results_A6000.json"
)

# ========= 1. INPUTS =====================================================

TOKENS_PER_BLOCK = 16          # system constant
BIG_M             = 1e6        # convenience big-M

class ProfileBasedEstimator:
    """
    Estimate per-token latency from pre-profiled polynomial fits.

    JSON format (one example):

    {
        "NoPrefetch": {
            "linear":      { "A": 1.23e-6, "B": 3.54e-2, "R2": 0.989 },
            "linear":  { "A": 8.38e-12, "B": 8.06e-7, "C": 4.46e-2, "R2": 0.895 }
        },
        "Prefetch32": { ... }
    }

    *   Top-level keys  →  “which” argument
    *   Second level     →  “mode” argument
    *   Each mode dict   →  polynomial coefficients + optional extras (e.g. R2)
    """

    _COEFF_ORDER = ["A", "B", "C", "D", "E"]        # extend if ever needed

    # ──────────────────────────────────────────────────────────────────
    # construction
    # ──────────────────────────────────────────────────────────────────
    def __init__(self, profiled_path: str | Path | None = None):
        if profiled_path is None:
            profiled_path = DEFAULT_PROFILED_PATH
        self.profile_path = Path(profiled_path).expanduser()
        with self.profile_path.open("r", encoding="utf-8") as f:
            self._data: Dict[str, Dict[str, Dict[str, float]]] = json.load(f)

        # pre–parse coefficient arrays for fast evaluation
        self._coeff_cache: Dict[tuple, List[float]] = {}  # (which, mode) -> coeff list
        for which, modes in self._data.items():
            for mode, params in modes.items():
                coeffs = [
                    params[c] for c in self._COEFF_ORDER if c in params
                ]
                if not coeffs:
                    raise ValueError(f"No coefficients found for {which}/{mode}")
                self._coeff_cache[(which, mode)] = coeffs

    # ──────────────────────────────────────────────────────────────────
    # main API
    # ──────────────────────────────────────────────────────────────────
    def estimate_by_profiled_results(
        self,
        tokens: int,
        which: str = "NoPrefetch",
        mode: str = "linear",
    ) -> float:
        """
        Parameters
        ----------
        tokens : int
            Number of output tokens.
        which  : str
            Top-level profile key (e.g. "NoPrefetch").
        mode   : str
            Fit name inside that profile (e.g. "linear").

        Returns
        -------
        float
            Estimated Δt in seconds for *one* token. (Multiply by tokens if
            you need total sequence time.)
        """
        coeffs = self._coeff_cache.get((which, mode))
        if coeffs is None:
            raise KeyError(f"Profile '{which}' with mode '{mode}' not found.")

        # polynomial evaluation: a_n x^n + ... + a1 x + a0
        x = float(tokens)
        estimate = 0.0
        for power, a in enumerate(reversed(coeffs)):  # constant term first
            estimate += a * x**power
        return estimate

    # ──────────────────────────────────────────────────────────────────
    # convenience helpers
    # ──────────────────────────────────────────────────────────────────
    def available_profiles(self) -> Dict[str, List[str]]:
        """Return {"which": [modes…], …}"""
        return {w: list(m.keys()) for w, m in self._data.items()}

    def r2(self, which: str, mode: str) -> float | None:
        """Return stored R² if present, else None."""
        return self._data.get(which, {}).get(mode, {}).get("R2")

# ────────────────── Data-classes ──────────────────
@dataclass
class Request:
    def __init__(self, id: str, context_len_in_blocks: int, layer_time: float,
                 deposit_count: int, slo: float, gpu_layers_on_gpu: int):
        self.id = id
        self.context_len_in_blocks = context_len_in_blocks
        self.layer_time = layer_time
        self.deposit_count = deposit_count
        self.slo = slo
        self.gpu_layers_on_gpu = gpu_layers_on_gpu
    def __repr__(self):
        return f"Request(id={self.id}, context_len_in_blocks={self.context_len_in_blocks}, layer_time={self.layer_time}, deposit_count={self.deposit_count}, slo={self.slo})"


@dataclass
class Result:
    def __init__(self, id: str, resume: bool, n: int, offload_num: int,
                 slo_fail: float,slo_fail_r:float, actual_time: float, window: int):
        self.id = id
        self.resume = resume
        self.n = n                  # (=prefetch_dist, 즉 distance)
        self.offload_num = offload_num
        self.slo_fail = slo_fail
        self.slo_fail_r = slo_fail_r 
        self.actual_time = actual_time
        self.window = window
    def __repr__(self):
        return f"Result(id={self.id}, resume={self.resume}, n={self.n}, offload_num={self.offload_num}, slo_fail={self.slo_fail}, actual_time={self.actual_time}, window={self.window})"
class ResultList(list):
    """A list[Result] that prints as a neat table."""
    # keep all list behaviour
    def __str__(self) -> str:                     # called by print()
        if not self:
            return "<empty RequestList>"
        rows = [vars(r) for r in self]            # each Result -> dict
        return tabulate(rows, headers="keys", tablefmt="github")

    __repr__ = list.__repr__                      # unambiguous fallback
    @property
    def batch_time(self) -> Optional[float]:
        """Actual-time values for requests that were resumed."""
        l = [r.actual_time for r in self if r.resume]
        if len(l) > 0: 
            assert(max(l) == min(l)) 
            return l[0]
        else:
            return 100
    # If you just need a scalar (e.g., max or average), expose that instead:
    @property
    def max_actual_time(self) -> float | None:
        times = self.batch_time
        return max(times) if times else None


# Old solver + communication latency fixed (still non-convex)
class Solver_v1:
    def __init__(self, profiled_path: str | Path | None = None):
        self.profiled_estimator = ProfileBasedEstimator(profiled_path)
        self.which = "NoPrefetch"
        self.mode = "linear"

    # @staticmethod
    def solve(
        self,
        requests_list: list[Request],
        *,
        layer_num: int = 32,
        block_bandwidth: float = 103_178.0 / 1_000,     # blocks / second (16 tokens per block)
        gpu_block_capacity: int = 49_152 // 80,         # total blocks the GPU can hold
        window_ub: int = 1_000,                         # upper bound on decode-window length
    ) -> Optional[list[Result]]:
        """Optimise KV-placement and pre-fetch distance for the current micro-batch.

        The latency model is pipeline-accurate:
        • per-layer communication is counted in **blocks** (not bytes);
        • a non-negative stall variable s_j captures comm > compute overlap;
        • batch latency = Σ compute + Σ stall; feeds all SLO logic unchanged.
        """

        # ----------------------------------------------------------------------------------
        # 0.  Gather per-request constants
        # ----------------------------------------------------------------------------------
        requests          = [r.id for r in requests_list]
        context_blocks    = {r.id: r.context_len_in_blocks for r in requests_list}   # blocks/layer
        deposit_count     = {r.id: r.deposit_count          for r in requests_list}
        SLO               = {r.id: r.slo                   for r in requests_list}
        blocks_per_layer  = context_blocks                                                     # alias

        L      = layer_num
        BIG_M  = 1_000_000.0                              # large number for indicator fallback

        # ----------------------------------------------------------------------------------
        # 1.  Enumerate admissible pre-fetch strides  (one best stride for each n_off)
        # ----------------------------------------------------------------------------------
        # NOTE(HONG): allowing distance 0
        # floor_val     = {d: L // (d + 1) for d in range(L)}          # d = 0 … 31
        # NOTE(HONG): distance 0 is not allowed, so we start from 1
        floor_val    = {d: L // (d + 1) for d in range(1, L)}           # 1 ≤ d < L - > distance ∈ {1,2,3,4,5,7,9,15,31}
        best_d_for_n  = {n_off: max(d for d, n in floor_val.items() if n == n_off)
                        for n_off in floor_val.values()}
        valid_dist    = sorted(best_d_for_n.values())                # final stride set

        # ----------------------------------------------------------------------------------
        # 2.  Pre-compute “is-offloaded” flag  a[r][d][j]  (1 ≤ j ≤ L)
        # ----------------------------------------------------------------------------------
        a = {
            r: {
                d: {j: int(j % (d + 1) == 0) for j in range(1, L + 1)}   # layers (d+1), 2(d+1), …
                for d in valid_dist
            }
            for r in requests
        }

        # ----------------------------------------------------------------------------------
        # 3.  Gurobi model, variables
        # ----------------------------------------------------------------------------------
        model = gp.Model("block_solver")
        model.Params.NonConvex = 2                    # allow bilinear equalities

        # -- binary “keep in batch” (fixed to 1 for now) -----------------------------------
        resume        = model.addVars(requests, lb=1, ub=1, vtype=GRB.BINARY,   name="resume")

        # -- stride selection (one-hot) -----------------------------------------------------
        onc = model.addVars(
            [(r, d) for r in requests for d in valid_dist],
            vtype=GRB.BINARY, name="onc"
        )

        prefetch_dist = model.addVars(requests, lb=1,  ub=L + 1, vtype=GRB.INTEGER, name="prefetch_dist")
        offload_num   = model.addVars(requests, lb=0,  ub=L,     vtype=GRB.INTEGER, name="offload_num")

        decode_steps  = model.addVar(lb=32, ub=window_ub, vtype=GRB.INTEGER, name="decode_steps")

        # -- latency-and-SLO variables ------------------------------------------------------
        stall        = model.addVars(range(1, L + 1), lb=0.0, name="stall")   # s_j
        token_time   = model.addVar(lb=0.0, name="token_time")
        actual_time  = model.addVars(requests, lb=0.0, ub=BIG_M, name="actual_time")
        ratio        = model.addVars(requests, lb=0.0, name="ratio")
        slo_fail_per_decode = model.addVars(requests, lb=0.0, name="slo_fail_per_decode")

        trans = model.addVars([(r, j) for r in requests for j in range(1, L + 1)],
                              lb=0.0, vtype=GRB.CONTINUOUS, name="trans")
        # goodput       = model.addVar(lb=0.0, name="obj")                          # objective

        # ----------------------------------------------------------------------------------
        # 4.  Fixed compute time per layer  (profiled “no-prefetch” curve)
        # ----------------------------------------------------------------------------------
        total_tokens = 16 * sum(context_blocks[r] for r in requests)              # rough estimate
        batch_layer  = (
            self.profiled_estimator
            .estimate_by_profiled_results(total_tokens, which="NoPrefetch", mode="linear")
            / 32.0
        )  # seconds per layer when every layer is GPU-resident

        # ----------------------------------------------------------------------------------
        # 5.  Communication bandwidth (blocks / second)
        # ----------------------------------------------------------------------------------
        per_block_time  = (
            self.profiled_estimator
            .estimate_by_profiled_results(total_tokens, which="Communication", mode="linear")
            / (32 * 16)                                       # 32 layers, 16 tokens per block
        )
        block_bandwidth = 1.0 / per_block_time                # blocks / second
        print(f"[solve] profiled PCIe/NVLink bandwidth  : {block_bandwidth:.2f} blocks/s")

        # ----------------------------------------------------------------------------------
        # 6.  Constraints ───────────────────────────────────────────────────────────────────
        # ----------------------------------------------------------------------------------
        
        # 6.1 one stride per request --------------------------------------------------------
        model.addConstrs(
            (gp.quicksum(onc[r, d] for d in valid_dist) == resume[r] for r in requests),
            name="one_stride"
        )

        # 6.2 derive stride (prefetch_dist) and CPU-layer count (offload_num) ---------------
        model.addConstrs(
            (gp.quicksum(onc[r, d] * (d + 1)   for d in valid_dist) == prefetch_dist[r]
            for r in requests),
            name="prefetch_dist_def"
        )
        model.addConstrs(
            (gp.quicksum(onc[r, d] * floor_val[d] for d in valid_dist) == offload_num[r]
            for r in requests),
            name="offload_num_def"
        )

        # 6.3 Bandwidth capacity per layer  (layer-budget model)
        for j in range(1, L + 1):
            model.addConstr(gp.quicksum(trans[r, j] for r in requests)
                            <= block_bandwidth * (batch_layer + stall[j]),
                            name=f"bw_cap_{j}")
        # demand expression  blocks_needed[r,j]
        blocks_needed = {
            (r, j): gp.quicksum(onc[r, d] * a[r][d][j] * blocks_per_layer[r]
                                for d in valid_dist)
            for r in requests for j in range(1, L + 1)
        }
        # prefix-flow constraints  (LB + UB = exact)
        for r in requests:
            for j in range(1, L + 1):
                # lower bound – must have transferred required data so far
                model.addConstr(
                    gp.quicksum(trans[r, k] for k in range(1, j + 1))
                    >= gp.quicksum(blocks_needed[r, t] for t in range(1, j + 1)),
                    name=f"flowLB_{r}_{j}"
                )
        # total transferred blocks = offloaded blocks
        for r in requests:
            model.addConstr(
                gp.quicksum(trans[r, j] for j in range(1, L + 1))
                == offload_num[r] * context_blocks[r],          # baseline block size
                name=f"flow_total_{r}")
        # 6.4 Stall definition (based on NEW comm_j)
        for j in range(1, L + 1):
            comm_j = gp.quicksum(trans[r, j] for r in requests) / block_bandwidth
            if j == 1:
                model.addConstr(stall[1] >= comm_j, name="stall_first")
            else:
                model.addConstr(stall[j] >= comm_j - batch_layer, name=f"stall_{j}")
            model.addConstr(stall[j] <= comm_j, name=f"stall_ub_{j}")
        # 6.5 batch latency definition
        model.addConstr(token_time == L * batch_layer
                        + gp.quicksum(stall[j] for j in range(1, L + 1)),
                        name="token_time_def")

        # 6.6 bind per-request actual_time via indicators -----------------------------------
        for r in requests:
            model.addGenConstrIndicator(resume[r], True,
                                        actual_time[r] == token_time,
                                        name=f"act_on_{r}")
            model.addGenConstrIndicator(resume[r], False,
                                        actual_time[r] >= BIG_M,
                                        name=f"act_off_{r}")

        # 6.7 SLO ratio and failure budget --------------------------------------------------
        model.addConstrs(
            (ratio[r] * actual_time[r] == decode_steps * SLO[r] for r in requests),
            name="ratio_qc"
        )
        model.addConstrs(
            (slo_fail_per_decode[r] * decode_steps
            >= decode_steps - ratio[r] - deposit_count[r]     for r in requests),
            name="slo_fail_def"
        )
        # adjust infeasible SLO failures here
        model.addConstr(gp.quicksum(slo_fail_per_decode[r] for r in requests) <= 2,
                        name="slo_fail_total")

        # 6.8 GPU memory capacity -----------------------------------------------------------
        model.addConstr(
            gp.quicksum(
                (L - offload_num[r]) * context_blocks[r] for r in requests
            ) <= gpu_block_capacity,
            name="gpu_memory_cap"
        )

        # ----------------------------------------------------------------------------------
        # 7.  Objective: maximise good-put (tokens / second that meet SLO) ------------------
        # ----------------------------------------------------------------------------------
        # model.addConstr(
        #     goodput * token_time ==
        #     gp.quicksum(resume[r] for r in requests) - slo_fail_per_decode.sum(),
        #     name="goodput_def"
        # )
        # model.setObjective(goodput, GRB.MAXIMIZE)
        model.setObjective(token_time, GRB.MINIMIZE)
        model.Params.OutputFlag = 1

        # ----------------------------------------------------------------------------------
        # 8.  Solve
        # ----------------------------------------------------------------------------------
        try:
            model.optimize()
        except gp.GurobiError as e:
            print(f"[solve] Gurobi Error: {e}")
            return None

        # ----------------------------------------------------------------------------------
        # 9.  Extract & print results
        # ----------------------------------------------------------------------------------
        if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            return None

        print("\n--- Optimal solution (or best found) ---")
        print(" id | offload | SLO-fail | actual_time")
        print("----|---------|----------|------------")
        for r in requests:
            print(f"{r:>3} | {int(offload_num[r].X):>7} |"
                f" {slo_fail_per_decode[r].X * decode_steps.X:8.2f} |"
                f" {actual_time[r].X:10.4f}s")

        print(f"\nGood-put           : {model.ObjVal:.4f} tokens/s")
        print(f"Batch latency      : {token_time.X:.4f}s "
            f"(includes {sum(stall[j].X for j in range(1, L + 1)):.4f}s stall)")
        print(f"GPU KV usage       : "
            f"{sum((L - offload_num[r].X) * context_blocks[r] for r in requests)} "
            f"/ {gpu_block_capacity} blocks")
        print(f"decode_window size : {decode_steps.X}")

        # Build ResultList for caller -------------------------------------------------------
        results = ResultList()
        for r in requests:
            stride    = int(prefetch_dist[r].X)
            distance  = -1 if offload_num[r].X == 0 else stride - 1
            results.append(
                Result(
                    id=r,
                    resume=True,
                    n=distance,
                    offload_num=int(offload_num[r].X),
                    slo_fail=slo_fail_per_decode[r].X * decode_steps.X,
                    slo_fail_r=slo_fail_per_decode[r].X,
                    actual_time=actual_time[r].X,
                    window=decode_steps.X,
                )
            )

        print(f"batch_time (token_time) : {results.batch_time:.4f}s")
        return results   

# New solver: MILP, accurate comm, accurate future forecast, and optimize for both token_time and decode_steps
class Solver_updated:
    def __init__(self, profiled_path: str | Path | None = None):
        self.profiled_estimator = ProfileBasedEstimator(profiled_path)
        self.which = "NoPrefetch"
        self.mode = "linear"

    # @staticmethod
    def solve(
        self,
        requests_list: list[Request],
        *,
        layer_num: int = 32,
        block_bandwidth: float = 103_178.0 / 1_000,   # blocks / second (16 tokens per block)
        gpu_block_capacity: int = 49_152 // 80,       # total blocks the GPU can hold
        window_ub: int = 130,                          # upper bound on decode-window length
    ) -> Optional[list[Result]]:
        """Optimise KV-placement and pre-fetch distance for the current micro-batch.

        Latency model v2  (layer-budget):
        • per-layer communication is a decision var  trans[r,j]  (blocks);
        • stall[j] captures residual comm > compute after overlap;
        • bw_cap_j couples trans, stall and compute to link bandwidth;
        • prefix LB+UB pins cumulative transfers exactly to cumulative demand;
        • optional prefetch-window filter ⇒ ≤ 1 live DMA stream / request.
        """

        # --------------------------------------------------------------------------
        # 0. Gather per-request constants
        # --------------------------------------------------------------------------
        requests          = [r.id for r in requests_list]
        context_blocks    = {r.id: r.context_len_in_blocks for r in requests_list}  # blocks/layer
        deposit_count     = {r.id: r.deposit_count          for r in requests_list}
        SLO               = {r.id: r.slo                   for r in requests_list}
        blocks_per_layer  = context_blocks                           # alias

        L      = layer_num
        BIG_M  = 1_000_000.0

        # --------------------------------------------------------------------------
        # 1. Enumerate admissible pre-fetch strides  (one best stride per n_off)
        # --------------------------------------------------------------------------
        # floor_val[d] = # CPU layers when stride = d+1
        floor_val = {d: layer_num // (d + 1) for d in range(1, layer_num)}   # 1 … 31
        floor_val[layer_num] = 0                                             # d = 32 ⇒ no off-load

        best_d_for_n = {n_off: max(d for d, n in floor_val.items() if n == n_off)
                        for n_off in floor_val.values()}                      # pick widest stride
        valid_dist    = sorted(best_d_for_n.values())                        # includes 32 now
        print(f"valid_dist: {valid_dist}")    # e.g. [1,2,3,4,5,7,9,15,31,32]

        # --------------------------------------------------------------------------
        # 1.b Prefetch-window mask  win[d][j]
        #     1 ⇔ layer j is inside the (d+1)-layer look-ahead window of some
        #         offloaded layer when stride = d+1.
        # --------------------------------------------------------------------------
        win = {d: {j: 0 for j in range(1, L + 1)} for d in valid_dist}
        for d in valid_dist:
            for l in range(d + 1, L + 1, d + 1):        # offloaded layers
                begin = max(1, l - d)                   # earliest legal copy layer
                for j in range(begin, l + 1):           # inclusive window [l-d, l]
                    win[d][j] = 1

        # --------------------------------------------------------------------------
        # 2. Pre-compute “is-offloaded” flag  a[r][d][j]
        #    a[r][d][j] = 1 if layer j of request r lives on CPU when stride = d+1.
        # --------------------------------------------------------------------------
        a = {
            r: {
                d: {j: int(j % (d + 1) == 0) for j in range(1, L + 1)}
                for d in valid_dist
            }
            for r in requests
        }

        # ---- outside the model: compute constants ----
        bucket_cnt   = (window_ub + 15) // 16          # number of 16-token buckets
        extra_comm_per16 = len(requests) / block_bandwidth        # sec
        tokens0 = 16 * sum(context_blocks.values())
        comp0 = self.profiled_estimator.estimate_by_profiled_results(
                    tokens0, "NoPrefetch", "linear") / 32.0
        comp16 = self.profiled_estimator.estimate_by_profiled_results(
                    tokens0 + 16, "NoPrefetch", "linear") / 32.0
        extra_comp_per16 = L * (comp16 - comp0)

        DELTA = extra_comm_per16 + extra_comp_per16                # sec / 16 tokens
        BASE_LAT = L * comp0                                       # baseline latency
        pos_slack = [SLO[r] - BASE_LAT for r in requests if SLO[r] - BASE_LAT > 0]
        M_val = min(pos_slack) if pos_slack else 1e3               # avoid 0 or neg

        PHI     = [1 + (k * DELTA) / M_val for k in range(bucket_cnt)]
        inv_PHI = [2.0 / v for v in PHI]

        # --------------------------------------------------------------------------
        # 3. Gurobi model & variables
        # --------------------------------------------------------------------------
        model = gp.Model("block_solver")
        # model.Params.NonConvex = 2              # (legacy) allow bilinear equalities

        # binary “keep in batch” (fixed 1 for all requests)
        resume = model.addVars(requests, lb=1, ub=1, vtype=GRB.BINARY, name="resume")

        # one-hot stride selector
        onc = model.addVars([(r, d) for r in requests for d in valid_dist],
                            vtype=GRB.BINARY, name="onc")

        prefetch_dist = model.addVars(requests, lb=1,  ub=L + 1, vtype=GRB.INTEGER,
                                    name="prefetch_dist")
        offload_num   = model.addVars(requests, lb=0,  ub=L,     vtype=GRB.INTEGER,
                                    name="offload_num")

        # decode-window length (secondary objective)
        decode_steps = model.addVar(lb=32, ub=window_ub, vtype=GRB.INTEGER,
                                    name="decode_steps")
        STEP   = 32
        k_mult = model.addVar(lb=1, ub=window_ub // STEP, vtype=GRB.INTEGER,
                            name="k_mult")
        #### !!!! ####
        extra_blocks = model.addVar(lb=0, ub=(window_ub + 15) // 16,
                                    vtype=GRB.INTEGER, name="extra_blocks")
        # ceil( decode_steps / 16 )
        model.addConstr(extra_blocks * 16 >= decode_steps,                       # lower
                        name="ceil_lb")
        model.addConstr(decode_steps >= 16 * (extra_blocks - 1) + 1,             # upper
                        name="ceil_ub")
        #### !!!! ####

        #### !!!! #### 
        # ---- inside the model (after extra_blocks link) ----
        seg = model.addVars(bucket_cnt, vtype=GRB.BINARY, name="seg")
        model.addConstr(seg.sum() == 1, name="seg_onehot")
        for k in range(bucket_cnt):
            model.addConstr(decode_steps >= 16*k      - window_ub * (1 - seg[k]))
            model.addConstr(decode_steps <= 16*(k+1)-1 + window_ub * (1 - seg[k]))

        inv_phi_factor = model.addVar(lb=min(inv_PHI), ub=max(inv_PHI),
                                    name="inv_phi_factor")
        model.addConstr(inv_phi_factor ==
                        gp.quicksum(inv_PHI[k] * seg[k] for k in range(bucket_cnt)),
                        name="invphi_def")
        #### !!!! #### 

        model.addConstr(decode_steps == STEP * k_mult, name="decode_steps_multiple")

        # latency & SLO vars
        stall        = model.addVars(range(1, L + 1), lb=0.0, name="stall")   # s_j
        token_time   = model.addVar(lb=0.0, name="token_time")
        actual_time  = model.addVars(requests, lb=0.0, ub=BIG_M, name="actual_time")
        slo_violate  = model.addVars(requests, vtype=GRB.BINARY, name="slo_violate")

        # NEW: layer-wise transfer amount (blocks)
        trans = model.addVars([(r, j) for r in requests for j in range(1, L + 1)],
                            lb=0.0, vtype=GRB.CONTINUOUS, name="trans")

        # --------------------------------------------------------------------------
        # 4. Fixed compute time per layer  (profiled “no-prefetch” curve)
        # --------------------------------------------------------------------------
        total_tokens = 16 * sum(context_blocks[r] for r in requests)   # rough estimate
        batch_layer  = (self.profiled_estimator
                        .estimate_by_profiled_results(total_tokens,
                                                    which="NoPrefetch",
                                                    mode="linear") / 32.0)

        # --------------------------------------------------------------------------
        # 5. Communication bandwidth  (blocks / second)
        # --------------------------------------------------------------------------
        per_block_time = (self.profiled_estimator
                        .estimate_by_profiled_results(total_tokens,
                                                        which="Communication",
                                                        mode="linear")
                        / (32 * 16))
        block_bandwidth = 1.0 / per_block_time
        print(f"[solve] profiled PCIe/NVLink bandwidth : {block_bandwidth:.2f} blocks/s")

        # --------------------------------------------------------------------------
        # 6. Constraints
        # --------------------------------------------------------------------------

        # 6.1 one stride per request (one-hot)
        model.addConstrs((gp.quicksum(onc[r, d] for d in valid_dist) == 1
                        for r in requests),
                        name="one_stride")

        # 6.2 derive stride length & #offloaded layers
        model.addConstrs((gp.quicksum(onc[r, d] * (d + 1) for d in valid_dist)
                        == prefetch_dist[r] for r in requests),
                        name="prefetch_dist_def")
        model.addConstrs((gp.quicksum(onc[r, d] * floor_val[d] for d in valid_dist)
                        == offload_num[r] for r in requests),
                        name="offload_num_def")

        # 6.3 Bandwidth capacity per layer  (layer-budget model)
        for j in range(1, L + 1):
            model.addConstr(gp.quicksum(trans[r, j] for r in requests)
                            <= block_bandwidth * (batch_layer + stall[j]),
                            name=f"bw_cap_{j}")

        # demand expression  blocks_needed[r,j]
        blocks_needed = {
            (r, j): gp.quicksum(onc[r, d] * a[r][d][j] * blocks_per_layer[r]
                                for d in valid_dist)
            for r in requests for j in range(1, L + 1)
        }

        # (legacy) spike-based comm time kept for reference -----------------------
        # t_comm_expr = {
        #     j: (
        #         gp.quicksum(
        #             onc[r, d] * a[r][d][j] * blocks_per_layer[r]
        #             for r in requests for d in valid_dist
        #         ) / block_bandwidth      # seconds
        #     )
        #     for j in range(1, L + 1)
        # }

        # 6.4 prefix-flow constraints  (LB + UB = exact)
        for r in requests:
            for j in range(1, L + 1):
                # lower bound – must have transferred required data so far
                model.addConstr(
                    gp.quicksum(trans[r, k] for k in range(1, j + 1))
                    >= gp.quicksum(blocks_needed[r, t] for t in range(1, j + 1)),
                    name=f"flowLB_{r}_{j}"
                )
        # --- NEW window-scoped upper bound  (kills dead slack, allows early prefetch) ---
        for r in requests:
            for j in range(1, L + 1):
                # how many off-load windows are already “open” at layer j
                windows_open = gp.quicksum(
                    onc[r, d] * win[d][j]          # 1 if window of stride d is open
                    for d in valid_dist
                )
                model.addConstr(
                    gp.quicksum(trans[r, k] for k in range(1, j + 1))
                    <= windows_open * blocks_per_layer[r],
                    name=f"flowUB_{r}_{j}"
                )

        # total transferred blocks = offloaded blocks
        # for r in requests:
        #     model.addConstr(gp.quicksum(trans[r, j] for j in range(1, L + 1))
        #                     == offload_num[r] * blocks_per_layer[r],
        #                     name=f"flow_total_{r}")
        for r in requests:
            model.addConstr(
                gp.quicksum(trans[r, j] for j in range(1, L + 1))
                == offload_num[r] * context_blocks[r],          # baseline block size
                name=f"flow_total_{r}")
        # 6.5 Stall definition (based on NEW comm_j)
        for j in range(1, L + 1):
            comm_j = gp.quicksum(trans[r, j] for r in requests) / block_bandwidth
            if j == 1:
                model.addConstr(stall[1] >= comm_j, name="stall_first")
            else:
                model.addConstr(stall[j] >= comm_j - batch_layer, name=f"stall_{j}")
            model.addConstr(stall[j] <= comm_j, name=f"stall_ub_{j}")

        # 6.6 Prefetch-window filter  (≤1 live DMA stream / request)
        for r in requests:
            for j in range(1, L + 1):
                model.addConstr(
                    trans[r, j]
                    <= gp.quicksum(onc[r, d] * win[d][j] * blocks_per_layer[r]
                                for d in valid_dist),
                    name=f"win_{r}_{j}"
                )

        # 6.7 batch latency definition
        model.addConstr(token_time == L * batch_layer
                        + gp.quicksum(stall[j] for j in range(1, L + 1)),
                        name="token_time_def")

        # 6.8 bind per-request actual_time
        model.addConstrs((actual_time[r] == token_time for r in requests),
                        name="actual_time_def")

        # 6.9 SLO logic -----------------------------------------------------------
        M_slo = window_ub * max(SLO.values())            # tight big-M
        model.addConstrs((actual_time[r] - SLO[r]
                        <= M_slo * slo_violate[r] for r in requests),
                        name="slo_violate_def")
        EPS = 1e-5
        model.addConstrs((actual_time[r] - SLO[r]
                        >= EPS - M_slo * (1 - slo_violate[r])
                        for r in requests),
                        name="slo_violate_eq")
        model.addConstr(gp.quicksum(slo_violate[r] for r in requests) <= 2,
                        name="slo_violate_total")
        # 6.9 SLO indicator logic ----------------------------------------------------
        M_slo = window_ub * max(SLO.values())          # tight big-M

        # slo_violate[r] = 1  ⇔  actual_time[r] > SLO[r]
        model.addConstrs(
            (actual_time[r] - SLO[r] <= M_slo * slo_violate[r]   for r in requests),
            name="slo_violate_def"
        )
        EPS = 1e-5
        model.addConstrs(
            (actual_time[r] - SLO[r] >= EPS - M_slo * (1 - slo_violate[r])
            for r in requests),
            name="slo_violate_eq"
        )

        # (legacy) global “≤ 2 offenders” cap — delete or comment out
        # model.addConstr(
        #     gp.quicksum(slo_violate[r] for r in requests) <= 2,
        #     name="slo_violate_total"
        # )

        # ---------------------------------------------------------------------------
        # 6.y  Average SLO-failure budget (MILP, no bilinear)
        # ---------------------------------------------------------------------------
        viol_tokens = model.addVars(requests, lb=0.0, ub=window_ub, name="viol_tokens")

        # C1: if slo_violate == 0  ⇒  viol_tokens = 0
        # model.addConstrs(
        #     (viol_tokens[r] <= decode_steps * slo_violate[r]   for r in requests),
        #     name="viol_tok_ub"
        # )
        # --- C1 & C1'  (indicator form → no bilinear term) -----------------
        for r in requests:
            # slo_violate == 0  ⇒  viol_tokens = 0
            model.addGenConstrIndicator(slo_violate[r], False,
                                        viol_tokens[r] == 0,
                                        name=f"viol_tok_zero_{r}")
            # slo_violate == 1  ⇒  viol_tokens ≤ decode_steps
            model.addGenConstrIndicator(slo_violate[r], True,
                                        viol_tokens[r] <= decode_steps,
                                        name=f"viol_tok_ub_{r}")

        # C2: if slo_violate == 1  ⇒  ≥ (decode_steps - deposit_count) late tokens
        M_tok = window_ub
        model.addConstrs(
            (viol_tokens[r] >= decode_steps - deposit_count[r] - M_tok * (1 - slo_violate[r])
            for r in requests),
            name="viol_tok_lb"
        )

        # C3: global average-failure budget  ≤ 2 tokens per decode window
        # model.addConstr(
        #     gp.quicksum(viol_tokens[r] for r in requests) <= 2,
        #     name="viol_tok_total"
        # )
        model.addConstr(
            gp.quicksum(viol_tokens[r] for r in requests) <= inv_phi_factor,
            name="viol_tok_total")
        # 6.10 GPU memory capacity
        # model.addConstr(gp.quicksum((L - offload_num[r]) * context_blocks[r]
        #                             for r in requests)
        #                 <= gpu_block_capacity,
        #                 name="gpu_memory_cap")
        # replace the old gpu_memory_cap block
        # grow_blocks = {r: context_blocks[r] + extra_blocks for r in requests}

        # model.addConstr(
        #     gp.quicksum((L - offload_num[r]) * grow_blocks[r] for r in requests)
        #     <= gpu_block_capacity,
        #     name="gpu_memory_cap")
        # --- helper: total #layers kept on-GPU across the batch --------------
        undecode_total = model.addVar(lb=0, ub=L * len(requests),
                                    vtype=GRB.INTEGER, name="undecode_total")
        model.addConstr(
            undecode_total == gp.quicksum(L - offload_num[r] for r in requests),
            name="undecode_total_def")
        # --- linear surrogate for  extra_blocks × undecode_total -------------
        mem_prod = model.addVar(lb=0, ub=gpu_block_capacity,
                                vtype=GRB.CONTINUOUS, name="mem_prod")

        # bounds for McCormick:  extra_blocks ∈ [2, bucket_cnt] ;
        #                        undecode_total ∈ [0, L * |R|]
        xL, xU = 2, bucket_cnt
        yL, yU = 0, L * len(requests)

        model.addConstr(mem_prod >= xL * undecode_total + yL * extra_blocks - xL * yL,
                        name="mem_mcc1")
        model.addConstr(mem_prod >= xU * undecode_total + yU * extra_blocks - xU * yU,
                    name="mem_mcc2")
        model.addConstr(mem_prod <= xU * undecode_total + yL * extra_blocks - xU * yL,
                        name="mem_mcc3")
        model.addConstr(mem_prod <= xL * undecode_total + yU * extra_blocks - xL * yU,
                    name="mem_mcc4")
        model.addConstr(
            gp.quicksum((L - offload_num[r]) * context_blocks[r] for r in requests)
            + mem_prod <= gpu_block_capacity,
            name="gpu_memory_cap")

        # --------------------------------------------------------------------------
        # 7. Objectives
        # --------------------------------------------------------------------------
        model.setObjectiveN(token_time, 0, 1, 1, GRB.MINIMIZE)   # primary – latency
        model.setObjectiveN(-decode_steps, 1, 2, 1, GRB.MINIMIZE)  # secondary – window

        # Gurobi parameters --------------------------------------------------------
        model.Params.OutputFlag    = 1
        model.Params.PoolSearchMode = 2
        model.Params.PoolSolutions  = 5000
        model.Params.Presolve       = 0
        model.Params.Aggregate      = 0
        model.Params.CutPasses      = 0

        # --------------------------------------------------------------------------
        # 8. Solve
        # --------------------------------------------------------------------------
        try:
            model.optimize()
        except gp.GurobiError as e:
            print(f"[solve] Gurobi Error: {e}")
            return None

        # --------------------------------------------------------------------------
        # 9. Extract results  (unchanged from previous version)
        # --------------------------------------------------------------------------
        if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            return None

        tokens_ok = sum(1.0 - slo_violate[r].X for r in requests)
        goodput   = tokens_ok / token_time.X

        print("\n--- Optimal solution (or best found) ---")
        print(" id | offload | SLO-fail | actual_time")
        print("----|---------|----------|------------")
        for r in requests:
            print(f"{r:>3} | {int(offload_num[r].X):>7} |"
                f" {slo_violate[r].X * decode_steps.X:8.2f} |"
                f" {actual_time[r].X:10.4f}s")

        print(f"\nGood-put           : {goodput:.4f} tokens/s")
        print(f"Batch latency      : {token_time.X:.4f}s "
            f"(includes {sum(stall[j].X for j in range(1, L + 1)):.4f}s stall)")
        print(f"GPU KV usage       : "
            f"{sum((L - offload_num[r].X) * context_blocks[r] for r in requests)} "
            f"/ {gpu_block_capacity} blocks")
        print(f"decode_window size : {decode_steps.X}")

        # Build ResultList ---------------------------------------------------------
        results = ResultList()
        for r in requests:
            stride    = int(prefetch_dist[r].X)
            distance  = -1 if offload_num[r].X == 0 else stride - 1
            results.append(
                Result(
                    id=r,
                    resume=True,
                    n=distance,
                    offload_num=int(offload_num[r].X),
                    slo_fail=slo_violate[r].X * decode_steps.X,
                    slo_fail_r=slo_violate[r].X,
                    actual_time=actual_time[r].X,
                    window=decode_steps.X,
                )
            )

        print(f"batch_time (token_time) : {results.batch_time:.4f}s")
        return results
    def compute_T_batch(
        self,                                        # <-- “self” so you can call it as a Solver method
        requests_list: List[Request],
        offload_decision: Dict[int, int],            # {req_id : distance (= d),  -1 ⇒ no offload}
        *,
        layer_num: int = 32,
        block_bandwidth: float = 103_178.0 / 1_000,  #   blocks · s⁻¹
        gpu_block_capacity: int = 49_152 // 80,      # *unused* but kept for API parity
        window_ub: int = 1_000                      # *unused*
    ) -> float:
        """
        Return the batch latency T_batch (seconds) for a *given* off-load plan.

        offload_decision[r]  ==  d   means pre-fetch distance d+1  (layers 0,d+1,2(d+1)… on CPU)
        offload_decision[r]  == -1   means no offload for request r (all KV on GPU).
        """
        # -----------------------------------------------------------------
        # 0. Sanity checks
        # -----------------------------------------------------------------
        req_ids = {r.id for r in requests_list}
        if req_ids != set(offload_decision):
            miss = req_ids - offload_decision.keys()
            extra = offload_decision.keys() - req_ids
            raise ValueError(f"offload_decision must cover every request "
                            f"(missing={miss}, extra={extra})")

        # -----------------------------------------------------------------
        # 1. Compute-time per layer (same profiler logic as in `solve`)
        # -----------------------------------------------------------------
        total_tokens = 16 * sum(r.context_len_in_blocks for r in requests_list)
        compute_per_layer = (
            self.profiled_estimator
                .estimate_by_profiled_results(total_tokens,
                                            which="NoPrefetch",
                                            mode="linear")
            / layer_num
        )

        # -----------------------------------------------------------------
        # 2. PCIe/NVLink bandwidth  (blocks / s)  — same derivation
        # -----------------------------------------------------------------
        per_block_time = (
            self.profiled_estimator
                .estimate_by_profiled_results(total_tokens,
                                            which="Communication",
                                            mode="linear")
            / (layer_num * 16)
        )
        eff_bandwidth = 1.0 / per_block_time if block_bandwidth is None else block_bandwidth
        
        # -----------------------------------------------------------------
        # 3. Layer-wise communication time  t_comm[j]
        # -----------------------------------------------------------------
        def stride(req_id: int) -> int:
            d = offload_decision[req_id]
            return 1 if d < 0 else d + 1            # 1 ⇒ fully resident

        t_comm = [0.0] * layer_num                 # index 0 ↔ layer 1
        for j in range(1, layer_num + 1):          # 1 … L
            blocks_this_layer = 0
            for r in requests_list:
                if j % stride(r.id) == 0:          # layer lives on CPU
                    blocks_this_layer += r.context_len_in_blocks
            t_comm[j - 1] = blocks_this_layer / eff_bandwidth

        # -----------------------------------------------------------------
        # 4. Stall
        # -----------------------------------------------------------------
        stall = [0.0] * layer_num
        stall[0] = t_comm[0]
        for j in range(1, layer_num):
            stall[j] = max(0.0, t_comm[j] - compute_per_layer)
        # -----------------------------------------------------------------
        # 5. Batch latency
        # -----------------------------------------------------------------
        T_batch = layer_num * compute_per_layer + sum(stall)
        return T_batch 
    
    
class LatencySolver:
    def __init__(self):
        self.profiled_estimator = ProfileBasedEstimator(profiled_path)
        self.which = "NoPrefetch"
        self.mode = "linear"
    # @staticmethod
    def solve(self, requests_list: list[Request], layer_num = 32, block_bandwidth = 103178.0 / 1000, gpu_block_capacity = 49152 / 80, window_ub = 1000) -> Optional[list[Result]]:
        requests = [r.id for r in requests_list]
        context_blocks = {r.id: r.context_len_in_blocks for r in requests_list}
        layer_time = {r.id: r.layer_time for r in requests_list}
        deposit_count = {r.id: r.deposit_count for r in requests_list}
        SLO = {r.id: r.slo for r in requests_list}
        gpu_layers = {r.id: r.gpu_layers_on_gpu for r in requests_list}
        # blocks_per_layer = {r.id: r.gpu_layers_per_seq   for r in requests_list}
        # gpu_cur_blocks = {
        #     rid: gpu_layers[rid] * blocks_per_layer[rid]   # (#레이어)×(블록/레이어)
        #     for rid in requests
        # }

        L = layer_num 
        M = 1e6                     # big-M

# === 2. floor/ceil 값 미리 계산 ===
        # floor_val = {d: L // (d + 1) for d in range(1, L+2)}
        floor_val = {d: L // (d + 1) for d in range(L)}          # d = 0 … 31
        # --- APPEND: enable the “no-offload” option ---------------------------------------
        floor_val[layer_num] = 0                                 # d = 32  →  n_off = 0

        best_d_for_n = {}                                        # n_off ➜ best (widest) distance
        for d, n_off in floor_val.items():
            if n_off not in best_d_for_n or d > best_d_for_n[n_off]:
                best_d_for_n[n_off] = d

        valid_dist = sorted(best_d_for_n.values())               # includes 0 and 32
        valid_dist = valid_dist[1:]                              # drop 0-distance entry only

        print(floor_val)      # debugging – now shows key 32 : 0
        print(valid_dist)     # e.g. [1, 2, 3, 4, 5, 7, 9, 15, 31, 32]
# === 3. 모델 생성 및 설정 ===
        model = gp.Model('block_solver')
        model.Params.NonConvex = 2    # 비선형 곱 제약 허용

# === 4. 의사결정 변수 ===
        prefetch_dist = model.addVars(requests, lb=1, ub=L+2, vtype=GRB.INTEGER, name='prefetch_dist')
        offload_num = model.addVars(requests, lb=0, ub=L, vtype=GRB.INTEGER, name='offload_num')

        decode_steps = model.addVar(lb=32, ub=window_ub, vtype=GRB.INTEGER, name='decode_steps')

        # ----- binary choice variables only for the preferred distances -----
        onc = {
            (r, d): model.addVar(vtype=GRB.BINARY, name=f"onc_{r}_{d}")
            for r in requests for d in valid_dist
        }

        # ----- exactly one distance per request -----
        model.addConstrs(
            (gp.quicksum(onc[r, d] for d in valid_dist) == 1
            for r in requests),
            name="one_distance"
        )

        # ----- link distance → model vars -----
        model.addConstrs(
            (gp.quicksum(onc[r, d] * (d + 1) for d in valid_dist) == prefetch_dist[r]
            for r in requests),
            name="prefetch_dist_def"
        )

        model.addConstrs(
            (gp.quicksum(onc[r, d] * (L // (d + 1)) for d in valid_dist) == offload_num[r]
            for r in requests),
            name="offload_num_def"
        )
        num_tokens = sum(context_blocks[r] for r in requests)*16
        batch_layer = self.profiled_estimator.estimate_by_profiled_results(num_tokens, which="NoPrefetch", mode="linear")/32
        # print(f"batch_layer: {batch_layer} for {num_tokens} tokens")
        
        # batch_layer  /= 32
        comm_time    = model.addVar(lb=0, name='comm_time')
        comp_time    = model.addVar(lb=0, name='comp_time')
        token_time   = model.addVar(lb=0, name='token_time')
        actual_time  = model.addVars(requests, lb=0, ub=M, name='actual_time')
        ratio        = model.addVars(requests, lb=0, name='ratio')
        slo_fail_per_decode     = model.addVars(requests, lb=0, name='slo_fail_per_decode')

# === 5. 제약식 ===

# 5.2 batch_layer = max(layer_time[r] * resume[r])
        # sum the sequence length, not the layer time!
                # batch_layer = (sum(context_blocks[r] for r in requests) * PROFILED_A + PROFILED_B) / 32
                # print("batch_layer", batch_layer)
        model.addConstr(1 <= gp.quicksum(resume[r] for r in requests), name="should execute more than 1")

# 5.3 comm_time 정의: 필요한 블록 수 ÷ block_bandwidth

        # # temp (xinyue)
        # per_block_time = self.profiled_estimator.estimate_by_profiled_results(num_tokens, which="Communication", mode="linear") / (32*16) # 32 layer, 16 tokens per block
        # block_bandwidth = 1 / per_block_time
        # print(f"block_bandwidth: {block_bandwidth} blks/s")
        # block_bandwidth = 1/per_block_time
        # model.addConstr(
        #     comm_time == gp.quicksum(
        #         resume[r] * offload_num[r] * context_blocks[r]
        #         for r in requests
        #     ) / block_bandwidth,
        #     name="comm_time_def"
        # )

# 5.4 comp_time = batch_layer * L
        model.addConstr(comp_time == batch_layer * L, name="comp_time_def")

# 5.4.1 move_overhead = max(0, move_blocks[r] * context_blocks[r] / block_bandwidth)        
        # move_overhead = model.addVar(lb=0, name='move_overhead')
        # model.addConstr(
        # move_overhead == gp.quicksum(
        #     resume[r] * move_blocks[r] * context_blocks[r]
        #     for r in requests
        # ) / block_bandwidth,
        # name="move_overhead_def"
        # )

# 5.5 token_time = max(comm_time, comp_time)
        model.addGenConstrMax(token_time, [comm_time, comp_time], name="token_time_max")

# 5.6 actual_time indicator
        for r in requests:
            model.addGenConstrIndicator(resume[r], True,
                                        actual_time[r] == token_time,
                                        name=f"act_on_{r}")
            model.addGenConstrIndicator(resume[r], False,
                                        actual_time[r] >= M,
                                        name=f"act_off_{r}")


# 5.8 ratio * H = deposit_timer (비선형 제약)
        for r in requests:
            model.addQConstr(ratio[r] * actual_time[r] == decode_steps * SLO[r], name=f"ratio_qc_{r}")

        for r in requests:
            model.addConstr(slo_fail_per_decode[r] * decode_steps >= decode_steps - ratio[r] - deposit_count[r],
                            name=f"slo_fail_{r}")

# 5.11 GPU 메모리 제약: 필요한 블록 수 ≤ gpu_block_capacity
        model.addConstr(
            gp.quicksum(
                (L - offload_num[r]) * context_blocks[r]
                for r in requests
            ) <= gpu_block_capacity,
            name="memory_limit"
        )
        # 5.12 SLO 실패 총합 ≤ decode_steps   ←★ 새로 추가
        model.addConstr(
            gp.quicksum(slo_fail_per_decode[r] for r in requests) <= 1,
            name="slo_fail_total_limit"
        )

        # 🆕 goodput 분모: token_time + move_overhead
        eff_latency = model.addVar(lb=0, name='effective_latency')
        model.addConstr(eff_latency == token_time,
                        name="eff_lat_def")

# === 6. 목적함수 및 최적화 ===
        # goodput = model.addVar(lb=0, name='obj')
        # model.addConstr(goodput * eff_latency == gp.quicksum(resume[r] for r in requests) - slo_fail_per_decode.sum())
        # model.setObjective(goodput, GRB.MAXIMIZE)
        model.setObjective(token_time, GRB.MINIMIZE)

        # model.setObjectiveN(token_time,0,1,1,GRB.MINIMIZE)
        # model.setObjectiveN(decode_steps,1,2,1,GRB.MAXIMIZE)  # minimize decode_steps as secondary objective
        # model.setObjective(token_time, GRB.MINIMIZE)
        model.Params.OutputFlag = 1
        try:
            model.optimize()
        except gp.GurobiError as e:
            print(f"Gurobi Error: {e}")
            return None
        print(f"\n--- solution pool ---")
        for k in range(model.SolCount):
            model.Params.SolutionNumber = k
            latency = token_time.Xn
            print(f"sol #{k:2d}: latency={latency:.5f}s  "
                f"stride={[offload_num[r].Xn for r in requests]} ")
# === 7. 결과 출력 ===
        if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT or model.Status == GRB.SUBOPTIMAL:
            print("\n--- Optimal Solution ---")
            print(" r | resume | offload_num | slo_fail | actual_time")
            print("---|--------|-------------|----------|-------------")
            for r in requests:
                print(f"{r:>2} |   {int(resume[r].X)}    |"
                      f"      {int(offload_num[r].X):2d}     |"
                      f" {slo_fail_per_decode[r].X * decode_steps.X:8.2f} | {actual_time[r].X:10.2f}")
            # print(f"\ngoodput = {model.ObjVal:.2f}")
            print(f"""\nMem Usage: {sum([
                (L - offload_num[r].X) * context_blocks[r]
                for r in requests
            ])}/{gpu_block_capacity}""")
            print(f"decode_steps: {decode_steps.X}")
            print(f"comm time: {comm_time.X}")
            print(f"comp time: {comp_time.X}")

            result = ResultList()

            for r in requests:
                stride = int(prefetch_dist[r].X)    # s
                distance = -1 if offload_num[r].X == 0 else stride - 1
                result.append(
                    Result(id=r,
                        resume=bool(resume[r].X),
                        n=distance,
                        offload_num=int(offload_num[r].X),
                        slo_fail=slo_fail_per_decode[r].X,
                        actual_time=actual_time[r].X,
                        window=decode_steps.X)
                )
            batch_time = result.batch_time
            print(f"batch time: {batch_time}")
            
            return result
        else:
            return None


# class Solver_uniform(Solver_updated):
#     def solve(
#         self,
#         requests_list: list[Request],
#         *,
#         layer_num: int = 32,
#         block_bandwidth: float = 103_178.0 / 1_000,   # blocks / second (16 tokens per block)
#         gpu_block_capacity: int = 49_152 // 80,       # total blocks the GPU can hold
#         window_ub: int = 130,                          # upper bound on decode-window length
#     ) -> Optional[list[Result]]:
#         """Optimise KV-placement and pre-fetch distance for the current micro-batch.

#         Latency model v2  (layer-budget):
#         • per-layer communication is a decision var  trans[r,j]  (blocks);
#         • stall[j] captures residual comm > compute after overlap;
#         • bw_cap_j couples trans, stall and compute to link bandwidth;
#         • prefix LB+UB pins cumulative transfers exactly to cumulative demand;
#         • optional prefetch-window filter ⇒ ≤ 1 live DMA stream / request.
#         """

#         # --------------------------------------------------------------------------
#         # 0. Gather per-request constants
#         # --------------------------------------------------------------------------
#         requests          = [r.id for r in requests_list]
#         context_blocks    = {r.id: r.context_len_in_blocks for r in requests_list}  # blocks/layer
#         deposit_count     = {r.id: r.deposit_count          for r in requests_list}
#         SLO               = {r.id: r.slo                   for r in requests_list}
#         blocks_per_layer  = context_blocks                           # alias

#         L      = layer_num
#         BIG_M  = 1_000_000.0

#         # --------------------------------------------------------------------------
#         # 1. Enumerate admissible pre-fetch strides  (one best stride per n_off)
#         # --------------------------------------------------------------------------
#         # floor_val[d] = # CPU layers when stride = d+1
#         floor_val = {d: layer_num // (d + 1) for d in range(1, layer_num)}   # 1 … 31
#         floor_val[layer_num] = 0                                             # d = 32 ⇒ no off-load

#         best_d_for_n = {n_off: max(d for d, n in floor_val.items() if n == n_off)
#                         for n_off in floor_val.values()}                      # pick widest stride
#         valid_dist    = sorted(best_d_for_n.values())                        # includes 32 now
#         print(f"valid_dist: {valid_dist}")    # e.g. [1,2,3,4,5,7,9,15,31,32]

#         # --------------------------------------------------------------------------
#         # 1.b Prefetch-window mask  win[d][j]
#         #     1 ⇔ layer j is inside the (d+1)-layer look-ahead window of some
#         #         offloaded layer when stride = d+1.
#         # --------------------------------------------------------------------------
#         win = {d: {j: 0 for j in range(1, L + 1)} for d in valid_dist}
#         for d in valid_dist:
#             for l in range(d + 1, L + 1, d + 1):        # offloaded layers
#                 begin = max(1, l - d)                   # earliest legal copy layer
#                 for j in range(begin, l + 1):           # inclusive window [l-d, l]
#                     win[d][j] = 1

#         # --------------------------------------------------------------------------
#         # 2. Pre-compute “is-offloaded” flag  a[r][d][j]
#         #    a[r][d][j] = 1 if layer j of request r lives on CPU when stride = d+1.
#         # --------------------------------------------------------------------------
#         a = {
#             r: {
#                 d: {j: int(j % (d + 1) == 0) for j in range(1, L + 1)}
#                 for d in valid_dist
#             }
#             for r in requests
#         }

#         # ---- outside the model: compute constants ----
#         bucket_cnt   = (window_ub + 15) // 16          # number of 16-token buckets
#         extra_comm_per16 = len(requests) / block_bandwidth        # sec
#         tokens0 = 16 * sum(context_blocks.values())
#         comp0 = self.profiled_estimator.estimate_by_profiled_results(
#                     tokens0, "NoPrefetch", "linear") / 32.0
#         comp16 = self.profiled_estimator.estimate_by_profiled_results(
#                     tokens0 + 16, "NoPrefetch", "linear") / 32.0
#         extra_comp_per16 = L * (comp16 - comp0)

#         DELTA = extra_comm_per16 + extra_comp_per16                # sec / 16 tokens
#         BASE_LAT = L * comp0                                       # baseline latency
#         pos_slack = [SLO[r] - BASE_LAT for r in requests if SLO[r] - BASE_LAT > 0]
#         M_val = min(pos_slack) if pos_slack else 1e3               # avoid 0 or neg

#         PHI     = [1 + (k * DELTA) / M_val for k in range(bucket_cnt)]
#         inv_PHI = [2.0 / v for v in PHI]

#         # --------------------------------------------------------------------------
#         # 3. Gurobi model & variables
#         # --------------------------------------------------------------------------
#         model = gp.Model("block_solver")
#         # model.Params.NonConvex = 2              # (legacy) allow bilinear equalities

#         # binary “keep in batch” (fixed 1 for all requests)
#         resume = model.addVars(requests, lb=1, ub=1, vtype=GRB.BINARY, name="resume")

#         # one-hot stride selector
#         onc = model.addVars([(r, d) for r in requests for d in valid_dist],
#                             vtype=GRB.BINARY, name="onc")

#         prefetch_dist = model.addVars(requests, lb=1,  ub=L + 1, vtype=GRB.INTEGER,
#                                     name="prefetch_dist")
#         offload_num   = model.addVars(requests, lb=0,  ub=L,     vtype=GRB.INTEGER,
#                                     name="offload_num")

#         # decode-window length (secondary objective)
#         decode_steps = model.addVar(lb=32, ub=window_ub, vtype=GRB.INTEGER,
#                                     name="decode_steps")
#         STEP   = 32
#         k_mult = model.addVar(lb=1, ub=window_ub // STEP, vtype=GRB.INTEGER,
#                             name="k_mult")
#         extra_blocks = model.addVar(lb=0, ub=(window_ub + 15) // 16,
#                                     vtype=GRB.INTEGER, name="extra_blocks")
#         # ceil( decode_steps / 16 )
#         model.addConstr(extra_blocks * 16 >= decode_steps,                       # lower
#                         name="ceil_lb")
#         model.addConstr(decode_steps >= 16 * (extra_blocks - 1) + 1,             # upper
#                         name="ceil_ub")

#         # ---- inside the model (after extra_blocks link) ----
#         seg = model.addVars(bucket_cnt, vtype=GRB.BINARY, name="seg")
#         model.addConstr(seg.sum() == 1, name="seg_onehot")
#         for k in range(bucket_cnt):
#             model.addConstr(decode_steps >= 16*k      - window_ub * (1 - seg[k]))
#             model.addConstr(decode_steps <= 16*(k+1)-1 + window_ub * (1 - seg[k]))

#         inv_phi_factor = model.addVar(lb=min(inv_PHI), ub=max(inv_PHI),
#                                     name="inv_phi_factor")
#         model.addConstr(inv_phi_factor ==
#                         gp.quicksum(inv_PHI[k] * seg[k] for k in range(bucket_cnt)),
#                         name="invphi_def")

#         model.addConstr(decode_steps == STEP * k_mult, name="decode_steps_multiple")

#         # latency & SLO vars
#         stall        = model.addVars(range(1, L + 1), lb=0.0, name="stall")   # s_j
#         token_time   = model.addVar(lb=0.0, name="token_time")
#         actual_time  = model.addVars(requests, lb=0.0, ub=BIG_M, name="actual_time")
#         slo_violate  = model.addVars(requests, vtype=GRB.BINARY, name="slo_violate")

#         # NEW: layer-wise transfer amount (blocks)
#         trans = model.addVars([(r, j) for r in requests for j in range(1, L + 1)],
#                             lb=0.0, vtype=GRB.CONTINUOUS, name="trans")

#         # --------------------------------------------------------------------------
#         # 4. Fixed compute time per layer  (profiled “no-prefetch” curve)
#         # --------------------------------------------------------------------------
#         total_tokens = 16 * sum(context_blocks[r] for r in requests)   # rough estimate
#         batch_layer  = (self.profiled_estimator
#                         .estimate_by_profiled_results(total_tokens,
#                                                     which="NoPrefetch",
#                                                     mode="linear") / 32.0)

#         # --------------------------------------------------------------------------
#         # 5. Communication bandwidth  (blocks / second)
#         # --------------------------------------------------------------------------
#         per_block_time = (self.profiled_estimator
#                         .estimate_by_profiled_results(total_tokens,
#                                                         which="Communication",
#                                                         mode="linear")
#                         / (32 * 16))
#         block_bandwidth = 1.0 / per_block_time
#         print(f"[solve] profiled PCIe/NVLink bandwidth : {block_bandwidth:.2f} blocks/s")

#         # --------------------------------------------------------------------------
#         # 6. Constraints
#         # --------------------------------------------------------------------------

#         # 6.1 one stride per request (one-hot)
#         model.addConstrs((gp.quicksum(onc[r, d] for d in valid_dist) == 1
#                         for r in requests),
#                         name="one_stride")

#         # 6.2 derive stride length & #offloaded layers
#         model.addConstrs((gp.quicksum(onc[r, d] * (d + 1) for d in valid_dist)
#                         == prefetch_dist[r] for r in requests),
#                         name="prefetch_dist_def")
#         model.addConstrs((gp.quicksum(onc[r, d] * floor_val[d] for d in valid_dist)
#                         == offload_num[r] for r in requests),
#                         name="offload_num_def")
#         # 6.2b  --- all requests must share the same stride ---
#         model.addConstrs(
#             (prefetch_dist[r] == prefetch_dist[requests[0]]  for r in requests),
#             name="global_stride"
#         )
#         # 6.3 Bandwidth capacity per layer  (layer-budget model)
#         for j in range(1, L + 1):
#             model.addConstr(gp.quicksum(trans[r, j] for r in requests)
#                             <= block_bandwidth * (batch_layer + stall[j]),
#                             name=f"bw_cap_{j}")

#         # demand expression  blocks_needed[r,j]
#         blocks_needed = {
#             (r, j): gp.quicksum(onc[r, d] * a[r][d][j] * blocks_per_layer[r]
#                                 for d in valid_dist)
#             for r in requests for j in range(1, L + 1)
#         }

#         # 6.4 prefix-flow constraints  (LB + UB = exact)
#         for r in requests:
#             for j in range(1, L + 1):
#                 # lower bound – must have transferred required data so far
#                 model.addConstr(
#                     gp.quicksum(trans[r, k] for k in range(1, j + 1))
#                     >= gp.quicksum(blocks_needed[r, t] for t in range(1, j + 1)),
#                     name=f"flowLB_{r}_{j}"
#                 )
#         # --- NEW window-scoped upper bound  (kills dead slack, allows early prefetch) ---
#         for r in requests:
#             for j in range(1, L + 1):
#                 # how many off-load windows are already “open” at layer j
#                 windows_open = gp.quicksum(
#                     onc[r, d] * win[d][j]          # 1 if window of stride d is open
#                     for d in valid_dist
#                 )
#                 model.addConstr(
#                     gp.quicksum(trans[r, k] for k in range(1, j + 1))
#                     <= windows_open * blocks_per_layer[r],
#                     name=f"flowUB_{r}_{j}"
#                 )

#         for r in requests:
#             model.addConstr(
#                 gp.quicksum(trans[r, j] for j in range(1, L + 1))
#                 == offload_num[r] * context_blocks[r],          # baseline block size
#                 name=f"flow_total_{r}")
#         # 6.5 Stall definition (based on NEW comm_j)
#         for j in range(1, L + 1):
#             comm_j = gp.quicksum(trans[r, j] for r in requests) / block_bandwidth
#             if j == 1:
#                 model.addConstr(stall[1] >= comm_j, name="stall_first")
#             else:
#                 model.addConstr(stall[j] >= comm_j - batch_layer, name=f"stall_{j}")
#             model.addConstr(stall[j] <= comm_j, name=f"stall_ub_{j}")

#         # 6.6 Prefetch-window filter  (≤1 live DMA stream / request)
#         for r in requests:
#             for j in range(1, L + 1):
#                 model.addConstr(
#                     trans[r, j]
#                     <= gp.quicksum(onc[r, d] * win[d][j] * blocks_per_layer[r]
#                                 for d in valid_dist),
#                     name=f"win_{r}_{j}"
#                 )

#         # 6.7 batch latency definition
#         model.addConstr(token_time == L * batch_layer
#                         + gp.quicksum(stall[j] for j in range(1, L + 1)),
#                         name="token_time_def")

#         # 6.8 bind per-request actual_time
#         model.addConstrs((actual_time[r] == token_time for r in requests),
#                         name="actual_time_def")

#         # 6.9 SLO logic -----------------------------------------------------------
#         M_slo = window_ub * max(SLO.values())            # tight big-M
#         model.addConstrs((actual_time[r] - SLO[r]
#                         <= M_slo * slo_violate[r] for r in requests),
#                         name="slo_violate_def")
#         EPS = 1e-5
#         model.addConstrs((actual_time[r] - SLO[r]
#                         >= EPS - M_slo * (1 - slo_violate[r])
#                         for r in requests),
#                         name="slo_violate_eq")
#         model.addConstr(gp.quicksum(slo_violate[r] for r in requests) <= 2,
#                         name="slo_violate_total")
#         # 6.9 SLO indicator logic ----------------------------------------------------
#         M_slo = window_ub * max(SLO.values())          # tight big-M

#         # slo_violate[r] = 1  ⇔  actual_time[r] > SLO[r]
#         model.addConstrs(
#             (actual_time[r] - SLO[r] <= M_slo * slo_violate[r]   for r in requests),
#             name="slo_violate_def"
#         )
#         EPS = 1e-5
#         model.addConstrs(
#             (actual_time[r] - SLO[r] >= EPS - M_slo * (1 - slo_violate[r])
#             for r in requests),
#             name="slo_violate_eq"
#         )

#         # ---------------------------------------------------------------------------
#         # 6.y  Average SLO-failure budget (MILP, no bilinear)
#         # ---------------------------------------------------------------------------
#         viol_tokens = model.addVars(requests, lb=0.0, ub=window_ub, name="viol_tokens")

#         # C1: if slo_violate == 0  ⇒  viol_tokens = 0
#         # model.addConstrs(
#         #     (viol_tokens[r] <= decode_steps * slo_violate[r]   for r in requests),
#         #     name="viol_tok_ub"
#         # )
#         # --- C1 & C1'  (indicator form → no bilinear term) -----------------
#         for r in requests:
#             # slo_violate == 0  ⇒  viol_tokens = 0
#             model.addGenConstrIndicator(slo_violate[r], False,
#                                         viol_tokens[r] == 0,
#                                         name=f"viol_tok_zero_{r}")
#             # slo_violate == 1  ⇒  viol_tokens ≤ decode_steps
#             model.addGenConstrIndicator(slo_violate[r], True,
#                                         viol_tokens[r] <= decode_steps,
#                                         name=f"viol_tok_ub_{r}")

#         # C2: if slo_violate == 1  ⇒  ≥ (decode_steps - deposit_count) late tokens
#         M_tok = window_ub
#         model.addConstrs(
#             (viol_tokens[r] >= decode_steps - deposit_count[r] - M_tok * (1 - slo_violate[r])
#             for r in requests),
#             name="viol_tok_lb"
#         )

#         model.addConstr(
#             gp.quicksum(viol_tokens[r] for r in requests) <= inv_phi_factor,
#             name="viol_tok_total")
#         # --- helper: total #layers kept on-GPU across the batch --------------
#         undecode_total = model.addVar(lb=0, ub=L * len(requests),
#                                     vtype=GRB.INTEGER, name="undecode_total")
#         model.addConstr(
#             undecode_total == gp.quicksum(L - offload_num[r] for r in requests),
#             name="undecode_total_def")
#         # --- linear surrogate for  extra_blocks × undecode_total -------------
#         mem_prod = model.addVar(lb=0, ub=gpu_block_capacity,
#                                 vtype=GRB.CONTINUOUS, name="mem_prod")

#         # bounds for McCormick:  extra_blocks ∈ [2, bucket_cnt] ;
#         #                        undecode_total ∈ [0, L * |R|]
#         xL, xU = 2, bucket_cnt
#         yL, yU = 0, L * len(requests)

#         model.addConstr(mem_prod >= xL * undecode_total + yL * extra_blocks - xL * yL,
#                         name="mem_mcc1")
#         model.addConstr(mem_prod >= xU * undecode_total + yU * extra_blocks - xU * yU,
#                     name="mem_mcc2")
#         model.addConstr(mem_prod <= xU * undecode_total + yL * extra_blocks - xU * yL,
#                         name="mem_mcc3")
#         model.addConstr(mem_prod <= xL * undecode_total + yU * extra_blocks - xL * yU,
#                     name="mem_mcc4")
#         model.addConstr(
#             gp.quicksum((L - offload_num[r]) * context_blocks[r] for r in requests)
#             + mem_prod <= gpu_block_capacity,
#             name="gpu_memory_cap")

#         # --------------------------------------------------------------------------
#         # 7. Objectives
#         # --------------------------------------------------------------------------
#         model.setObjectiveN(token_time, 0, 1, 1, GRB.MINIMIZE)   # primary – latency
#         model.setObjectiveN(-decode_steps, 1, 2, 1, GRB.MINIMIZE)  # secondary – window

#         # Gurobi parameters --------------------------------------------------------
#         model.Params.OutputFlag    = 1
#         model.Params.PoolSearchMode = 2
#         model.Params.PoolSolutions  = 5000
#         model.Params.Presolve       = 0
#         model.Params.Aggregate      = 0
#         model.Params.CutPasses      = 0

#         # --------------------------------------------------------------------------
#         # 8. Solve
#         # --------------------------------------------------------------------------
#         try:
#             model.optimize()
#         except gp.GurobiError as e:
#             print(f"[solve] Gurobi Error: {e}")
#             return None

#         # --------------------------------------------------------------------------
#         # 9. Extract results  (unchanged from previous version)
#         # --------------------------------------------------------------------------
#         if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
#             return None

#         tokens_ok = sum(1.0 - slo_violate[r].X for r in requests)
#         goodput   = tokens_ok / token_time.X

#         print("\n--- Optimal solution (or best found) ---")
#         print(" id | offload | SLO-fail | actual_time")
#         print("----|---------|----------|------------")
#         for r in requests:
#             print(f"{r:>3} | {int(offload_num[r].X):>7} |"
#                 f" {slo_violate[r].X * decode_steps.X:8.2f} |"
#                 f" {actual_time[r].X:10.4f}s")

#         print(f"\nGood-put           : {goodput:.4f} tokens/s")
#         print(f"Batch latency      : {token_time.X:.4f}s "
#             f"(includes {sum(stall[j].X for j in range(1, L + 1)):.4f}s stall)")
#         print(f"GPU KV usage       : "
#             f"{sum((L - offload_num[r].X) * context_blocks[r] for r in requests)} "
#             f"/ {gpu_block_capacity} blocks")
#         print(f"decode_window size : {decode_steps.X}")

#         # Build ResultList ---------------------------------------------------------
#         results = ResultList()
#         for r in requests:
#             stride    = int(prefetch_dist[r].X)
#             distance  = -1 if offload_num[r].X == 0 else stride - 1
#             results.append(
#                 Result(
#                     id=r,
#                     resume=True,
#                     n=distance,
#                     offload_num=int(offload_num[r].X),
#                     slo_fail=slo_violate[r].X * decode_steps.X,
#                     slo_fail_r=slo_violate[r].X,
#                     actual_time=actual_time[r].X,
#                     window=decode_steps.X,
#                 )
#             )

#         print(f"batch_time (token_time) : {results.batch_time:.4f}s")
#         return results

class Solver_uniform(Solver_v1):
    def solve(
        self,
        requests_list: list[Request],
        *,
        layer_num: int = 32,
        block_bandwidth: float = 103_178.0 / 1_000,     # blocks / second (16 tokens per block)
        gpu_block_capacity: int = 49_152 // 80,         # total blocks the GPU can hold
        window_ub: int = 1_000,                         # upper bound on decode-window length
    ) -> Optional[list[Result]]:
        """Optimise KV-placement and pre-fetch distance for the current micro-batch.

        The latency model is pipeline-accurate:
        • per-layer communication is counted in **blocks** (not bytes);
        • a non-negative stall variable s_j captures comm > compute overlap;
        • batch latency = Σ compute + Σ stall; feeds all SLO logic unchanged.
        """

        # ----------------------------------------------------------------------------------
        # 0.  Gather per-request constants
        # ----------------------------------------------------------------------------------
        requests          = [r.id for r in requests_list]
        context_blocks    = {r.id: r.context_len_in_blocks for r in requests_list}   # blocks/layer
        deposit_count     = {r.id: r.deposit_count          for r in requests_list}
        SLO               = {r.id: r.slo                   for r in requests_list}
        blocks_per_layer  = context_blocks                                                     # alias

        L      = layer_num
        BIG_M  = 1_000_000.0                              # large number for indicator fallback

        # ----------------------------------------------------------------------------------
        # 1.  Enumerate admissible pre-fetch strides  (one best stride for each n_off)
        # ----------------------------------------------------------------------------------
        # NOTE(HONG): allowing distance 0
        # floor_val     = {d: L // (d + 1) for d in range(L)}          # d = 0 … 31
        # NOTE(HONG): distance 0 is not allowed, so we start from 1
        floor_val    = {d: L // (d + 1) for d in range(1, L)}           # 1 ≤ d < L - > distance ∈ {1,2,3,4,5,7,9,15,31}
        best_d_for_n  = {n_off: max(d for d, n in floor_val.items() if n == n_off)
                        for n_off in floor_val.values()}
        valid_dist    = sorted(best_d_for_n.values())                # final stride set

        # ----------------------------------------------------------------------------------
        # 2.  Pre-compute “is-offloaded” flag  a[r][d][j]  (1 ≤ j ≤ L)
        # ----------------------------------------------------------------------------------
        a = {
            r: {
                d: {j: int(j % (d + 1) == 0) for j in range(1, L + 1)}   # layers (d+1), 2(d+1), …
                for d in valid_dist
            }
            for r in requests
        }

        # ----------------------------------------------------------------------------------
        # 3.  Gurobi model, variables
        # ----------------------------------------------------------------------------------
        model = gp.Model("block_solver")
        model.Params.NonConvex = 2                    # allow bilinear equalities

        # -- binary “keep in batch” (fixed to 1 for now) -----------------------------------
        resume        = model.addVars(requests, lb=1, ub=1, vtype=GRB.BINARY,   name="resume")

        # -- stride selection (one-hot) -----------------------------------------------------
        onc = model.addVars(
            [(r, d) for r in requests for d in valid_dist],
            vtype=GRB.BINARY, name="onc"
        )

        prefetch_dist = model.addVars(requests, lb=1,  ub=L + 1, vtype=GRB.INTEGER, name="prefetch_dist")
        offload_num   = model.addVars(requests, lb=0,  ub=L,     vtype=GRB.INTEGER, name="offload_num")

        decode_steps  = model.addVar(lb=32, ub=window_ub, vtype=GRB.INTEGER, name="decode_steps")

        # -- latency-and-SLO variables ------------------------------------------------------
        stall        = model.addVars(range(1, L + 1), lb=0.0, name="stall")   # s_j
        token_time   = model.addVar(lb=0.0, name="token_time")
        actual_time  = model.addVars(requests, lb=0.0, ub=BIG_M, name="actual_time")
        ratio        = model.addVars(requests, lb=0.0, name="ratio")
        slo_fail_per_decode = model.addVars(requests, lb=0.0, name="slo_fail_per_decode")

        trans = model.addVars([(r, j) for r in requests for j in range(1, L + 1)],
                              lb=0.0, vtype=GRB.CONTINUOUS, name="trans")
        # goodput       = model.addVar(lb=0.0, name="obj")                          # objective

        # ----------------------------------------------------------------------------------
        # 4.  Fixed compute time per layer  (profiled “no-prefetch” curve)
        # ----------------------------------------------------------------------------------
        total_tokens = 16 * sum(context_blocks[r] for r in requests)              # rough estimate
        batch_layer  = (
            self.profiled_estimator
            .estimate_by_profiled_results(total_tokens, which="NoPrefetch", mode="linear")
            / 32.0
        )  # seconds per layer when every layer is GPU-resident

        # ----------------------------------------------------------------------------------
        # 5.  Communication bandwidth (blocks / second)
        # ----------------------------------------------------------------------------------
        per_block_time  = (
            self.profiled_estimator
            .estimate_by_profiled_results(total_tokens, which="Communication", mode="linear")
            / (32 * 16)                                       # 32 layers, 16 tokens per block
        )
        block_bandwidth = 1.0 / per_block_time                # blocks / second
        print(f"[solve] profiled PCIe/NVLink bandwidth  : {block_bandwidth:.2f} blocks/s")

        # ----------------------------------------------------------------------------------
        # 6.  Constraints ───────────────────────────────────────────────────────────────────
        # ----------------------------------------------------------------------------------
        
        # 6.1 one stride per request --------------------------------------------------------
        model.addConstrs(
            (gp.quicksum(onc[r, d] for d in valid_dist) == resume[r] for r in requests),
            name="one_stride"
        )

        # 6.2 derive stride (prefetch_dist) and CPU-layer count (offload_num) ---------------
        model.addConstrs(
            (gp.quicksum(onc[r, d] * (d + 1)   for d in valid_dist) == prefetch_dist[r]
            for r in requests),
            name="prefetch_dist_def"
        )
        model.addConstrs(
            (gp.quicksum(onc[r, d] * floor_val[d] for d in valid_dist) == offload_num[r]
            for r in requests),
            name="offload_num_def"
        )
        ###### UNIFORM STRIDE #####
        model.addConstrs(
            (prefetch_dist[r] == prefetch_dist[requests[0]]  for r in requests),
            name="global_stride"
        )
        ###### UNIFORM STRIDE #####
        # 6.3 Bandwidth capacity per layer  (layer-budget model)
        for j in range(1, L + 1):
            model.addConstr(gp.quicksum(trans[r, j] for r in requests)
                            <= block_bandwidth * (batch_layer + stall[j]),
                            name=f"bw_cap_{j}")
        # demand expression  blocks_needed[r,j]
        blocks_needed = {
            (r, j): gp.quicksum(onc[r, d] * a[r][d][j] * blocks_per_layer[r]
                                for d in valid_dist)
            for r in requests for j in range(1, L + 1)
        }
        # prefix-flow constraints  (LB + UB = exact)
        for r in requests:
            for j in range(1, L + 1):
                # lower bound – must have transferred required data so far
                model.addConstr(
                    gp.quicksum(trans[r, k] for k in range(1, j + 1))
                    >= gp.quicksum(blocks_needed[r, t] for t in range(1, j + 1)),
                    name=f"flowLB_{r}_{j}"
                )
        # total transferred blocks = offloaded blocks
        for r in requests:
            model.addConstr(
                gp.quicksum(trans[r, j] for j in range(1, L + 1))
                == offload_num[r] * context_blocks[r],          # baseline block size
                name=f"flow_total_{r}")
        # 6.4 Stall definition (based on NEW comm_j)
        for j in range(1, L + 1):
            comm_j = gp.quicksum(trans[r, j] for r in requests) / block_bandwidth
            if j == 1:
                model.addConstr(stall[1] >= comm_j, name="stall_first")
            else:
                model.addConstr(stall[j] >= comm_j - batch_layer, name=f"stall_{j}")
            model.addConstr(stall[j] <= comm_j, name=f"stall_ub_{j}")
        # 6.5 batch latency definition
        model.addConstr(token_time == L * batch_layer
                        + gp.quicksum(stall[j] for j in range(1, L + 1)),
                        name="token_time_def")

        # 6.6 bind per-request actual_time via indicators -----------------------------------
        for r in requests:
            model.addGenConstrIndicator(resume[r], True,
                                        actual_time[r] == token_time,
                                        name=f"act_on_{r}")
            model.addGenConstrIndicator(resume[r], False,
                                        actual_time[r] >= BIG_M,
                                        name=f"act_off_{r}")

        # 6.7 SLO ratio and failure budget --------------------------------------------------
        model.addConstrs(
            (ratio[r] * actual_time[r] == decode_steps * SLO[r] for r in requests),
            name="ratio_qc"
        )
        model.addConstrs(
            (slo_fail_per_decode[r] * decode_steps
            >= decode_steps - ratio[r] - deposit_count[r]     for r in requests),
            name="slo_fail_def"
        )
        # adjust infeasible SLO failures here
        model.addConstr(gp.quicksum(slo_fail_per_decode[r] for r in requests) <= 2,
                        name="slo_fail_total")

        # 6.8 GPU memory capacity -----------------------------------------------------------
        model.addConstr(
            gp.quicksum(
                (L - offload_num[r]) * context_blocks[r] for r in requests
            ) <= gpu_block_capacity,
            name="gpu_memory_cap"
        )

        # ----------------------------------------------------------------------------------
        # 7.  Objective: maximise good-put (tokens / second that meet SLO) ------------------
        # ----------------------------------------------------------------------------------
        # model.addConstr(
        #     goodput * token_time ==
        #     gp.quicksum(resume[r] for r in requests) - slo_fail_per_decode.sum(),
        #     name="goodput_def"
        # )
        # model.setObjective(goodput, GRB.MAXIMIZE)
        model.setObjective(token_time, GRB.MINIMIZE)
        model.Params.OutputFlag = 1

        # ----------------------------------------------------------------------------------
        # 8.  Solve
        # ----------------------------------------------------------------------------------
        try:
            model.optimize()
        except gp.GurobiError as e:
            print(f"[solve] Gurobi Error: {e}")
            return None

        # ----------------------------------------------------------------------------------
        # 9.  Extract & print results
        # ----------------------------------------------------------------------------------
        if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            return None

        print("\n--- Optimal solution (or best found) ---")
        print(" id | offload | SLO-fail | actual_time")
        print("----|---------|----------|------------")
        for r in requests:
            print(f"{r:>3} | {int(offload_num[r].X):>7} |"
                f" {slo_fail_per_decode[r].X * decode_steps.X:8.2f} |"
                f" {actual_time[r].X:10.4f}s")

        print(f"\nGood-put           : {model.ObjVal:.4f} tokens/s")
        print(f"Batch latency      : {token_time.X:.4f}s "
            f"(includes {sum(stall[j].X for j in range(1, L + 1)):.4f}s stall)")
        print(f"GPU KV usage       : "
            f"{sum((L - offload_num[r].X) * context_blocks[r] for r in requests)} "
            f"/ {gpu_block_capacity} blocks")
        print(f"decode_window size : {decode_steps.X}")

        # Build ResultList for caller -------------------------------------------------------
        results = ResultList()
        for r in requests:
            stride    = int(prefetch_dist[r].X)
            distance  = -1 if offload_num[r].X == 0 else stride - 1
            results.append(
                Result(
                    id=r,
                    resume=True,
                    n=distance,
                    offload_num=int(offload_num[r].X),
                    slo_fail=slo_fail_per_decode[r].X * decode_steps.X,
                    slo_fail_r=slo_fail_per_decode[r].X,
                    actual_time=actual_time[r].X,
                    window=decode_steps.X,
                )
            )

        print(f"batch_time (token_time) : {results.batch_time:.4f}s")
        return results   
class LatencySolver:
    def __init__(self):
        self.which = "NoPrefetch"
        self.mode = "linear"
    # @staticmethod
    def solve(self, requests_list: list[Request], layer_num = 32, block_bandwidth = 103178.0 / 1000, gpu_block_capacity = 49152 / 80, window_ub = 1000) -> Optional[list[Result]]:
        requests = [r.id for r in requests_list]
        context_blocks = {r.id: r.context_len_in_blocks for r in requests_list}
        layer_time = {r.id: r.layer_time for r in requests_list}
        deposit_count = {r.id: r.deposit_count for r in requests_list}
        SLO = {r.id: r.slo for r in requests_list}
        gpu_layers = {r.id: r.gpu_layers_on_gpu for r in requests_list}
        # blocks_per_layer = {r.id: r.gpu_layers_per_seq   for r in requests_list}
        # gpu_cur_blocks = {
        #     rid: gpu_layers[rid] * blocks_per_layer[rid]   # (#레이어)×(블록/레이어)
        #     for rid in requests
        # }

        L = layer_num 
        M = 1e6                     # big-M

# === 2. floor/ceil 값 미리 계산 ===
        # floor_val = {d: L // (d + 1) for d in range(1, L+2)}
        floor_val = {d: L // (d + 1) for d in range(L)} 
        best_d_for_n = {}                                   # n_off ➜ best distance
        for d, n_off in floor_val.items():
            if n_off not in best_d_for_n or d > best_d_for_n[n_off]:
                best_d_for_n[n_off] = d

        valid_dist = sorted(best_d_for_n.values())  
        valid_dist = valid_dist[1:]  # remove 0 
        print(floor_val)
        print(valid_dist)  # valid strides
# === 3. 모델 생성 및 설정 ===
        model = gp.Model('block_solver')
        model.Params.NonConvex = 2    # 비선형 곱 제약 허용

# === 4. 의사결정 변수 ===
        # MOD: resume 고정(lb=ub=1)  -------------------------------
        # resume        = model.addVars(requests, vtype=GRB.BINARY, name='resume')
        resume = model.addVars(requests, lb=1, ub=1, vtype=GRB.BINARY, name='resume')

        prefetch_dist = model.addVars(requests, lb=1, ub=L+2, vtype=GRB.INTEGER, name='prefetch_dist')
        offload_num = model.addVars(requests, lb=0, ub=L, vtype=GRB.INTEGER, name='offload_num')

        decode_steps = model.addVar(lb=32, ub=window_ub, vtype=GRB.INTEGER, name='decode_steps')

        # ----- binary choice variables only for the preferred distances -----
        onc = {
            (r, d): model.addVar(vtype=GRB.BINARY, name=f"onc_{r}_{d}")
            for r in requests for d in valid_dist
        }

        # ----- exactly one distance per request -----
        model.addConstrs(
            (gp.quicksum(onc[r, d] for d in valid_dist) == 1
            for r in requests),
            name="one_distance"
        )

        # ----- link distance → model vars -----
        model.addConstrs(
            (gp.quicksum(onc[r, d] * (d + 1) for d in valid_dist) == prefetch_dist[r]
            for r in requests),
            name="prefetch_dist_def"
        )

        model.addConstrs(
            (gp.quicksum(onc[r, d] * (L // (d + 1)) for d in valid_dist) == offload_num[r]
            for r in requests),
            name="offload_num_def"
        )
        num_tokens = sum(context_blocks[r] for r in requests)*16
        batch_layer = self.profiled_estimator.estimate_by_profiled_results(num_tokens, which="NoPrefetch", mode="linear")/32
        # print(f"batch_layer: {batch_layer} for {num_tokens} tokens")
        
        # batch_layer  /= 32
        comm_time    = model.addVar(lb=0, name='comm_time')
        comp_time    = model.addVar(lb=0, name='comp_time')
        token_time   = model.addVar(lb=0, name='token_time')
        actual_time  = model.addVars(requests, lb=0, ub=M, name='actual_time')
        ratio        = model.addVars(requests, lb=0, name='ratio')
        slo_fail_per_decode     = model.addVars(requests, lb=0, name='slo_fail_per_decode')

# === 5. 제약식 ===

# 5.2 batch_layer = max(layer_time[r] * resume[r])
        # sum the sequence length, not the layer time!
                # batch_layer = (sum(context_blocks[r] for r in requests) * PROFILED_A + PROFILED_B) / 32
                # print("batch_layer", batch_layer)
        model.addConstr(1 <= gp.quicksum(resume[r] for r in requests), name="should execute more than 1")

# 5.3 comm_time 정의: 필요한 블록 수 ÷ block_bandwidth

        # # temp (xinyue)
        # per_block_time = self.profiled_estimator.estimate_by_profiled_results(num_tokens, which="Communication", mode="linear") / (32*16) # 32 layer, 16 tokens per block
        # block_bandwidth = 1 / per_block_time
        # print(f"block_bandwidth: {block_bandwidth} blks/s")
        # block_bandwidth = 1/per_block_time
        # model.addConstr(
        #     comm_time == gp.quicksum(
        #         resume[r] * offload_num[r] * context_blocks[r]
        #         for r in requests
        #     ) / block_bandwidth,
        #     name="comm_time_def"
        # )

# 5.4 comp_time = batch_layer * L
        model.addConstr(comp_time == batch_layer * L, name="comp_time_def")

# 5.4.1 move_overhead = max(0, move_blocks[r] * context_blocks[r] / block_bandwidth)        
        # move_overhead = model.addVar(lb=0, name='move_overhead')
        # model.addConstr(
        # move_overhead == gp.quicksum(
        #     resume[r] * move_blocks[r] * context_blocks[r]
        #     for r in requests
        # ) / block_bandwidth,
        # name="move_overhead_def"
        # )

# 5.5 token_time = max(comm_time, comp_time)
        model.addGenConstrMax(token_time, [comm_time, comp_time], name="token_time_max")

# 5.6 actual_time indicator
        for r in requests:
            model.addGenConstrIndicator(resume[r], True,
                                        actual_time[r] == token_time,
                                        name=f"act_on_{r}")
            model.addGenConstrIndicator(resume[r], False,
                                        actual_time[r] >= M,
                                        name=f"act_off_{r}")


# 5.8 ratio * H = deposit_timer (비선형 제약)
        for r in requests:
            model.addQConstr(ratio[r] * actual_time[r] == decode_steps * SLO[r], name=f"ratio_qc_{r}")

        for r in requests:
            model.addConstr(slo_fail_per_decode[r] * decode_steps >= decode_steps - ratio[r] - deposit_count[r],
                            name=f"slo_fail_{r}")

# 5.11 GPU 메모리 제약: 필요한 블록 수 ≤ gpu_block_capacity
        model.addConstr(
            gp.quicksum(
                (L - offload_num[r]) * context_blocks[r]
                for r in requests
            ) <= gpu_block_capacity,
            name="memory_limit"
        )
        # 5.12 SLO 실패 총합 ≤ decode_steps   ←★ 새로 추가
        model.addConstr(
            gp.quicksum(slo_fail_per_decode[r] for r in requests) <= 1,
            name="slo_fail_total_limit"
        )

        # 🆕 goodput 분모: token_time + move_overhead
        eff_latency = model.addVar(lb=0, name='effective_latency')
        model.addConstr(eff_latency == token_time,
                        name="eff_lat_def")

# === 6. 목적함수 및 최적화 ===
        # goodput = model.addVar(lb=0, name='obj')
        # model.addConstr(goodput * eff_latency == gp.quicksum(resume[r] for r in requests) - slo_fail_per_decode.sum())
        # model.setObjective(goodput, GRB.MAXIMIZE)
        model.setObjective(token_time, GRB.MINIMIZE)
        model.Params.OutputFlag = 1
        try:
            model.optimize()
        except gp.GurobiError as e:
            print(f"Gurobi Error: {e}")
            return None
# === 7. 결과 출력 ===
        if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT or model.Status == GRB.SUBOPTIMAL:
            print("\n--- Optimal Solution ---")
            print(" r | resume | offload_num | slo_fail | actual_time")
            print("---|--------|-------------|----------|-------------")
            for r in requests:
                print(f"{r:>2} |   {int(resume[r].X)}    |"
                      f"      {int(offload_num[r].X):2d}     |"
                      f" {slo_fail_per_decode[r].X * decode_steps.X:8.2f} | {actual_time[r].X:10.2f}")
            # print(f"\ngoodput = {model.ObjVal:.2f}")
            print(f"""\nMem Usage: {sum([
                (L - offload_num[r].X) * context_blocks[r]
                for r in requests
            ])}/{gpu_block_capacity}""")
            print(f"decode_steps: {decode_steps.X}")
            print(f"comm time: {comm_time.X}")
            print(f"comp time: {comp_time.X}")

            result = ResultList()

            for r in requests:
                stride = int(prefetch_dist[r].X)    # s
                distance = -1 if offload_num[r].X == 0 else stride - 1
                result.append(
                    Result(id=r,
                        resume=bool(resume[r].X),
                        n=distance,
                        offload_num=int(offload_num[r].X),
                        slo_fail=slo_fail_per_decode[r].X,
                        actual_time=actual_time[r].X,
                        window=decode_steps.X)
                )
            batch_time = result.batch_time
            print(f"batch time: {batch_time}")
            
            return result
        else:
            return None

    # requests = [
    #     Request(id="req1", context_len_in_blocks=10, layer_time=0.5, deposit_count=2, slo=1.0, gpu_layers_on_gpu=4),
    #     Request(id="req2", context_len_in_blocks=20, layer_time=0.6, deposit_count=3, slo=1.5, gpu_layers_on_gpu=5)
    # ]
    # solver = Solver_updated()
    # result = solver.solve(requests)
    # if result:
    #     for res in result:
    #         print(res.__dict__)
    # else:
    #     print("No solution found.") 