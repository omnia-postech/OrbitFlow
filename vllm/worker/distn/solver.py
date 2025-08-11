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
class SolverV1:
    """
    MILP solver (v1) that chooses a *per-request (non‑uniform)* prefetch distance
    to minimise **batch token latency** under:
      • per-layer PCIe/NVLink bandwidth budgets,
      • GPU KV‑memory capacity, and
      • a simple SLO‑failure budget.

    Glossary
    --------
    - Distance `d`  ⇒  stride `d+1` (layers `d+1, 2(d+1), …` live on CPU).
    - Offloaded layers per request = `floor(L / (d+1))` (where `L` is #layers).
    - We search only one representative distance per offloaded‑layer count to
      reduce symmetry (pick the **widest** distance for each `n_off`).

    Notes
    -----
    * This class fixes `resume[r] = 1` (keep all requests), matching the
      original behavior.
    * The objective is **latency minimisation** (token_time). We no longer print
      a misleading "Good-put" value here.
    """

    def __init__(self, profiled_path: str | Path | None = None):
        self.profiled_estimator = ProfileBasedEstimator(profiled_path)
        self.which = "NoPrefetch"
        self.mode = "linear"

    def solve(
        self,
        requests_list: list[Request],
        *,
        layer_num: int = 32,
        block_bandwidth: float | None = 103_178.0 / 1_000,  # blocks / second (16 tokens per block)
        gpu_block_capacity: int = 49_152 // 80,            # total blocks the GPU can hold
        window_ub: int = 1_000,                            # upper bound on decode-window length
    ) -> Optional[list[Result]]:
        """Optimise KV placement and per-request prefetch distance for a micro‑batch.

        Latency model (layer‑budget):
          • communication per layer is measured in **blocks** via `trans[r,j]`,
          • non‑negative `stall[j]` captures comm>compute after overlap,
          • batch latency = `L * compute_per_layer + Σ_j stall[j]`.
        """
        if not requests_list:
            return None

        # ------------------------------------------------------------------
        # 0) Gather per‑request constants
        # ------------------------------------------------------------------
        requests         = [r.id for r in requests_list]
        context_blocks   = {r.id: r.context_len_in_blocks for r in requests_list}  # blocks/layer
        deposit_count    = {r.id: r.deposit_count          for r in requests_list}
        SLO              = {r.id: r.slo                   for r in requests_list}
        blocks_per_layer = context_blocks  # alias

        L = int(layer_num)
        if L <= 0:
            raise ValueError(f"layer_num must be > 0, got {layer_num}")

        # ------------------------------------------------------------------
        # 1) Enumerate admissible distances (pick *widest* for each n_off)
        #    Exclude distance 0 (all‑GPU is represented as offload_num == 0).
        # ------------------------------------------------------------------
        floor_val: dict[int, int] = {d: L // (d + 1) for d in range(1, L)}  # d = 1..L-1
        best_d_for_n: dict[int, int] = {}
        for d, n_off in floor_val.items():
            if n_off not in best_d_for_n or d > best_d_for_n[n_off]:
                best_d_for_n[n_off] = d
        valid_dist: list[int] = sorted(best_d_for_n.values())  # e.g., [1,2,3,4,5,7,9,15,31]

        # Offload mask a[r][d][j]: 1 if layer j of request r is on CPU for distance d.
        a: dict[int, dict[int, dict[int, int]]] = {
            r: {d: {j: int(j % (d + 1) == 0) for j in range(1, L + 1)} for d in valid_dist}
            for r in requests
        }

        # ------------------------------------------------------------------
        # 2) Model & variables
        # ------------------------------------------------------------------
        model = gp.Model("block_solver")
        model.Params.NonConvex = 2  # allow indicator/auxiliary nonconvex bits

        # Keep all requests (fixed 1)
        resume = model.addVars(requests, lb=1, ub=1, vtype=GRB.BINARY, name="resume")

        # Distance selection per request (one‑hot over valid_dist)
        onc = model.addVars([(r, d) for r in requests for d in valid_dist], vtype=GRB.BINARY, name="onc")

        prefetch_dist = model.addVars(requests, lb=1,  ub=L + 1, vtype=GRB.INTEGER, name="prefetch_dist")
        offload_num   = model.addVars(requests, lb=0,  ub=L,     vtype=GRB.INTEGER, name="offload_num")
        decode_steps  = model.addVar(lb=32, ub=window_ub, vtype=GRB.INTEGER, name="decode_steps")

        # Latency & SLO vars
        stall        = model.addVars(range(1, L + 1), lb=0.0, name="stall")
        token_time   = model.addVar(lb=0.0, name="token_time")
        actual_time  = model.addVars(requests, lb=0.0, ub=BIG_M, name="actual_time")
        ratio        = model.addVars(requests, lb=0.0, name="ratio")
        slo_fail_per_decode = model.addVars(requests, lb=0.0, name="slo_fail_per_decode")
        trans = model.addVars([(r, j) for r in requests for j in range(1, L + 1)], lb=0.0, name="trans")

        # ------------------------------------------------------------------
        # 3) Profile‑based timings (compute per layer, comm bandwidth)
        # ------------------------------------------------------------------
        total_tokens = 16 * sum(context_blocks[r] for r in requests)
        compute_per_layer = (
            self.profiled_estimator.estimate_by_profiled_results(total_tokens, which="NoPrefetch", mode="linear")
            / 32.0
        )  # seconds per layer when KV is fully GPU‑resident

        per_block_time = (
            self.profiled_estimator.estimate_by_profiled_results(total_tokens, which="Communication", mode="linear")
            / (32 * 16)
        )
        # Safer block_bandwidth handling: use profiled if None or <= 0
        if block_bandwidth is None or block_bandwidth <= 0:
            block_bandwidth = 1.0 / per_block_time  # blocks / second
        print(f"[solve] effective PCIe/NVLink bandwidth : {block_bandwidth:.2f} blocks/s")

        # ------------------------------------------------------------------
        # 4) Constraints
        # ------------------------------------------------------------------
        # 4.1 One distance per request
        model.addConstrs((gp.quicksum(onc[r, d] for d in valid_dist) == resume[r] for r in requests),
                         name="one_stride")

        # 4.2 Link distance → stride and #offloaded layers
        model.addConstrs((gp.quicksum(onc[r, d] * (d + 1) for d in valid_dist) == prefetch_dist[r]
                          for r in requests), name="prefetch_dist_def")
        model.addConstrs((gp.quicksum(onc[r, d] * floor_val[d] for d in valid_dist) == offload_num[r]
                          for r in requests), name="offload_num_def")

        # 4.3 Per‑layer bandwidth capacity (layer‑budget model)
        for j in range(1, L + 1):
            model.addConstr(gp.quicksum(trans[r, j] for r in requests)
                            <= block_bandwidth * (compute_per_layer + stall[j]),
                            name=f"bw_cap_{j}")

        # 4.4 Demand expression and prefix‑flow lower bounds
        blocks_needed = {
            (r, j): gp.quicksum(onc[r, d] * a[r][d][j] * blocks_per_layer[r] for d in valid_dist)
            for r in requests for j in range(1, L + 1)
        }
        for r in requests:
            for j in range(1, L + 1):
                model.addConstr(
                    gp.quicksum(trans[r, k] for k in range(1, j + 1))
                    >= gp.quicksum(blocks_needed[r, t] for t in range(1, j + 1)),
                    name=f"flowLB_{r}_{j}"
                )

        # 4.5 Total transferred blocks = offloaded blocks
        for r in requests:
            model.addConstr(gp.quicksum(trans[r, j] for j in range(1, L + 1)) == offload_num[r] * context_blocks[r],
                            name=f"flow_total_{r}")

        # 4.6 Stall definition and batch latency
        for j in range(1, L + 1):
            comm_j = gp.quicksum(trans[r, j] for r in requests) / block_bandwidth
            if j == 1:
                model.addConstr(stall[1] >= comm_j, name="stall_first")
            else:
                model.addConstr(stall[j] >= comm_j - compute_per_layer, name=f"stall_{j}")
            model.addConstr(stall[j] <= comm_j, name=f"stall_ub_{j}")
        model.addConstr(token_time == L * compute_per_layer + gp.quicksum(stall[j] for j in range(1, L + 1)),
                        name="token_time_def")

        # 4.7 Bind per‑request actual_time via indicators
        for r in requests:
            model.addGenConstrIndicator(resume[r], True,  actual_time[r] == token_time, name=f"act_on_{r}")
            model.addGenConstrIndicator(resume[r], False, actual_time[r] >= BIG_M,     name=f"act_off_{r}")

        # 4.8 SLO ratio and failure budget
        model.addConstrs((ratio[r] * actual_time[r] == decode_steps * SLO[r] for r in requests), name="ratio_qc")
        model.addConstrs((slo_fail_per_decode[r] * decode_steps >= decode_steps - ratio[r] - deposit_count[r]
                          for r in requests), name="slo_fail_def")
        model.addConstr(gp.quicksum(slo_fail_per_decode[r] for r in requests) <= 2, name="slo_fail_total")

        # 4.9 GPU KV memory capacity
        model.addConstr(gp.quicksum((L - offload_num[r]) * context_blocks[r] for r in requests) <= gpu_block_capacity,
                        name="gpu_memory_cap")

        # ------------------------------------------------------------------
        # 5) Objective & solve
        # ------------------------------------------------------------------
        model.setObjective(token_time, GRB.MINIMIZE)
        model.Params.OutputFlag = 1
        try:
            model.optimize()
        except gp.GurobiError as e:
            print(f"[solve] Gurobi Error: {e}")
            return None

        # ------------------------------------------------------------------
        # 6) Extract results
        # ------------------------------------------------------------------
        if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            return None

        print("\n--- Optimal solution (or best found) ---")
        print(" id | offload | SLO-fail | actual_time")
        print("----|---------|----------|------------")
        for r in requests:
            print(f"{r:>3} | {int(offload_num[r].X):>7} |"
                  f" {slo_fail_per_decode[r].X * decode_steps.X:8.2f} |"
                  f" {actual_time[r].X:10.4f}s")

        print(f"\nObjective (latency): {token_time.X:.4f}s")
        print(f"Batch latency      : {token_time.X:.4f}s "
              f"(includes {sum(stall[j].X for j in range(1, L + 1)):.4f}s stall)")
        print(f"GPU KV usage       : "
              f"{sum((L - offload_num[r].X) * context_blocks[r] for r in requests)} / {gpu_block_capacity} blocks")
        print(f"decode_window size : {decode_steps.X}")

        results = ResultList()
        for r in requests:
            stride   = int(prefetch_dist[r].X)
            distance = -1 if offload_num[r].X == 0 else stride - 1
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

"""
New solver: MILP, accurate comm, accurate future forecast, and optimize for both token_time and decode_steps
"""
class SolverV2:
    """
    MILP solver (v2): pipeline‑accurate communication scheduling with
    prefetch‑window awareness and a secondary objective to enlarge the
    decode window.

    Differences vs v1
    ------------------
    • Still minimises *batch token latency* (primary objective),
      but also encourages a larger decode window (secondary objective).
    • Models per‑layer transfer `trans[r,j]` (in **blocks**) and a prefetch
      window mask to avoid unrealistic unlimited look‑ahead.
    • Keeps the public API and printed diagnostics compatible with the
      current code path.

    Notation
    --------
    L            : number of transformer layers (e.g., 32)
    distance d   : prefetch stride − 1 (so stride = d + 1)
    offload_num  : ⌊ L / (d+1) ⌋ (layers per sequence that live on CPU)
    win[d][j]    : 1 if layer j is inside the (d+1)-layer look‑ahead window
                   for *some* offloaded layer at stride d+1, else 0.
    a[r][d][j]   : 1 if layer j of request r is CPU‑resident under distance d.

    Objective
    ---------
    Primary: minimise total token time `token_time`.
    Secondary: maximise `decode_steps` (tie‑breaker; implemented via a
    second priority objective with negative sign).
    """

    def __init__(self, profiled_path: str | Path | None = None):
        self.profiled_estimator = ProfileBasedEstimator(profiled_path)
        self.which = "NoPrefetch"
        self.mode = "linear"

    def solve(
        self,
        requests_list: list[Request],
        *,
        layer_num: int = 32,
        block_bandwidth: float = 103_178.0 / 1_000,   # blocks / second (16 tokens per block)
        gpu_block_capacity: int = 49_152 // 80,       # total blocks the GPU can hold
        window_ub: int = 130,                         # upper bound on decode-window length
    ) -> Optional[list[Result]]:
        """Optimise KV placement and prefetch distance for the current micro‑batch."""
        if not requests_list:
            return None

        # ───────────────────────────────────────────────────────────────────
        # 0) Gather constants per request
        # ───────────────────────────────────────────────────────────────────
        requests          = [r.id for r in requests_list]
        context_blocks    = {r.id: r.context_len_in_blocks for r in requests_list}  # blocks/layer
        deposit_count     = {r.id: r.deposit_count          for r in requests_list}
        SLO               = {r.id: r.slo                   for r in requests_list}
        blocks_per_layer  = context_blocks  # alias

        L = int(layer_num)
        BIG_M = 1_000_000.0

        # ───────────────────────────────────────────────────────────────────
        # 1) Enumerate admissible distances (pick *widest* for each n_off)
        #    Include d = L as an alias meaning “no offload” (n_off = 0).
        # ───────────────────────────────────────────────────────────────────
        floor_val = {d: L // (d + 1) for d in range(1, L)}
        floor_val[L] = 0  # d = L ⇒ no offload alias
        best_d_for_n = {
            n_off: max(d for d, n in floor_val.items() if n == n_off)
            for n_off in floor_val.values()
        }
        valid_dist = sorted(best_d_for_n.values())  # e.g. [1,2,3,4,5,7,9,15,31,32]
        print(f"valid_dist: {valid_dist}")

        # 1.b) Prefetch‑window mask win[d][j]
        win: dict[int, dict[int, int]] = {d: {j: 0 for j in range(1, L + 1)} for d in valid_dist}
        for d in valid_dist:
            for l in range(d + 1, L + 1, d + 1):          # offloaded layers
                begin = max(1, l - d)
                for j in range(begin, l + 1):             # inclusive window [l-d, l]
                    win[d][j] = 1

        # 2) Pre‑compute CPU‑residency flag a[r][d][j]
        a = {
            r: {d: {j: int(j % (d + 1) == 0) for j in range(1, L + 1)} for d in valid_dist}
            for r in requests
        }

        # ───────────────────────────────────────────────────────────────────
        # 3) Build model and variables
        # ───────────────────────────────────────────────────────────────────
        model = gp.Model("block_solver")
        # model.Params.NonConvex = 2  # keep disabled unless indicators require it

        # Keep all requests (fixed 1)
        resume = model.addVars(requests, lb=1, ub=1, vtype=GRB.BINARY, name="resume")

        # Distance selection (one‑hot across valid_dist)
        onc = model.addVars([(r, d) for r in requests for d in valid_dist], vtype=GRB.BINARY, name="onc")
        prefetch_dist = model.addVars(requests, lb=1,  ub=L + 1, vtype=GRB.INTEGER, name="prefetch_dist")
        offload_num   = model.addVars(requests, lb=0,  ub=L,     vtype=GRB.INTEGER, name="offload_num")

        # Decode window (secondary objective shaping)
        decode_steps  = model.addVar(lb=32, ub=window_ub, vtype=GRB.INTEGER, name="decode_steps")
        STEP          = 32
        k_mult        = model.addVar(lb=1, ub=max(1, window_ub // STEP), vtype=GRB.INTEGER, name="k_mult")
        extra_blocks  = model.addVar(lb=0, ub=(window_ub + 15) // 16, vtype=GRB.INTEGER, name="extra_blocks")
        model.addConstr(extra_blocks * 16 >= decode_steps, name="ceil_lb")
        model.addConstr(decode_steps >= 16 * (extra_blocks - 1) + 1, name="ceil_ub")

        # Piecewise coefficient for window benefit (inv_phi_factor)
        bucket_cnt = (window_ub + 15) // 16
        seg = model.addVars(bucket_cnt, vtype=GRB.BINARY, name="seg")
        model.addConstr(seg.sum() == 1, name="seg_onehot")
        inv_phi_factor = model.addVar(lb=0.0, ub=2.0, name="inv_phi_factor")

        # Latency & SLO variables
        stall        = model.addVars(range(1, L + 1), lb=0.0, name="stall")
        token_time   = model.addVar(lb=0.0, name="token_time")
        actual_time  = model.addVars(requests, lb=0.0, ub=BIG_M, name="actual_time")
        slo_violate  = model.addVars(requests, vtype=GRB.BINARY, name="slo_violate")

        # Per‑layer transferred blocks
        trans = model.addVars([(r, j) for r in requests for j in range(1, L + 1)], lb=0.0, vtype=GRB.CONTINUOUS, name="trans")

        # ───────────────────────────────────────────────────────────────────
        # 4) Profile‑based compute and communication timing
        # ───────────────────────────────────────────────────────────────────
        total_tokens = 16 * sum(context_blocks[r] for r in requests)
        batch_layer  = (
            self.profiled_estimator.estimate_by_profiled_results(total_tokens, which="NoPrefetch", mode="linear")
            / 32.0
        )

        per_block_time = (
            self.profiled_estimator.estimate_by_profiled_results(total_tokens, which="Communication", mode="linear")
            / (32 * 16)
        )
        # Respect user‑supplied bandwidth if positive; otherwise use profiled value
        if block_bandwidth is None or block_bandwidth <= 0:
            block_bandwidth = 1.0 / per_block_time
        print(f"[solve] profiled PCIe/NVLink bandwidth : {block_bandwidth:.2f} blocks/s")

        # Window shaping constants (outside model arithmetic)
        extra_comm_per16 = len(requests) / max(block_bandwidth, 1e-9)
        comp0  = batch_layer
        comp16 = (
            self.profiled_estimator.estimate_by_profiled_results(total_tokens + 16, which="NoPrefetch", mode="linear")
            / 32.0
        )
        DELTA = extra_comm_per16 + L * (comp16 - comp0)  # seconds / 16 tokens
        BASE_LAT = L * comp0
        pos_slack = [SLO[r] - BASE_LAT for r in requests if SLO[r] - BASE_LAT > 0]
        M_val = min(pos_slack) if pos_slack else 1e3
        PHI     = [1 + (k * DELTA) / M_val for k in range(bucket_cnt)]
        inv_PHI = [2.0 / v for v in PHI]
        # Bind piece‑wise coefficient: decode_steps ∈ [16k, 16(k+1)-1] ⇒ inv_phi_factor = 2/PHI[k]
        for k in range(bucket_cnt):
            model.addConstr(decode_steps >= 16 * k      - window_ub * (1 - seg[k]))
            model.addConstr(decode_steps <= 16 * (k+1) - 1 + window_ub * (1 - seg[k]))
        model.addConstr(inv_phi_factor == gp.quicksum(inv_PHI[k] * seg[k] for k in range(bucket_cnt)), name="invphi_def")
        model.addConstr(decode_steps == STEP * k_mult, name="decode_steps_multiple")

        # ───────────────────────────────────────────────────────────────────
        # 5) Constraints
        # ───────────────────────────────────────────────────────────────────
        # 5.1 One distance per request
        model.addConstrs((gp.quicksum(onc[r, d] for d in valid_dist) == 1 for r in requests), name="one_stride")

        # 5.2 Link distance → stride and #offloaded layers
        model.addConstrs((gp.quicksum(onc[r, d] * (d + 1) for d in valid_dist) == prefetch_dist[r] for r in requests), name="prefetch_dist_def")
        model.addConstrs((gp.quicksum(onc[r, d] * floor_val[d] for d in valid_dist) == offload_num[r] for r in requests), name="offload_num_def")

        # 5.3 Per‑layer bandwidth capacity (layer‑budget model)
        for j in range(1, L + 1):
            model.addConstr(gp.quicksum(trans[r, j] for r in requests) <= block_bandwidth * (batch_layer + stall[j]), name=f"bw_cap_{j}")

        # 5.4 Prefix‑flow lower bounds: cumulative transfers must cover demand so far
        blocks_needed = {
            (r, j): gp.quicksum(onc[r, d] * a[r][d][j] * blocks_per_layer[r] for d in valid_dist)
            for r in requests for j in range(1, L + 1)
        }
        for r in requests:
            for j in range(1, L + 1):
                model.addConstr(gp.quicksum(trans[r, k] for k in range(1, j + 1)) >= gp.quicksum(blocks_needed[r, t] for t in range(1, j + 1)), name=f"flowLB_{r}_{j}")

        # 5.5 Window‑scoped upper bounds: at most one live DMA stream per request
        for r in requests:
            for j in range(1, L + 1):
                windows_open = gp.quicksum(onc[r, d] * win[d][j] for d in valid_dist)
                model.addConstr(gp.quicksum(trans[r, k] for k in range(1, j + 1)) <= windows_open * blocks_per_layer[r], name=f"flowUB_{r}_{j}")

        # 5.6 Total transferred blocks equals offloaded blocks
        for r in requests:
            model.addConstr(gp.quicksum(trans[r, j] for j in range(1, L + 1)) == offload_num[r] * context_blocks[r], name=f"flow_total_{r}")

        # 5.7 Stall definition and batch latency
        for j in range(1, L + 1):
            comm_j = gp.quicksum(trans[r, j] for r in requests) / max(block_bandwidth, 1e-9)
            if j == 1:
                model.addConstr(stall[1] >= comm_j, name="stall_first")
            else:
                model.addConstr(stall[j] >= comm_j - batch_layer, name=f"stall_{j}")
            model.addConstr(stall[j] <= comm_j, name=f"stall_ub_{j}")
        model.addConstr(token_time == L * batch_layer + gp.quicksum(stall[j] for j in range(1, L + 1)), name="token_time_def")

        # 5.8 Actual time binding (all requests share the same token_time)
        model.addConstrs((actual_time[r] == token_time for r in requests), name="actual_time_def")

        # 5.9 SLO indicator logic (deduplicated Big‑M)
        M_slo = window_ub * max(SLO.values()) if SLO else 1e6
        EPS = 1e-5
        model.addConstrs((actual_time[r] - SLO[r] <= M_slo * slo_violate[r] for r in requests), name="slo_violate_def")
        model.addConstrs((actual_time[r] - SLO[r] >= EPS - M_slo * (1 - slo_violate[r]) for r in requests), name="slo_violate_eq")

        # 5.10 Average SLO‑failure budget (no bilinear terms)
        viol_tokens = model.addVars(requests, lb=0.0, ub=window_ub, name="viol_tokens")
        for r in requests:
            model.addGenConstrIndicator(slo_violate[r], False, viol_tokens[r] == 0, name=f"viol_tok_zero_{r}")
            model.addGenConstrIndicator(slo_violate[r], True,  viol_tokens[r] <= decode_steps, name=f"viol_tok_ub_{r}")
        M_tok = window_ub
        model.addConstrs((viol_tokens[r] >= decode_steps - deposit_count[r] - M_tok * (1 - slo_violate[r]) for r in requests), name="viol_tok_lb")
        model.addConstr(gp.quicksum(viol_tokens[r] for r in requests) <= inv_phi_factor, name="viol_tok_total")

        # 5.11 GPU memory capacity with decode‑growth surrogate (McCormick envelope)
        undecode_total = model.addVar(lb=0, ub=L * len(requests), vtype=GRB.INTEGER, name="undecode_total")
        model.addConstr(undecode_total == gp.quicksum(L - offload_num[r] for r in requests), name="undecode_total_def")
        mem_prod = model.addVar(lb=0, ub=gpu_block_capacity, vtype=GRB.CONTINUOUS, name="mem_prod")
        xL, xU = 2, bucket_cnt
        yL, yU = 0, L * len(requests)
        model.addConstr(mem_prod >= xL * undecode_total + yL * extra_blocks - xL * yL, name="mem_mcc1")
        model.addConstr(mem_prod >= xU * undecode_total + yU * extra_blocks - xU * yU, name="mem_mcc2")
        model.addConstr(mem_prod <= xU * undecode_total + yL * extra_blocks - xU * yL, name="mem_mcc3")
        model.addConstr(mem_prod <= xL * undecode_total + yU * extra_blocks - xL * yU, name="mem_mcc4")
        model.addConstr(gp.quicksum((L - offload_num[r]) * context_blocks[r] for r in requests) + mem_prod <= gpu_block_capacity, name="gpu_memory_cap")

        # ───────────────────────────────────────────────────────────────────
        # 6) Objectives & solve
        # ───────────────────────────────────────────────────────────────────
        model.setObjectiveN(token_time, 0, 1, 1, GRB.MINIMIZE)   # primary – latency
        model.setObjectiveN(-decode_steps, 1, 2, 1, GRB.MINIMIZE)  # secondary – window

        model.Params.OutputFlag    = 1
        model.Params.PoolSearchMode = 2
        model.Params.PoolSolutions  = 5000
        model.Params.Presolve       = 0
        model.Params.Aggregate      = 0
        model.Params.CutPasses      = 0

        try:
            model.optimize()
        except gp.GurobiError as e:
            print(f"[solve] Gurobi Error: {e}")
            return None

        # ───────────────────────────────────────────────────────────────────
        # 7) Extract results (unchanged semantics)
        # ───────────────────────────────────────────────────────────────────
        if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            return None

        tokens_ok = sum(1.0 - slo_violate[r].X for r in requests)
        goodput   = tokens_ok / token_time.X if token_time.X > 0 else 0.0

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
              f"{sum((L - offload_num[r].X) * context_blocks[r] for r in requests)} / {gpu_block_capacity} blocks")
        print(f"decode_window size : {decode_steps.X}")

        # Build ResultList --------------------------------------------------
        results = ResultList()
        for r in requests:
            stride   = int(prefetch_dist[r].X)
            distance = -1 if offload_num[r].X == 0 else stride - 1
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
        self,
        requests_list: List[Request],
        offload_decision: Dict[int, int],            # {req_id : distance (= d),  -1 ⇒ no offload}
        *,
        layer_num: int = 32,
        block_bandwidth: float = 103_178.0 / 1_000,  # blocks · s⁻¹
        gpu_block_capacity: int = 49_152 // 80,      # unused (API parity)
        window_ub: int = 1_000                       # unused
    ) -> float:
        """Return batch latency `T_batch` (seconds) under a fixed offload plan."""
        # 0) Sanity: plan must cover every request id
        req_ids = {r.id for r in requests_list}
        if req_ids != set(offload_decision):
            miss  = req_ids - offload_decision.keys()
            extra = offload_decision.keys() - req_ids
            raise ValueError(f"offload_decision must cover every request (missing={miss}, extra={extra})")

        # 1) Compute time per layer (profiled)
        total_tokens = 16 * sum(r.context_len_in_blocks for r in requests_list)
        compute_per_layer = (
            self.profiled_estimator.estimate_by_profiled_results(total_tokens, which="NoPrefetch", mode="linear")
            / layer_num
        )

        # 2) Bandwidth (blocks/s)
        per_block_time = (
            self.profiled_estimator.estimate_by_profiled_results(total_tokens, which="Communication", mode="linear")
            / (layer_num * 16)
        )
        eff_bandwidth = 1.0 / per_block_time if block_bandwidth is None else block_bandwidth

        # 3) Layer‑wise communication time
        def stride(req_id: int) -> int:
            d = offload_decision[req_id]
            return 1 if d < 0 else d + 1  # 1 ⇒ fully GPU‑resident

        t_comm = [0.0] * layer_num
        for j in range(1, layer_num + 1):
            blocks_this_layer = 0
            for r in requests_list:
                if j % stride(r.id) == 0:  # layer lives on CPU
                    blocks_this_layer += r.context_len_in_blocks
            t_comm[j - 1] = blocks_this_layer / max(eff_bandwidth, 1e-9)

        # 4) Stall
        stall = [0.0] * layer_num
        stall[0] = t_comm[0]
        for j in range(1, layer_num):
            stall[j] = max(0.0, t_comm[j] - compute_per_layer)

        # 5) Batch latency
        return layer_num * compute_per_layer + sum(stall)
    
    
class LatencySolver:
    """
    Simple latency-first solver (aggregate model).

    This variant is intentionally lighter-weight than SolverV1/V2:
      • Picks one prefetch *distance* per request (from a reduced candidate set).
      • Approximates communication time as a single bulk transfer over the whole
        batch:  B_total / bandwidth, where B_total = Σ_r offload_num[r] · blocks[r].
      • Compute time uses the profiled no-prefetch curve per layer.
      • Token latency is token_time = max(comm_time, comp_time).

    It preserves the high-level SLO formulation from the original code but
    removes the broken/undefined parts (e.g., `resume`/`comm_time` not linked).
    """

    def __init__(self, profiled_path: str | Path | None = None):
        # Use env default if not provided
        if profiled_path is None:
            profiled_path = DEFAULT_PROFILED_PATH
        self.profiled_estimator = ProfileBasedEstimator(profiled_path)
        self.which = "NoPrefetch"
        self.mode = "linear"

    def solve(
        self,
        requests_list: list[Request],
        *,
        layer_num: int = 32,
        block_bandwidth: float | None = 103_178.0 / 1_000,  # blocks / sec (16 tokens per block)
        gpu_block_capacity: int = 49_152 // 80,
        window_ub: int = 1_000,
    ) -> Optional[list[Result]]:
        if not requests_list:
            return None

        # 0) Gather constants --------------------------------------------------------------
        requests       = [r.id for r in requests_list]
        context_blocks = {r.id: r.context_len_in_blocks for r in requests_list}  # blocks/layer
        deposit_count  = {r.id: r.deposit_count          for r in requests_list}
        SLO            = {r.id: r.slo                   for r in requests_list}
        L = int(layer_num)
        BIG_M = 1_000_000.0
        if L <= 0:
            raise ValueError(f"layer_num must be > 0, got {layer_num}")

        # 1) Candidate distances -----------------------------------------------------------
        # Use one representative distance per #offloaded layers; include d=L as no-offload.
        floor_val = {d: L // (d + 1) for d in range(1, L)}
        floor_val[L] = 0
        best_d_for_n = {n_off: max(d for d, n in floor_val.items() if n == n_off)
                         for n in floor_val.values() for n_off in [n]}
        valid_dist = sorted(best_d_for_n.values())  # e.g., [1,2,3,4,5,7,9,15,31,32]

        # 2) Model & variables -------------------------------------------------------------
        model = gp.Model('block_solver')
        model.Params.NonConvex = 2

        # Keep-all binary (fixed to 1 for parity with other solvers)
        resume = model.addVars(requests, lb=1, ub=1, vtype=GRB.BINARY, name='resume')

        # Distance selection (one-hot among valid_dist)
        onc = {(r, d): model.addVar(vtype=GRB.BINARY, name=f"onc_{r}_{d}")
               for r in requests for d in valid_dist}
        model.addConstrs((gp.quicksum(onc[r, d] for d in valid_dist) == 1 for r in requests),
                         name='one_distance')

        prefetch_dist = model.addVars(requests, lb=1, ub=L + 2, vtype=GRB.INTEGER, name='prefetch_dist')
        offload_num   = model.addVars(requests, lb=0, ub=L,     vtype=GRB.INTEGER, name='offload_num')
        model.addConstrs((gp.quicksum(onc[r, d] * (d + 1) for d in valid_dist) == prefetch_dist[r] for r in requests),
                         name='prefetch_dist_def')
        model.addConstrs((gp.quicksum(onc[r, d] * (L // (d + 1)) for d in valid_dist) == offload_num[r] for r in requests),
                         name='offload_num_def')

        # Decode window and SLO variables
        decode_steps = model.addVar(lb=32, ub=window_ub, vtype=GRB.INTEGER, name='decode_steps')
        actual_time  = model.addVars(requests, lb=0, ub=BIG_M, name='actual_time')
        ratio        = model.addVars(requests, lb=0, name='ratio')
        slo_fail_pd  = model.addVars(requests, lb=0, name='slo_fail_per_decode')

        # Latency components
        comp_time  = model.addVar(lb=0, name='comp_time')
        comm_time  = model.addVar(lb=0, name='comm_time')
        token_time = model.addVar(lb=0, name='token_time')

        # 3) Profile-based timings ---------------------------------------------------------
        num_tokens = 16 * sum(context_blocks[r] for r in requests)
        batch_layer = (self.profiled_estimator
                       .estimate_by_profiled_results(num_tokens, which='NoPrefetch', mode='linear')
                       / 32.0)
        per_block_time = (self.profiled_estimator
                          .estimate_by_profiled_results(num_tokens, which='Communication', mode='linear')
                          / (32 * 16))
        if block_bandwidth is None or block_bandwidth <= 0:
            block_bandwidth = 1.0 / per_block_time  # blocks / sec

        # 4) Constraints -------------------------------------------------------------------
        # 4.1 comp_time = L * batch_layer
        model.addConstr(comp_time == batch_layer * L, name='comp_time_def')

        # 4.2 comm_time ≥ total blocks moved / bandwidth (aggregate model)
        #     B_total = Σ_r offload_num[r] · context_blocks[r]
        total_blocks = gp.quicksum(offload_num[r] * context_blocks[r] for r in requests)
        model.addConstr(comm_time * block_bandwidth >= total_blocks, name='comm_time_lb')

        # 4.3 token_time = max(comm_time, comp_time)
        model.addGenConstrMax(token_time, [comm_time, comp_time], name='token_time_max')

        # 4.4 actual_time indicators (all kept)
        for r in requests:
            model.addGenConstrIndicator(resume[r], True,  actual_time[r] == token_time, name=f'act_on_{r}')
            model.addGenConstrIndicator(resume[r], False, actual_time[r] >= BIG_M,     name=f'act_off_{r}')

        # 4.5 SLO ratio and failure budget
        model.addConstrs((ratio[r] * actual_time[r] == decode_steps * SLO[r] for r in requests), name='ratio_qc')
        model.addConstrs((slo_fail_pd[r] * decode_steps >= decode_steps - ratio[r] - deposit_count[r] for r in requests),
                         name='slo_fail_def')
        model.addConstr(gp.quicksum(slo_fail_pd[r] for r in requests) <= 1, name='slo_fail_total_limit')

        # 4.6 GPU memory capacity
        model.addConstr(gp.quicksum((L - offload_num[r]) * context_blocks[r] for r in requests) <= gpu_block_capacity,
                        name='memory_limit')

        # 5) Objective & solve --------------------------------------------------------------
        model.setObjective(token_time, GRB.MINIMIZE)
        model.Params.OutputFlag = 1
        try:
            model.optimize()
        except gp.GurobiError as e:
            print(f"Gurobi Error: {e}")
            return None

        # 6) Results -----------------------------------------------------------------------
        if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            return None

        print("\n--- Optimal Solution (LatencySolver) ---")
        print(" r | offload | slo_fail | actual_time")
        print("---|---------|---------|-------------")
        for r in requests:
            print(f"{r:>2} | {int(offload_num[r].X):>7} | {slo_fail_pd[r].X * decode_steps.X:8.2f} | {actual_time[r].X:10.4f}")
        print(f"comp_time: {comp_time.X:.4f}s, comm_time: {comm_time.X:.4f}s, token_time: {token_time.X:.4f}s")
        print(f"GPU KV usage: {sum((L - offload_num[r].X) * context_blocks[r] for r in requests)} / {gpu_block_capacity} blocks")
        print(f"decode_steps: {decode_steps.X}")

        results = ResultList()
        for r in requests:
            stride   = int(prefetch_dist[r].X)
            distance = -1 if offload_num[r].X == 0 else stride - 1
            results.append(
                Result(
                    id=r,
                    resume=True,
                    n=distance,
                    offload_num=int(offload_num[r].X),
                    slo_fail=slo_fail_pd[r].X * decode_steps.X,
                    slo_fail_r=slo_fail_pd[r].X,
                    actual_time=actual_time[r].X,
                    window=decode_steps.X,
                )
            )

        print(f"batch_time (token_time): {results.batch_time:.4f}s")
        return results

class SolverV1_Uniform(SolverV1):
    """
    Uniform‑stride MILP (derived from SolverV1).

    Chooses **one** prefetch distance (stride−1) shared by all requests in the
    micro‑batch, then minimises batch token latency under:
      • per‑layer PCIe/NVLink bandwidth budgets,
      • GPU KV‑memory capacity, and
      • a simple SLO‑failure budget.

    Differences from SolverV1
    -------------------------
    - Adds a *global stride* constraint: `prefetch_dist[r] == prefetch_dist[r0]` for all r.
    - Otherwise the formulation (variables/constraints/objective) is the same.
    """

    def solve(
        self,
        requests_list: list[Request],
        *,
        layer_num: int = 32,
        block_bandwidth: float | None = 103_178.0 / 1_000,  # blocks / second (16 tokens per block)
        gpu_block_capacity: int = 49_152 // 80,             # total blocks the GPU can hold
        window_ub: int = 1_000,                             # upper bound on decode-window length
    ) -> Optional[list[Result]]:
        """Optimise KV placement with a **single shared stride** for all requests."""
        if not requests_list:
            return None

        # ───────────────────────────────────────────────────────────────────
        # 0) Gather constants per request
        # ───────────────────────────────────────────────────────────────────
        requests          = [r.id for r in requests_list]
        context_blocks    = {r.id: r.context_len_in_blocks for r in requests_list}  # blocks/layer
        deposit_count     = {r.id: r.deposit_count          for r in requests_list}
        SLO               = {r.id: r.slo                   for r in requests_list}
        blocks_per_layer  = context_blocks  # alias

        L = int(layer_num)
        if L <= 0:
            raise ValueError(f"layer_num must be > 0, got {layer_num}")

        # ───────────────────────────────────────────────────────────────────
        # 1) Candidate distances (representative per offloaded‑layer count)
        #    Exclude d=0; offload_num==0 is represented via decision later.
        # ───────────────────────────────────────────────────────────────────
        floor_val: dict[int, int] = {d: L // (d + 1) for d in range(1, L)}
        best_d_for_n: dict[int, int] = {}
        for d, n_off in floor_val.items():
            if n_off not in best_d_for_n or d > best_d_for_n[n_off]:
                best_d_for_n[n_off] = d
        valid_dist: list[int] = sorted(best_d_for_n.values())

        # Offload mask a[r][d][j]: 1 if layer j of request r is on CPU for distance d.
        a: dict[int, dict[int, dict[int, int]]] = {
            r: {d: {j: int(j % (d + 1) == 0) for j in range(1, L + 1)} for d in valid_dist}
            for r in requests
        }

        # ───────────────────────────────────────────────────────────────────
        # 2) Model & variables
        # ───────────────────────────────────────────────────────────────────
        model = gp.Model("block_solver")
        model.Params.NonConvex = 2  # allow indicator/auxiliary nonconvex bits

        # Keep all requests (fixed 1)
        resume = model.addVars(requests, lb=1, ub=1, vtype=GRB.BINARY, name="resume")

        # Distance selection per request (one‑hot over valid_dist)
        onc = model.addVars([(r, d) for r in requests for d in valid_dist], vtype=GRB.BINARY, name="onc")

        prefetch_dist = model.addVars(requests, lb=1,  ub=L + 1, vtype=GRB.INTEGER, name="prefetch_dist")
        offload_num   = model.addVars(requests, lb=0,  ub=L,     vtype=GRB.INTEGER, name="offload_num")
        decode_steps  = model.addVar(lb=32, ub=window_ub, vtype=GRB.INTEGER, name="decode_steps")

        # Latency & SLO variables
        stall        = model.addVars(range(1, L + 1), lb=0.0, name="stall")
        token_time   = model.addVar(lb=0.0, name="token_time")
        actual_time  = model.addVars(requests, lb=0.0, ub=BIG_M, name="actual_time")
        ratio        = model.addVars(requests, lb=0.0, name="ratio")
        slo_fail_per_decode = model.addVars(requests, lb=0.0, name="slo_fail_per_decode")
        trans = model.addVars([(r, j) for r in requests for j in range(1, L + 1)], lb=0.0, vtype=GRB.CONTINUOUS, name="trans")

        # ───────────────────────────────────────────────────────────────────
        # 3) Profile‑based timings (compute per layer, comm bandwidth)
        # ───────────────────────────────────────────────────────────────────
        total_tokens = 16 * sum(context_blocks[r] for r in requests)
        batch_layer  = (
            self.profiled_estimator.estimate_by_profiled_results(total_tokens, which="NoPrefetch", mode="linear")
            / 32.0
        )  # seconds per layer when KV is fully GPU‑resident

        per_block_time = (
            self.profiled_estimator.estimate_by_profiled_results(total_tokens, which="Communication", mode="linear")
            / (32 * 16)
        )
        if block_bandwidth is None or block_bandwidth <= 0:
            block_bandwidth = 1.0 / per_block_time  # blocks / second
        print(f"[solve][uniform] effective PCIe/NVLink bandwidth : {block_bandwidth:.2f} blocks/s")

        # ───────────────────────────────────────────────────────────────────
        # 4) Constraints
        # ───────────────────────────────────────────────────────────────────
        # 4.1 One distance per request
        model.addConstrs((gp.quicksum(onc[r, d] for d in valid_dist) == resume[r] for r in requests), name="one_stride")

        # 4.2 Link distance → stride and #offloaded layers
        model.addConstrs((gp.quicksum(onc[r, d] * (d + 1) for d in valid_dist) == prefetch_dist[r] for r in requests), name="prefetch_dist_def")
        model.addConstrs((gp.quicksum(onc[r, d] * floor_val[d] for d in valid_dist) == offload_num[r] for r in requests), name="offload_num_def")

        # 4.3 **Uniform stride**: all requests share the same prefetch_dist
        r0 = requests[0]
        model.addConstrs((prefetch_dist[r] == prefetch_dist[r0] for r in requests), name="global_stride")

        # 4.4 Per‑layer bandwidth capacity (layer‑budget model)
        for j in range(1, L + 1):
            model.addConstr(gp.quicksum(trans[r, j] for r in requests) <= block_bandwidth * (batch_layer + stall[j]), name=f"bw_cap_{j}")

        # 4.5 Demand expression and prefix‑flow lower bounds
        blocks_needed = { (r, j): gp.quicksum(onc[r, d] * a[r][d][j] * blocks_per_layer[r] for d in valid_dist)
                          for r in requests for j in range(1, L + 1) }
        for r in requests:
            for j in range(1, L + 1):
                model.addConstr(gp.quicksum(trans[r, k] for k in range(1, j + 1)) >= gp.quicksum(blocks_needed[r, t] for t in range(1, j + 1)), name=f"flowLB_{r}_{j}")

        # 4.6 Total transferred blocks = offloaded blocks
        for r in requests:
            model.addConstr(gp.quicksum(trans[r, j] for j in range(1, L + 1)) == offload_num[r] * context_blocks[r], name=f"flow_total_{r}")

        # 4.7 Stall definition and batch latency
        for j in range(1, L + 1):
            comm_j = gp.quicksum(trans[r, j] for r in requests) / block_bandwidth
            if j == 1:
                model.addConstr(stall[1] >= comm_j, name="stall_first")
            else:
                model.addConstr(stall[j] >= comm_j - batch_layer, name=f"stall_{j}")
            model.addConstr(stall[j] <= comm_j, name=f"stall_ub_{j}")
        model.addConstr(token_time == L * batch_layer + gp.quicksum(stall[j] for j in range(1, L + 1)), name="token_time_def")

        # 4.8 Bind per‑request actual_time via indicators
        for r in requests:
            model.addGenConstrIndicator(resume[r], True,  actual_time[r] == token_time, name=f"act_on_{r}")
            model.addGenConstrIndicator(resume[r], False, actual_time[r] >= BIG_M,     name=f"act_off_{r}")

        # 4.9 SLO ratio and failure budget
        model.addConstrs((ratio[r] * actual_time[r] == decode_steps * SLO[r] for r in requests), name="ratio_qc")
        model.addConstrs((slo_fail_per_decode[r] * decode_steps >= decode_steps - ratio[r] - deposit_count[r] for r in requests), name="slo_fail_def")
        model.addConstr(gp.quicksum(slo_fail_per_decode[r] for r in requests) <= 2, name="slo_fail_total")

        # 4.10 GPU KV memory capacity
        model.addConstr(gp.quicksum((L - offload_num[r]) * context_blocks[r] for r in requests) <= gpu_block_capacity, name="gpu_memory_cap")

        # ───────────────────────────────────────────────────────────────────
        # 5) Objective & solve
        # ───────────────────────────────────────────────────────────────────
        model.setObjective(token_time, GRB.MINIMIZE)
        model.Params.OutputFlag = 1
        try:
            model.optimize()
        except gp.GurobiError as e:
            print(f"[solve][uniform] Gurobi Error: {e}")
            return None

        # ───────────────────────────────────────────────────────────────────
        # 6) Extract results
        # ───────────────────────────────────────────────────────────────────
        if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            return None

        print("\n--- Optimal solution (uniform stride) ---")
        print(" id | offload | SLO-fail | actual_time")
        print("----|---------|----------|------------")
        for r in requests:
            print(f"{r:>3} | {int(offload_num[r].X):>7} | {slo_fail_per_decode[r].X * decode_steps.X:8.2f} | {actual_time[r].X:10.4f}s")

        print(f"\nObjective (latency): {token_time.X:.4f}s")
        print(f"Batch latency      : {token_time.X:.4f}s (includes {sum(stall[j].X for j in range(1, L + 1)):.4f}s stall)")
        print(f"GPU KV usage       : {sum((L - offload_num[r].X) * context_blocks[r] for r in requests)} / {gpu_block_capacity} blocks")
        print(f"decode_window size : {decode_steps.X}")

        results = ResultList()
        for r in requests:
            stride   = int(prefetch_dist[r].X)
            distance = -1 if offload_num[r].X == 0 else stride - 1
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