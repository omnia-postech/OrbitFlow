import math
import gurobipy as gp
from gurobipy import GRB
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
from typing     import Iterable, Iterator
from tabulate   import tabulate          # pip install tabulate
from pathlib import Path
import json
from vllm.logger import init_logger
logger = init_logger(__name__)
profiled_path = "~/vllm/benchmark/scripts/profiled_results.json"
class ProfileBasedEstimator:
    """
    Estimate per-token latency from pre-profiled polynomial fits.

    JSON format (one example):

    {
        "NoPrefetch": {
            "linear":      { "A": 1.23e-6, "B": 3.54e-2, "R2": 0.989 },
            "upper_quad":  { "A": 8.38e-12, "B": 8.06e-7, "C": 4.46e-2, "R2": 0.895 }
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
    def __init__(self, profiled_path: str | Path):
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
            Fit name inside that profile (e.g. "upper_quad").

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
                 slo_fail: float, actual_time: float, window: int):
        self.id = id
        self.resume = resume
        self.n = n                  # (=prefetch_dist, 즉 distance)
        self.offload_num = offload_num
        self.slo_fail = slo_fail
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

class Solver:
    def __init__(self):
        self.profiled_estimator = ProfileBasedEstimator(profiled_path)
        self.which = "NoPrefetch"
        self.mode = "upper_quad"
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
        floor_val = {d: L // (d + 1) for d in range(L+2)}
        unique = {d: n for n, d in {n: max(d for d, k in floor_val.items() if k == n) for n in set(floor_val.values())}.items()}
        print(dict(sorted(unique.items())))  # unique-preferred map
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

        offload_num_constr = {(r, i): model.addVar(vtype=GRB.BINARY, name=f"onc_{i}") for i in range(1, L+2) for r in requests}

        # move_blocks   = model.addVars(requests, lb=0, ub=L, vtype=GRB.INTEGER, name='move_blocks')

        onc = {(r, i): model.addVar(vtype=GRB.BINARY, name=f"onc_{r}_{i}")
               for r in requests for i in range(1, L + 2)}

        print(floor_val)

        for r in requests:
            model.addConstr(gp.quicksum(offload_num_constr[(r, i)] for i in range(1, L+2)) == 1)
            model.addConstr(gp.quicksum(offload_num_constr[(r, i)] * i for i in range(1, L+2)) == prefetch_dist[r])
            model.addConstr(gp.quicksum(offload_num_constr[(r, i)] * floor_val[i] for i in range(1, L+2)) == offload_num[r])
            # model.addConstr(move_blocks[r]ㄴ >= (L - offload_num[r]) - gpu_layers[r], name=f"mv_pos_{r}")

        num_tokens = sum(context_blocks[r] for r in requests)
        batch_layer = self.profiled_estimator.estimate_by_profiled_results(num_tokens, which="NoPrefetch", mode="upper_quad")
        batch_layer  /= 32
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
        model.addConstr(
            comm_time == gp.quicksum(
                resume[r] * offload_num[r] * context_blocks[r]
                for r in requests
            ) / block_bandwidth,
            name="comm_time_def"
        )

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
        goodput = model.addVar(lb=0, name='obj')
        model.addConstr(goodput * eff_latency == gp.quicksum(resume[r] for r in requests) - slo_fail_per_decode.sum())
        model.setObjective(goodput, GRB.MAXIMIZE)
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
            print(f"\ngoodput = {model.ObjVal:.2f}")
            print(f"""\nMem Usage: {sum([
                (L - offload_num[r].X) * context_blocks[r]
                for r in requests
            ])}/{gpu_block_capacity}""")
            print(f"decode_steps: {decode_steps.X}")
            print(f"comm time: {comm_time.X}")
            print(f"comp time: {comp_time.X}")

            result = []

            for r in requests:
                result.append(Result(r, bool(resume[r].X), -1 if offload_num[r].X == 0 else int(prefetch_dist[r].X), int(offload_num[r].X), slo_fail_per_decode[r].X, actual_time[r].X, decode_steps.X))

            return result
        else:
            return None


class Solver_updated:
    def __init__(self):
        self.profiled_estimator = ProfileBasedEstimator(profiled_path)
        self.which = "NoPrefetch"
        self.mode = "upper_quad"
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
        floor_val     = {d: L // (d + 1) for d in range(L)}          # n_off layers on CPU
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
        stall         = model.addVars(range(1, L + 1), lb=0.0, name="stall")      # s_j  (1-based)
        token_time    = model.addVar(lb=0.0, name="token_time")
        actual_time   = model.addVars(requests, lb=0.0, ub=BIG_M, name="actual_time")
        ratio         = model.addVars(requests, lb=0.0, name="ratio")
        slo_fail_per_decode = model.addVars(requests, lb=0.0, name="slo_fail_per_decode")

        goodput       = model.addVar(lb=0.0, name="obj")                          # objective

        # ----------------------------------------------------------------------------------
        # 4.  Fixed compute time per layer  (profiled “no-prefetch” curve)
        # ----------------------------------------------------------------------------------
        total_tokens = 16 * sum(context_blocks[r] for r in requests)              # rough estimate
        batch_layer  = (
            self.profiled_estimator
            .estimate_by_profiled_results(total_tokens, which="NoPrefetch", mode="upper_quad")
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

        # 6.3 layer-wise communication time  t_comm_expr[j] --------------------------------
        t_comm_expr = {
            j: (
                gp.quicksum(
                    onc[r, d] * a[r][d][j] * blocks_per_layer[r]
                    for r in requests for d in valid_dist
                ) / block_bandwidth          # seconds
            )
            for j in range(1, L + 1)
        }

        # 6.4 stall overlap:  s₁ ≥ comm₁ ;  s_j ≥ comm_j − compute_{j-1}  -------------------
        model.addConstr(stall[1] >= t_comm_expr[1], name="stall_first")
        for j in range(2, L + 1):
            model.addConstr(stall[j] >= t_comm_expr[j] - batch_layer,
                            name=f"stall_{j}")

        # 6.5 define batch latency  token_time = Σ compute + Σ stall ------------------------
        model.addConstr(
            token_time == L * batch_layer + gp.quicksum(stall[j] for j in range(1, L + 1)),
            name="token_time_def"
        )

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
        model.addConstr(
            goodput * token_time ==
            gp.quicksum(resume[r] for r in requests) - slo_fail_per_decode.sum(),
            name="goodput_def"
        )
        model.setObjective(goodput, GRB.MAXIMIZE)
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
                    actual_time=actual_time[r].X,
                    window=decode_steps.X,
                )
            )

        print(f"batch_time (token_time) : {results.batch_time:.4f}s")
        return results   
class LatencySolver:
    def __init__(self):
        self.profiled_estimator = ProfileBasedEstimator(profiled_path)
        self.which = "NoPrefetch"
        self.mode = "upper_quad"
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
        batch_layer = self.profiled_estimator.estimate_by_profiled_results(num_tokens, which="NoPrefetch", mode="upper_quad")/32
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
        # per_block_time = self.profiled_estimator.estimate_by_profiled_results(num_tokens, which="Communication", mode="upper_quad") / (32*16) # 32 layer, 16 tokens per block
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
            print(f"\ngoodput = {model.ObjVal:.2f}")
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

if __name__ == "__main__":
    # Test the Solver class
    requests = [
        Request(id="req1", context_len_in_blocks=10, layer_time=0.5, deposit_count=2, slo=1.0, gpu_layers_on_gpu=4),
        Request(id="req2", context_len_in_blocks=20, layer_time=0.6, deposit_count=3, slo=1.5, gpu_layers_on_gpu=5)
    ]
    solver = Solver()
    result = solver.solve(requests)
    if result:
        for res in result:
            print(res.__dict__)
    else:
        print("No solution found.") 