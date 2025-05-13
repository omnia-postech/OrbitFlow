import math
import gurobipy as gp
from gurobipy import GRB
from typing import Optional, List


# ────────────────── Data-classes ──────────────────
class Request:
    def __init__(self, id: str, context_len_in_blocks: int, layer_time: float,
                 deposit_count: int, slo: float, gpu_layers_on_gpu: int):
        self.id = id
        self.context_len_in_blocks = context_len_in_blocks
        self.layer_time = layer_time
        self.deposit_count = deposit_count
        self.slo = slo
        self.gpu_layers_on_gpu = gpu_layers_on_gpu


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

class Solver:
    @staticmethod
    def solve(requests_list: list[Request], layer_num = 32, block_bandwidth = 103178.0 / 1000, gpu_block_capacity = 49152 / 80, window_ub = 1000) -> Optional[list[Result]]:
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
        floor_val = {n: math.floor(L / n) for n in range(1, L+2)}

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

        move_blocks   = model.addVars(requests, lb=0, ub=L, vtype=GRB.INTEGER, name='move_blocks')

        onc = {(r, i): model.addVar(vtype=GRB.BINARY, name=f"onc_{r}_{i}")
               for r in requests for i in range(1, L + 2)}

        print(floor_val)

        for r in requests:
            model.addConstr(gp.quicksum(offload_num_constr[(r, i)] for i in range(1, L+2)) == 1)
            model.addConstr(gp.quicksum(offload_num_constr[(r, i)] * i for i in range(1, L+2)) == prefetch_dist[r])
            model.addConstr(gp.quicksum(offload_num_constr[(r, i)] * floor_val[i] for i in range(1, L+2)) == offload_num[r])
            model.addConstr(move_blocks[r] >= (L - offload_num[r]) - gpu_layers[r], name=f"mv_pos_{r}")

        batch_layer  = model.addVar(lb=0, name='batch_layer')
        comm_time    = model.addVar(lb=0, name='comm_time')
        comp_time    = model.addVar(lb=0, name='comp_time')
        token_time   = model.addVar(lb=0, name='token_time')
        actual_time  = model.addVars(requests, lb=0, ub=M, name='actual_time')
        ratio        = model.addVars(requests, lb=0, name='ratio')
        slo_fail_per_decode     = model.addVars(requests, lb=0, name='slo_fail_per_decode')

# === 5. 제약식 ===

# 5.2 batch_layer = max(layer_time[r] * resume[r])
        for r in requests:
            model.addConstr(batch_layer >= layer_time[r] * resume[r],
                            name=f"batch_layer_{r}")

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
        move_overhead = model.addVar(lb=0, name='move_overhead')
        model.addConstr(
        move_overhead == gp.quicksum(
            resume[r] * move_blocks[r] * context_blocks[r]
            for r in requests
        ) / block_bandwidth,
        name="move_overhead_def"
        )

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

        # 🆕 goodput 분모: token_time + move_overhead
        eff_latency = model.addVar(lb=0, name='effective_latency')
        model.addConstr(eff_latency == token_time + move_overhead,
                        name="eff_lat_def")

# === 6. 목적함수 및 최적화 ===
        goodput = model.addVar(lb=0, name='obj')
        model.addConstr(goodput * eff_latency == gp.quicksum(resume[r] for r in requests) - slo_fail_per_decode.sum())
        model.setObjective(goodput, GRB.MAXIMIZE)
        model.Params.OutputFlag = 1
        model.optimize()

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

class BetterSolver:
    @staticmethod
    def solve(requests_list: list[Request], layer_num = 32, block_bandwidth = 103178.0 / 1000, gpu_block_capacity = 49152 / 80, window_ub = 1000) -> Optional[list[Result]]:
        requests = [r.id for r in requests_list]
        context_blocks = {r.id: r.context_len_in_blocks for r in requests_list}
        layer_time = {r.id: r.layer_time for r in requests_list}
        deposit_count = {r.id: r.deposit_count for r in requests_list}
        SLO = {r.id: r.slo for r in requests_list}
        gpu_layers = {r.id: r.gpu_layers_on_gpu for r in requests_list}

        L = layer_num 
        M = 1e6                     # big-M

# === 2. floor/ceil 값 미리 계산 ===
        floor_val = {n: math.floor(L / n) for n in range(1, L+2)}

# === 3. 모델 생성 및 설정 ===
        model = gp.Model('block_solver')
        model.Params.NonConvex = 2    # 비선형 곱 제약 허용

        prefetch_dist = model.addVars(requests, lb=1, ub=L+2, vtype=GRB.INTEGER, name='prefetch_dist')
        offload_num = model.addVars(requests, lb=0, ub=L, vtype=GRB.INTEGER, name='offload_num')

        decode_steps = model.addVar(lb=32, ub=window_ub, vtype=GRB.INTEGER, name='decode_steps')

        offload_num_constr = {(r, i): model.addVar(vtype=GRB.BINARY, name=f"onc_{i}") for i in range(1, L+2) for r in requests}

        move_blocks   = model.addVars(requests, lb=0, ub=L, vtype=GRB.INTEGER, name='move_blocks')

        print(floor_val)

        for r in requests:
            model.addConstr(gp.quicksum(offload_num_constr[(r, i)] for i in range(1, L+2)) == 1)
            model.addConstr(gp.quicksum(offload_num_constr[(r, i)] * i for i in range(1, L+2)) == prefetch_dist[r])
            model.addConstr(gp.quicksum(offload_num_constr[(r, i)] * floor_val[i] for i in range(1, L+2)) == offload_num[r])
            model.addConstr(move_blocks[r] >= (L - offload_num[r]) - gpu_layers[r], name=f"mv_pos_{r}")

        batch_layer  = model.addVar(lb=0, name='batch_layer')
        comm_time    = model.addVar(lb=0, name='comm_time')
        comp_time    = model.addVar(lb=0, name='comp_time')
        token_time   = model.addVar(lb=0, name='token_time')
        ratio        = model.addVars(requests, lb=0, name='ratio')
        slo_fail_per_decode     = model.addVars(requests, lb=0, name='slo_fail_per_decode')

# === 5. 제약식 ===

        for r in requests:
            model.addConstr(batch_layer >= layer_time[r],
                            name=f"batch_layer_{r}")
# 5.3 comm_time 정의: 필요한 블록 수 ÷ block_bandwidth
        model.addConstr(
            comm_time == gp.quicksum(
                offload_num[r] * context_blocks[r]
                for r in requests
            ) / block_bandwidth,
            name="comm_time_def"
        )

# 5.4 comp_time = batch_layer * L
        model.addConstr(comp_time == batch_layer * L, name="comp_time_def")

# 5.4.1 move_overhead = max(0, move_blocks[r] * context_blocks[r] / block_bandwidth)        
        move_overhead = model.addVar(lb=0, name='move_overhead')
        model.addConstr(
        move_overhead == gp.quicksum(
            move_blocks[r] * context_blocks[r]
            for r in requests
        ) / block_bandwidth,
        name="move_overhead_def"
        )

# 5.5 token_time = max(comm_time, comp_time)
        model.addGenConstrMax(token_time, [comm_time, comp_time], name="token_time_max")



# 5.8 ratio * H = deposit_timer (비선형 제약)
        for r in requests:
            model.addQConstr(ratio[r] * batch_layer == decode_steps * SLO[r], name=f"ratio_qc_{r}")

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

        # 🆕 goodput 분모: token_time + move_overhead
        eff_latency = model.addVar(lb=0, name='effective_latency')
        model.addConstr(eff_latency * decode_steps == token_time * decode_steps + move_overhead,
                        name="eff_lat_def")
        
        model.addConstr(slo_fail_per_decode.sum() <= 1, name="stoping condition")

# === 6. 목적함수 및 최적화 ===
        goodput = model.addVar(lb=0, name='obj')
        model.addConstr(goodput * eff_latency == len(requests) - slo_fail_per_decode.sum())
        model.setObjective(goodput, GRB.MAXIMIZE)
        model.Params.OutputFlag = 1
        model.Params.TimeLimit = 0.1
        model.optimize()        

# === 7. 결과 출력 ===
        if model.Status == GRB.TIME_LIMIT: 
            return None
        if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
            print("\n--- Optimal Solution ---")
            print(" r | resume | offload_num | slo_fail | actual_time")
            print("---|--------|-------------|----------|-------------")
            for r in requests:
                print(f"{r:>2} |   1    |"
                      f"      {int(offload_num[r].X):2d}     |"
                      f" {slo_fail_per_decode[r].X * decode_steps.X:8.2f} | ")
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
                result.append(Result(r, 1, -1 if offload_num[r].X == 0 else int(prefetch_dist[r].X), int(offload_num[r].X), slo_fail_per_decode[r].X, batch_layer.X, decode_steps.X))

            return result
        else:
            return None



class SolverV2:
    """
    * distance 후보를 1·2·4·8·… 식으로 축소하여
      ‟같은 offload_num 을 만드는 작은 distance” 를 제거
    * KeyError 수정
    * move_overhead 포함 여부를 플래그로 제어
    """

    # ---------- helper --------------------------------------------------
    @staticmethod
    def _build_floor_map(L: int) -> tuple[dict[int, int], List[int]]:
        """
        Returns
        -------
        floor_val : dict  (i -> ⌊L/i⌋)         # 모든 i ∈ 1..L+1
        valid_i   : list  (1,2,4,8,…,L)        # **중복 offload 제거된** distance
        """
        raw = {i: L // i for i in range(1, L + 2)}

        # offload_num 동일하면 가장 큰 distance 만 남긴다
        picked: dict[int, int] = {}
        for i, v in raw.items():          # v = offload_num
            if v not in picked or i > picked[v]:
                picked[v] = i
        valid_i = sorted(picked.values())  # ex) [1,2,4,8,16,32]

        return raw, valid_i

    # ---------- main ----------------------------------------------------
    @staticmethod
    def solve(
        requests_list: List[Request],
        *,
        layer_num: int = 32,
        block_bandwidth: float = 103178.0 / 1000,   # blocks / ms
        gpu_block_capacity: float = 49152 / 80,
        window_ub: int = 1000,
        use_move_overhead: bool = False,
        verbose: bool = False,
    ) -> Optional[List[Result]]:

        # 1. 입력 파싱 ----------------------------------------------------
        ids = [r.id for r in requests_list]
        ctx_blk = {r.id: r.context_len_in_blocks for r in requests_list}
        layer_t = {r.id: r.layer_time            for r in requests_list}
        deposit = {r.id: r.deposit_count         for r in requests_list}
        slo     = {r.id: r.slo                   for r in requests_list}
        gpu_L   = {r.id: r.gpu_layers_on_gpu     for r in requests_list}

        L = layer_num
        floor_val, valid_i = SolverV2._build_floor_map(L)

        # 2. Gurobi 모델 --------------------------------------------------
        m = gp.Model("better_solver")
        m.Params.NonConvex = 2
        if not verbose:
            m.Params.OutputFlag = 0

        # 3. 변수 ---------------------------------------------------------
        # distance (prefetch_dist)
        d = m.addVars(ids, lb=1, ub=L + 1, vtype=GRB.INTEGER, name="dist")
        # offload_num
        off = m.addVars(ids, lb=0, ub=L, vtype=GRB.INTEGER, name="off")
        # one-hot 선택 변수 (valid_i 에 한정)
        z = {(r, i): m.addVar(vtype=GRB.BINARY, name=f"z_{r}_{i}")
             for r in ids for i in valid_i}

        # 창(window) 길이
        H = m.addVar(lb=32, ub=window_ub, vtype=GRB.INTEGER, name="decode_steps")

        # 시간 관련
        batch_layer = m.addVar(lb=0, name="batch_layer")
        comm_time   = m.addVar(lb=0, name="comm_time")
        comp_time   = m.addVar(lb=0, name="comp_time")
        token_time  = m.addVar(lb=0, name="token_time")
        if use_move_overhead:
            mv_over = m.addVar(lb=0, name="move_overhead")

        ratio = m.addVars(ids, lb=0, name="ratio")
        fail  = m.addVars(ids, lb=0, name="slo_fail_per_decode")

        # 4. distance ↔ offload 매핑 -------------------------------------
        for r in ids:
            m.addConstr(gp.quicksum(z[r, i] for i in valid_i) == 1)
            m.addConstr(gp.quicksum(i * z[r, i] for i in valid_i) == d[r])
            m.addConstr(gp.quicksum(floor_val[i] * z[r, i]
                                    for i in valid_i) == off[r])

        # 5. 시간 계산 ----------------------------------------------------
        for r in ids:
            m.addConstr(batch_layer >= layer_t[r])

        m.addConstr(
            comm_time == gp.quicksum(off[r] * ctx_blk[r] for r in ids)
            / block_bandwidth)

        m.addConstr(comp_time == batch_layer * L)
        m.addGenConstrMax(token_time, [comm_time, comp_time])

        if use_move_overhead:
            mv_blocks = m.addVars(ids, lb=0, vtype=GRB.INTEGER, name="mv_blocks")
            for r in ids:
                m.addConstr(mv_blocks[r] >= (L - off[r]) - gpu_L[r])
            m.addConstr(
                mv_over == gp.quicksum(mv_blocks[r] * ctx_blk[r] for r in ids)
                / block_bandwidth)

        # 6. SLO 실패 계산 -----------------------------------------------
        for r in ids:
            m.addQConstr(ratio[r] * batch_layer == H * slo[r])
            m.addConstr(fail[r] * H >= H - ratio[r] - deposit[r])

        # 7. 메모리 제약 ---------------------------------------------------
        m.addConstr(
            gp.quicksum((L - off[r]) * ctx_blk[r] for r in ids)
            <= gpu_block_capacity)

        m.addConstr(fail.sum() <= 1)

        # 8. 목적식 --------------------------------------------------------
        eff_lat = m.addVar(lb=0, name="effective_latency")
        if use_move_overhead:
            m.addConstr(eff_lat * H == token_time * H + mv_over)
        else:
            m.addConstr(eff_lat == token_time)

        goodput = m.addVar(lb=0, name="obj")
        m.addConstr(goodput * eff_lat == len(ids) - fail.sum())
        m.setObjective(goodput, GRB.MAXIMIZE)

        # 9. 최적화 --------------------------------------------------------
        m.optimize()
        if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
            return None

        # 10. 결과 변환 ----------------------------------------------------
        window = int(H.X)
        results: List[Result] = []
        for r in ids:
            results.append(Result(
                id          = r,
                resume      = True,
                n           = int(d[r].X),
                offload_num = int(off[r].X),
                slo_fail    = fail[r].X,
                actual_time = float(batch_layer.X),
                window      = window,
            ))
        return results

if __name__ == "__main__":
    requests = ['r1', 'r2', 'r3', 'r4']
    context_blocks   = {'r1':  int(300 / 16), 'r2': int(200 / 16), 'r3':  int(150 / 16), 'r4': int(350 / 16)}  # 각 요청에 대해 읽어야 할 메타데이터 블록 수
    layer_time       = {'r1': 0.12, 'r2': 0.12, 'r3': 0.19, 'r4': 0.12}  # ms
    deposit_count    = {'r1': 0,   'r2': 3,   'r3': 100, 'r4': 0}
    SLO              = {'r1': 2,  'r2': 1,  'r3': 3, 'r4': 4}   # ms

    requests_list = []

    for id in requests:
        requests_list.append(Request(id, context_blocks[id], layer_time[id], deposit_count[id], SLO[id]))

    match Solver.solve(requests_list, gpu_block_capacity = 40152 / 40):
        case None:
            print("No optimal solution found.")
        case result:
            pass
