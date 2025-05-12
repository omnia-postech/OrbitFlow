import math
import gurobipy as gp
from gurobipy import GRB
from typing import Optional

class Request:
    def __init__(self, id: str, context_len_in_blocks: int, layer_time: float, deposit_count: int, slo: float):
        self.id = id
        self.context_len_in_blocks = context_len_in_blocks
        self.layer_time = layer_time
        self.deposit_count = deposit_count
        self.slo = slo

class Result:
    def __init__(self, id: str, resume: bool, n: int, offload_num: int, slo_fail: float, actual_time: float, window: int):
        self.id = id
        self.n = n
        self.resume = resume
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

        L = layer_num 
        M = 1e6                     # big-M

# === 2. floor/ceil 값 미리 계산 ===
        floor_val = {n: math.floor(L / n) for n in range(1, L+2)}

# === 3. 모델 생성 및 설정 ===
        model = gp.Model('block_solver')
        model.Params.NonConvex = 2    # 비선형 곱 제약 허용

# === 4. 의사결정 변수 ===
        resume        = model.addVars(requests, vtype=GRB.BINARY, name='resume')

        prefetch_dist = model.addVars(requests, lb=1, ub=L+2, vtype=GRB.INTEGER, name='prefetch_dist')
        offload_num = model.addVars(requests, lb=0, ub=L, vtype=GRB.INTEGER, name='offload_num')

        decode_steps = model.addVar(lb=32, ub=window_ub, vtype=GRB.INTEGER, name='decode_steps')

        offload_num_constr = {(r, i): model.addVar(vtype=GRB.BINARY, name=f"onc_{i}") for i in range(1, L+2) for r in requests}

        print(floor_val)

        for r in requests:
            model.addConstr(gp.quicksum(offload_num_constr[(r, i)] for i in range(1, L+2)) == 1)
            model.addConstr(gp.quicksum(offload_num_constr[(r, i)] * i for i in range(1, L+2)) == prefetch_dist[r])
            model.addConstr(gp.quicksum(offload_num_constr[(r, i)] * floor_val[i] for i in range(1, L+2)) == offload_num[r])

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

# === 6. 목적함수 및 최적화 ===
        goodput = model.addVar(lb=0, name='obj')
        model.addConstr(goodput * token_time == gp.quicksum(resume[r] for r in requests) - slo_fail_per_decode.sum())
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

