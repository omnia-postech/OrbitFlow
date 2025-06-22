# -------------------------------------------------------------------------
#  Solver: KV-placement & prefetch-distance optimiser  (minimal version)
# -------------------------------------------------------------------------
#  ┌─────────┐
#  │ INPUTS  │     ─ hardware/model constants (system-level)
#  │         │     ─ per-batch request descriptors (instance-level)
#  ├─────────┤
#  │ MODEL   │     ─ decision vars
#  │         │     ─ constraints
#  │         │     ─ latency sub-model (stall & comm)
#  ├─────────┤
#  │ SOLVE   │     ─ MILP+bilinear optimisation
#  └─────────┘
#
#  Anything not strictly required for the analytical description
#  (e.g. resume/off-batch logic, global SLO cap) has been removed.
# -------------------------------------------------------------------------

import gurobipy as gp
from gurobipy import GRB
from typing import List, Optional


# ========= 1. INPUTS =====================================================

TOKENS_PER_BLOCK = 16          # system constant
BIG_M             = 1e6        # convenience big-M

def solve_minimal(
    requests_list: List["Request"],
    *,
    # --- system-level parameters (usually fixed for a given cluster/model)
    layer_num          : int   = 32,        # transformer depth  (L)
    gpu_block_capacity : int   = 614,       # KV-block budget on GPU
    # profiled curves injected by caller (avoid hard-coding in algorithm)
    sec_per_layer_on_gpu : float,           # compute time when all KV in GPU
    blocks_per_sec_comm : float,            # PCIe/NVLink throughput
    # --- tuning knob
    window_ub          : int   = 1000,      # max decode window
) -> Optional["ResultList"]:

    # ---- 1.1 Extract per-request constants ------------------------------
    R = [r.id for r in requests_list]

    C_block = {r.id: r.context_len_in_blocks for r in requests_list}  # KV blocks / layer
    SLO     = {r.id: r.slo                     for r in requests_list}
    deposit = {r.id: r.deposit_count           for r in requests_list}

    L  = layer_num
    B  = blocks_per_sec_comm                  # rename for brevity
    Tc = sec_per_layer_on_gpu                 # compute time of one layer

    # ========= 2. DECISION VARIABLES =====================================
    m = gp.Model("kv_solver_min")
    m.Params.NonConvex = 2

    # 2.1 One-hot stride selector -----------------------------------------
    #    Valid strides = divisors of L **minus 1**  (1,2,3,4,5,7,9,15,31 for L=32)
    valid_stride = [d for d in range(1, L) if L % (d + 1) == 0]
    onc = m.addVars([(r, d) for r in R for d in valid_stride],
                    vtype=GRB.BINARY, name="onc")

    # Derivatives: stride (prefetch_dist) and # offloaded layers -----------
    # ------- NEW (continuous) ------------------------------
    stride       = m.addVars(R, lb=1,  ub=L, vtype=GRB.CONTINUOUS, name="stride")
    n_off        = m.addVars(R, lb=0,  ub=L, vtype=GRB.CONTINUOUS, name="n_off")


    m.addConstrs((gp.quicksum(onc[r, d] for d in valid_stride) == 1 for r in R),
                 name="one_stride")
    m.addConstrs((gp.quicksum(onc[r, d] * (d + 1)   for d in valid_stride) == stride[r]
                 for r in R), name="stride_def")
    m.addConstrs((gp.quicksum(onc[r, d] * (L // (d + 1)) for d in valid_stride) == n_off[r]
                 for r in R), name="n_off_def")

    # 2.2 Latency-model vars ----------------------------------------------
    stall = m.addVars(range(1, L + 1), lb=0.0, name="stall")
    token_time = m.addVar(lb=0.0, name="token_time")

    # ========= 3. COMMUNICATION & STALL ==================================
    #
    #  pre-compute a_rdj  (1 if layer j is *off-GPU* under stride d)
    #
    a = {
        r: {d: {j: int(j % (d + 1) == 0) for j in range(1, L + 1)}
            for d in valid_stride}
        for r in R
    }

    # 3.1 per-layer communication time  (sum over active requests) --------
    t_comm = {}
    for j in range(1, L + 1):
        # blocks transferred at layer j, divided by bandwidth B
        t_comm[j] = (
            gp.quicksum(onc[r, d] * a[r][d][j] * C_block[r]
                        for r in R for d in valid_stride) / B
        )

    # 3.2 stall >= comm – overlap -----------------------------------------
    m.addConstr(stall[1] >= t_comm[1], name="stall_first")
    for j in range(2, L + 1):
        # single-layer look-ahead overlap (may refine later)
        m.addConstr(stall[j] >= t_comm[j] - Tc * 1, name=f"stall_{j}")

    # 3.3 batch latency ----------------------------------------------------
    m.addConstr(token_time == L * Tc + gp.quicksum(stall[j] for j in range(1, L + 1)),
                name="token_time_def")

    # ========= 4. CAPACITY CONSTRAINT ====================================
    m.addConstr(
        gp.quicksum((L - n_off[r]) * C_block[r] for r in R) <= gpu_block_capacity,
        name="gpu_mem"
    )

    # ========= 5. SLO / WINDOW LOGIC (minimal) ===========================
    decode_steps = m.addVar (          lb=32, ub=window_ub,
                                    vtype=GRB.CONTINUOUS,      name="decode_steps")

    # ratio_r = actual_time / (decode_steps * SLO_r)
    ratio = m.addVars(R, lb=0.0, name="ratio")
    m.addConstrs((ratio[r] * token_time == decode_steps * SLO[r] for r in R),
                 name="ratio_qc")

    # slo_fail_r = max(0, 1 – ratio – deposit/decode_steps)
    slo_fail = m.addVars(R, lb=0.0, name="slo_fail")
    m.addConstrs(
        (slo_fail[r] * decode_steps >= decode_steps - ratio[r] - deposit[r]
         for r in R), name="slo_fail_def"
    )

    # ========= 6. OBJECTIVE =============================================
    # minimise per-token latency  (commented line shows good-put alt.)
    m.setObjective(token_time, GRB.MINIMIZE)
    # m.setObjective(gp.quicksum(1 - slo_fail[r] for r in R) / token_time,
    #                GRB.MAXIMIZE)

    # ========= 7. SOLVE ==================================================
    m.Params.OutputFlag = 1
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        return None

    # ========= 8. RETURN PARSED RESULTS =================================
    results = ResultList()
    for r in R:
        results.append(Result(
            id=r,
            resume=True,
            n=stride[r].X - 1,
            offload_num=int(n_off[r].X),
            slo_fail=slo_fail[r].X * decode_steps.X,
            actual_time=token_time.X,
            window=decode_steps.X,
        ))
    return results


# -------------------------------------------------------------------------
#  KV-placement & Prefetch MILP  (McCormick linearised, decode_steps free)
# -------------------------------------------------------------------------
import gurobipy as gp
from gurobipy import GRB
from typing import List, Optional


TOKENS_PER_BLOCK = 16
BIG_M            = 1e6          # coarse upper bound for times

def solve_milp(
    requests_list: List["Request"],
    *,
    # ---- system-level constants ----------------------------------------
    layer_num             : int   = 32,
    gpu_block_capacity    : int   = 614,
    sec_per_layer_on_gpu  : float,
    blocks_per_sec_comm   : float,
    # ---- tuning knobs ---------------------------------------------------
    window_ub             : int   = 1000,
) -> Optional["ResultList"]:

    # ========= 1.  INSTANCE CONSTANTS ===================================
    R = [r.id for r in requests_list]

    C_block = {r.id: r.context_len_in_blocks for r in requests_list}   # blocks / layer
    SLO     = {r.id: r.slo                     for r in requests_list}
    deposit = {r.id: r.deposit_count           for r in requests_list}

    L  = layer_num
    B  = blocks_per_sec_comm
    Tc = sec_per_layer_on_gpu

    # ----------
    # Tight bounds needed for McCormick envelopes
    # ----------
    token_L  = L * Tc                         # no-stall latency lower bound
    token_U  = BIG_M                          # coarse but constant
    decode_L = 32                             # per spec
    decode_U = window_ub
    ratio_L, ratio_U     = 0.0, 1.5           # generous upper bound ≥ 1
    slofail_L, slofail_U = 0.0, 1.0           # failure rate is a fraction

    # ========= 2.  MODEL & VARIABLES ====================================
    m = gp.Model("kv_solver_milp")            # <- now pure MILP!

    # 2.1 stride selection ------------------------------------------------
    valid_stride = [d for d in range(1, L) if L % (d + 1) == 0]
    onc   = m.addVars([(r, d) for r in R for d in valid_stride],
                      vtype=GRB.BINARY,  name="onc")

    stride = m.addVars(R, lb=1,  ub=L, vtype=GRB.INTEGER, name="stride")
    n_off  = m.addVars(R, lb=0,  ub=L, vtype=GRB.INTEGER, name="n_off")

    m.addConstrs((gp.quicksum(onc[r, d] for d in valid_stride) == 1 for r in R),
                 name="one_stride")
    m.addConstrs((gp.quicksum(onc[r, d] * (d + 1)   for d in valid_stride) == stride[r]
                 for r in R), name="stride_def")
    m.addConstrs((gp.quicksum(onc[r, d] * (L // (d + 1)) for d in valid_stride) == n_off[r]
                 for r in R), name="n_off_def")

    # 2.2 latency variables ----------------------------------------------
    stall      = m.addVars(range(1, L + 1), lb=0.0, name="stall")
    token_time = m.addVar(lb=token_L, ub=token_U, name="token_time")

    # 2.3 SLO-related vars (now with McCormick products) ------------------
    ratio    = m.addVars(R, lb=ratio_L,  ub=ratio_U,  name="ratio")
    slo_fail = m.addVars(R, lb=slofail_L, ub=slofail_U, name="slo_fail")

    decode_steps = m.addVar(lb=decode_L, ub=decode_U,
                            vtype=GRB.INTEGER, name="decode_steps")

    # ---- McCormick auxiliary variables ---------------------------------
    w_rt = m.addVars(R, lb=0.0, name="w_ratio*token")           # ⇦ NEW (MILP)
    w_sd = m.addVars(R, lb=0.0, name="w_slofail*decode")        # ⇦ NEW (MILP)

    # ========= 3.  COMMUNICATION & STALL ================================
    a = {
        r: {d: {j: int(j % (d + 1) == 0) for j in range(1, L + 1)}
            for d in valid_stride}
        for r in R
    }

    t_comm = {}
    for j in range(1, L + 1):
        t_comm[j] = (
            gp.quicksum(onc[r, d] * a[r][d][j] * C_block[r]
                        for r in R for d in valid_stride) / B
        )

    m.addConstr(stall[1] >= t_comm[1], name="stall_first")
    for j in range(2, L + 1):
        m.addConstr(stall[j] >= t_comm[j] - Tc,
                    name=f"stall_{j}")

    m.addConstr(token_time == L * Tc + gp.quicksum(stall[j] for j in range(1, L + 1)),
                name="token_time_def")

    # ========= 4.  GPU CAPACITY =========================================
    m.addConstr(
        gp.quicksum((L - n_off[r]) * C_block[r] for r in R) <= gpu_block_capacity,
        name="gpu_mem"
    )

    # ========= 5.  SLO  (McCormick linearisation) =======================
    # 5.1  w_rt[r] = ratio[r] * token_time  (envelope) -------------------
    for r in R:
        m.addConstr(w_rt[r] >= ratio_L * token_time + token_L * ratio[r] - ratio_L * token_L,
                    name=f"mcc1_rt_{r}")                        # ⇦ NEW (MILP)
        m.addConstr(w_rt[r] >= ratio_U * token_time + token_U * ratio[r] - ratio_U * token_U,
                    name=f"mcc2_rt_{r}")                        # ⇦
        m.addConstr(w_rt[r] <= ratio_U * token_time + token_L * ratio[r] - ratio_U * token_L,
                    name=f"mcc3_rt_{r}")                        # ⇦
        m.addConstr(w_rt[r] <= ratio_L * token_time + token_U * ratio[r] - ratio_L * token_U,
                    name=f"mcc4_rt_{r}")                        # ⇦
        # now equate to RHS (decode_steps * SLO_r) – linear
        m.addConstr(w_rt[r] == decode_steps * SLO[r],
                    name=f"ratio_def_{r}")                      # ⇦ NEW (MILP)

    # 5.2  w_sd[r] = slo_fail[r] * decode_steps  (envelope) --------------
    for r in R:
        m.addConstr(w_sd[r] >= slofail_L * decode_steps + decode_L * slo_fail[r]
                                - slofail_L * decode_L,
                    name=f"mcc1_sd_{r}")                        # ⇦
        m.addConstr(w_sd[r] >= slofail_U * decode_steps + decode_U * slo_fail[r]
                                - slofail_U * decode_U,
                    name=f"mcc2_sd_{r}")                        # ⇦
        m.addConstr(w_sd[r] <= slofail_U * decode_steps + decode_L * slo_fail[r]
                                - slofail_U * decode_L,
                    name=f"mcc3_sd_{r}")                        # ⇦
        m.addConstr(w_sd[r] <= slofail_L * decode_steps + decode_U * slo_fail[r]
                                - slofail_L * decode_U,
                    name=f"mcc4_sd_{r}")                        # ⇦
        # inequality ( ≥ ) from original formulation
        m.addConstr(w_sd[r] >= decode_steps - ratio[r] - deposit[r],
                    name=f"slo_def_{r}")                        # ⇦ NEW (MILP)

    # ========= 6.  OBJECTIVE ============================================
    m.setObjective(token_time, GRB.MINIMIZE)

    # ========= 7.  SOLVE ===============================================
    m.Params.OutputFlag = 1
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        return None

    # ========= 8.  PACK RESULTS ========================================
    results = ResultList()
    for r in R:
        results.append(Result(
            id=r,
            resume=True,
            n=stride[r].X - 1,
            offload_num=int(n_off[r].X),
            slo_fail=slo_fail[r].X * decode_steps.X,
            actual_time=token_time.X,
            window=decode_steps.X,
        ))
    return results
