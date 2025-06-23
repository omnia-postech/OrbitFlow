""" 
Solver inputs: list of Request 
layer_num: 32 
block_bandwidth: 103178.0 / 1000 (# blocks per second? or miliseconds)
gpu_block_capacity: block_manager.num_total_gpu_blocks 
window_ub: window upper bound? 
 
Solver outputs: list of Result 
class Request:
    def __init__(self, id: str, context_len_in_blocks: int, layer_time: float,
                 deposit_count: int, slo: float, gpu_layers_on_gpu: int):
        self.id = id
        self.context_len_in_blocks = context_len_in_blocks
        self.layer_time = layer_time
        self.deposit_count = deposit_count
        self.slo = slo
        self.gpu_layers_on_gpu = gpu_layers_on_gpu
"""
import torch 
from solver  import Solver_updated as Solver
# from solver import LatencySolver as Solver
from solver import Request
# from solver  import Solver as Solver
import time

from math import floor
from typing import List, Dict, Tuple, Union, Optional
PROFILED_A = 1.0017431830666432e-06
PROFILED_B = 0.049519613282613506
RequestDesc = Dict[str, Union[str, bool, int, float]]

def compute_batch_latency(
    requests: List["Request"],
    offload_num: Optional[Dict[int, int]] = None,
    layer_num: int = 32,
    block_bandwidth: float = 103_178.0,   # blocks per second
    epsilon: float = 1e-12
) -> Tuple[float, Dict[str, Optional[float]]]:
    """
    Return (token_time, {req.id: actual_time or None})
    All time units must be consistent (s or ms).

    Each Request may carry extra attributes:
        • resume : bool   (default True)
        • offload_num      –int
          OR
        • prefetch_dist    –int   (offload_num = floor(layer_num / prefetch_dist))
    """
    # compute batch layer time 
    batch_req_len = sum(r.context_len_in_blocks for r in requests)* 16 
    batch_layer = (PROFILED_A * batch_req_len + PROFILED_B ) / 32
    print("batch_layer", batch_layer)
    comp_time = batch_layer * 32 
    print("comm_time", comp_time) 
    
    comm_time = 0 
    token_time = max(comp_time, comm_time)
    print("token_time", token_time) 
    print("requests", requests)
path = "/home/xinyuema/vllm/vllm/worker/distn/snapshot/step{}.pt" 



steps = list(range(49))
steps = [0]

solver = Solver()
for step in steps: 
    solver_req = torch.load(path.format(step), weights_only=False)
    request_list, block_bandwidth, gpu_block_capacity = solver_req 
    if len(request_list) == 0: 
        # make a dummy request\
        request_list.append(Request(
            id="dummy1", context_len_in_blocks=10, layer_time=0.0000125, deposit_count=1000, slo=100, gpu_layers_on_gpu=3
        ))
        request_list.append(Request(
            id="dummy2", context_len_in_blocks=30, layer_time=0.0000125, deposit_count=1000, slo=100, gpu_layers_on_gpu=2
        ))        
    # request_list[0].layer_time = 0.10
    print("request_list", request_list)
    print("block_bandwidth", block_bandwidth)
    print("gpu_block_capacity", gpu_block_capacity)
    import time 
    print('---------solver original-----------')
    start = time.time() 
    output = solver.solve(
        request_list, 
        layer_num=32,
        gpu_block_capacity=100*32)
    end = time.time()
    print("time", end-start)
    end = time.time()
    print(output)
    solver_dists = {s.id: s.n for s in output}  
    print(solver_dists)
    print("solver time", end-start)
    print('-----------solver original end-------------')

    # print('----------solver minimal--------------')
    # start = time.time() 
    # output = solver.solve_minimal(
    #     request_list, 
    #     layer_num=32,
    #     gpu_block_capacity=30*32,
    #     )
    # end = time.time()
    # print("time", end-start)
    # end = time.time()
    # print(output)
    # solver_dists = {s.id: s.n for s in output}  
    # print(solver_dists)
    # print("solver time", end-start)
    
    # print('---------solver minimal end---------------')

    
    # offload_num = {r.id: list(range(31)) for r in request_list}
    # compute_batch_latency(request_list, offload_num=offload_num, block_bandwidth=block_bandwidth)
    
    # latency = solver.compute_T_batch(
    #     requests_list=request_list,
    #     offload_decision=solver_dists,
    #     layer_num=32,
    # )
    # uni_dist = {r.id: 2 for r in request_list}
    # print(latency)
    # latency = solver.compute_T_batch(
    #     requests_list=request_list,
    #     offload_decision=uni_dist,
    #     layer_num=32,
    # )
    # print(latency)