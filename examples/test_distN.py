import os
import time
import json
from collections import defaultdict

from vllm.engine.llm_engine import LLMEngine
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams
from vllm.logger import init_logger
from vllm.inputs import TokensPrompt

from trace_generator import Trace
logger = init_logger("vllm")
import logging

import pandas as pd 
import torch 
import bisect
import torch
import csv, os, pathlib

torch.set_printoptions(edgeitems=2, linewidth=120, sci_mode=True)
# --- Config ---
MODEL = "/home/jongseop/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
PROMPT_DIR = "./prompts"
USE_DEFAULT_SAMPLES = True
BATCH_SIZE = 4  # Serving batch size set to 4
MAX_MODEL_LEN = 13000
BLOCK_SIZE  = 16
SLO_THRESHOLD = 0.5      
CSV_OUTPUT_FILE = "metrics.csv"
PROFILED_A = 1.0017431830666432e-06
PROFILED_B = 0.049519613282613506
test_trace = {
    "description": "Continuous batching; BS = 4; A6000 with 48GB memory; Continuous batched request too large to fit in available memory.", 
    "batch_size": 4, 
    "max_model_len": 10000,
    "num_gpu_blocks_override": 400,
    "samples": {
        "sample1":  {"prompt": 250, "max_tokens": 1500, "arrive_at_step": 0, "schedule_at_step": 0, "wait_time": 0},
        "sample2":  {"prompt": 500, "max_tokens": 2000, "arrive_at_step": 0, "schedule_at_step": 0, "wait_time": 0},
        "sample3":  {"prompt": 250, "max_tokens": 2000, "arrive_at_step": 0, "schedule_at_step": 0, "wait_time": 0},
        "sample4":  {"prompt": 500, "max_tokens": 2000, "arrive_at_step": 0, "schedule_at_step": 0, "wait_time": 0},
        "sample5":  {"prompt": 5000, "max_tokens": 2000, "arrive_at_step": 1450, "schedule_at_step": 1500, "wait_time": 50},
        "sample6":  {"prompt": 1000, "max_tokens": 1500, "arrive_at_step": 1475, "schedule_at_step": 2000, "wait_time": 525}
    }
}
DEFAULT_PROMPTS = test_trace

class DelaySimulator:
    def __init__(self, v_tps: float, slo_ratio: float = 0.5, deposit_enabled: bool = True):        
        # v_tps: tokens per second
        self.deposit_enabled = deposit_enabled # if disabled, deposit is empty and the simulator only keeps track of slo violations
        self.v_default = v_tps          # keep for backwards compatibility
        self.v = {}                           # rid -> tokens/sec
        self.last_time = {}
        self.last_decode_time = 0.0
        # self.interval = 1.0 / v_tps       # 토큰 하나가 배출되어야 하는 간격(초)
        # self.next_deadline = self.interval
        self.deposit = defaultdict(int)   # request_id → 누적된 토큰 수
        self.violations = defaultdict(int)      #  #of slo violations for each request
        self.slo_ratio = slo_ratio
        logger.info(f"[Simulator __init__] v_tps={self.v_default:.3f} tokens/sec")

    def register(self, rid: str, slo_s_per_token: float):
        if slo_s_per_token <= 0:
            raise ValueError("SLO must be > 0")
        slo_s_per_token = slo_s_per_token / self.slo_ratio # slo = 1.0, ratio = 0.5 => real slo = 2 
        self.v[rid] = 1.0 / slo_s_per_token
        self.deposit[rid] = 0
        self.violations[rid] = 0

    def on_token(self, rid: str, step_time: float):
        before = self.deposit[rid]
        if self.deposit_enabled:
            self.deposit[rid] += 1
        else: 
            self.deposit[rid] = 1 # exactly 1 token, since we assume whatever we had is sent right away
        after = self.deposit[rid]
        
        if rid not in self.last_time: 
            self.last_time[rid] = step_time
            logger.debug(f"[on_token] first token for {rid}, setting last_time[{rid}] = {step_time:.6f}")
        logger.debug(f"[on_token] rid={rid} @ {step_time:.6f}: deposit {before} -> {after}")


    def pop(self, step_time: float, solver_time: float = 0.0):
        deposits_snapshot = {k: int(v) for k, v in self.deposit.items()}
        pops: list[tuple[str, int]] = []
        rid_log: list[str] = []

        for rid, dep in list(self.deposit.items()):
            v  = self.v.get(rid, None)  # default v_tps  
            if dep == 0:
                self.last_time[rid] = step_time
                continue 
            
            
            last   = self.last_time.get(rid, step_time)
            dt     = step_time - last - solver_time           
            n      = int(dt * v)             
            if n <= 0:
                continue

            # release n tokens, but not more than deposit
            if dep < n:
                self.violations[rid] += 1 
                rid_log.append(f"{rid}(SLO {dep}/{n}), {self.violations[rid]} violations so far")
            else:
                rid_log.append(f"{rid}({n})@{v:.2f}t/s)")
            to_rel = min(n, dep)
            # deposit 차감
            self.deposit[rid] -= to_rel


            if self.deposit[rid] == 0:         # backlog cleared → reset stopwatch
                self.last_time[rid] = step_time   #  ← NEW LINE
            else: 
                # last_time[rid] 을 방출된 토큰 시간만큼 앞으로 이동
                # 즉, to_rel tokens / v_tps 만큼 경과시킨 것처럼
                advance = to_rel / v
                self.last_time[rid] = last + advance                
            pops.append((rid, to_rel))
        # single-line summary
        if rid_log:
            logger.info(
                "[pop] deposits=%s | released: %s",
                deposits_snapshot,
                " | ".join(rid_log),
            )
        return pops
    def violation_count(self, rid: str) -> int:
        return self.violations.get(rid, 0)
    def finish(self, rid: str):
        """요청 완료 시 호출: 남은 deposit 제거, tracking 변수 cleanup"""
        if rid in self.deposit:
            logger.info(f"[finish] clearing deposit for {rid}: was {self.deposit[rid]}")
            del self.deposit[rid]
        if rid in self.last_time:
            logger.info(f"[finish] removing last_time entry for {rid}")
            del self.last_time[rid]
        if rid in self.v:
            self.v.pop(rid, None)
        if rid in self.violations:
            self.violations.pop(rid,None)

    def stats(self) -> dict:
        """현재 deposit 맵(rid→남은 토큰 수) 반환."""
        stats = dict(self.deposit)
        # logger.info(f"[stats] deposit map: {stats}")
        return dict(stats)

def load_prompts(path=None):
    use_default = path is None 
    if use_default:
        return DEFAULT_PROMPTS
    
    with open(path, "r") as f:
        data = json.load(f)
        description = data.get("description", "No description")
        logging.info(f"Prompt description: {description}")
    return data

def enqueue_batch(engine, batch, request_metadata):
    """Enqueues a batch of requests into the engine."""
    for req_id, prompt in batch:
        arrival_time = time.time()  # Record arrival time
        request_metadata[req_id] = {
            "arrival_time": 0,
            "scheduled_time": None,
            "first_token_time": None,
            "finished_time": None,
            "token_timestamps": [],
            "decode_length": 0,
            "stall_times": [],
            "stall_durations": [],
            "stall_duration": 0,
        }

        # Build and add the prompt to the engine.
        prompt_len = prompt["prompt"]
        token_id_range = (200, 20000)
        prompt['prompt_ids'] =  torch.randint(low=token_id_range[0],
                          high=token_id_range[1],
                          size=(prompt_len,),
                          dtype=torch.int).tolist()
        prompt_obj = TokensPrompt(prompt_token_ids=prompt['prompt_ids'])
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=prompt['max_tokens'],
            stop=[],
            stop_token_ids=[],
            ignore_eos=True,
        )
        engine.add_request(req_id, prompt_obj, sampling_params)

def run_inference_step_mode(engine, trace_obj, csv_path=None, enable_deposit=False):
    """
    Step-based inference driver that consumes a Trace object (dictionary-based).
    Each request's arrival_time is interpreted as the 'arrive_at_step'.

    We assume:
      - trace_obj.requests is a dict: {"request_0": Request, "request_1": Request, ...}
      - Each Request object has arrival_time, input_length, output_length, etc.
      - Scheduling logic is already done in the trace if needed (sched_time, wait_time),
        but here we demonstrate a step-based approach for how you might integrate
        with vLLM's engine.step() loop.

    The rest of the logic follows the original step-by-step approach, except
    references to 'prompt["arrive_at_step"]', 'prompt["prompt"]', and 'prompt["max_tokens"]'
    now map to request.arrival_time, request.input_length, request.output_length, respectively.
    """

    import time
    import os
    import json
    import pandas as pd
    from collections import defaultdict
    import logging
    import torch
    from vllm.sampling_params import SamplingParams
    logger = logging.getLogger("vllm")
    consecutive_no_output = 0
    solver_invocations = 0
    # 1) Convert Trace dictionary -> sorted list by arrival_time
    #    E.g. [("request_0", req0), ("request_1", req1), ...]
    #    We'll interpret arrival_time as "arrive_at_step".
    requests_sorted = sorted(
        trace_obj.requests.items(),
        key=lambda x: x[1].arrival_time
    )
    # 2) Build an initial queue of all requests (similar to prompt_dict usage)
    #    We'll keep them in a list but only "activate" them once
    #    cumulative_steps >= arrival_time.
    queue = list(requests_sorted)

    # 3) Data structures for the step loop
    request_metadata = {}
    request_output = defaultdict(list)
    metrics_data = []

    # In the original, we track times / tokens / steps
    start_time = None
    end_time = None
    overall_wall    = 0.0     # s
    prefill_wall    = 0.0     # s (non-overlapping)
    decode_wall     = 0.0     # s (non-overlapping)
    cumulative_tokens = 0
    cumulative_steps = 0
    finished_tokens = 0
    finished_decode_tokens = 0
    finished_prefill_tokens = 0

    running_requests = set()
    received_requests = []

    def enqueue_batch(engine, batch, request_metadata):
        """Enqueue a batch of requests into the vLLM engine."""
        for req_id, req_obj in batch:
            # Keep track of request-level metadata
            request_metadata[req_id] = {
                "arrival_time": time.time(),  # or 0 if you prefer
                "scheduled_time": None,
                "first_token_time": None,
                "finished_time": None,
                "token_timestamps": [],
                "profiled_tbt": [],
                "time_between_tokens": [],
                "solver_time": [],
                "decode_length": 0,
                "expected_output_length": req_obj.output_length,
                "stall_times": [],
                "stall_durations": [],
                "stall_duration": 0,
                "prompt_length": req_obj.input_length,
            }

            # Build the random input tokens:
            # (just for demonstration; in reality you might do something else)
            # in case token ids are empty [Deprecated, now Trace object generate this]
            if not hasattr(req_obj, "token_ids"):
                # Randomly generate token IDs for the prompt
                # This is a placeholder; replace with actual token generation
                token_id_range = (200, 20000)
                req_obj.token_ids = torch.randint(
                    low=token_id_range[0],
                    high=token_id_range[1],
                    size=(req_obj.input_length,),
                    dtype=torch.int
                ).tolist()

            # Prepare the vLLM tokens prompt
            prompt_obj = TokensPrompt(prompt_token_ids=req_obj.token_ids)

            # Prepare sampling params
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=req_obj.output_length,
                stop=[],
                stop_token_ids=[],
                ignore_eos=True
            )

            # Enqueue with the engine
            engine.add_request(req_id, prompt_obj, sampling_params)

    SLO_THRESHOLD = 0.5 # TBT SLO (seconds per token)
    sim = DelaySimulator(v_tps=1/SLO_THRESHOLD, slo_ratio=0.8, deposit_enabled=enable_deposit)

    if csv_path is None:
        csv_path = "metrics.csv"
    csv_path = pathlib.Path(csv_path)

    # True if we need to write the header row first
    write_header = not csv_path.exists()
    csv_fh = csv_path.open("a", newline="", buffering=1)   # line-buffered I/O
    csv_writer = csv.DictWriter(csv_fh, fieldnames=[
        "request_id", "arrival_time", "first_scheduled_time",
        "finished_time", "stall_times",
        "time_to_first_token", "slo_threshold",
        "slo_violations",
        "stall_duration", "decode_length",
        "end_to_end_time", "decode_time",
        "time_per_output_token", "time_between_tokens",
        "finish_reason", "solver_time",
        "profiled_tbt", "expected_output_length",
        "stall_durations",
    ])
    if write_header:
        csv_writer.writeheader()
    step_count = 1
    # The main simulation loop
    while queue or request_metadata:
        # 4) Find all requests that have arrival_time <= cumulative_steps
        #    -> these are ready to enqueue
        ready = [(req_id, req_obj) for (req_id, req_obj) in queue
                 if req_obj.arrival_time <= cumulative_steps]
        if ready:
            # We enqueue *all* that are <= cumulative_steps
            enqueue_batch(engine, ready, request_metadata)
            for (req_id, req_obj )in ready: 
                if hasattr(req_obj, "slo") and req_obj.slo is not None: 
                    max_slo = req_obj.slo
                    slo = sim.register(req_id, max_slo)
                else: 
                    max_slo = PROFILED_A*(req_obj.input_length + req_obj.output_length) + PROFILED_B
                    slo = sim.register(req_id, max_slo)
                logger.critical(f"Enqueued request {req_id} with max_slo {max_slo} and SLO {(1/sim.v[req_id]):.3f} ms per token")
            # Remove them from the queue
            queue = [(req_id, req_obj) for (req_id, req_obj) in queue
                     if req_obj.arrival_time > cumulative_steps]

        # TODO(Heelim): need to send metadata to scheduler to make decision of which request and when to pause and resume
        deposit_map = sim.stats()  
        engine.scheduler[0].deposit_map = deposit_map
        engine.scheduler[0].slo_from_delaysim = sim.v 
        # NOTE(HONG): use below for pipelining parallelism
        step_start = time.time()
        step_outputs = engine.step()
        step_end = time.time()
        step_count += 1
        elapsed_time_step = step_end - step_start
        overall_wall += elapsed_time_step
        
        # sim.last_decode_time = elapsed_time_step
        # If no outputs come back, we still increment steps
        cumulative_steps += 1
        
        # NOTE(HONG): neet to use step_time to unify the time for all events at step-level
        torch.cuda.synchronize()

        # 6) Process each output
        # gather rids so we can log them in one line
        prefill_rids: list[str] = []
        decode_rids:  list[str] = []
        finished_rids: list[str] = []
        step_tokens = 0
        # to test whether it was decode 
        if len(step_outputs) > 0:
            if step_outputs[0].request_id in received_requests:
                decode_rids = list(set([output.request_id for output in step_outputs])) 
                decode_wall += elapsed_time_step
                step_tokens = sum([request_metadata[rid]["prompt_length"] + request_metadata[rid]['decode_length']+1 for rid in decode_rids])
                profiled_res = PROFILED_A * step_tokens + PROFILED_B 
                if not hasattr(step_outputs[0], "solver_time"):
                    solver_time = 0.0
                else: 
                    solver_time = step_outputs[0].solver_time
            else: 
                prefill_rids = list(set([output.request_id for output in step_outputs])) 
                prefill_wall += elapsed_time_step
                profiled_res = PROFILED_A * step_tokens + PROFILED_B 
                step_tokens = sum([request_metadata[rid]["prompt_length"] for rid in prefill_rids])
                solver_time = 0.0
            consecutive_no_output = 0
        else: 
            solver_time = 0.0
            if queue: 
                time.sleep(0.1) # wait for a while to avoid busy waiting 
            else: 
                consecutive_no_output += 1 
                if consecutive_no_output > 5:
                    # pop everything in request_metadata into finished_requests 
                    # and stop the simulation
                    logger.critical("No output for 5 consecutive steps, stopping simulation.")
                    rids = list(request_metadata.keys())
                    for rid in rids:
                        logger.critical(f"[finish] clearing deposit for {rid}: was {sim.deposit[rid]}")
                        sim.finish(rid)
                        request_metadata.pop(rid)
                        logger.critical(f"[finish] removing last_time entry for {rid}")
                        print(f"Failure (due to memory limits): {len(rids)}")
                    break
        
        assert(elapsed_time_step > solver_time) 

        for output in step_outputs:
            rid = output.request_id
            # Prefill
            if len(prefill_rids) > 0:
            # if rid not in received_requests:
                sim.last_decode_time = 0
                # This is the first token for that request
                received_requests.append(rid)
                prefill_rids.append(rid)
                # We'll treat the entire prompt as prefill
                # i.e., output.prompt_token_ids is the input
                finished_tokens += len(output.prompt_token_ids)
                finished_prefill_tokens += len(output.prompt_token_ids)
                running_requests.add(rid)
            # Decoding
            else:
                sim.last_decode_time = elapsed_time_step
                # Another token
                decode_rids.append(rid)
                # now = time.time()

                sim.on_token(rid, step_end)

                # NOTE(HONG): not useing output.metrics.last_token_time since we are using step_time
                request_metadata[rid]["token_timestamps"].append(step_end)
                request_metadata[rid]["time_between_tokens"].append(elapsed_time_step)
                request_metadata[rid]["solver_time"].append(solver_time)
                request_metadata[rid]["profiled_tbt"].append(profiled_res)
                request_metadata[rid]["decode_length"] += 1
                finished_tokens += 1
                finished_decode_tokens += 1
                request_output[rid].append(output)
                
            step_tokens += request_metadata[rid]["prompt_length"]
            step_tokens += request_metadata[rid]["decode_length"]

            # If the request is finished:
            if output.finished:

                if len(output.outputs[0].token_ids) < request_metadata[rid]["expected_output_length"]:
                    finish_reason = "length_capped"
                else: 
                    finish_reason = output.outputs[0].finish_reason
                logger.critical(f"Finished request {rid} at step {cumulative_steps}, finish_reason={finish_reason}")
                logger.critical(f"{rid} prompt: {len(output.prompt_token_ids)} tokens; {output.prompt_token_ids[:20]}")
                logger.critical(f"{rid} output: {len(output.outputs[0].token_ids)} {output.outputs[0].token_ids[:20]}")
                logger.critical(f"{rid} text: {output.outputs[0].text[:20]}")
                running_requests.remove(rid)

                # ------------------------------------------------------------------
                # LOCAL WALL‑CLOCK MEASUREMENTS (no out.metrics)
                # ------------------------------------------------------------------
                arrival_time_local = request_metadata[rid]["arrival_time"]
                finished_time_local = step_end
                m = output.metrics

                token_ts = request_metadata[rid]["token_timestamps"]
                if token_ts:
                    first_token_time_local = token_ts[0]
                    per_token_latencies = [j - i for i, j in zip(token_ts[:-1], token_ts[1:])]
                    avg_token_latency = sum(per_token_latencies) / len(per_token_latencies) if per_token_latencies else 0.0
                else:
                    first_token_time_local = finished_time_local
                    per_token_latencies = []
                    avg_token_latency = 0.0

                decode_length = request_metadata[rid]["decode_length"]

                if start_time is None or arrival_time_local < start_time:
                    start_time = arrival_time_local
                if end_time is None or finished_time_local > end_time:
                    end_time = finished_time_local

                row = {
                    "request_id": rid,
                    "arrival_time": arrival_time_local - (start_time or 0),
                    "first_scheduled_time": 0,                 # not available locally
                    "finished_time": finished_time_local - (start_time or 0),
                    "stall_times": json.dumps(request_metadata[rid]["stall_times"]),
                    "time_to_first_token": first_token_time_local - arrival_time_local,
                    "slo_threshold": 1/sim.v[rid],
                    "slo_violations": sim.violation_count(rid),
                    "stall_duration": request_metadata[rid]["stall_duration"],
                    "decode_length": decode_length,
                    "end_to_end_time": finished_time_local - arrival_time_local,
                    "decode_time": finished_time_local - first_token_time_local,
                    "time_per_output_token": avg_token_latency,
                    "finish_reason": finish_reason,
                    # "time_between_tokens": json.dumps(per_token_latencies),
                    "solver_time": json.dumps(request_metadata[rid]["solver_time"]),
                    "time_between_tokens": json.dumps(request_metadata[rid]["time_between_tokens"]),
                    "profiled_tbt": request_metadata[rid]["profiled_tbt"],
                    "expected_output_length": request_metadata[rid]["expected_output_length"],
                    "stall_durations": json.dumps(request_metadata[rid]["stall_durations"]),
                }
                metrics_data.append(row)
                csv_writer.writerow(row)
                csv_fh.flush()  
                finished_rids.append(rid)
                logger.info(f"Finished request {rid} with {decode_length} decode tokens")
                sim.finish(rid)
        if prefill_rids:
            logger.info("Prefill step %d: %s",
                        cumulative_steps, ", ".join(map(str, prefill_rids)))
        if decode_rids:
            logger.info("Decode  step %d: %s",
                        cumulative_steps, ", ".join(map(str, decode_rids)))
        if not running_requests and queue:
            # queue is always kept sorted by arrival_time
            next_arrival_step = queue[0][1].arrival_time
            if cumulative_steps < next_arrival_step:
                logger.debug(
                    "Fast-forwarding from step %d → %d (idle gap)",
                    cumulative_steps, next_arrival_step,
                )
                cumulative_steps = next_arrival_step
                # also reset the “no-output” watchdog
                consecutive_no_output = 0
            # continue so that the enqueue logic at the top of the loop
            # will pick up the new request(s) immediately.
            continue
        engine.scheduler[0].last_decode_time = sim.last_decode_time
        engine.scheduler[0].step_tokens = step_tokens          
        
        for rid in finished_rids:
            request_metadata.pop(rid)
        tokens_to_release = sim.pop(step_end,solver_time=solver_time)
        

    # 7) After all requests are done, save to CSV
    print("All requests completed. Now saving CSV...")

    import pandas as pd
    df = pd.DataFrame(metrics_data)
    # If your request IDs are like "request_12", you might sort by the numeric part:
    def numeric_part(x):
        # from "request_12" -> 12
        return int(x.split("_")[-1]) if "_" in x else 999999
    df = df.sort_values(by=["request_id"], key=lambda x: x.apply(numeric_part))
    if not csv_path:
        csv_path = "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")

    # 8) Compute overall throughput
    if start_time and end_time and (end_time > start_time):
        print("------Wall-clock stats (non-overlapping)------")
        print(f"Overall runtime  : {overall_wall:.3f} s")
        print(f"Prefill time     : {finished_prefill_tokens} tokens over {prefill_wall:.3f} s "
            f"({finished_prefill_tokens / prefill_wall: .2f} t/s)")
        print(f"Decode  time     : {finished_decode_tokens } tokens over {decode_wall:.3f} s "
            f"({finished_decode_tokens / decode_wall: .2f} t/s)")
        print(f"Preemptions  time     : {engine.scheduler[0].num_cumulative_preemption}")
    else:
        print("No valid start/end time for throughput calculation.")
def main(configs):
    prompt_path = configs.config_file
    num_gpu_blocks_override = None
    trace = Trace.load_from_json(prompt_path)
    for attribute in dir(trace):
        if not attribute.startswith("__") and not callable(getattr(trace, attribute)):
            if "requests" not in attribute:
                print(f"{attribute}: {getattr(trace, attribute)}") 
    if hasattr(trace, "gpu_memory_utilization"): 
        gpu_memory_utilization = trace.gpu_memory_utilization if trace.gpu_memory_utilization else 0.01 # dummy to work around with vllm's _verify_args()
    if gpu_memory_utilization == 0.01:
        num_gpu_blocks_override = trace.num_gpu_blocks_override
    max_model_len = trace.max_model_len if hasattr(trace, "max_model_len") else  trace.num_gpu_blocks_override * BLOCK_SIZE
    assert gpu_memory_utilization != 0.01 or num_gpu_blocks_override is not None, "No gpu_memory_utilization or num_gpu_blocks_override found in prompt_dict"

    prefetch_mode = "none"
    is_monolithic_distn = True 
    prefetch_distance = 0

    if hasattr(configs, "prefetch_mode"):
        prefetch_mode = configs.prefetch_mode
    if prefetch_mode == "distn":
        if hasattr(configs, "is_monolithic_distn"):
            is_monolithic_distn = configs.is_monolithic_distn 
        else: 
            is_monolithic_distn = True 
    elif prefetch_mode == "static":
        if hasattr(configs, "prefetch_distance"):
            prefetch_distance = configs.prefetch_distance 
        else: 
            prefetch_distance = 0 
    
    flattened_cache = configs.flattened_cache if hasattr(configs, "flattened_cache") else False
    merge_prefetch_buffer = configs.merge_prefetch_buffer if hasattr(configs, "merge_prefetch_buffer") else True
    pause_and_resume = configs.pause_and_resume if hasattr(configs, "pause_and_resume") else False
    batch_size = trace.batch_size if hasattr(trace, "batch_size") else BATCH_SIZE
    # prompts = trace.requests if hasattr(trace, "requests") else trace.samples
    static_batching = configs.static_batching if hasattr(configs, "static_batching") else False
    
    print(f"batch_size: {batch_size}")
    print(f"prefetch_mode: {prefetch_mode}")
    print(f"prefetch_distance: {prefetch_distance}")
    print(f"gpu_memory_utilization: {gpu_memory_utilization}")
    print(f"num_gpu_blocks_override: {num_gpu_blocks_override}")
    print(f"merge_prefetch_buffer: {merge_prefetch_buffer}")
    print(f"pause_and_resume: {pause_and_resume}")
    print(f"static_batching: {static_batching}")
    
    if flattened_cache and num_gpu_blocks_override is not None:
        num_gpu_blocks_override *= 32 
    print(f"max_model_len: {max_model_len}")
    args = EngineArgs(
        model=MODEL,
        max_model_len=max_model_len,
        tensor_parallel_size=1,
        pipeline_parallel_size = 1,
        max_num_seqs=batch_size,  # Updated batch size for serving
        max_num_batched_tokens=max_model_len,
        disable_log_stats=True,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        num_gpu_blocks_override=num_gpu_blocks_override,
        preemption_mode="pause",
        is_monolithic_distn=is_monolithic_distn,
        prefetch_mode = prefetch_mode,
        prefetch_distance = prefetch_distance,
        enable_chunked_prefill=False,
        flattened_cache=flattened_cache,
        merge_prefetch_buffer=merge_prefetch_buffer,
        pause_and_resume=pause_and_resume,
        static_batching=static_batching,
        disable_sliding_window=True,
    )
    print(f"Logging to {configs.output_log}")
    import sys 
    sys.stdout = open(configs.output_log, 'w')
    engine = LLMEngine.from_engine_args(args)  
    
    csv_path = configs.output_log.replace(".log", ".csv")

    run_inference_step_mode(engine, trace, csv_path=csv_path,enable_deposit=configs.enable_deposit)

if __name__ == "__main__":
    from vllm.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser(description="distN test.")
    parser.add_argument("--config-file",
                        type=str,
                        default="/home/xinyuema/vllm/samples/large_new_request.json",
                        help="Configurations file.")
    parser.add_argument("--prefetch-mode",
                        type=str,
                        default="none",
                        help="prefetch method: none, static, distn, static_req_wise, selectn, flexgen")
    parser.add_argument("--is-monolithic-distn",
                        type=bool,
                        default=True,
                        help="is monolithic distn")
    parser.add_argument("--prefetch-distance",
                        type=int,
                        default=0,
                        help="prefetch distance")
    parser.add_argument("--output-log",
                        type=str,
                        default="/home/xinyuema/vllm/outputs/default.log",
                        help="output log file")
    parser.add_argument("--flattened-cache",
                        type=bool,
                        default=False,
                        help="whether the kv cache is flattened")
    parser.add_argument("--merge-prefetch-buffer",
                        type=bool,
                        default=False,
                        help="whether the prefetch buffer is merged")
    parser.add_argument("--pause-and-resume",
                        action="store_true",
                        default=False,
                        help="whether to use pause and resume")
    parser.add_argument("--enable-deposit",
                        action="store_true",
                        default=False,
                        help="whether to use use token deposit")
    parser.add_argument("--static-batching",
                        action="store_true",
                        default=False,
                        help="whether to use use static batching instead of continuous batching")
    args = parser.parse_args()    
    print(args)
    # --- Setup Logging ---
    logging.basicConfig(filename=args.output_log, level=logging.INFO, format="%(message)s")
    main(args)
