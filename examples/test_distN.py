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
    def __init__(self, v_tps: float):        
        # v_tps: tokens per second
        self.v = v_tps
        self.last_time = {}
        # self.interval = 1.0 / v_tps       # 토큰 하나가 배출되어야 하는 간격(초)
        # self.next_deadline = self.interval
        self.deposit = defaultdict(int)   # request_id → 누적된 토큰 수

        logger.info(f"[Simulator __init__] v_tps={self.v:.3f} tokens/sec")

    def on_token(self, rid: str, step_time: float):
        before = self.deposit[rid]
        self.deposit[rid] += 1
        after = self.deposit[rid]
        
        if rid not in self.last_time: 
            self.last_time[rid] = step_time
            logger.debug(f"[on_token] first token for {rid}, setting last_time[{rid}] = {step_time:.6f}")

        logger.debug(f"[on_token] rid={rid} @ {step_time:.6f}: deposit {before} -> {after}")


    def pop(self, step_time: float):
        deposits_snapshot = {k: int(v) for k, v in self.deposit.items()}
        pops: list[tuple[str, int]] = []
        rid_log: list[str] = []

        for rid, dep in list(self.deposit.items()):
            last   = self.last_time.get(rid, step_time)
            dt     = step_time - last             
            n      = int(dt * self.v)             
            if n <= 0:
                continue

            # release n tokens, but not more than deposit
            if dep < n:
                rid_log.append(f"{rid}(SLO {dep}/{n})")
            else:
                rid_log.append(f"{rid}({n})")
            to_rel = min(n, dep)
            # deposit 차감
            self.deposit[rid] -= to_rel
            pops.append((rid, to_rel))
            
            # last_time[rid] 을 방출된 토큰 시간만큼 앞으로 이동
            # 즉, to_rel tokens / v_tps 만큼 경과시킨 것처럼
            advance = to_rel / self.v
            self.last_time[rid] = last + advance

        # single-line summary
        if rid_log:
            logger.info(
                "[pop] deposits=%s | released: %s",
                deposits_snapshot,
                " | ".join(rid_log),
            )
        return pops
    
    def finish(self, rid: str):
        """요청 완료 시 호출: 남은 deposit 제거, tracking 변수 cleanup"""
        if rid in self.deposit:
            logger.info(f"[finish] clearing deposit for {rid}: was {self.deposit[rid]}")
            del self.deposit[rid]
        if rid in self.last_time:
            logger.info(f"[finish] removing last_time entry for {rid}")
            del self.last_time[rid]

    def stats(self) -> dict:
        """현재 deposit 맵(rid→남은 토큰 수) 반환."""
        stats = dict(self.deposit)
        logger.info(f"[stats] deposit map: {stats}")
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
            ignore_eos=True
        )
        engine.add_request(req_id, prompt_obj, sampling_params)

def run_inference_step_mode(engine, trace_obj, csv_path=None):
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
                "decode_length": 0,
                "stall_times": [],
                "stall_durations": [],
                "stall_duration": 0,
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
    sim = DelaySimulator(v_tps=1/SLO_THRESHOLD)

    # The main simulation loop
    while queue or request_metadata:
        # 4) Find all requests that have arrival_time <= cumulative_steps
        #    -> these are ready to enqueue
        ready = [(req_id, req_obj) for (req_id, req_obj) in queue
                 if req_obj.arrival_time <= cumulative_steps]
        if ready:
            # We enqueue *all* that are <= cumulative_steps
            enqueue_batch(engine, ready, request_metadata)
            # Remove them from the queue
            queue = [(req_id, req_obj) for (req_id, req_obj) in queue
                     if req_obj.arrival_time > cumulative_steps]

        # TODO(Heelim): need to send metadata to scheduler to make decision of which request and when to pause and resume
        deposit_map = sim.stats()  
        engine.scheduler[0].deposit_map = deposit_map
        # NOTE(HONG): use below for pipelining parallelism
        # for sched in engine.scheduler:
        #     sched.deposit_map = deposit_map

        # 5) Perform one engine step
        step_outputs = engine.step()
        # If no outputs come back, we still increment steps
        cumulative_steps += 1
        
        # NOTE(HONG): neet to use step_time to unify the time for all events at step-level
        step_time = time.time()

        # 6) Process each output
        # gather rids so we can log them in one line
        prefill_rids: list[str] = []
        decode_rids:  list[str] = []

        for output in step_outputs:
            rid = output.request_id
            logger.debug("step %d  rid=%s", cumulative_steps, rid)
            # Prefill
            if rid not in received_requests:
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
                # Another token
                decode_rids.append(rid)
                # now = time.time()

                sim.on_token(rid, step_time)

                # NOTE(HONG): not useing output.metrics.last_token_time since we are using step_time
                request_metadata[rid]["token_timestamps"].append(step_time)
                request_metadata[rid]["decode_length"] += 1
                finished_tokens += 1
                finished_decode_tokens += 1
                request_output[rid].append(output)

            # If the request is finished:
            if output.finished:
                logger.info(f"Finished request {rid} at step {cumulative_steps}, finish_reason={output.outputs[0].finish_reason}")
                logger.info(f"{rid} prompt: {len(output.prompt_token_ids)} tokens; {output.prompt_token_ids[:20]}")
                logger.info(f"{rid} output: {len(output.prompt_token_ids)} {output.outputs[0].token_ids[:20]}")
                logger.info(f"{rid} text: {output.outputs[0].text[:20]}")
                running_requests.remove(rid)
                m = output.metrics

                # Per-token latencies
                token_ts = request_metadata[rid]["token_timestamps"]
                if len(token_ts) < 2:
                    per_token_latencies = []
                    avg_token_latency = 0.0
                else:
                    per_token_latencies = [
                        j - i for i, j in zip(token_ts[:-1], token_ts[1:])
                    ]
                    avg_token_latency = sum(per_token_latencies) / len(per_token_latencies)

                decode_length = request_metadata[rid]["decode_length"]
                # finished_tokens += decode_length
                # finished_decode_tokens += decode_length

                # Track global start/end times
                if start_time is None or (m.arrival_time and m.arrival_time < start_time):
                    start_time = m.arrival_time
                if end_time is None or (m.finished_time and m.finished_time > end_time):
                    end_time = m.finished_time

                # Collect row of metrics
                row = {
                    "request_id": rid,
                    "arrival_time": m.arrival_time - (start_time or 0),
                    "first_scheduled_time": m.first_scheduled_time - (start_time or 0),
                    "finished_time": m.finished_time - (start_time or 0),
                    "stall_times": json.dumps(request_metadata[rid]["stall_times"]),
                    "wait_duration": m.time_in_queue,
                    "time_to_first_token": m.first_token_time - m.first_scheduled_time,
                    "scheduler_overehad": m.scheduler_time,
                    "stall_duration": request_metadata[rid]["stall_duration"],
                    "decode_length": decode_length,
                    "end_to_end_time": (
                        m.finished_time - m.arrival_time
                        if m.finished_time and m.arrival_time
                        else None
                    ),
                    "time_per_output_token": avg_token_latency,
                    "time_between_tokens": json.dumps(per_token_latencies),
                    "stall_durations": json.dumps(request_metadata[rid]["stall_durations"]),
                }
                metrics_data.append(row)

                # done with this request
                request_metadata.pop(rid)
                logger.info(f"Finished request {rid} with {decode_length} decode tokens")

                sim.finish(rid)
        if prefill_rids:
            logger.info("Prefill step %d: %s",
                        cumulative_steps, ", ".join(map(str, prefill_rids)))
        if decode_rids:
            logger.info("Decode  step %d: %s",
                        cumulative_steps, ", ".join(map(str, decode_rids)))
        tokens_to_release = sim.pop(step_time)
        
        # We'll sleep a bit
        time.sleep(0.01)

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
        total_runtime = end_time - start_time
        throughput = finished_tokens / total_runtime
        print("------Overall------")
        print(f"Finished tokens: {finished_tokens} over {total_runtime:.3f} s")
        print(f"System throughput: {throughput:.3f} tokens/s")
        logger.info(f"System throughput: {throughput:.3f} tokens/s")

        # Summation for time_to_first_token
        time_to_first_token_sum = df["time_to_first_token"].sum()
        print("------Prefill------")
        print(f"Finished prefill tokens: {finished_prefill_tokens} over {time_to_first_token_sum:.3f} s")
        prefill_throughput = finished_prefill_tokens / time_to_first_token_sum if time_to_first_token_sum > 0 else 0
        print(f"Prefill throughput: {prefill_throughput:.3f} tokens/s")
        logger.info(f"prefill throughput: {prefill_throughput:.3f} tokens/s")

        # Summation for decode latencies
        time_per_output_token_sum = (df["time_per_output_token"] * df["decode_length"]).sum()
        print("------Decode------")
        print(f"Finished decode tokens: {finished_decode_tokens} over {time_per_output_token_sum:.3f} s")
        decode_throughput = (finished_decode_tokens / time_per_output_token_sum) if time_per_output_token_sum > 0 else 0
        print(f"Decode throughput: {decode_throughput:.3f} tokens/s")
        logger.info(f"decode throughput: {decode_throughput:.3f} tokens/s")
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
        max_model_len = min(num_gpu_blocks_override * BLOCK_SIZE, 128000  )
    else: 
        max_model_len = trace.max_model_len if hasattr(trace, "max_model_len") else MAX_MODEL_LEN
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
    
    print(f"max_model_len: {max_model_len}")
    print(f"batch_size: {batch_size}")
    print(f"prefetch_mode: {prefetch_mode}")
    print(f"prefetch_distance: {prefetch_distance}")
    print(f"gpu_memory_utilization: {gpu_memory_utilization}")
    print(f"num_gpu_blocks_override: {num_gpu_blocks_override}")
    print(f"merge_prefetch_buffer: {merge_prefetch_buffer}")
    print(f"pause_and_resume: {pause_and_resume}")
    
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
        num_gpu_blocks_override=num_gpu_blocks_override*32 if flattened_cache else num_gpu_blocks_override,
        preemption_mode="pause",
        is_monolithic_distn=is_monolithic_distn,
        prefetch_mode = prefetch_mode,
        prefetch_distance = prefetch_distance,
        enable_chunked_prefill=False,
        flattened_cache=flattened_cache,
        merge_prefetch_buffer=merge_prefetch_buffer,
        pause_and_resume=pause_and_resume,
        # No prefetch, (N=1,static), (N=dynamic,mono), (N=dynamic,dyn), the last two version, N only decreases 
        # multi-request version (might decrease, or increase)
        # num_gpu_blocks_override: Optional[int] = None
    )
    print(f"Logging to {configs.output_log}")
    import sys 
    sys.stdout = open(configs.output_log, 'w')
    engine = LLMEngine.from_engine_args(args)
    
    run_inference_step_mode(engine, trace, csv_path=configs.output_log.replace(".log", ".csv"))

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
    args = parser.parse_args()    
    print(args)
    # --- Setup Logging ---
    logging.basicConfig(filename=args.output_log, level=logging.INFO, format="%(message)s")
    main(args)
