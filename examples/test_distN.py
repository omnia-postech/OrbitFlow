import os
import time
import json
from collections import defaultdict

from vllm.engine.llm_engine import LLMEngine
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams
from vllm.logger import init_logger
logger = init_logger("vllm")
import logging

import pandas as pd 
import torch 
# --- Config ---
MODEL = "/home/jongseop/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
PROMPT_DIR = "./prompts"
USE_DEFAULT_SAMPLES = True
BATCH_SIZE = 4  # Serving batch size set to 4
MAX_MODEL_LEN = 13000
SLO_THRESHOLD = 0.5      
CSV_OUTPUT_FILE = "metrics.csv"
Moderately_Uniform_Set = {
    "sample1":  {"prompt": "ModUniform prompt 1",  "max_tokens": 1100, "schedule_at_step": 0},
    "sample2":  {"prompt": "ModUniform prompt 2",  "max_tokens": 1000, "schedule_at_step": 0},
    "sample3":  {"prompt": "ModUniform prompt 3",  "max_tokens": 1100, "schedule_at_step": 0},
    "sample4":  {"prompt": "ModUniform prompt 4",  "max_tokens": 1000, "schedule_at_step": 0},
    "sample5":  {"prompt": "ModUniform prompt 5",  "max_tokens": 1100, "schedule_at_step": 500},
    "sample6":  {"prompt": "ModUniform prompt 6",  "max_tokens": 1000, "schedule_at_step": 700},
    "sample7":  {"prompt": "ModUniform prompt 7",  "max_tokens": 1100, "schedule_at_step": 900},
    "sample8":  {"prompt": "ModUniform prompt 8",  "max_tokens": 1000, "schedule_at_step": 1100},
    "sample9":  {"prompt": "ModUniform prompt 9",  "max_tokens": 1100, "schedule_at_step": 1300},
    "sample10": {"prompt": "ModUniform prompt 10", "max_tokens": 1000, "schedule_at_step": 1500},
    "sample11": {"prompt": "ModUniform prompt 11", "max_tokens": 1100, "schedule_at_step": 1700},
    "sample12": {"prompt": "ModUniform prompt 12", "max_tokens": 1000, "schedule_at_step": 1900},
    "sample13": {"prompt": "ModUniform prompt 13", "max_tokens": 1100, "schedule_at_step": 2100},
    "sample14": {"prompt": "ModUniform prompt 14", "max_tokens": 1000, "schedule_at_step": 2300},
    "sample15": {"prompt": "ModUniform prompt 15", "max_tokens": 1100, "schedule_at_step": 2500},
    "sample16": {"prompt": "ModUniform prompt 16", "max_tokens": 1000, "schedule_at_step": 2700},
    "sample17": {"prompt": "ModUniform prompt 17", "max_tokens": 1100, "schedule_at_step": 2900},
    "sample18": {"prompt": "ModUniform prompt 18", "max_tokens": 1000, "schedule_at_step": 3100},
    "sample19": {"prompt": "ModUniform prompt 19", "max_tokens": 1100, "schedule_at_step": 3300},
    "sample20": {"prompt": "ModUniform prompt 20", "max_tokens": 1000, "schedule_at_step": 3500},
}
test_trace = {
    'sample1':  {'prompt': 754, 'max_tokens': 3228, 'arrive_at_step': 0, 'schedule_at_step': 0, 'wait_time': 0},
    'sample2':  {'prompt': 125, 'max_tokens': 4518, 'arrive_at_step': 0, 'schedule_at_step': 0, 'wait_time': 0},
    'sample3':  {'prompt': 381, 'max_tokens': 3501, 'arrive_at_step': 0, 'schedule_at_step': 0, 'wait_time': 0},
    'sample4':  {'prompt': 328, 'max_tokens': 3285, 'arrive_at_step': 0, 'schedule_at_step': 0, 'wait_time': 0},
    'sample5':  {'prompt': 854, 'max_tokens': 3209, 'arrive_at_step': 3229, 'schedule_at_step': 3502, 'wait_time': 273},
    'sample6':  {'prompt': 792, 'max_tokens': 4516, 'arrive_at_step': 1651, 'schedule_at_step': 3229, 'wait_time': 1578},
    'sample7':  {'prompt': 658, 'max_tokens': 3178, 'arrive_at_step': 2070, 'schedule_at_step': 3286, 'wait_time': 1216},
    'sample8':  {'prompt': 704, 'max_tokens': 3864, 'arrive_at_step': 4519, 'schedule_at_step': 4519, 'wait_time': 0},
    'sample9':  {'prompt': 132, 'max_tokens': 3061, 'arrive_at_step': 6439, 'schedule_at_step': 6465, 'wait_time': 26},
    'sample10': {'prompt': 195, 'max_tokens': 3447, 'arrive_at_step': 6681, 'schedule_at_step': 6712, 'wait_time': 31},
    'sample11': {'prompt': 338, 'max_tokens': 4034, 'arrive_at_step': 7803, 'schedule_at_step': 7803, 'wait_time': 0},
    'sample12': {'prompt': 716, 'max_tokens': 3054, 'arrive_at_step': 8384, 'schedule_at_step': 8384, 'wait_time': 0},
    'sample13': {'prompt': 674, 'max_tokens': 3407, 'arrive_at_step': 9501, 'schedule_at_step': 9527, 'wait_time': 26},
    'sample14': {'prompt': 833, 'max_tokens': 4330, 'arrive_at_step': 10129, 'schedule_at_step': 11439, 'wait_time': 1310},
    'sample15': {'prompt': 818, 'max_tokens': 4116, 'arrive_at_step': 9522, 'schedule_at_step': 10160, 'wait_time': 638},
    'sample16': {'prompt': 529, 'max_tokens': 3451, 'arrive_at_step': 11838, 'schedule_at_step': 11838, 'wait_time': 0},
    'sample17': {'prompt': 559, 'max_tokens': 4206, 'arrive_at_step': 12909, 'schedule_at_step': 12935, 'wait_time': 26},
    'sample18': {'prompt': 384, 'max_tokens': 4657, 'arrive_at_step': 13602, 'schedule_at_step': 14277, 'wait_time': 675},
    'sample19': {'prompt': 990, 'max_tokens': 3013, 'arrive_at_step': 14512, 'schedule_at_step': 15290, 'wait_time': 778},
    'sample20': {'prompt': 877, 'max_tokens': 4650, 'arrive_at_step': 15556, 'schedule_at_step': 15770, 'wait_time': 214},
    'sample21': {'prompt': 263, 'max_tokens': 4429, 'arrive_at_step': 17116, 'schedule_at_step': 17142, 'wait_time': 26},
    'sample22': {'prompt': 532, 'max_tokens': 3696, 'arrive_at_step': 18304, 'schedule_at_step': 18304, 'wait_time': 0},
    'sample23': {'prompt': 384, 'max_tokens': 3318, 'arrive_at_step': 19118, 'schedule_at_step': 19118, 'wait_time': 0},
    'sample24': {'prompt': 320, 'max_tokens': 4960, 'arrive_at_step': 20207, 'schedule_at_step': 21572, 'wait_time': 1365},
    'sample25': {'prompt': 881, 'max_tokens': 3689, 'arrive_at_step': 19786, 'schedule_at_step': 20421, 'wait_time': 635},
    'sample26': {'prompt': 204, 'max_tokens': 3189, 'arrive_at_step': 22001, 'schedule_at_step': 22001, 'wait_time': 0},
    'sample27': {'prompt': 489, 'max_tokens': 3198, 'arrive_at_step': 22437, 'schedule_at_step': 22437, 'wait_time': 0},
    'sample28': {'prompt': 467, 'max_tokens': 4735, 'arrive_at_step': 25168, 'schedule_at_step': 25168, 'wait_time': 0},
    'sample29': {'prompt': 452, 'max_tokens': 4236, 'arrive_at_step': 25191, 'schedule_at_step': 25191, 'wait_time': 0},
    'sample30': {'prompt': 370, 'max_tokens': 4652, 'arrive_at_step': 25236, 'schedule_at_step': 25636, 'wait_time': 400},
    'sample31': {'prompt': 144, 'max_tokens': 4494, 'arrive_at_step': 25636, 'schedule_at_step': 26533, 'wait_time': 897},
    'sample32': {'prompt': 570, 'max_tokens': 4098, 'arrive_at_step': 29428, 'schedule_at_step': 29428, 'wait_time': 0},
    'sample33': {'prompt': 227, 'max_tokens': 4992, 'arrive_at_step': 29889, 'schedule_at_step': 29904, 'wait_time': 15},
    'sample34': {'prompt': 487, 'max_tokens': 3161, 'arrive_at_step': 29904, 'schedule_at_step': 30289, 'wait_time': 385},
    'sample35': {'prompt': 665, 'max_tokens': 3600, 'arrive_at_step': 30131, 'schedule_at_step': 31028, 'wait_time': 897},
    'sample36': {'prompt': 949, 'max_tokens': 4287, 'arrive_at_step': 33066, 'schedule_at_step': 33451, 'wait_time': 385},
    'sample37': {'prompt': 733, 'max_tokens': 4813, 'arrive_at_step': 33527, 'schedule_at_step': 33527, 'wait_time': 0},
    'sample38': {'prompt': 982, 'max_tokens': 3740, 'arrive_at_step': 33732, 'schedule_at_step': 34629, 'wait_time': 897},
    'sample39': {'prompt': 691, 'max_tokens': 3393, 'arrive_at_step': 34116, 'schedule_at_step': 34897, 'wait_time': 781},
    'sample40': {'prompt': 821, 'max_tokens': 3142, 'arrive_at_step': 37354, 'schedule_at_step': 37739, 'wait_time': 385}
}
DEFAULT_PROMPTS = test_trace
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

def run_inference_step_mode(engine, prompt_dict, csv_path=None):
    # Local counters and timers.
    start_time = None        # Earliest scheduled time among enqueued requests.
    end_time = None          # Latest finished time among completed requests.
    cumulative_tokens = 0    # Total tokens produced by the engine (for scheduling new requests).
    cumulative_steps = 0      # Total number of steps taken by the engine.
    finished_tokens = 0      # Sum of tokens lengths for finished requests (for throughput).
    finished_decode_tokens = 0  
    finished_prefill_tokens = 0  
    # Prepare queues and metrics.
    queue = list(prompt_dict.items())
    request_metadata = {}
    request_output = defaultdict(list)
    metrics_data = []

    received_requests = []
    running_requests = set()
    while queue or request_metadata:
        # Check pending samples against the cumulative token count.
        ready = [(req_id, sample) for req_id, sample in queue if sample["arrive_at_step"] <= cumulative_steps]
        if ready:
            # Enqueue all samples with the smallest scheduled threshold.
            min_threshold = min(sample["arrive_at_step"] for _, sample in ready)
            group = [(req_id, sample) for req_id, sample in ready if sample["arrive_at_step"] == min_threshold]
            # Remove those samples from the pending queue.
            queue = [(req_id, sample) for req_id, sample in queue if sample["arrive_at_step"] > min_threshold]

            # Enqueue requests; we don't record time here, because vLLM now tracks arrival time internally.
            enqueue_batch(engine, group, request_metadata)

        # Call one generation step.
        step_outputs = engine.step()

        # add stall time if new requests cause running ones to pause 
        rids = [output.request_id for output in step_outputs]
        if len(rids)>0 and rids[0] not in received_requests: # continuous batching 
            for req in running_requests:
                stall_time =  output.metrics.first_token_time - output.metrics.first_scheduled_time
                request_metadata[req]["stall_durations"].append(stall_time)
                request_metadata[req]["stall_duration"] += stall_time
                request_metadata[req]["stall_times"].append(output.metrics.first_scheduled_time - start_time)

        # Process each token output.
        for output in step_outputs:
            rid = output.request_id
            if rid not in received_requests:
                # prefill 
                received_requests.append(rid)
                finished_tokens += len(output.prompt_token_ids) 
                finished_prefill_tokens += len(output.prompt_token_ids)
                running_requests.add(rid)
            else:
                now = time.time()
                request_metadata[rid]["token_timestamps"].append(now)
                request_metadata[rid]["decode_length"] += 1
                request_output[rid].append(output)

            # If this request is finished, collect the metrics.
            if output.finished:
                running_requests.remove(rid)
                # Use built-in metrics from vLLM.
                m = output.metrics

                # If you want per‐token latencies, compute them here.
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
                finished_tokens += decode_length
                finished_decode_tokens += decode_length

                # If you keep track of global start/end times, you can do:
                if start_time is None or (m.arrival_time and m.arrival_time < start_time):
                    start_time = m.arrival_time
                if end_time is None or (m.finished_time and m.finished_time > end_time):
                    end_time = m.finished_time
                    
                # Construct the row using RequestMetrics fields.
                row = {
                    "request_id": rid,
                    "arrival_time": m.arrival_time - start_time,
                    "first_scheduled_time": m.first_scheduled_time - start_time,
                    "finished_time": m.finished_time - start_time,
                    "stall_times": json.dumps(request_metadata[rid]["stall_times"]),
                    "wait_duration": m.time_in_queue,
                    "time_to_first_token": m.first_token_time-m.first_scheduled_time,
                    "scheduler_overehad": m.scheduler_time,
                    # "model_execute_duration": m.model_execute_time,
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

                # Remove finished requests.
                request_metadata.pop(rid)
                logger.info(f"Finished request {rid} with {decode_length} decode tokens")
        # Update the cumulative token count using the number of tokens produced in this step.
        cumulative_tokens += len(step_outputs)
        cumulative_steps += 1

        # This small sleep can be retained if you want a less tight loop
        # (it does not affect vLLM’s internal metrics).
        time.sleep(0.01)
    print("All requests completed. Now saving CSV...")

    # Dump per-request metrics to CSV.
    df = pd.DataFrame(metrics_data)
    df = df.sort_values(by=["request_id"],key=lambda x: x.str.split("sample").str[1].astype(int))
    if csv_path is None:
        csv_path = CSV_OUTPUT_FILE
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")

    # Compute overall throughput based on finished tokens.
    if start_time and end_time and (end_time > start_time):
        total_runtime = end_time - start_time
        throughput = finished_tokens / total_runtime
        print("------Overall------")
        print(f"Finished tokens: {finished_tokens} over {total_runtime:.3f} s")
        print(f"System throughput: {throughput:.3f} tokens/s")
        logging.info(f"System throughput: {throughput:.3f} tokens/s")
        
        # collect sum of time_to_first_token from df 
        time_to_first_token_sum = df["time_to_first_token"].sum()
        print("------Prefill------")
        print(f"Finished prefill tokens: {finished_prefill_tokens} over {time_to_first_token_sum:.3f} s")
        print(f"Prefill throughput: {(finished_prefill_tokens/time_to_first_token_sum):.3f} tokens/s")
        logging.info(f"prefill throughput: {finished_prefill_tokens / time_to_first_token_sum:.3f} tokens/s")
        
        # collect sum of timer_per_output_token from df
        time_per_output_token_sum = (df["time_per_output_token"]*df['decode_length']).sum()
        print("------Decode------")
        print(f"Finished decode tokens: {finished_decode_tokens} over {time_per_output_token_sum:.3f} s")
        print(f"Decode throughput: {(finished_decode_tokens/time_per_output_token_sum):.3f} tokens/s")
        logging.info(f"decode throughput: {finished_decode_tokens / time_per_output_token_sum:.3f} tokens/s")
        
    else:
        throughput = None

def main(configs):
    prompt_path = configs.config
    gpu_memory_utilization = 0.5
    num_gpu_blocks_override = None
    prompt_dict = load_prompts(prompt_path)
    
    if "gpu_memory_utilization" in prompt_dict:
        gpu_memory_utilization = prompt_dict["gpu_memory_utilization"]
    elif "num_gpu_blocks_override" in prompt_dict:
        num_gpu_blocks_override = prompt_dict["num_gpu_blocks_override"]
    else: 
        raise ValueError("No gpu_memory_utilization or num_gpu_blocks_override found in prompt_dict") 
    
    
    
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
    max_model_len = prompt_dict.get("max_model_len", MAX_MODEL_LEN) 
    batch_size = prompt_dict.get("batch_size", BATCH_SIZE)
    prompts = prompt_dict.get("samples", Moderately_Uniform_Set)
    args = EngineArgs(
        model=MODEL,
        max_model_len=max_model_len,
        tensor_parallel_size=1,
        max_num_seqs=batch_size,  # Updated batch size for serving
        max_num_batched_tokens=max_model_len,
        disable_log_stats=True,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        num_gpu_blocks_override=num_gpu_blocks_override,
        is_monolithic_distn=is_monolithic_distn, 
        prefetch_mode = prefetch_mode,
        prefetch_distance = prefetch_distance,
        # No prefetch, (N=1,static), (N=dynamic,mono), (N=dynamic,dyn), the last two version, N only decreases 
        # multi-request version (might decrease, or increase)
        # num_gpu_blocks_override: Optional[int] = None
    )
    print(f"Logging to {configs.output_log}")
    import sys 
    sys.stdout = open(configs.output_log, 'w')
    engine = LLMEngine.from_engine_args(args)
    
    run_inference_step_mode(engine, prompts, csv_path=configs.output_log.replace(".log", ".csv"))

if __name__ == "__main__":
    from vllm.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser(description="distN test.")
    parser.add_argument("--config",
                        type=str,
                        default="/home/xinyuema/vllm/samples/large_new_request.json",
                        help="Configurations file.")
    parser.add_argument("--prefetch-mode",
                        type=str,
                        default="none",
                        help="prefetch method: none, static, distn")
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
    args = parser.parse_args()    
    # --- Setup Logging ---
    logging.basicConfig(filename=args.output_log, level=logging.INFO, format="%(message)s")
    main(args)
