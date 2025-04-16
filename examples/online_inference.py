import os
import asyncio
import time
import logging
import requests
import threading
from argparse import Namespace
from vllm.entrypoints.api_server import run_server
from vllm.engine.arg_utils import AsyncEngineArgs

# --- Configuration ---
MODEL = "facebook/opt-125m"
PROMPT_DIR = "./prompts"
API_URL = "http://localhost:8000/generate"
INTERVAL = 1.5  # seconds between requests
LOG_FILE = "request_metrics.log"
SLO_THRESHOLD = 0.5  # seconds per token

# --- Setup Logging ---
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(message)s")

# --- Start vLLM Server ---
def start_server():
    args = AsyncEngineArgs(
        model=MODEL,
        port=8000,
        disable_log_requests=True,
    )
    args = Namespace(**vars(args), host="0.0.0.0", log_level="error", root_path=None,
                     ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_cert_reqs=0)
    asyncio.run(run_server(args))

# --- Load Prompts into Memory ---
def load_prompts(prompt_dir):
    prompts = {}
    for fname in sorted(os.listdir(prompt_dir)):
        if fname.endswith(".txt"):
            with open(os.path.join(prompt_dir, fname), "r") as f:
                prompts[fname] = f.read()
    return prompts

# --- Send Request and Track Token Latencies ---
def send_prompt(file_name, prompt):
    payload = {
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 32,
        "stream": True
    }

    token_times = []
    start = time.time()

    with requests.post(API_URL, json=payload, stream=True) as resp:
        for line in resp.iter_lines():
            if line:
                now = time.time()
                token_times.append(now)
    
    if len(token_times) < 2:
        return
    
    per_token_latencies = [j - i for i, j in zip(token_times[:-1], token_times[1:])]
    avg_latency = sum(per_token_latencies) / len(per_token_latencies)
    slo_violation = any(lat > SLO_THRESHOLD for lat in per_token_latencies)

    logging.info(f"{file_name}, tokens={len(per_token_latencies)}, avg_latency={avg_latency:.3f}, violates_slo={slo_violation}")

# --- Main Flow ---
if __name__ == "__main__":
    # Start API server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(5)

    # Load all prompts first
    prompt_dict = load_prompts(PROMPT_DIR)

    # Send one-by-one at fixed interval
    for file_name, content in prompt_dict.items():
        send_prompt(file_name, content)
        time.sleep(INTERVAL)

    print("All requests completed. Metrics stored in:", LOG_FILE)
