import json
import random
import math
import heapq
import re
import numpy as np

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Callable, Any
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

PROFILED_A = 1.0017431830666432e-06
PROFILED_B = 0.049519613282613506

class ArrivalPattern(ABC):
    """
    Abstract base class for arrival patterns. Each subclass is responsible for
    implementing a method to generate arrival times for a specified number of requests.
    """

    @abstractmethod
    def generate_arrival_times(self, num_requests: int) -> List[float|int]:
        """
        Must return a list of arrival times (floats or ints) for `num_requests` requests.
        """
        raise NotImplementedError

# -------------------------------------------------------
# Continuous Uniform
# -------------------------------------------------------

class ContinuousUniformArrival(ArrivalPattern):
    """
    Requests arrive uniformly in continuous time, within [0, total_time].
    """

    def __init__(self, total_time: float):
        """
        total_time: The time window in which all requests will arrive [0, total_time].
        """
        self.total_time = total_time

    def generate_arrival_times(self, num_requests: int) -> List[float]:
        """
        Generate num_requests times uniformly in [0, total_time].
        """
        return [random.uniform(0, self.total_time) for _ in range(num_requests)]

    def __str__(self):
        return f"ContinuousUniformArrival(total_time={self.total_time})"


# -------------------------------------------------------
# Discrete Uniform
# -------------------------------------------------------

class DiscreteUniformArrival(ArrivalPattern):
    """
    Requests arrive at discrete time steps, each chosen uniformly in [0, max_step].
    So arrival times are integers in [0, max_step].
    """

    def __init__(self, max_step: int):
        """
        max_step: The largest integer time step at which a request can arrive.
        """
        self.max_step = max_step

    def generate_arrival_times(self, num_requests: int) -> List[int]:
        """
        Generate num_requests integer times in [0, max_step].
        """
        return [random.randint(0, self.max_step) for _ in range(num_requests)]

    def __str__(self):
        return f"DiscreteUniformArrival(max_step={self.max_step})"


# -------------------------------------------------------
# Continuous Periodic
# -------------------------------------------------------

class ContinuousPeriodicArrival(ArrivalPattern):
    """
    Continuous periodic: each arrival is exactly interval apart.
    E.g., if interval=2.0, arrivals at times 0, 2, 4, 6, ...
    """

    def __init__(self, interval: float):
        self.interval = interval

    def generate_arrival_times(self, num_requests: int) -> List[float]:
        return [i * self.interval for i in range(num_requests)]

    def __str__(self):
        return f"ContinuousPeriodicArrival(interval={self.interval})"


# -------------------------------------------------------
# Discrete Periodic
# -------------------------------------------------------

class DiscretePeriodicArrival(ArrivalPattern):
    """
    Discrete periodic: arrivals at integer multiples of an interval, but
    we store them as integers. For example, if interval=2, the times are
    [0,2,4,6], etc., all stored as int.
    """

    def __init__(self, interval: int):
        """
        interval: The fixed integer step between consecutive arrivals.
        """
        self.interval = interval

    def generate_arrival_times(self, num_requests: int) -> List[int]:
        return [i * self.interval for i in range(num_requests)]

    def __str__(self):
        return f"DiscretePeriodicArrival(interval={self.interval})"


# -------------------------------------------------------
# Continuous Bimodal
# -------------------------------------------------------

class ContinuousBimodalArrival(ArrivalPattern):
    """
    A bimodal continuous-time arrival process: each inter-arrival time is drawn
    from Exp(rate1) with probability p, or Exp(rate2) with probability (1-p).
    """

    def __init__(self, rate1: float, rate2: float, p: float):
        self.rate1 = rate1
        self.rate2 = rate2
        self.p = p

    def generate_arrival_times(self, num_requests: int) -> List[float]:
        arrivals = []
        current_time = 0.0
        for _ in range(num_requests):
            if random.random() < self.p:
                interarrival = random.expovariate(self.rate1)
            else:
                interarrival = random.expovariate(self.rate2)
            current_time += interarrival
            arrivals.append(current_time)
        return arrivals

    def __str__(self):
        return (f"ContinuousBimodalArrival("
                f"rate1={self.rate1}, rate2={self.rate2}, p={self.p})")


# -------------------------------------------------------
# Discrete Bimodal
# -------------------------------------------------------

class DiscreteBimodalArrival(ArrivalPattern):
    """
    A discrete-time bimodal process: at each integer step t:
     - with probability p, we draw K ~ Poisson(lambda1)
     - with probability (1-p), we draw K ~ Poisson(lambda2)
    Then we place K arrivals at time t.
    """

    def __init__(self, lambda1: float, lambda2: float, p: float, max_steps: int = 100000):
        """
        lambda1, lambda2: The two Poisson rates
        p: fraction of steps that use lambda1 vs. lambda2
        max_steps: safety limit for the # of steps
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.p = p
        self.max_steps = max_steps

    def generate_arrival_times(self, num_requests: int) -> List[int]:
        arrivals = []
        total_requests = 0
        t = 0
        while total_requests < num_requests and t < self.max_steps:
            # decide which distribution to use this step
            if random.random() < self.p:
                k = self._sample_poisson(self.lambda1)
            else:
                k = self._sample_poisson(self.lambda2)

            for _ in range(k):
                arrivals.append(t)
                total_requests += 1
                if total_requests >= num_requests:
                    break
            t += 1
        return arrivals

    @staticmethod
    def _sample_poisson(lmbd: float) -> int:
        """Knuth's algorithm for Poisson(lmbd)."""
        L = math.exp(-lmbd)
        p = 1.0
        k = 0
        while p > L:
            k += 1
            p *= random.random()
        return k - 1

    def __str__(self):
        return (f"DiscreteBimodalArrival("
                f"lambda1={self.lambda1}, lambda2={self.lambda2}, "
                f"p={self.p}, max_steps={self.max_steps})")


# -------------------------------------------------------
# Poisson Bursty
# -------------------------------------------------------
class PoissonBurstyArrivalPattern(ArrivalPattern):
    """
    Generate requests in bursts:
    - Each burst has size ~ Poisson(lambda_burst)
    - Each burst is separated by gap ~ Poisson(lambda_gap)
    """

    def __init__(self, lambda_burst: float, lambda_gap: float = 0):
        self.lambda_burst = lambda_burst
        self.lambda_gap = lambda_gap

    @staticmethod
    def _sample_poisson(lmbd: float) -> int:
        """Knuth's algorithm for Poisson sampling."""
        L = math.exp(-lmbd)
        p = 1.0
        k = 0
        while p > L:
            k += 1
            p *= random.random()
        return k - 1

    def generate_arrival_times(self, num_requests: int) -> List[int]:
        arrivals = []
        current_time = 0
        total_requests = 0
        t = 0

        while total_requests < num_requests:
            burst_size = min(
                self._sample_poisson(self.lambda_burst),
                num_requests - total_requests
            )

            for i in range(burst_size):
                offset = random.randint(int(self.lambda_burst * 0.5), self.lambda_burst)
                arrivals.append(current_time + offset)

            total_requests += burst_size
            # 다음 burst까지의 간격
            gap = int(random.uniform(self.lambda_gap * 0.8, self.lambda_gap))

            current_time += gap
            t += 1

        return sorted(arrivals)

    def __str__(self):
        return (f"PoissonBurstyArrivalPattern("
                f"lambda_burst={self.lambda_burst}, lambda_gap={self.lambda_gap}")


# -------------------------------------------------------
# Continuous Poisson
# -------------------------------------------------------

class ContinuousPoissonArrival(ArrivalPattern):
    """
    Represents a continuous-time Poisson arrival process, sampling inter-arrival
    times from an exponential distribution with parameter 'rate'.
    """

    def __init__(self, rate: float):
        self.rate = rate

    def generate_arrival_times(self, num_requests: int) -> List[float]:
        arrivals = []
        current_time = 0.0
        for _ in range(num_requests):
            interarrival = random.expovariate(self.rate)
            current_time += interarrival
            arrivals.append(current_time)
        return arrivals

    def __str__(self):
        return f"ContinuousPoissonArrival(rate={self.rate})"


# -------------------------------------------------------
# Discrete Poisson
# -------------------------------------------------------

class DiscretePoissonArrival(ArrivalPattern):
    """
    A discrete-time Poisson model where time is considered in discrete steps
    (0, 1, 2, ...). At each integer step t:
      - We draw how many arrivals occur using Poisson(lambda_per_step).
      - All arrivals happening at step t have arrival_time = t.
    """

    def __init__(self, lambda_per_step: float, max_steps: int = 100000):
        self.lambda_per_step = lambda_per_step
        self.max_steps = max_steps

    def generate_arrival_times(self, num_requests: int) -> List[int]:
        arrivals = []
        total_requests = 0
        t = 0
        while total_requests < num_requests and t < self.max_steps:
            k = self._sample_poisson(self.lambda_per_step)
            for _ in range(k):
                arrivals.append(t)
                total_requests += 1
                if total_requests >= num_requests:
                    break
            t += 1
        return arrivals

    @staticmethod
    def _sample_poisson(lmbd: float) -> int:
        L = math.exp(-lmbd)
        p = 1.0
        k = 0
        while p > L:
            k += 1
            p *= random.random()
        return k - 1

    def __str__(self):
        return (f"DiscretePoissonArrival("
                f"lambda_per_step={self.lambda_per_step}, "
                f"max_steps={self.max_steps})")

@dataclass
class Request:
    """
    Represents a single request instance in the generated trace.
    
    Attributes:
        category (RequestType): The category or 'type' of this request.
        input_length (int): Number of tokens in the request input.
        output_length (int): Number of tokens in the request output.
        arrival_time (float|int, optional): The time when this request arrives.
        token_ids (List[int]): A list of token IDs representing the request.
        sched_time (float|int, optional): The time when this request starts being processed.
        wait_time (float|int, optional): How long this request waited from arrival until processing began.
    """
    category: str
    input_length: int
    output_length: int
    arrival_time: Optional[float|int]
    token_ids: List[int]
    sched_time: Optional[float|int] = None
    wait_time: Optional[float|int] = None
    slo: Optional[float|int] = None
    def __repr__(self):
        """
        Custom string representation, including scheduling details if they're available.
        """
        if sched_time := getattr(self, 'sched_time', None):
            if self.slo is not None:
                return (f"Request(category={self.category}, prompt={self.input_length}, "
                    f"max_tokens={self.output_length}, slo={self.slo}, arrive_time={self.arrival_time}, "
                    f"sched_time={sched_time}, wait_time={self.wait_time})")
            else:
                return (f"Request(category={self.category}, prompt={self.input_length}, "
                    f"max_tokens={self.output_length}, arrive_time={self.arrival_time}, "
                    f"sched_time={sched_time}, wait_time={self.wait_time})")
        else:
            return (f"Request(category={self.category}, prompt={self.input_length}, "
                    f"max_tokens={self.output_length}, t={self.arrival_time})")

def compress_idle_steps(requests):
        """
        For discrete-time arrival traces, this function compresses any idle periods between
        consecutive groups of requests so that each new batch starts shortly after the previous
        batch finishes. This helps avoid large time gaps when no requests are incoming.
        """
        requests.sort(key=lambda r: r.arrival_time)
        compressed = []
        batch_map = defaultdict(list)

        for req in requests:
            batch_map[req.arrival_time].append(req)

        unique_arrivals = sorted(batch_map.keys())
        last_finish = 0.0

        for arr_time in unique_arrivals:
            batch = batch_map[arr_time]

            max_out = max(r.output_length for r in batch)
            batch_run_time = 1 + max_out

            if arr_time > last_finish:
                new_arr_time = last_finish + 1
                shift = arr_time - new_arr_time
                for r in batch:
                    r.arrival_time -= shift
                arr_time = new_arr_time

            finish_time = arr_time + batch_run_time
            last_finish = finish_time
            compressed.extend(batch)

        compressed.sort(key=lambda r: r.arrival_time)
        return compressed

def _arrival_pattern_from_dict(d: dict) -> str:
    """
    Convert a stored arrival_pattern dict back into a single string, e.g.:
      {
        "type": "DiscretePoissonArrival",
        "params": { "lambda_per_step": 0.01, "max_steps": 4000 }
      }
    -> "DiscretePoissonArrival(lambda_per_step=0.01, max_steps=4000)"

    If no params found, just return d["type"].
    """
    pattern_type = d.get("type", "UnknownPattern")
    params = d.get("params", None)
    if not params:
        # no params -> just return the type
        return pattern_type

    # build "key=val" in some stable order
    param_strs = []
    for k, v in params.items():
        if isinstance(v, (int, float)):
            param_strs.append(f"{k}={v}")
        else:
            param_strs.append(f"{k}={repr(v)}")

    inside = ", ".join(param_strs)
    return f"{pattern_type}({inside})"

def _arrival_pattern_to_dict(pattern_name: str) -> dict:
    """
    Parse a string of the form:
      SomeArrivalClassName(key1=val1, key2=val2, ...)
    into {"type": "SomeArrivalClassName", "params": {...}}.

    If it fails to parse, returns {"type": pattern_name}.
    """
    # Regex to capture:
    # 1) The class name (one or more word chars)  -> group(1)
    # 2) The inside of parentheses (everything until the final ) ) -> group(2)
    #
    # e.g. "ContinuousPoissonArrival(rate=1.0)"
    #   -> group(1) = "ContinuousPoissonArrival"
    #   -> group(2) = "rate=1.0"
    #
    # If it doesn't match, we can't parse generically.
    match = re.match(r'^(\w+)\s*\(([^)]*)\)$', pattern_name.strip())
    if not match:
        return {"type": pattern_name}  # fallback

    cls_name = match.group(1)
    inside = match.group(2).strip()

    # If there are no params inside, e.g. "MyArrival()", just return type:
    if not inside:
        return {"type": cls_name, "params": {}}

    params_dict = _parse_key_value_pairs(inside)
    if params_dict is None:
        # Could not parse the inside
        return {"type": pattern_name}

    return {"type": cls_name, "params": params_dict}

def _parse_key_value_pairs(param_str: str) -> dict:
    """
    Naive parsing of a comma-separated list of key=value pairs, like:
      "rate1=0.5, rate2=1.5, p=0.3"
    or
      "lambda_per_step=0.01, max_steps=4000"

    Returns a dict of {key: converted_value} or None if parse fails badly.
    """
    params = {}
    # split by commas at the top level (naive approach, won't handle nested commas)
    chunks = param_str.split(",")
    for chunk in chunks:
        chunk = chunk.strip()
        if "=" not in chunk:
            # can't parse
            return None
        k, v = chunk.split("=", 1)
        k = k.strip()
        v = v.strip()
        # Try to convert v -> float or int if possible
        v_converted = _convert_str_value(v)
        params[k] = v_converted
    return params

def _convert_str_value(s: str):
    """
    Attempt to convert a string to int or float, otherwise leave as string.
    """
    # If it has a decimal or 'e', try float
    if "." in s or "e" in s.lower():
        try:
            return float(s)
        except ValueError:
            pass
    # If it's purely digits (possibly with a leading '-'), parse as int
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)

    # Otherwise, return original string
    return s

@dataclass
class Trace:
    """
    Represents a collection (or 'trace') of generated requests, now stored as a dictionary:
      { "request_0": Request, "request_1": Request, ... }

    Attributes:
        requests (Dict[str, Request]): The requests dictionary.
        arrival_pattern_name (str): A descriptive name of the arrival pattern used.
        batch_size (int): The initial batch size used in generation.
        request_type_probs: The distribution of RequestTypes used.
        vocab (Tuple[int, int]): The range of token IDs used.
    """
    requests: Dict[str, "Request"]
    arrival_pattern_name: str
    batch_size: int
    request_type_probs: any  # could be a dict or list of tuples, depending on your design
    vocab: Tuple[int, int]
    num_gpu_blocks_override: Optional[int] = field(default=None, init=True) 
    max_model_len: Optional[int] = field(default=None, init=True) 
    gpu_memory_utilization: Optional[float] = field(default=None, init=True)
    
    def __iter__(self):
        """
        Allows iteration over the Trace object in ascending order of arrival_time.
        """
        # Sort dictionary values by each request's arrival_time
        sorted_requests = sorted(self.requests.values(), key=lambda r: r.arrival_time)
        return iter(sorted_requests)

    def __repr__(self):
        # For the display of request_type_probs, handle it if it's a dict or list
        if isinstance(self.request_type_probs, dict):
            info = [(rt.category_name, p) for rt, p in self.request_type_probs.items()]
        else:
            info = [(rt.category_name, p) for rt, p in self.request_type_probs]

        return (f"Trace(\n"
                f"  arrival_pattern={self.arrival_pattern_name},\n"
                f"  batch_size={self.batch_size},\n"
                f"  max_model_len={self.max_model_len},\n"
                f"  num_gpu_blocks_override={self.num_gpu_blocks_override},\n"
                f"  request_type_probs={info},\n"
                f"  vocab={self.vocab},\n"
                f"  requests=[{len(self.requests)} dictionary entries]\n"
                f")")
    
    @classmethod
    def load_from_json(cls, filename: str) -> "Trace":
        """
        Reads a JSON file (in the compact format from save_to_json) and reconstructs a Trace object.
        If token_ids are missing or their lengths do not match input_length, they are regenerated.

        Args:
            filename (str): Path to the JSON file to load.

        Returns:
            Trace: The reconstructed Trace object.
        """
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Rebuild request_type_probs from something like [["Short Q&A", 0.7], ["Summarization", 0.3]]
        # We'll just keep them as a list of (category_name, probability)
        request_type_probs_data = data["request_type_probs"]

        # Convert the arrival_pattern dictionary back into a single string
        # e.g. {"type": "DiscretePoissonArrival", "params": {"lambda_per_step":0.01,"max_steps":4000}}
        # -> "DiscretePoissonArrival(lambda_per_step=0.01, max_steps=4000)"
        arrival_pattern_str = _arrival_pattern_from_dict(data["arrival_pattern"])

        requests_dict = {}
        vocab_min, vocab_max = data["vocab"]
        num_gpu_blocks_override = data.get("num_gpu_blocks_override", None) 
        gpu_memory_utilization = data.get("gpu_memory_utilization", None)

        for req_id, rdict in data["requests"].items():
            # If 'token_ids' missing or length mismatch, regenerate
            actual_token_ids = rdict.get("token_ids", None)
            input_len = rdict["input_length"]
            if (actual_token_ids is None) or (len(actual_token_ids) != input_len):
                rdict["token_ids"] = [
                    random.randint(vocab_min, vocab_max)
                    for _ in range(input_len)
                ]

            request_obj = Request(
                category=rdict["category"],
                input_length=rdict["input_length"],
                output_length=rdict["output_length"],
                arrival_time=rdict["arrival_time"],
                token_ids=rdict["token_ids"],
                sched_time=rdict.get("sched_time"),
                wait_time=rdict.get("wait_time"),
            )
            requests_dict[req_id] = request_obj

        return cls(
            requests=requests_dict,
            arrival_pattern_name=arrival_pattern_str,
            batch_size=data["batch_size"],
            max_model_len=data["max_model_len"],
            request_type_probs=request_type_probs_data,  # we keep it as-is
            vocab=tuple(data["vocab"]),
            num_gpu_blocks_override=num_gpu_blocks_override,
            gpu_memory_utilization=gpu_memory_utilization
        )
    
    def add_estimate_sched(self,
                           num_gpu_blocks: int,
                           block_size: int = 16,
                           max_parallel: int = 1) -> None:
        """
        A scheduling simulation, adapted for a requests dict. The logic is the same,
        but we'll convert the dictionary to a list (sorted by arrival_time),
        do the scheduling, then store the updated requests back in the dict.

        Scheduling rules:
          - Sort requests by arrival_time, but schedule them individually.
          - Processing duration = (1 + output_length).
          - GPU memory usage is based on input_length, claimed at the start and freed upon completion.
          - If memory or parallel constraints are exceeded, we must wait.
          - Each request's sched_time and wait_time are updated.
        """
        self.num_gpu_blocks_override = num_gpu_blocks
        self.batch_size = max_parallel
        self.max_model_len = num_gpu_blocks*block_size
        # 1) Convert to a list of (req_key, Request), sorted by arrival_time
        items_sorted = sorted(self.requests.items(), key=lambda x: x[1].arrival_time)
        # e.g. items_sorted = [("request_0", reqObj), ("request_1", reqObj), ...]

        tokens_capacity = num_gpu_blocks * block_size
        running = []
        heapq.heapify(running)

        current_time = 0

        # We'll group requests by arrival_time
        groups = defaultdict(list)
        for key, req in items_sorted:
            groups[req.arrival_time].append((key, req))

        arrival_times = sorted(groups.keys())

        for arrival_time in arrival_times:
            batch = groups[arrival_time]

            # (a) Free any requests that finished by this arrival_time
            while running and running[0][0] <= arrival_time:
                finish_time, in_usage, finished_key = heapq.heappop(running)
                tokens_capacity += in_usage

            # (b) For each request in this batch, schedule
            for req_key, req_obj in batch:
                req_in = req_obj.input_length

                # If it can't fit an empty GPU, fail
                if req_in > tokens_capacity and not running:
                    raise RuntimeError(
                        f"Request {req_obj} has input_length={req_in}, "
                        f"exceeds total capacity={tokens_capacity} with no tasks running."
                    )

                # free tasks finishing exactly by arrival_time
                while running and running[0][0] <= req_obj.arrival_time:
                    finish_time, in_usage, finished_key = heapq.heappop(running)
                    tokens_capacity += in_usage

                # Wait if needed
                can_run_now = False
                while not can_run_now:
                    if len(running) >= max_parallel:
                        finish_time, in_usage, finished_key = heapq.heappop(running)
                        current_time = finish_time
                        tokens_capacity += in_usage
                    else:
                        if req_in <= tokens_capacity:
                            can_run_now = True
                        else:
                            if not running:
                                raise RuntimeError(
                                    f"Request {req_obj} can't fit in capacity={tokens_capacity} "
                                    f"and no tasks to wait for."
                                )
                            finish_time, in_usage, finished_key = heapq.heappop(running)
                            current_time = finish_time
                            tokens_capacity += in_usage

                start_time = max(req_obj.arrival_time, current_time)
                req_obj.sched_time = start_time
                req_obj.wait_time = req_obj.sched_time - req_obj.arrival_time

                finish_time = start_time + (1 + req_obj.output_length)
                tokens_capacity -= req_in
                heapq.heappush(running, (finish_time, req_in, req_key))

        # end of scheduling
        # The `requests` dict is updated in-place since we changed each req_obj


    def save_to_json(self, filename: str, skip_token_ids: bool = False) -> None:
        """
        Serialize this Trace object into a JSON file in a compact form:
        - "arrival_pattern" is a one-liner dict with "type" and (optionally) "params".
        - "batch_size", "request_type_probs", "vocab" are also written in a compact style.
        - "requests" is a dictionary. Each key is "request_i", each value is either:
            - single-line if token_ids are skipped or absent
            - two-line if token_ids exist
        """
        # 1) Convert request_type_probs into a list of (category_name, prob) for JSON
        # if isinstance(self.request_type_probs, dict):
        #     request_type_probs_data = [
        #         (rt.category, prob) for (rt, prob) in self.request_type_probs.items()
        #     ]
        # else:
        #     request_type_probs_data = [
        #         (rt.category, prob) for (rt, prob) in self.request_type_probs
        #     ]

        request_type_probs_data = [
            (cat, prob) for cat, prob in self.request_type_probs
        ]

        data = {
            "arrival_pattern": _arrival_pattern_to_dict(self.arrival_pattern_name),
            "batch_size": self.batch_size,
            "max_model_len": self.max_model_len,
            "num_gpu_blocks_override": self.num_gpu_blocks_override,
            "request_type_probs": request_type_probs_data,
            "vocab": self.vocab
        }

        # 2) Prepare each request in a dictionary
        #    We'll do a manual approach to keep it compact, especially for token_ids.
        requests_items = list(self.requests.items())  # [(req_id, RequestObj), ...]
        # We'll need them in a stable order (e.g. sorted by req_id or arrival_time).
        # If you prefer arrival-time order, do:
        # requests_items.sort(key=lambda x: x[1].arrival_time)
        # For now, let's assume they're already in ascending "request_i" keys.

        # 3) Write top-level JSON keys manually
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("{\n")

            # a) arrival_pattern
            # "arrival_pattern" is a dict like {"type": "...", "params": {...}}
            pattern_json = json.dumps(
                data["arrival_pattern"], separators=(',', ':'),
                ensure_ascii=False
            )
            f.write(f'  "arrival_pattern":{pattern_json},\n')

            # b) batch_size
            f.write(f'  "batch_size":{data["batch_size"]},\n')

            f.write(f'  "max_model_len":{data["max_model_len"]},\n')

            f.write(f'  "num_gpu_blocks_override":{data["num_gpu_blocks_override"]},\n')

            # c) request_type_probs (one-liner)
            rtp_json = json.dumps(data["request_type_probs"], separators=(',', ':'))
            f.write(f'  "request_type_probs":{rtp_json},\n')

            # d) vocab
            vocab_json = json.dumps(data["vocab"], separators=(',', ':'))
            f.write(f'  "vocab":{vocab_json},\n')

            # e) "requests" dictionary
            f.write('  "requests":{\n')

            # We'll iterate over each request key/value
            for i, (req_id, req_obj) in enumerate(requests_items):
                is_last = (i == len(requests_items) - 1)

                # Build a dictionary for the request minus token_ids
                req_dict = {
                    "category": req_obj.category,
                    "input_length": req_obj.input_length,
                    "output_length": req_obj.output_length,
                    "arrival_time": req_obj.arrival_time,
                    "sched_time": req_obj.sched_time,
                    "wait_time": req_obj.wait_time
                }
                token_ids = None
                if not skip_token_ids:
                    token_ids = req_obj.token_ids

                main_line = json.dumps(req_dict, separators=(',', ':'), ensure_ascii=False)

                # We'll write something like:
                # "request_0":{...},
                # or if token_ids exist => two-line format
                f.write(f'    "{req_id}":')
                if token_ids is None:
                    # Single-line
                    f.write(main_line)
                    if not is_last:
                        f.write(',')
                    f.write('\n')
                else:
                    # Two-line format:
                    # e.g. "request_0":{"category":"ChatBot Q&A","input_length":10,...
                    #                    "token_ids":[...]}
                    main_no_brace = main_line[:-1]  # remove closing '}'
                    token_ids_str = json.dumps(token_ids, separators=(',', ':'), ensure_ascii=False)

                    f.write(main_no_brace)
                    f.write(',')  # comma after the last field in main
                    f.write(f'"token_ids":{token_ids_str}}}')
                    if not is_last:
                        f.write(',')
                    f.write('\n')

            # close the requests dict
            f.write('  }\n')
            # close the entire JSON
            f.write("}\n")


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def create_instances_from_json(json_data):
    pattern_map = {
        "DiscreteBimodalArrival": DiscreteBimodalArrival,
        "DiscretePoissonArrival": DiscretePoissonArrival,
        "DiscretePeriodicArrival": DiscretePeriodicArrival,
        "DiscreteUniformArrival": DiscreteUniformArrival,
        "PoissonBurstyArrivalPattern": PoissonBurstyArrivalPattern
    }

    output_map = {
        "DiscreteBimodalArrival": "bimodal",
        "DiscretePoissonArrival": 'poisson',
        "DiscretePeriodicArrival": "periodic",
        "DiscreteUniformArrival": "uniform",
        "PoissonBurstyArrivalPattern": "bursty"
    }
        
    instance_list = []
    output_list = []
    for item in json_data:
        cls_name = item.pop("class")
        cls = pattern_map[cls_name]
        instance = cls(**item)
        instance_list.append(instance)
        output_list.append(output_map[cls_name])

    return instance_list, output_list

def generate_trace(
        request_data, 
        arrival_pattern_obj,
        static: bool = False,
    ):

    json_requests = request_data["requests"]
    batch_size = request_data["batch_size"]
    vocab = request_data["vocab"]
    request_type_probs = request_data["request_type_probs"]
    
    num_requests = len(json_requests)
    arrival_times = arrival_pattern_obj.generate_arrival_times(num_requests)
    arrival_times.sort()

    for i in range(min(batch_size, num_requests)):
        arrival_times[i] = 0
    
    if static:
        for i in range(0, num_requests, batch_size):
            group_value = arrival_times[i]
            for j in range(i, min(i + batch_size, num_requests)):
                arrival_times[j] = group_value

    print(f"arrival_times: {arrival_times}")
    sorted_keys = sorted(json_requests.keys(), key=lambda x: int(x.split('_')[1]))
    requests = []

    for i, key in enumerate(sorted_keys):
        req = json_requests[key]
        arrival_time = arrival_times[i]
        sched_time = (i // batch_size) * 1.0  # 예시 스케줄링: 배치당 1.0 step
        wait_time = max(0.0, sched_time - arrival_time)
        total_length = req["input_length"] + req["output_length"]
        slo = total_length * PROFILED_A + PROFILED_B 

        requests.append(Request(
            category=req["category"],
            input_length=req["input_length"],
            output_length=req["output_length"],
            arrival_time=arrival_time,
            token_ids=req.get("token_ids", []),
            sched_time=sched_time,
            wait_time=wait_time,
            slo=slo
        ))

    requests = compress_idle_steps(requests)
    requests.sort(key=lambda r: r.arrival_time)

    requests_dict = {}
    for i, req in enumerate(requests):
        requests_dict[f"request_{i}"] = req

    trace = Trace(
        requests=requests_dict,  # pass the dict instead of a list
        arrival_pattern_name=str(arrival_pattern_obj),
        batch_size=batch_size,
        request_type_probs=request_type_probs,
        vocab=vocab
    )
    return trace

def build_sched_save(
    request_json_path: str, 
    arrival_json_path: str,
    num_gpu_blocks: int,
    block_size: int,
    static: bool = False,
    plot_distributions: bool = True,            # ← new flag
    skip_token_ids: bool = True,
    postfix_trace: str  = "_trace",
    postfix_model: str  = "_model"
):
    import json, random, matplotlib.pyplot as plt
    from itertools import accumulate
    from pathlib import Path
    """
    1) Make a TraceType
    2) Generate a trace
    3) Call add_estimate_sched(...) with the global GPU parameters
    4) Save trace to JSON  (skip_token_ids=True keeps files small)
    5) [optional] Plot & save distributions of the generated trace
       vs. the underlying generative model.
    """
    request_data = load_json(request_json_path)
    arrival_config = load_json(arrival_json_path)
    arrival_patterns, output_list = create_instances_from_json(arrival_config)

    batch_size = request_data["batch_size"]
    request_type_probs = request_data["request_type_probs"]

    request_path = Path(request_json_path)

    for arrival_pattern, output in zip(arrival_patterns, output_list):
        print(arrival_pattern)
        output_dir = request_path.parent / output
        output_dir.mkdir(exist_ok=True)  # 디렉토리 없으면 생성
        output_json_path = output_dir / "trace.json"
    
        trace_obj = generate_trace(
            request_data, arrival_pattern, static
        )

        # GPU-capacity scheduling simulation
        trace_obj.add_estimate_sched(
            num_gpu_blocks=num_gpu_blocks, block_size=block_size,
            max_parallel=batch_size
        )

        # # ---------- 4. persist JSON ----------
        trace_obj.save_to_json(output_json_path, skip_token_ids=skip_token_ids)

        # # -------- NEW: write a summary .txt with total token counts -----------------
        # tot_in  = sum(r.input_length  for r in trace_obj.requests.values())
        # tot_out = sum(r.output_length for r in trace_obj.requests.values())

        # txt_path = Path(output_json_path).with_suffix("")  # strip “.json”
        # with open(f"{txt_path}_token_totals.txt", "w") as fh:
        #     fh.write(f"total_input_tokens  {tot_in}\n")
        #     fh.write(f"total_output_tokens {tot_out}\n")

        # # ---------- 5. optional plotting ----------
        # if not plot_distributions:
        #     return                                            # nothing else to do
        # # ---------------- empirical data ----------------
        # reqs          = list(trace_obj.requests.values())
        # input_lens    = np.array([r.input_length  for r in reqs])
        # output_lens   = np.array([r.output_length for r in reqs])
        # arrivals      = np.sort([r.arrival_time for r in reqs])
        # inter_arr_emp = np.diff(arrivals)          # empty if len<2

        # # ---------------- analytic model ----------------
        # # 1) helper → uniform-mixture PMF
        # def mixture_uniform_pmf(bounds, probs):
        #     lo = min(b[0] for b in bounds.values())
        #     hi = max(b[1] for b in bounds.values())
        #     xs = np.arange(lo, hi + 1)
        #     pmf = np.zeros_like(xs, dtype=float)
        #     for cat, p in probs.items():
        #         a, b = bounds[cat]
        #         pmf[(xs >= a) & (xs <= b)] += p / (b - a + 1)
        #     return xs, pmf

        # from collections import defaultdict

        # input_by_cat = defaultdict(list)
        # output_by_cat = defaultdict(list)

        # for r in reqs:
        #     input_by_cat[r.category].append(r.input_length)
        #     output_by_cat[r.category].append(r.output_length)

        # # 3) compute in_bounds, out_bounds per category
        # in_bounds = {
        #     cat: (min(lengths), max(lengths))
        #     for cat, lengths in input_by_cat.items()
        # }
        # out_bounds = {
        #     cat: (min(lengths), max(lengths))
        #     for cat, lengths in output_by_cat.items()
        # }

        # request_type_probs_list = request_data["request_type_probs"]
        # probs = {cat: prob for cat, prob in request_type_probs_list}

        # xin,  pin  = mixture_uniform_pmf(in_bounds,  probs)
        # xout, pout = mixture_uniform_pmf(out_bounds, probs)

        # # 2) geometric PMF: P(k) = (1-λ)^{k-1} λ   for k≥1
        # lam = getattr(arrival_pattern, "lambda_per_step", None)
        # k_max = (inter_arr_emp.max() if inter_arr_emp.size else 1) + 20
        # k_vals = np.arange(1, k_max + 1)
        # if lam is not None and 0 < lam < 1:
        #     pgeom = (1 - lam) ** (k_vals - 1) * lam
        # else:                      # continuous pattern or λ invalid → flat 0
        #     pgeom = np.zeros_like(k_vals, dtype=float)

        # # ---------------- save helper ----------------
        # stem = Path(output_json_path).with_suffix("")  # remove .json

        # def savefig(tag, postfix):
        #     out = f"{stem}{postfix}_{tag}.png"
        #     plt.savefig(out, dpi=150, bbox_inches="tight")
        #     plt.close()

        # # ---------------- trace histograms ----------------
        # plt.figure()
        # plt.hist(input_lens, bins=50)
        # plt.xlabel("input length (tokens)"); plt.ylabel("count")
        # plt.title("Trace: input-length distribution")
        # savefig("input", postfix_trace)

        # plt.figure()
        # plt.hist(output_lens, bins=50)
        # plt.xlabel("output length (tokens)"); plt.ylabel("count")
        # plt.title("Trace: output-length distribution")
        # savefig("output", postfix_trace)

        # if inter_arr_emp.size:
        #     plt.figure()
        #     plt.hist(inter_arr_emp, bins=50)
        #     plt.xlabel("inter-arrival gap"); plt.ylabel("count")
        #     plt.title("Trace: inter-arrival distribution")
        #     savefig("inter", postfix_trace)

        # # ---------------- analytic PDFs / PMF ----------------
        # plt.figure()
        # plt.step(xin, pin, where="mid")
        # plt.xlabel("input length (tokens)"); plt.ylabel("probability")
        # plt.title("Model: input-length PDF (mixture of uniforms)")
        # savefig("input", postfix_model)

        # plt.figure()
        # plt.step(xout, pout, where="mid")
        # plt.xlabel("output length (tokens)"); plt.ylabel("probability")
        # plt.title("Model: output-length PDF (mixture of uniforms)")
        # savefig("output", postfix_model)

        # if lam is not None and 0 < lam < 1:
        #     plt.figure()
        #     plt.stem(k_vals, pgeom)
        #     plt.xlabel("inter-arrival gap (steps)"); plt.ylabel("probability")
        #     plt.title(f"Model: geometric PMF (λ={lam:g})")
        #     savefig("inter", postfix_model)


# 사용 예시
if __name__ == "__main__":
    max_model_len = 8000
    block_size = 16
    num_gpu_blocks = max_model_len//block_size
    max_parallel = 4 # batch size 

    build_sched_save(
        request_json_path="/home/sychoy/vllm/trace_pool/type2_32k/request.json",
        arrival_json_path="/home/sychoy/vllm/trace_pool/type2_32k/arrivals.json",
        num_gpu_blocks=num_gpu_blocks,
        block_size=block_size,
        plot_distributions=True,  # Set to True to plot distributions
        skip_token_ids=True,
        postfix_trace="_trace",
        postfix_model="_model",
        # static=True
    )
