import random
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from abc import abstractmethod
import heapq
from collections import defaultdict
import json

# -------------------------------------------------------
# Request + RequestType
# -------------------------------------------------------

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
    category: "RequestType"
    input_length: int
    output_length: int
    arrival_time: Optional[float|int]
    token_ids: List[int]
    sched_time: Optional[float|int] = None
    wait_time: Optional[float|int] = None

    def __repr__(self):
        """
        Custom string representation, including scheduling details if they're available.
        """
        if sched_time := getattr(self, 'sched_time', None):
            return (f"Request(category={self.category}, prompt={self.input_length}, "
                    f"max_tokens={self.output_length}, arrive_time={self.arrival_time}, "
                    f"sched_time={sched_time}, wait_time={self.wait_time})")
        else:
            return (f"Request(category={self.category}, prompt={self.input_length}, "
                    f"max_tokens={self.output_length}, t={self.arrival_time})")


class RequestType:
    """
    Defines a particular 'type' of request, specifying minimum/maximum
    token lengths for both inputs and outputs, along with a descriptive name.
    """

    def __init__(
        self,
        category_name: str,
        min_input_tokens: int,
        max_input_tokens: int,
        min_output_tokens: int,
        max_output_tokens: int
    ):
        """
        Args:
            category_name (str): A descriptive name for this request type.
            min_input_tokens (int): Minimum input token length.
            max_input_tokens (int): Maximum input token length.
            min_output_tokens (int): Minimum output token length.
            max_output_tokens (int): Maximum output token length.
        """
        self.category_name = category_name
        self.min_in = min_input_tokens
        self.max_in = max_input_tokens
        self.min_out = min_output_tokens
        self.max_out = max_output_tokens

    def generate_request(
        self,
        arrival_time: float = 0.0,
        vocab: Tuple[int, int] = (200, 30000),
        trace_limit: int = None
    ) -> "Request":
        """
        Creates a single `Request` object using uniform random sampling for
        input and output lengths, subject to an optional upper bound (`trace_limit`).
        
        Args:
            arrival_time (float): The time at which this request arrives.
            vocab (Tuple[int, int]): The range of token IDs to sample from.
            trace_limit (int, optional): If provided, ensures that the sum of
                input_length and output_length does not exceed this limit.
        
        Raises:
            ValueError: If the minimum possible size (min_in + min_out) is larger than trace_limit.

        Returns:
            Request: A new Request instance with randomly determined sizes and token IDs.
        """
        # Check if the smallest possible request is still too large given trace_limit.
        if trace_limit is not None:
            if (self.min_in + self.min_out) > trace_limit:
                raise ValueError(
                    f"No valid request can fit: minimum size "
                    f"{self.min_in + self.min_out} > trace_limit={trace_limit}."
                )

        # Keep drawing random input/output lengths until (input + output) is within trace_limit (if given).
        while True:
            input_length = random.randint(self.min_in, self.max_in)
            output_length = random.randint(self.min_out, self.max_out)
            if trace_limit is not None:
                if (input_length + output_length) > trace_limit:
                    continue
            break

        # Generate a placeholder list of token IDs for the request.
        token_ids = [random.randint(vocab[0], vocab[1]) for _ in range(input_length)]

        return Request(
            category=self.category_name,
            input_length=input_length,
            output_length=output_length,
            arrival_time=arrival_time,
            token_ids=token_ids
        )


# -------------------------------------------------------
# ArrivalPattern Interface
# -------------------------------------------------------

@abstractmethod
class ArrivalPattern:
    """
    Abstract base class for arrival patterns. Each subclass is responsible for
    implementing a method to generate arrival times for a specified number of requests.
    """

    def generate_arrival_times(self, num_requests: int) -> List[float|int]:
        """
        Must return a list of arrival times (floats or ints) for `num_requests` requests.
        """
        raise NotImplementedError


# -------------------------------------------------------
# Continuous-Time Poisson Arrival
# -------------------------------------------------------

class ContinuousPoissonArrival(ArrivalPattern):
    """
    Represents a continuous-time Poisson arrival process, sampling inter-arrival
    times from an exponential distribution with parameter 'rate'.
    """

    def __init__(self, rate: float):
        """
        Args:
            rate (float): The average arrival rate (λ) per unit time.
        """
        self.rate = rate

    def generate_arrival_times(self, num_requests: int) -> List[float]:
        """
        Generates a list of arrival times by summing exponentially distributed inter-arrivals.
        
        Args:
            num_requests (int): How many requests to generate.

        Returns:
            List[float]: A list of ascending arrival times.
        """
        arrivals = []
        current_time = 0.0
        for _ in range(num_requests):
            interarrival = random.expovariate(self.rate)
            current_time += interarrival
            arrivals.append(current_time)
        return arrivals


# -------------------------------------------------------
# Discrete-Time Poisson Arrival
# -------------------------------------------------------

class DiscretePoissonArrival(ArrivalPattern):
    """
    A discrete-time Poisson model where time is considered in discrete steps
    (0, 1, 2, ...). At each integer step t:
      - We draw how many arrivals occur using Poisson(lambda_per_step).
      - All arrivals happening at step t have arrival_time = t.
    """

    def __init__(self, lambda_per_step: float, max_steps: int = 100000):
        """
        Args:
            lambda_per_step (float): The expected number of arrivals per time step.
            max_steps (int): A cap on how many time steps to simulate if num_requests is large.
        """
        self.lambda_per_step = lambda_per_step
        self.max_steps = max_steps

    def generate_arrival_times(self, num_requests: int) -> List[int]:
        """
        Generates up to `num_requests` arrival times, each an integer step value.
        Continues sampling discrete steps until we have enough arrivals or reach max_steps.

        Args:
            num_requests (int): Desired number of arrival times.

        Returns:
            List[int]: List of arrival times (non-decreasing).
        """
        arrivals = []
        total_requests = 0
        t = 0
        while total_requests < num_requests and t < self.max_steps:
            # Number of arrivals in this time step
            k = self._sample_poisson(self.lambda_per_step)
            for _ in range(k):
                arrivals.append(int(t))
                total_requests += 1
                if total_requests >= num_requests:
                    break
            t += 1
        return arrivals

    @staticmethod
    def _sample_poisson(lmbd: float) -> int:
        """
        Minimal Poisson sampling using Knuth's algorithm (efficient for small λ).

        Args:
            lmbd (float): The Poisson rate parameter (expected value).

        Returns:
            int: The sampled number of arrivals.
        """
        L = math.exp(-lmbd)
        p = 1.0
        k = 0
        while p > L:
            k += 1
            p *= random.random()
        return k - 1


# -------------------------------------------------------
# Helper functions to parse / reformat arrival_pattern_name
# into a structured dictionary and back.
# -------------------------------------------------------

def _arrival_pattern_to_dict(pattern_name: str) -> dict:
    """
    Attempt to parse the trace's arrival_pattern_name (a string) into a structured dict.
    Expected forms:
        "DiscretePoissonArrival(lambda_per_step=0.01, max_steps=4000)"
        "ContinuousPoissonArrival(rate=1.0)"
    Fallback: just store {"type": <pattern_name>} if it doesn't match or parse well.
    
    Returns:
        dict: For recognized patterns, e.g. {
                  "type": "DiscretePoissonArrival",
                  "params": {"lambda_per_step": 0.01, "max_steps": 4000}
              }
              Otherwise {"type": <unparsed string>}
    """
    # Quick check for recognized pattern prefixes
    if "DiscretePoissonArrival(" in pattern_name:
        # e.g. "DiscretePoissonArrival(lambda_per_step=0.01, max_steps=4000)"
        return _parse_specific_pattern(pattern_name, "DiscretePoissonArrival")
    elif "ContinuousPoissonArrival(" in pattern_name:
        # e.g. "ContinuousPoissonArrival(rate=1.0)"
        return _parse_specific_pattern(pattern_name, "ContinuousPoissonArrival")
    else:
        # Could be a memory address string or something else
        return {"type": pattern_name}


def _parse_specific_pattern(full_str: str, prefix: str) -> dict:
    """
    Parse a pattern string like "DiscretePoissonArrival(lambda_per_step=0.01, max_steps=4000)"
    extracting the arguments inside parentheses into a dict.
    
    If parsing fails, returns {"type": full_str}.
    """
    # Expected form: prefix + '(' + stuff + ')'
    # We'll try a naive parse of the parentheses.
    try:
        start = full_str.index("(")
        end = full_str.rindex(")")
        inside = full_str[start+1:end].strip()
        # inside might be "lambda_per_step=0.01, max_steps=4000"
        params_dict = {}

        for chunk in inside.split(","):
            chunk = chunk.strip()
            if "=" not in chunk:
                # Invalid chunk, skip
                continue
            k, v = chunk.split("=", 1)
            k = k.strip()
            v = v.strip()
            # Attempt float or int conversion
            if "." in v or "e" in v.lower():
                try:
                    v = float(v)
                except ValueError:
                    pass
            else:
                # Might be int
                if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
                    v = int(v)
            params_dict[k] = v

        return {
            "type": prefix,
            "params": params_dict
        }
    except (ValueError, IndexError):
        # If anything went wrong, just store the whole string
        return {"type": full_str}


def _arrival_pattern_from_dict(d: dict) -> str:
    """
    In the load function, we will take the stored arrival_pattern dictionary
    and convert it back into a single string for arrival_pattern_name, e.g.:
      {
        "type": "DiscretePoissonArrival",
        "params": { "lambda_per_step": 0.01, "max_steps": 4000 }
      }
    becomes
      "DiscretePoissonArrival(lambda_per_step=0.01, max_steps=4000)"
    If d lacks "params", we just use d["type"].
    """
    pattern_type = d.get("type", "UnknownPattern")
    params = d.get("params", None)
    if not params:
        return pattern_type  # e.g. "UnknownPattern"
    # Rebuild something like: DiscretePoissonArrival(lambda_per_step=0.01, max_steps=4000)
    param_parts = []
    for k, v in params.items():
        if isinstance(v, (int, float)):
            param_parts.append(f"{k}={v}")
        else:
            param_parts.append(f"{k}={repr(v)}")
    inside = ", ".join(param_parts)
    return f"{pattern_type}({inside})"


@dataclass
class Trace:
    """
    Represents a collection (or 'trace') of generated requests, along with
    metadata describing how those requests were created.
    
    Attributes:
        requests (List[Request]): The sequence of requests generated.
        arrival_pattern_name (str): A descriptive name of the arrival pattern used.
        batch_size (int): The initial batch size used in generation (first requests set to time=0).
        request_type_probs (List[Tuple[RequestType, float]]): RequestType objects with their cumulative probability.
        vocab (Tuple[int, int]): The range of token IDs used to generate each request's tokens.
    """
    requests: List["Request"]
    arrival_pattern_name: str
    batch_size: int
    # NOTE: user replaced request_type_probs with a dict in generate_requests,
    # but let's keep the type annotation to match the original usage.
    request_type_probs: Dict["RequestType", float]  # or List[Tuple["RequestType", float]]
    vocab: Tuple[int, int]

    def __iter__(self):
        """
        Allows iteration over the Trace object, yielding each Request in self.requests.
        """
        return iter(self.requests)

    def __repr__(self):
        """
        Custom string output showing the summary of the Trace.
        """
        # If request_type_probs is a dict, we might convert it to a list for display:
        if isinstance(self.request_type_probs, dict):
            info = [(rt.category_name, p) for rt, p in self.request_type_probs.items()]
        else:
            # already a list of (RequestType, float)
            info = [(rt.category_name, p) for rt, p in self.request_type_probs]
        return (f"Trace(\n"
                f"  arrival_pattern={self.arrival_pattern_name},\n"
                f"  batch_size={self.batch_size},\n"
                f"  request_type_probs={info},\n"
                f"  vocab={self.vocab},\n"
                f"  requests=[{len(self.requests)} Requests total]\n"
                f")")

    def save_to_json(self, filename: str, skip_token_ids: bool = False) -> None:
        """
        Serialize this Trace object into a JSON file with a more compact layout:
          - arrival_pattern is a dict with "type" and optionally "params" if we can parse them.
          - Top-level keys occupy minimal lines.
          - Each request is on one or two lines:
              - One line if token_ids are skipped or absent.
              - Two lines if token_ids exist (line #1: main fields, line #2: token_ids array).
        
        Args:
            filename (str): Path of the JSON file to be created.
            skip_token_ids (bool): If True, exclude token_ids from each request to reduce file size.
        """
        # Convert arrival_pattern_name to a structured dict
        arrival_pattern_info = _arrival_pattern_to_dict(self.arrival_pattern_name)

        # Convert (RequestType, float) to (category_name, float) for saving
        if isinstance(self.request_type_probs, dict):
            request_type_probs_data = [
                (rt.category_name, prob) for (rt, prob) in self.request_type_probs.items()
            ]
        else:
            # It's already a list of tuples
            request_type_probs_data = [
                (rt.category_name, prob) for (rt, prob) in self.request_type_probs
            ]

        data = {
            "arrival_pattern": arrival_pattern_info,
            "batch_size": self.batch_size,
            "request_type_probs": request_type_probs_data,
            "vocab": self.vocab
        }

        # Prepare each request as a dictionary
        requests_data = []
        for req in self.requests:
            req_dict = {
                "category": req.category,
                "input_length": req.input_length,
                "output_length": req.output_length,
                "arrival_time": req.arrival_time,
                "sched_time": req.sched_time,
                "wait_time": req.wait_time,
            }
            # Add token_ids unless told to skip them
            if not skip_token_ids:
                req_dict["token_ids"] = req.token_ids
            requests_data.append(req_dict)

        # Write out top-level keys manually to control format
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("{\n")

            # arrival_pattern as a JSON object
            pattern_json = json.dumps(data["arrival_pattern"], separators=(',', ':'), ensure_ascii=False)
            f.write(f'  "arrival_pattern": {pattern_json},\n')

            # batch_size
            f.write(f'  "batch_size": {data["batch_size"]},\n')

            # request_type_probs: e.g. [["Short Q&A",0.7],["Summarization",0.3]]
            rtp_json = json.dumps(data["request_type_probs"], separators=(',', ':'))
            f.write(f'  "request_type_probs": {rtp_json},\n')

            # vocab
            vocab_json = json.dumps(data["vocab"], separators=(',', ':'))
            f.write(f'  "vocab": {vocab_json},\n')

            # "requests" array
            f.write('  "requests": [\n')

            # Now handle each request in a custom compact format
            for i, rdict in enumerate(requests_data):
                is_last = (i == len(requests_data) - 1)

                token_ids = rdict.pop("token_ids", None)
                main_line = json.dumps(rdict, separators=(',', ':'), ensure_ascii=False)

                if token_ids is None:
                    # Single line
                    line = f'    {main_line}'
                    line += ',' if not is_last else ''
                    f.write(line + '\n')
                else:
                    # Two-line format
                    main_line_no_brace = main_line[:-1]  # remove final '}'
                    token_ids_str = json.dumps(token_ids, separators=(',', ':'), ensure_ascii=False)
                    line_tokens = f'      "token_ids":{token_ids_str}'
                    multi_line_req = (f'    {main_line_no_brace},\n'
                                      f'{line_tokens}')
                    if not is_last:
                        multi_line_req += ','
                    f.write(multi_line_req + '\n')

            f.write('  ]\n')
            f.write("}\n")

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

        requests_list = []
        vocab_min, vocab_max = data["vocab"]

        for rdict in data["requests"]:
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
                wait_time=rdict.get("wait_time")
            )
            requests_list.append(request_obj)

        return cls(
            requests=requests_list,
            arrival_pattern_name=arrival_pattern_str,
            batch_size=data["batch_size"],
            request_type_probs=request_type_probs_data,  # we keep it as-is
            vocab=tuple(data["vocab"])
        )

    def add_estimate_sched(self,
                           num_gpu_blocks: int,
                           block_size: int = 16,
                           max_parallel: int = 1) -> None:
        """
        Adds a rudimentary scheduling simulation to the requests.

        Scheduling rules:
          - Sort requests by arrival_time.
          - Group requests by arrival_time, but schedule them individually.
          - Processing duration = (1 + output_length). The '1' is a simplified 'forward pass' time,
            then we assume each token is generated in 1 unit of time.
          - GPU memory usage is based on input_length, claimed at the start and freed upon completion.
          - If memory or parallel constraints are exceeded, we must wait until at least one ongoing
            request finishes.
          - Once a request is scheduled, it's not preempted (no mid-run interruption).
          - Each request's sched_time and wait_time are updated.

        Args:
            num_gpu_blocks (int): The number of 'blocks' of GPU memory available.
            block_size (int): The size (in tokens) of each GPU block.
            max_parallel (int): Maximum number of requests to run simultaneously.
        """
        self.max_model_len = num_gpu_blocks * block_size
        self.num_gpu_blocks = num_gpu_blocks 
        self.block_size = block_size 

        tokens_capacity = num_gpu_blocks * block_size

        self.requests.sort(key=lambda r: r.arrival_time)
        groups = defaultdict(list)
        for req in self.requests:
            groups[req.arrival_time].append(req)
        arrival_times = sorted(groups.keys())

        running = []
        heapq.heapify(running)

        current_time = 0

        for arrival_time in arrival_times:
            batch = groups[arrival_time]

            # Free any requests finished by arrival_time
            while running and running[0][0] <= arrival_time:
                finish_time, in_usage, finished_req = heapq.heappop(running)
                tokens_capacity += in_usage

            # Try scheduling each request in the batch
            for req in batch:
                req_in = req.input_length

                # If it's impossible to fit an empty GPU, error out
                if req_in > tokens_capacity and not running:
                    raise RuntimeError(
                        f"Request {req} has input_length={req_in}, "
                        f"exceeds total capacity={tokens_capacity} with no tasks running."
                    )

                # Free requests finishing by this arrival_time
                while running and running[0][0] <= req.arrival_time:
                    finish_time, in_usage, finished_req = heapq.heappop(running)
                    tokens_capacity += in_usage

                # Wait if parallel/memory is full
                can_run_now = False
                while not can_run_now:
                    if len(running) >= max_parallel:
                        # free up a slot
                        finish_time, in_usage, finished_req = heapq.heappop(running)
                        current_time = finish_time
                        tokens_capacity += in_usage
                    else:
                        if req_in <= tokens_capacity:
                            can_run_now = True
                        else:
                            if not running:
                                raise RuntimeError(
                                    f"Request {req} with prompt={req_in} can't fit in capacity={tokens_capacity} "
                                    f"and no tasks to wait for."
                                )
                            finish_time, in_usage, finished_req = heapq.heappop(running)
                            current_time = finish_time
                            tokens_capacity += in_usage

                start_time = max(req.arrival_time, current_time)
                req.sched_time = start_time
                req.wait_time = req.sched_time - req.arrival_time

                finish_time = start_time + (1 + req.output_length)

                tokens_capacity -= req_in
                heapq.heappush(running, (finish_time, req_in, req))


# -------------------------------------------------------
# TraceType
# -------------------------------------------------------

class TraceType:
    """
    Defines how multiple RequestTypes (with associated probabilities) are combined,
    plus how their arrival times are generated via an ArrivalPattern.
    """

    def __init__(
        self,
        request_type_probs: Dict[RequestType, float],
        arrival_pattern: ArrivalPattern,
        vocab: Tuple[int] = (200,30000)
    ):
        """
        Args:
            request_type_probs (Dict[RequestType, float]): A mapping from RequestType
                to its probability (probabilities must total ~1.0).
            arrival_pattern (ArrivalPattern): An instance of either continuous or discrete
                Poisson arrival patterns (or any other subclass).
            vocab (Tuple[int], optional): Range of token IDs for random token generation.
        """
        self.request_type_probs = request_type_probs
        self.arrival_pattern = arrival_pattern
        self.vocab = vocab

        total = sum(self.request_type_probs.values())
        if not math.isclose(total, 1.0, rel_tol=1e-5):
            raise ValueError("RequestType probabilities must sum to 1.0")

        # Build a cumulative distribution for random draws
        self.cumulative_probs = []
        running_sum = 0.0
        sorted_rt = sorted(self.request_type_probs.items(), key=lambda x: x[1], reverse=True)
        for rt, p in sorted_rt:
            running_sum += p
            self.cumulative_probs.append((rt, running_sum))

    def compress_idle_steps(self, requests):
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

    def generate_requests(
        self,
        num_requests: int,
        batch_size: int = 1,
        trace_limit: int = None
    ) -> "Trace":
        """
        Generates a list of requests by:
          - Sampling arrival times from the chosen ArrivalPattern.
          - Randomly picking a RequestType for each arrival, weighted by request_type_probs.
          - Enforcing (input_length + output_length) <= trace_limit if provided.
          - Force first `batch_size` arrivals at time=0.
          - Optionally compress idle steps if we're in discrete time.

        Returns:
            Trace: Contains the generated requests plus the relevant metadata.
        """
        arrival_times = self.arrival_pattern.generate_arrival_times(num_requests)

        for i in range(min(batch_size, num_requests)):
            arrival_times[i] = 0

        requests = []
        for arrival_time in arrival_times:
            r = random.random()
            for (rt, cum_p) in self.cumulative_probs:
                if r <= cum_p:
                    chosen_rt = rt
                    break
            req = chosen_rt.generate_request(
                arrival_time=arrival_time,
                vocab=self.vocab,
                trace_limit=trace_limit
            )
            requests.append(req)

        requests = self.compress_idle_steps(requests)

        # The user specifically doesn't want us to change how we store
        # arrival_pattern_name in the Trace, so we keep using str(self.arrival_pattern).
        trace = Trace(
            requests=requests,
            arrival_pattern_name=str(self.arrival_pattern),
            batch_size=batch_size,
            request_type_probs=self.request_type_probs,  # storing the original dict
            vocab=self.vocab
        )
        return trace


# -------------------------------------------------------
# Example usage
# -------------------------------------------------------
if __name__ == "__main__":
    # 1) Define some RequestTypes
    short_qa = RequestType("Short Q&A", 10, 200, 10, 200)
    summarization = RequestType("Summarization", 2000, 10000, 100, 1000)

    # 2) Combine them with probabilities
    req_type_probs = {
        short_qa: 0.70,
        summarization: 0.30
    }

    # Example A: Continuous-time Poisson with rate=1.0
    continuous_arrival = ContinuousPoissonArrival(rate=1.0)
    continuous_trace_type = TraceType(req_type_probs, continuous_arrival)
    # Generate 5 requests in continuous-time
    continuous_requests = continuous_trace_type.generate_requests(num_requests=5)
    print("\nContinuous-time Poisson arrivals:")
    for r in continuous_requests:
        print(r)

    # Example B: Discrete-time Poisson with lambda_per_step=0.01
    discrete_arrival = DiscretePoissonArrival(lambda_per_step=0.01, max_steps=4000)
    discrete_trace_type = TraceType(req_type_probs, discrete_arrival)
    # Generate 10 requests in discrete-time with an initial batch_size of 4
    discrete_requests = discrete_trace_type.generate_requests(
        num_requests=10,
        batch_size=4,
        trace_limit=10000
    )
    print("\nDiscrete-time Poisson arrivals:")
    for r in discrete_requests:
        print(r)

    print('-----------------------')
    # Run the scheduling estimator
    discrete_requests.add_estimate_sched(
        num_gpu_blocks=(10000 // 16),
        block_size=16,
        max_parallel=4
    )
    print("\nAfter scheduling:")
    for r in discrete_requests:
        print(r)

    # Save to JSON (skipping token_ids to keep file small)
    discrete_requests.save_to_json("trace.json", skip_token_ids=True)

    print("\nLoaded from JSON:")
    trace = Trace.load_from_json("trace.json")
    for r in trace:
        print(r)
