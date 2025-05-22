import random
from typing import List, Dict, Tuple, Optional, Callable, Any
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import os, math
import warnings
from dataclasses import dataclass, field

PROFILED_A = 1.0017431830666432e-06
PROFILED_B = 0.049519613282613506

@dataclass
class Request:
    """
    Represents a single request instance in the generated trace.
    
    Attributes:
        category (RequestType): The category or 'type' of this request.
        input_length (int): Number of tokens in the request input.
        output_length (int): Number of tokens in the request output.
        token_ids (List[int]): A list of token IDs representing the request.
    """
    category: "RequestType"
    input_length: int
    output_length: int
    token_ids: List[int]
    slo: Optional[float|int] = None
    def __repr__(self):
        """
        Custom string representation, including scheduling details if they're available.
        """
        return (f"Request(category={self.category}, prompt={self.input_length}, "
                f"max_tokens={self.output_length}")

def clamp(value, low, high):
    return max(low, min(value, high))

def approximate_lognormal_params(min_val: int, max_val: int, percentile: float = 0.95):
    """
    Approximate mu and sigma for a lognormal distribution so that:
      - The median is roughly the geometric mean of [min_val, max_val]
      - 'percentile' fraction of the distribution is <= max_val
    Returns (mu, sigma).

    This is a heuristic, not an exact closed-form solution.
    """
    # 1) Median near geometric mean => median = exp(mu) = sqrt(min_val * max_val)
    median = math.sqrt(min_val * max_val)
    mu = math.log(median)

    # 2) Assume that percentile quantile is near max_val => 
    #    max_val = exp(mu + z*sigma)
    #    => sigma = (ln(max_val) - mu)/z
    z_values = {
        0.90: 1.282,
        0.95: 1.645,
        0.975: 1.96,
        0.99: 2.33,
    }
    z = z_values.get(percentile, 1.645)  # default ~95%
    sigma = (math.log(max_val) - mu) / z
    if sigma < 0:
        sigma = 0.5  # fallback if the logic yields negative sigma

    return mu, sigma

def build_token_bank(
    text_path: str,
    tokenizer_name: str = "gpt2",
    out_path: str | None = None,
    dtype=np.uint32,
):
    """
    Read raw text, tokenize with `tokenizer_name`, and store a flat numpy
    array of token ids (uint32 by default) plus a small JSON side-car.

    >>> build_token_bank("war_and_peace.txt", "gpt2")
    """
    text = Path(text_path).read_text(encoding="utf-8")
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokens = tok(text, add_special_tokens=False)["input_ids"]
    tokens_np = np.asarray(tokens, dtype=dtype)

    out_path = out_path or (Path(text_path).with_suffix(".npy"))
    np.save(out_path, tokens_np)

    meta = {
        "tokenizer": tokenizer_name,
        "num_tokens": int(tokens_np.shape[0]),
        "source": str(text_path),
        "dtype": str(dtype),
    }
    with open(Path(out_path).with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {meta['num_tokens']:,} tokens ➜ {out_path}")

class RequestType:
    """
    Defines a particular 'type' of request, with min/max token lengths
    and a sampling method (e.g. 'uniform', 'lognormal_approx', 'ratio', etc.).
    """

    SUPPORTED_METHODS = {"uniform", "lognormal_approx", "ratio", "default"}

    def __init__(
        self,
        category_name: str,
        min_input_tokens: int = 0,
        max_input_tokens: int = 0,
        min_output_tokens: int = 0,
        max_output_tokens: int = 0,
        sampling_method: str = "default",
        dataset_name: str = None,
        **kwargs: Any
    ):
        """
        Args:
            category_name (str): A descriptive name for this request type.
            min_input_tokens (int): Minimum input token length.
            max_input_tokens (int): Maximum input token length.
            min_output_tokens (int): Minimum output token length.
            max_output_tokens (int): Maximum output token length.
            sampling_method (str): How to sample input/output lengths. One of:
                - "uniform": simple randint within [min, max]
                - "lognormal_approx": approximate lognormal fit based on [min, max]
                - "ratio": for tasks where output is a fraction of input
                - "default": same as uniform or your existing approach
            **kwargs: Optional parameters to control the sampling. For example:
                - mu_in, sigma_in, mu_out, sigma_out for lognormal
                - ratio_min, ratio_max for ratio-based, etc.
        """
        self.category_name = category_name
        self.min_in = min_input_tokens
        self.max_in = max_input_tokens
        self.min_out = min_output_tokens
        self.max_out = max_output_tokens

        if (dataset_name == "ShareGPT"):
            token_bank_path = os.path.join("/home/sychoy/vllm/samples", f"{dataset_name}.json")

            if not os.path.exists(token_bank_path):
                print(f"Token bank {token_bank_path} not found. Start to make token bank.")
                bank = self.save_token_bank(dataset_name, token_bank_path)
            else: 
                with open(token_bank_path, "r", encoding="utf-8") as f:
                    bank = [json.loads(line) for line in f]

            # trace_limit 조건을 만족하는 샘플만 필터링
            self.bank = [
                sample for sample in bank
                if (self.min_in <= sample["input_length"] <= self.max_in) and
                    (self.min_out <= sample["output_length"] <= self.max_out)
            ]

            print(f"{self.category_name} has token bank with {len(self.bank)} samples.")
        
        if sampling_method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported sampling_method: {sampling_method}")

        self.sampling_method = sampling_method
        self.sampler_kwargs = kwargs  # store them; parse inside _build_sampler_function

        # Internally configure the sampler for input and output
        self.sampler = self._build_sampler_function()

    def _build_sampler_function(self) -> Callable[[], Tuple[int, int]]:
        """
        Decide which sampler function to use based on self.sampling_method.
        Validate kwargs relevant to that method and warn about unknown keys.
        """
        if self.sampling_method == "uniform":
            self._check_extra_kwargs(valid_keys=set(), method="uniform")
            return self._uniform_sampler

        elif self.sampling_method == "lognormal_approx":
            # Known valid keys: mu_in, sigma_in, mu_out, sigma_out (all optional)
            self._check_extra_kwargs(
                valid_keys={"mu_in", "sigma_in", "mu_out", "sigma_out"},
                method="lognormal_approx"
            )
            mu_in, sigma_in = self._resolve_lognormal_params(
                side="in", min_val=self.min_in, max_val=self.max_in
            )
            mu_out, sigma_out = self._resolve_lognormal_params(
                side="out", min_val=self.min_out, max_val=self.max_out
            )
            return lambda: self._lognormal_sampler(mu_in, sigma_in, mu_out, sigma_out)

        elif self.sampling_method == "ratio":
            # Known valid keys: ratio_min, ratio_max
            self._check_extra_kwargs(valid_keys={"ratio_min", "ratio_max"}, method="ratio")
            return self._ratio_sampler

        else:
            # "default" or anything else -> treat as uniform
            self._check_extra_kwargs(valid_keys=set(), method="default")
            return self._uniform_sampler

    def _check_extra_kwargs(self, valid_keys: set, method: str):
        """
        Warns if self.sampler_kwargs contain keys that are not in valid_keys.
        """
        extra = set(self.sampler_kwargs.keys()) - valid_keys
        if extra:
            warnings.warn(
                f"Invalid kwargs for method '{method}': {extra}. "
                "They will be ignored."
            )

    def _resolve_lognormal_params(
        self, side: str, min_val: int, max_val: int
    ) -> Tuple[float, float]:
        """
        side: "in" or "out"
        Attempts to parse `mu_in`, `sigma_in`, or `mu_out`, `sigma_out` from self.sampler_kwargs.
        If not provided or invalid, approximate them from min_val, max_val.
        """
        # e.g., if side=="in", we look for "mu_in" and "sigma_in" in self.sampler_kwargs
        mu_key = f"mu_{side}"
        sigma_key = f"sigma_{side}"

        # check if user provided them
        if mu_key in self.sampler_kwargs or sigma_key in self.sampler_kwargs:
            # we only use them if both are present and are valid
            provided_mu = self.sampler_kwargs.get(mu_key, None)
            provided_sigma = self.sampler_kwargs.get(sigma_key, None)
            if (isinstance(provided_mu, (int, float)) and
                isinstance(provided_sigma, (int, float)) and
                provided_sigma > 0):
                return float(provided_mu), float(provided_sigma)
            else:
                warnings.warn(
                    f"Invalid or missing '{mu_key}'/'{sigma_key}' in sampler_kwargs; "
                    f"falling back to approximate lognormal params."
                )

        # Fallback: approximate
        mu, sigma = approximate_lognormal_params(min_val, max_val, percentile=0.95)
        return mu, sigma

    def _uniform_sampler(self) -> Tuple[int, int]:
        """Return (input_length, output_length) by uniform randint in each range."""
        input_length = random.randint(self.min_in, self.max_in)
        output_length = random.randint(self.min_out, self.max_out)
        return input_length, output_length

    def _lognormal_sampler(self, mu_in: float, sigma_in: float,
                           mu_out: float, sigma_out: float) -> Tuple[int, int]:
        """
        Sample from lognormal for input and output, then clamp.
        If user supplied mu_in/sigma_in in kwargs, we skip approximation.
        """
        # sample input
        while True:
            in_val = int(random.lognormvariate(mu_in, sigma_in))
            if self.min_in <= in_val <= self.max_in:
                break
        # sample output
        while True:
            out_val = int(random.lognormvariate(mu_out, sigma_out))
            if self.min_out <= out_val <= self.max_out:
                break
        return in_val, out_val

    def _ratio_sampler(self) -> Tuple[int, int]:
        """
        Example ratio-based approach:
          - sample input uniform in [min_in, max_in]
          - output is some fraction (e.g. ratio_min to ratio_max) of the input
          - fallback ratio if none is provided
        """
        input_length = random.randint(self.min_in, self.max_in)

        ratio_min = self.sampler_kwargs.get("ratio_min", 0.1)
        ratio_max = self.sampler_kwargs.get("ratio_max", 0.3)
        if ratio_min > ratio_max:
            warnings.warn(f"ratio_min ({ratio_min}) > ratio_max ({ratio_max}); swapping them.")
            ratio_min, ratio_max = ratio_max, ratio_min

        ratio = random.uniform(ratio_min, ratio_max)
        out_length = int(input_length * ratio)
        out_length = clamp(out_length, self.min_out, self.max_out)
        return input_length, out_length

    def generate_request(
        self,
        vocab: Tuple[int, int] = (200, 30_000),
        trace_limit: Optional[int] = None,
        token_bank_path: Optional[str] = None,
        slo_max = True,
    ):
        """
        Creates a single `Request` object using the configured sampler method
        to select input_length and output_length. Then optionally checks trace_limit.
        """
        # 1) Sample the lengths
        while True:
            input_length, output_length = self.sampler()
            if trace_limit is not None:
                if (input_length + output_length) > trace_limit:
                    continue  # re-sample
            break
        if slo_max: 
            total_length = input_length + output_length 
            slo = total_length * PROFILED_A + PROFILED_B 
        # 2) get the token source
        
        if self.bank is not None:
            bank_size = len(self.bank)

            idx = np.random.randint(0, bank_size)
            input_length = self.bank[idx]["input_length"]
            output_length = self.bank[idx]["output_length"]
            token_ids = self.bank[idx]["input_token_ids"]

        if token_bank_path is None:
            token_ids = [random.randint(*vocab) for _ in range(input_length)]
        
        if slo_max:
            return Request(
                category=self.category_name,
                input_length=input_length,
                output_length=output_length,
                token_ids=token_ids,
                slo=slo,
            ) 
        else:
            return Request(
                category=self.category_name,
                input_length=input_length,
                output_length=output_length,
                token_ids=token_ids
            )
    
    def save_token_bank(self, token_bank, token_bank_path):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        parsed_data = []

        test = 1
        if token_bank == "ShareGPT":
            json_path = "/home/sychoy/vllm/downloads/ShareGPT_V3_unfiltered_cleaned_split.json"  # JSON 다운로드 위치
            with open(json_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)

            # Filter out the conversations with less than 2 turns.
            dataset = [data for data in dataset if len(data["conversations"]) >= 2]
            # Only keep the first two turns of each conversation.
            dataset = [(data["conversations"][0]["value"],
            data["conversations"][1]["value"]) for data in dataset]

            for i in tqdm(range(len(dataset))):
                prompt = dataset[i][0]
                completion = dataset[i][1]

                if (test < 3):
                    print(f"print {test}th data")
                    print(dataset[i])
                    print()
                    print()
                    test += 1

                if not prompt or not completion:
                    continue

                input_tokens = tokenizer.encode(prompt, truncation=True, max_length=2048)

                parsed_data.append({
                    "input_length": len(prompt),
                    "output_length": len(completion),
                    "input_token_ids": input_tokens,
                })

        else:
            raise ValueError(f"Wrong token bank name: {token_bank}")

        os.makedirs(os.path.dirname(token_bank_path), exist_ok=True)
        with open(token_bank_path, "w", encoding="utf-8") as f:
            for item in parsed_data:
                f.write(json.dumps(item) + "\n")

        return parsed_data

def save_to_json(
    filename: str, 
    batch_size: int,
    vocab: Tuple[int],
    request_type_probs: dict,
    requests: Dict[str, Request],
    skip_token_ids: bool = False,
) -> None:
    """
    Serialize this Trace object into a JSON file in a compact form:
    - "arrival_pattern" is a one-liner dict with "type" and (optionally) "params".
    - "batch_size", "request_type_probs", "vocab" are also written in a compact style.
    - "requests" is a dictionary. Each key is "request_i", each value is either:
        - single-line if token_ids are skipped or absent
        - two-line if token_ids exist
    """
    # 1) Convert request_type_probs into a list of (category_name, prob) for JSON
    if isinstance(request_type_probs, dict):
        request_type_probs_data = [
            (rt.category_name, prob) for (rt, prob) in request_type_probs.items()
        ]
    else:
        request_type_probs_data = [
            (rt.category_name, prob) for (rt, prob) in request_type_probs
        ]

    data = {
        "batch_size": batch_size,
        "request_type_probs": request_type_probs_data,
        "vocab": vocab
    }

    # 2) Prepare each request in a dictionary
    #    We'll do a manual approach to keep it compact, especially for token_ids.
    requests_items = list(requests.items())  # [(req_id, RequestObj), ...]
    # We'll need them in a stable order (e.g. sorted by req_id or arrival_time).
    # If you prefer arrival-time order, do:
    # requests_items.sort(key=lambda x: x[1].arrival_time)
    # For now, let's assume they're already in ascending "request_i" keys.

    # 3) Write top-level JSON keys manually
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("{\n")

        # a) arrival_pattern

        # b) batch_size
        f.write(f'  "batch_size":{data["batch_size"]},\n')

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
                "output_length": req_obj.output_length
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


import random

def save_requests(
    filename: str,
    request_type_dict: dict,
    num_requests: int,
    max_parallel,
    skip_token_ids: bool = True,
    static: bool = False,  # 추가된 플래그
):
    requests = []

    cumulative_probs = []
    running_sum = 0.0
    sorted_rt = sorted(request_type_dict.items(), key=lambda x: x[1], reverse=True)
    for rt, p in sorted_rt:
        running_sum += p
        cumulative_probs.append((rt, running_sum))

    if static:
        num_unique_requests = (num_requests + max_parallel - 1) // max_parallel  # ceil
        unique_requests = []

        for _ in range(num_unique_requests):
            r = random.random()
            for (rt, cum_p) in cumulative_probs:
                if r <= cum_p:
                    chosen_rt = rt
                    break
            req = chosen_rt.generate_request(
                vocab=(200, 30000),
                slo_max=True
            )
            unique_requests.append(req)

        for req in unique_requests:
            requests.extend([req] * max_parallel)
        requests = requests[:num_requests]  # 정확히 num_requests 개수만 유지

    else:
        for i in range(num_requests):
            r = random.random()
            for (rt, cum_p) in cumulative_probs:
                if r <= cum_p:
                    chosen_rt = rt
                    break
            req = chosen_rt.generate_request(
                vocab=(200, 30000),
                slo_max=True
            )
            requests.append(req)

    requests_dict = {}
    for i, req in enumerate(requests):
        requests_dict[f"request_{i}"] = req

    # Save to JSON
    save_to_json(
        filename,
        batch_size=max_parallel,
        vocab=(200, 30000),
        request_type_probs=request_type_dict,
        requests=requests_dict,
        skip_token_ids=skip_token_ids
    )


if __name__ == "__main__":
    
    # predefine some requests 
    # shortshort = RequestType(
    #     category_name="Short-Short ShareGPT",
    #     min_input_tokens=600,   # short question
    #     max_input_tokens=1000,
    #     min_output_tokens=600,  # short answer
    #     max_output_tokens=1000,
    #     sampling_method="uniform",
    #     dataset_name="ShareGPT"
    # )
    shortlong = RequestType(
        category_name="Short-Long ShareGPT",
        min_input_tokens=700,
        max_input_tokens=1500,
        min_output_tokens=3500,
        max_output_tokens=5500,
        sampling_method="uniform",
        dataset_name="ShareGPT"
    )
    # longlong = RequestType(
    #     category_name="Long-Long ShareGPT",
    #     min_input_tokens=2000,
    #     max_input_tokens=8000,
    #     min_output_tokens=2000,
    #     max_output_tokens=10000,
    #     sampling_method="uniform",
    #     dataset_name="ShareGPT"
    # )
    longshort = RequestType(
        category_name="Long-Short ShareGPT",
        min_input_tokens=3500,
        max_input_tokens=5500,
        min_output_tokens=700,
        max_output_tokens=1500,
        sampling_method="uniform",
        dataset_name="ShareGPT"
    )

    probs_dict = {t:prob for (t,prob) in zip([shortlong, longshort], [0.5, 0.5])}
    for k,v in probs_dict.items():
        if v == 0:
            del probs_dict[k]
    skip_token_ids = True

    save_requests(
        filename="/home/sychoy/vllm/trace_pool/static_8k_pressure/request.json",
        request_type_dict=probs_dict,
        num_requests=12,
        max_parallel=4,
        skip_token_ids=skip_token_ids,
        # static=True
    )