import os
import time
import json
import logging
import csv
from collections import defaultdict
import pathlib
import psutil
import tracemalloc
import pandas as pd
import torch
import sys

from vllm.engine.llm_engine import LLMEngine
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams
from vllm.logger import init_logger
from vllm.worker.distn.solver import ProfileBasedEstimator
from trace_generator import Trace

torch.set_printoptions(edgeitems=2, linewidth=120, sci_mode=True)

logger = init_logger("vllm")

def log_memory_usage(tag="", process=None):
    """Log current CPU memory usage"""
    if process is None:
        process = psutil.Process()
    
    memory_info = process.memory_info()
    rss_mb = memory_info.rss / 1024 / 1024  # MB
    vms_mb = memory_info.vms / 1024 / 1024  # MB
    
    # System memory
    sys_memory = psutil.virtual_memory()
    available_mb = sys_memory.available / 1024 / 1024
    used_percent = sys_memory.percent
    total_mb = sys_memory.total / 1024 / 1024
    used_mb = sys_memory.used / 1024 / 1024
    free_mb = sys_memory.free / 1024 / 1024
    active_mb = getattr(sys_memory, "active", 0) / 1024 / 1024
    inactive_mb = getattr(sys_memory, "inactive", 0) / 1024 / 1024
    buffers_mb = getattr(sys_memory, "buffers", 0) / 1024 / 1024
    cached_mb = getattr(sys_memory, "cached", 0) / 1024 / 1024
    shared_mb = getattr(sys_memory, "shared", 0) / 1024 / 1024
    slab_mb = getattr(sys_memory, "slab", 0) / 1024 / 1024

    logger.critical(
        f"[MEMORY {tag}] RSS: {rss_mb:.1f}MB, VMS: {vms_mb:.1f}MB, "
        f"System Used: {used_percent:.1f}%, Available: {available_mb:.1f}MB, "
        f"Total: {total_mb:.1f}MB, Used: {used_mb:.1f}MB, Free: {free_mb:.1f}MB, "
        f"Active: {active_mb:.1f}MB, Inactive: {inactive_mb:.1f}MB, "
        f"Buffers: {buffers_mb:.1f}MB, Cached: {cached_mb:.1f}MB, "
        f"Shared: {shared_mb:.1f}MB, Slab: {slab_mb:.1f}MB"
    )
    return rss_mb

def start_memory_profiling():
    """Start tracemalloc for detailed memory tracking"""
    tracemalloc.start()
    logger.critical("[MEMORY] Started tracemalloc profiling")

def log_top_memory_allocations(limit=10):
    """Log top memory allocations"""
    if not tracemalloc.is_tracing():
        return
    
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    logger.critical(f"[MEMORY] Top {limit} memory allocations:")
    for stat in top_stats[:limit]:
        logger.critical(f"  {stat}")

def get_memory_diff(snapshot_before):
    """Get memory difference since snapshot_before"""
    if not tracemalloc.is_tracing():
        return None
    
    snapshot_after = tracemalloc.take_snapshot()
    top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    
    logger.critical("[MEMORY] Top memory differences:")
    for stat in top_stats[:10]:
        logger.critical(f"  {stat}")

MODEL = os.environ.get("MODEL")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))
BLOCK_SIZE  = 16 
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", 32)) # Xinyue: HARDCODE, should be passed from model config to block manager

class DelaySimulator:
    """
    Simulates per-request token emission and tracks SLO (Service Level Objective) violations.

    The simulator maintains a simple per-request rate limiter where the nominal
    per-token SLO is scaled by `slo_ratio`. Each decode step, tokens produced
    by the engine are first deposited via `on_token()`. Later, `pop()` releases
    the amount that would have been emitted by a virtual queue running at `v`
    (tokens per second), potentially registering SLO violations when backlog
    exceeds the releasable tokens.

    Public attributes are intentionally kept for compatibility with existing code:
        - deposit_enabled (bool)
        - v (dict[str, float]): request id -> tokens/second
        - last_time (dict[str, float]): request id -> last accounting time
        - last_decode_time (float)
        - deposit (defaultdict[str, int]): request id -> queued tokens
        - violations (defaultdict[str, int]): request id -> violation count
        - slo_ratio (float)
    """

    def __init__(self, slo_ratio: float = 2.5, deposit_enabled: bool = True) -> None:
        """
        Args:
            slo_ratio: Multiplier applied to the per-token SLO to obtain the
                effective deadline. Larger values relax the SLO.
            deposit_enabled: If False, each `on_token()` call behaves as if any
                existing backlog was flushed immediately and sets deposit to 1.
        """
        self.deposit_enabled: bool = deposit_enabled
        self.v: dict[str, float] = {}              # rid -> tokens/sec
        self.last_time: dict[str, float] = {}      # rid -> last time accounting was done
        self.last_decode_time: float = 0.0
        self.deposit: defaultdict[str, int] = defaultdict(int)
        self.violations: defaultdict[str, int] = defaultdict(int)
        self.slo_ratio: float = float(slo_ratio)

    # ------------------------------- Registration -------------------------------

    def register(self, rid: str, slo_s_per_token: float) -> float:
        """
        Register a request with its base SLO (seconds per token).

        The effective SLO used by the simulator is `slo_s_per_token * slo_ratio`.

        Returns:
            The effective SLO (seconds per token) used for this rid.
        """
        if slo_s_per_token <= 0:
            raise ValueError("SLO must be > 0 (seconds per token).")
        eff_s_per_token = slo_s_per_token * self.slo_ratio
        self.v[rid] = 1.0 / eff_s_per_token
        # initialize tracking
        self.deposit[rid] = 0
        self.violations[rid] = 0
        # do not set last_time yet; it will be set on first token/pop
        return eff_s_per_token

    # ------------------------------- Accounting -------------------------------

    def on_token(self, rid: str, step_time: float) -> None:
        """
        Record that one token was produced for `rid` at wall-clock `step_time` (seconds).

        If deposit is enabled, backlog increases by 1; otherwise, it is set to 1,
        modeling immediate flushing of prior backlog.
        """
        if rid not in self.v:
            # Allow late registration for robustness, but warn loudly.
            logger.warning("[DelaySimulator] on_token for unregistered rid=%s; "
                           "auto-registering with default v=inf (no throttling).", rid)
            # Set a huge rate (no throttling) to avoid blocking the pipeline.
            self.v[rid] = float("inf")

        before = self.deposit[rid]
        if self.deposit_enabled:
            self.deposit[rid] = before + 1
        else:
            # exactly one token; we assume anything pending is flushed immediately
            self.deposit[rid] = 1

        if rid not in self.last_time:
            self.last_time[rid] = step_time
            logger.debug("[DelaySimulator.on_token] first token for %s, last_time=%.6f",
                         rid, step_time)

        logger.debug("[DelaySimulator.on_token] rid=%s @ %.6f: deposit %d -> %d",
                     rid, step_time, before, self.deposit[rid])

    def pop(self, step_time: float, solver_time: float = 0.0) -> list[tuple[str, int]]:
        """
        Release tokens according to each request's emission rate.

        Args:
            step_time: Current wall-clock time (seconds).
            solver_time: Time spent in solver this step (seconds). This portion
                is subtracted from available time for emission accounting.

        Returns:
            A list of (rid, released_count) for requests that released > 0 tokens.
        """
        pops: list[tuple[str, int]] = []
        # snapshot only for compact logging
        deposits_snapshot = {k: int(v) for k, v in self.deposit.items()}

        rid_log: list[str] = []
        for rid, dep in list(self.deposit.items()):
            v_tps = self.v.get(rid)
            if v_tps is None or v_tps <= 0:
                # If we have no rate, we cannot release; keep last_time fresh.
                self.last_time[rid] = step_time
                continue
            if dep <= 0:
                self.last_time[rid] = step_time
                continue

            last = self.last_time.get(rid, step_time)
            dt = max(0.0, step_time - last - max(0.0, solver_time))
            n = int(dt * v_tps)  # releasable tokens in this step
            if n <= 0:
                continue

            to_rel = min(n, dep)
            # violation if backlog exceeded the releasable amount
            if dep > n:
                self.violations[rid] += 1
                rid_log.append(f"{rid}(SLO {dep}/{n}), {self.violations[rid]} violations so far")
            else:
                rid_log.append(f"{rid}({to_rel})@{v_tps:.2f}t/s")

            # drain deposit
            self.deposit[rid] = dep - to_rel

            # move accounting time
            if self.deposit[rid] == 0:
                # backlog cleared → reset stopwatch
                self.last_time[rid] = step_time
            else:
                # advance by the amount of time that would emit `to_rel` tokens
                self.last_time[rid] = last + (to_rel / v_tps)

            pops.append((rid, to_rel))

        if rid_log:
            logger.info("[DelaySimulator.pop] deposits=%s | released: %s",
                        deposits_snapshot, " | ".join(rid_log))
        return pops

    # ------------------------------- Introspection -----------------------------

    def violation_count(self, rid: str) -> int:
        """Return the number of recorded SLO violations for `rid`."""
        return int(self.violations.get(rid, 0))

    def finish(self, rid: str) -> None:
        """
        Called when a request finishes: clear remaining deposit and clean up tracking state.
        """
        if rid in self.deposit:
            logger.info("[DelaySimulator.finish] clearing deposit for %s: was %d",
                        rid, self.deposit[rid])
            del self.deposit[rid]
        if rid in self.last_time:
            logger.info("[DelaySimulator.finish] removing last_time entry for %s", rid)
            del self.last_time[rid]
        self.v.pop(rid, None)
        self.violations.pop(rid, None)

    def stats(self) -> dict[str, int]:
        """Return a copy of the current deposit map (rid → remaining tokens)."""
        return dict(self.deposit)

def run_inference_step_mode(engine,
                            trace_obj,
                            csv_path: str | os.PathLike | None = None,
                            enable_deposit: bool = False,
                            estimator=None,
                            slo_ratio: float = 2.5):
    """
    Step-based inference driver that consumes a Trace-like object and iterates
    `engine.step()` until all requests complete.

    Expectations on `trace_obj`:
      - `trace_obj.requests` is a dict: {request_id -> Request}
      - Each Request has: arrival_time, input_length, output_length, (optional) token_ids, (optional) slo
      - `trace_obj.num_gpu_blocks_override` may exist and is used to compute a global token_limit

    CSV output:
      - Appends one row per finished request to `csv_path` (header written if file is new)
      - Columns match the previous implementation for backward compatibility
    """
    assert estimator is not None, "estimator is required"
    logger = logging.getLogger("vllm")

    # --- Prepare requests in arrival order ---
    requests_sorted = sorted(trace_obj.requests.items(),
                             key=lambda kv: kv[1].arrival_time)
    queue: list[tuple[str, object]] = list(requests_sorted)

    # Per-request bookkeeping
    request_metadata: dict[str, dict] = {}
    request_output: defaultdict[str, list] = defaultdict(list)

    # Timing accumulators (non-overlapping)
    overall_wall = prefill_wall = decode_wall = 0.0
    cumulative_steps = 0
    finished_tokens = finished_decode_tokens = finished_prefill_tokens = 0

    running_requests: set[str] = set()
    received_requests: list[str] = []        # first-appearance tracker to split prefill/decode
    consecutive_no_output = 0
    sum_solver_time = 0.0

    # --- Local helpers ---------------------------------------------------------
    def _enqueue_batch(batch: list[tuple[str, object]]) -> None:
        """Enqueue a list of (request_id, request_obj) into the engine and register to DelaySimulator."""
        for rid, req in batch:
            # Metadata scaffold
            request_metadata[rid] = {
                "arrival_time": time.time(),
                "scheduled_time": None,
                "first_token_time": None,
                "finished_time": None,
                "token_timestamps": [],
                "profiled_tbt": [],
                "time_between_tokens": [],
                "solver_time": [],
                "solver_estimated_time": [],
                "decode_length": 0,
                "expected_output_length": req.output_length,
                "stall_times": [],
                "stall_durations": [],
                "stall_duration": 0,
                "prompt_length": req.input_length,
            }

            # Ensure prompt token ids exist (Trace may already populate these)
            if not hasattr(req, "token_ids") or not req.token_ids:
                token_id_range = (200, 20000)
                req.token_ids = torch.randint(low=token_id_range[0],
                                              high=token_id_range[1],
                                              size=(req.input_length,),
                                              dtype=torch.int).tolist()

            prompt_obj = TokensPrompt(prompt_token_ids=req.token_ids)
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=req.output_length,
                stop=[],
                stop_token_ids=[],
                ignore_eos=True,
            )
            engine.add_request(rid, prompt_obj, sampling_params)

            # Register to DelaySimulator with either request-provided SLO or profiled estimate
            if hasattr(req, "slo") and req.slo is not None:
                max_slo = req.slo
            else:
                max_slo = estimator.estimate_by_profiled_results(
                    tokens=token_limit, which="NoPrefetch", mode="linear"
                )
            effective_slo = sim.register(rid, max_slo)
            logger.debug("Enqueued %s | max_slo=%.6f s/tok | effective_slo=%.6f s/tok",
                         rid, max_slo, effective_slo)

    # DelaySimulator
    sim = DelaySimulator(slo_ratio=slo_ratio, deposit_enabled=enable_deposit)

    # CSV writer
    csv_path = pathlib.Path(csv_path or "metrics.csv")
    write_header = not csv_path.exists()
    csv_fh = csv_path.open("a", newline="", buffering=1)   # line-buffered
    csv_writer = csv.DictWriter(csv_fh, fieldnames=[
        "request_id", "arrival_time", "first_scheduled_time",
        "finished_time", "stall_times",
        "time_to_first_token", "slo_threshold",
        "slo_violations",
        "stall_duration", "decode_length",
        "end_to_end_time", "decode_time",
        "time_per_output_token", "time_between_tokens",
        "finish_reason", "solver_time",
        "solver_estimated_time",
        "profiled_tbt", "expected_output_length",
        "stall_durations",
    ])
    if write_header:
        csv_writer.writeheader()

    # --- Provide global hints to cache engine (kept from original behavior) ---
    trace_token_count = sum(req.input_length + req.output_length for _, req in requests_sorted)
    trace_avg_token_count = trace_token_count / max(1, len(requests_sorted))
    engine.model_executor.driver_worker.cache_engine[0].flexgen_tok_estimate = trace_avg_token_count

    token_limit = (trace_obj.num_gpu_blocks_override or 500) * BLOCK_SIZE
    max_comp_time = estimator.estimate_by_profiled_results(tokens=token_limit,
                                                           which="NoPrefetch",
                                                           mode="linear")
    engine.model_executor.driver_worker.cache_engine[0].max_slo = max_comp_time * sim.slo_ratio
    engine.model_executor.driver_worker.cache_engine[0].max_comp_time = max_comp_time
    engine.model_executor.driver_worker.cache_engine[0].estimator = estimator

    # Step loop
    step_count = 1
    memory_check_interval = 1   # keep frequent checks for debugging
    start_time = end_time = None

    while queue or request_metadata:
        # Memory monitoring
        if step_count % memory_check_interval == 0:
            log_memory_usage(f"STEP_{step_count}")

        # Activate arrivals whose arrival_time <= current step
        ready = [(rid, req) for (rid, req) in queue if req.arrival_time <= cumulative_steps]
        if ready:            
            _enqueue_batch(ready)            
            # Remove enqueued from queue
            queue = [(rid, req) for (rid, req) in queue if req.arrival_time > cumulative_steps]

        # push DelaySimulator state into scheduler for policy decisions
        engine.scheduler[0].deposit_map = sim.stats()
        engine.scheduler[0].slo_from_delaysim = sim.v

        # One engine step
        torch.cuda.synchronize()
        step_start = time.time()
        step_outputs = engine.step()
        torch.cuda.synchronize()
        step_end = time.time()
        step_count += 1
        elapsed_time_step = step_end - step_start
        overall_wall += elapsed_time_step
        cumulative_steps += 1

        # Classify this step as prefill or decode by first-appearance rule
        prefill_rids: list[str] = []
        decode_rids: list[str] = []
        finished_rids: list[str] = []
        step_tokens = 0

        # Defaults for this step
        solver_time = 0.0
        solver_estimated_time = 0.0
        profiled_res = 0.0

        if step_outputs:
            first_rid = step_outputs[0].request_id
            is_decode = first_rid in received_requests

            if is_decode:
                decode_rids = list({o.request_id for o in step_outputs})
                decode_wall += elapsed_time_step

                # total tokens so far for those requests (prompt + decode so far + this token each)
                step_tokens = sum(
                    request_metadata[rid]["prompt_length"] + request_metadata[rid]["decode_length"] + 1
                    for rid in decode_rids
                )
                temp_step_tokens = {
                    rid: request_metadata[rid]["prompt_length"] + request_metadata[rid]["decode_length"] + 1
                    for rid in decode_rids
                }
                logger.critical("Step %d, step_tokens = %s", step_count, temp_step_tokens)

                profiled_res = estimator.estimate_by_profiled_results(tokens=step_tokens,
                                                                      which="NoPrefetch",
                                                                      mode="linear")
                # pull solver timings if provided
                solver_time = getattr(step_outputs[0], "solver_time", 0.0) or 0.0
                solver_estimated_time = getattr(step_outputs[0], "solver_estimated_time", 0.0) or 0.0
            else:
                prefill_rids = list({o.request_id for o in step_outputs})
                prefill_wall += elapsed_time_step
                step_tokens = sum(request_metadata[rid]["prompt_length"] for rid in prefill_rids)
                profiled_res = estimator.estimate_by_profiled_results(tokens=step_tokens,
                                                                      which="NoPrefetch",
                                                                      mode="linear")
                solver_time = 0.0
                solver_estimated_time = 100.0

            consecutive_no_output = 0
        else:
            # idle step
            solver_time = 0.0
            solver_estimated_time = 100.0
            if queue:
                time.sleep(0.1)  # avoid busy waiting while waiting for next arrivals
            else:
                consecutive_no_output += 1
                if consecutive_no_output > 5:
                    logger.critical("No output for 5 consecutive steps, stopping simulation.")
                    for rid in list(request_metadata.keys()):
                        logger.critical("[finish] clearing deposit for %s: was %d", rid, sim.deposit.get(rid, 0))
                        sim.finish(rid)
                        request_metadata.pop(rid, None)
                        logger.critical("[finish] removing last_time entry for %s", rid)
                    print(f"Failure (due to memory limits): {len(request_metadata)}")
                    break

        if elapsed_time_step < solver_time and step_outputs:
            # Keep original safety log (do not hard fail)
            sched_time = getattr(step_outputs[0].metrics, "scheduler_time", 0.0)
            logger.critical(
                "Elapsed time %.3f s < solver time %.3f s (scheduler_time=%.3f); skipping sanity assertion.",
                elapsed_time_step, solver_time, sched_time
            )

        sum_solver_time += solver_time

        # ---- Process outputs ---------------------------------------------------
        for output in step_outputs:
            rid = output.request_id

            # Prefill: first token seen for rid in received_requests bookkeeping
            if prefill_rids:
                sim.last_decode_time = 0.0
                if rid not in received_requests:
                    received_requests.append(rid)
                finished_tokens += len(output.prompt_token_ids)
                finished_prefill_tokens += len(output.prompt_token_ids)
                running_requests.add(rid)
            else:
                # Decode
                sim.last_decode_time = elapsed_time_step
                sim.on_token(rid, step_end)

                # timestamps & metrics
                request_metadata[rid]["token_timestamps"].append(step_end)
                request_metadata[rid]["time_between_tokens"].append(elapsed_time_step)
                request_metadata[rid]["solver_time"].append(solver_time)
                request_metadata[rid]["solver_estimated_time"].append(solver_estimated_time)
                request_metadata[rid]["profiled_tbt"].append(profiled_res)
                request_metadata[rid]["decode_length"] += 1
                finished_tokens += 1
                finished_decode_tokens += 1
                request_output[rid].append(output)

            step_tokens += request_metadata[rid]["prompt_length"]
            step_tokens += request_metadata[rid]["decode_length"]

            # Finished?
            if output.finished:
                # derive finish_reason compatible with previous version
                if len(output.outputs[0].token_ids) < request_metadata[rid]["expected_output_length"]:
                    finish_reason = "length_capped"
                else:
                    finish_reason = output.outputs[0].finish_reason

                logger.critical("Finished request %s at step %d, finish_reason=%s",
                                rid, cumulative_steps, finish_reason)
                logger.critical("%s prompt: %d tokens; %s",
                                rid, len(output.prompt_token_ids), str(output.prompt_token_ids[:20]))
                logger.critical("%s output: %d %s",
                                rid, len(output.outputs[0].token_ids), str(output.outputs[0].token_ids[:20]))
                logger.critical("%s text: %s", rid, output.outputs[0].text[:20])

                if len(output.outputs[0].token_ids) > 0:
                    running_requests.discard(rid)
                else:
                    logger.critical("Request %s finished with no output tokens, skipping metrics row.", rid)

                # Metrics row (preserve previous schema)
                if len(output.outputs[0].token_ids) > 0:
                    arrival_time_local = request_metadata[rid]["arrival_time"]
                    finished_time_local = step_end
                    token_ts = request_metadata[rid]["token_timestamps"]
                    if token_ts:
                        first_token_time_local = token_ts[0]
                    else:
                        first_token_time_local = finished_time_local

                    decode_len = request_metadata[rid]["decode_length"]
                    if start_time is None or arrival_time_local < start_time:
                        start_time = arrival_time_local
                    if end_time is None or finished_time_local > end_time:
                        end_time = finished_time_local

                    row = {
                        "request_id": rid,
                        "arrival_time": arrival_time_local - (start_time or 0),
                        "first_scheduled_time": 0,
                        "finished_time": finished_time_local - (start_time or 0),
                        "stall_times": json.dumps(request_metadata[rid]["stall_times"]),
                        "time_to_first_token": first_token_time_local - arrival_time_local,
                        "slo_threshold": 1 / sim.v[rid],
                        "slo_violations": sim.violation_count(rid),
                        "stall_duration": request_metadata[rid]["stall_duration"],
                        "decode_length": decode_len,
                        "end_to_end_time": finished_time_local - arrival_time_local,
                        "decode_time": finished_time_local - first_token_time_local,
                        "time_per_output_token": (
                            sum(request_metadata[rid]["time_between_tokens"]) / decode_len if decode_len > 0 else 0.0
                        ),
                        "finish_reason": finish_reason,
                        "solver_time": json.dumps(request_metadata[rid]["solver_time"]),
                        "solver_estimated_time": json.dumps(request_metadata[rid]["solver_estimated_time"]),
                        "time_between_tokens": json.dumps(request_metadata[rid]["time_between_tokens"]),
                        "profiled_tbt": request_metadata[rid]["profiled_tbt"],
                        "expected_output_length": request_metadata[rid]["expected_output_length"],
                        "stall_durations": json.dumps(request_metadata[rid]["stall_durations"]),
                    }
                    csv_writer.writerow(row)
                else:
                    row = {
                        "request_id": rid,
                        "arrival_time": -1,
                        "first_scheduled_time": 0,
                        "finished_time": -1,
                        "stall_times": [-1],
                        "time_to_first_token": -1,
                        "slo_threshold": 1 / sim.v[rid],
                        "slo_violations": -1,
                        "stall_duration": -1,
                        "decode_length": 0,
                        "end_to_end_time": -1,
                        "decode_time": -1,
                        "time_per_output_token": [-1],
                        "finish_reason": "length_capped",
                        "solver_time": [-1],
                        "solver_estimated_time": [-1],
                        "time_between_tokens": [-1],
                        "profiled_tbt": [-1],
                        "expected_output_length": request_metadata[rid]["expected_output_length"],
                        "stall_durations": [-1],
                    }
                    csv_writer.writerow(row)

                finished_rids.append(rid)
                logger.info("Finished request %s with %d decode tokens", rid, request_metadata[rid]["decode_length"])
                sim.finish(rid)

        # Step-level logs
        if prefill_rids:
            logger.info("Prefill step %d: %s", cumulative_steps, ", ".join(map(str, prefill_rids)))
        if decode_rids:
            logger.info("Decode  step %d: %s", cumulative_steps, ", ".join(map(str, decode_rids)))

        # Idle fast-forward if nothing is running but future arrivals exist
        if not running_requests and queue:
            next_arrival = queue[0][1].arrival_time
            if cumulative_steps < next_arrival:
                logger.debug("Fast-forwarding from step %d → %d (idle gap)", cumulative_steps, next_arrival)
                cumulative_steps = next_arrival
                consecutive_no_output = 0
            continue

        # Feed scheduler with recent step stats and clear finished
        engine.scheduler[0].last_decode_time = sim.last_decode_time
        engine.scheduler[0].step_tokens = step_tokens
        for rid in finished_rids:
            request_metadata.pop(rid, None)

        # Apply DelaySimulator releases at the end of the step
        _ = sim.pop(step_end, solver_time=solver_time)

    # --- Post processing: persist metrics and show summary ---------------------
    print("All requests completed. Now saving CSV...")
    df = pd.DataFrame(csv.DictReader(csv_path.open("r")))
    # Sort by numeric suffix if request IDs are like "request_12"
    def _numeric_part(x: str) -> int:
        try:
            return int(x.split("_")[-1])
        except Exception:
            return 999999
    df = df.sort_values(by=["request_id"], key=lambda s: s.apply(_numeric_part))
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")

    # Overall throughput (non-overlapping)
    if start_time and end_time and (end_time > start_time):
        print("------Wall-clock stats (non-overlapping)------")
        print(f"Overall runtime  : {overall_wall:.3f} s")
        if prefill_wall > 0:
            print(f"Prefill time     : {finished_prefill_tokens} tokens over {prefill_wall:.3f} s "
                  f"({finished_prefill_tokens / prefill_wall: .2f} t/s)")
        if decode_wall > 0:
            print(f"Decode  time     : {finished_decode_tokens} tokens over {decode_wall:.3f} s "
                  f"({finished_decode_tokens / decode_wall: .2f} t/s)")
        print(f"Preemptions count: {engine.scheduler[0].num_cumulative_preemption}")
        print(f"Overall solver time: {sum_solver_time:.3f} s")
    else:
        print("No valid start/end time for throughput calculation.")
        
def main(configs):
    prompt_path = configs.config_file
    num_gpu_blocks_override = None
    trace = Trace.load_from_json(prompt_path)

    # Print trace attributes excluding methods and private attributes
    for attribute in dir(trace):
        if not attribute.startswith("__") and not callable(getattr(trace, attribute)) and "requests" not in attribute:
            print(f"{attribute}: {getattr(trace, attribute)}")

    # Handle gpu_memory_utilization and num_gpu_blocks_override (robust to None)
    num_gpu_blocks_override = getattr(trace, "num_gpu_blocks_override", None)

    # Use trace value if present; otherwise pick a safe default (no CLI override).
    trace_gmu = getattr(trace, "gpu_memory_utilization", None)
    if trace_gmu is not None:
        gpu_memory_utilization = float(trace_gmu)
    else:
        # If user specified explicit block override, pass a tiny utilization to bypass vLLM's allocator check;
        # otherwise choose a conservative default.
        gpu_memory_utilization = 0.01 if num_gpu_blocks_override is not None else 0.90

    # Determine max_model_len
    trace_mml = getattr(trace, "max_model_len", None)
    if trace_mml is not None:
        max_model_len = int(trace_mml)
    elif num_gpu_blocks_override is not None:
        max_model_len = int(num_gpu_blocks_override) * BLOCK_SIZE
    else:
        # final fallback
        max_model_len = 8096

    # Determine prefetch mode and related configurations (set robust defaults first)
    prefetch_mode = getattr(configs, "prefetch_mode", "none")
    prefetch_distance = getattr(configs, "prefetch_distance", 0)
    is_monolithic_distn = getattr(configs, "is_monolithic_distn", True)
    if prefetch_mode == "distn":
        # keep is_monolithic_distn as provided
        pass
    elif prefetch_mode == "static":
        # keep prefetch_distance as provided
        pass
    
    flattened_cache = configs.flattened_cache if hasattr(configs, "flattened_cache") else False
    merge_prefetch_buffer = configs.merge_prefetch_buffer if hasattr(configs, "merge_prefetch_buffer") else True
    pause_and_resume = configs.pause_and_resume if hasattr(configs, "pause_and_resume") else False
    batch_size = trace.batch_size if hasattr(trace, "batch_size") else BATCH_SIZE
    static_batching = configs.static_batching if hasattr(configs, "static_batching") else False
    removable_cache = configs.removable_cache if hasattr(configs, "removable_cache") else False
    uniform_solver = configs.uniform_solver if hasattr(configs, "uniform_solver") else False
    pause_strategy = getattr(configs, "pause_strategy", "longest")

    if configs.profiled_results:
        p_path = configs.profiled_results
    else:
        p_path = "/home/heelim/vllm/benchmark/scripts/profiling_data/profiled_results_A6000.json"
    estimator = ProfileBasedEstimator(p_path)
    print("Available Profiled Fitters:", estimator.available_profiles())

    t_bt = estimator.estimate_by_profiled_results(tokens=2048,
                                            which="NoPrefetch",
                                            mode="linear")    
    print(f"Estimated Δt per token @2048 tokens ≈ {t_bt:.4f} s")

    # Retrieve goodness-of-fit if you wish
    print("R² for that fit:", estimator.r2("NoPrefetch", "linear"))
    print(f"model: {configs.model}")
    print(f"batch_size: {batch_size}")
    print(f"prefetch_mode: {prefetch_mode}")
    print(f"prefetch_distance: {prefetch_distance}")
    print(f"gpu_memory_utilization: {gpu_memory_utilization}")
    print(f"num_gpu_blocks_override: {num_gpu_blocks_override}")
    print(f"merge_prefetch_buffer: {merge_prefetch_buffer}")
    print(f"pause_and_resume: {pause_and_resume}")
    print(f"pause_strategy: {pause_strategy}")
    print(f"static_batching: {static_batching}")
    print(f"removable_cache: {removable_cache}")
    print(f"uniform_solver: {uniform_solver}")
    print(f"slo_ratio: {configs.slo_ratio}")
    
    if flattened_cache and num_gpu_blocks_override is not None:
        num_gpu_blocks_override *= NUM_LAYERS
    print(f"max_model_len: {max_model_len}")
    args = EngineArgs(
        model=configs.model if getattr(configs, "model", None) else MODEL,
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
        pause_strategy=pause_strategy,
        uniform_solver=uniform_solver,
        removable_cache=removable_cache,
        # static_batching=static_batching,
        disable_sliding_window=True,
    )
    print(f"Logging to {configs.output_log}")
     
    sys.stdout = open(configs.output_log, 'w')
    
    # Start memory profiling
    start_memory_profiling()
    initial_snapshot = tracemalloc.take_snapshot() if tracemalloc.is_tracing() else None
    log_memory_usage("BEFORE_ENGINE_INIT")
    
    engine = LLMEngine.from_engine_args(args)  
    
    log_memory_usage("AFTER_ENGINE_INIT")
    log_top_memory_allocations(10)
    
    csv_path = configs.output_log.replace(".log", ".csv")

    run_inference_step_mode(engine, trace, csv_path=csv_path,enable_deposit=configs.enable_deposit, estimator=estimator, slo_ratio=configs.slo_ratio)
    
    # Final memory check
    log_memory_usage("AFTER_INFERENCE")
    if initial_snapshot:
        get_memory_diff(initial_snapshot)

if __name__ == "__main__":
    from vllm.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser(description="distN test.")
    parser.add_argument("--config-file",
                        type=str,
                        default="/home/heelim/vllm/samples/large_new_request.json",
                        help="Configurations file.")
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="Model path or HuggingFace model ID (e.g., /path/to/model or meta-llama/Meta-Llama-3-8B-Instruct)")
    parser.add_argument("--prefetch-mode",
                        type=str,
                        default="none",
                        help="prefetch method: none, static, distn, solver, static_req_wise, selectn, flexgen, flexgen_orig")
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
                        default="/home/heelim/vllm/outputs/default.log",
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
    parser.add_argument("--uniform-solver",
                        action="store_true",
                        default=False,
                        help="whether to use use uniform decision making sovler")
    parser.add_argument("--removable-cache",
                        action="store_true",
                        default=False,
                        help="whether to use use static batching instead of continuous batching")
    parser.add_argument("--static-batching",
                        action="store_true",
                        default=False,
                        help="whether to use use static batching instead of continuous batching")
    parser.add_argument("--profiled-results",
                        type=str,
                        default="/home/heelim/vllm/benchmark/scripts/profiling_data/profiled_results.json",
                        help="profiling results. If not provided, use the default ones.")
    parser.add_argument("--pause-strategy", 
                        type=str, 
                        default="longest", 
                        choices=["longest", "shortest", "random", "slo_loose", "slo_strict", "no_pause"], 
                        help="pause strategy for scheduler")
    parser.add_argument("--slo-ratio",
                        type=float,
                        default=2.5,
                        help="delay simulator slo ratio (default: 2.5)")
    args = parser.parse_args()    
    print(args)
    # --- Setup Logging ---
    logging.basicConfig(filename=args.output_log, level=logging.INFO, format="%(message)s")

    main(args)
