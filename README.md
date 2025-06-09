# vLLM Benchmark Script README

This guide explains how to use `run_benchmarks.sh` to execute vLLM benchmarks for specific prefetch methods, traces, and SLO ratios.

## Overview

The script automates benchmarking with `test_distN.py`. It loops over:

- **SLO ratios**: Latency scaling factors (e.g., 1.5).
- **Experiments**: Named groups (e.g., `TestBestAndWorst`).
- **Methods**: Prefetch strategies (e.g., `Ours`, `OursUniformSolver`, `DistNSingle`) from `supported_methods.json`.
- **Traces**: Workload inputs (e.g., `test_shortshort_enough`).

**Modes**:

- `FIGURE_ONLY=0` (default): Run benchmarks.
- `FIGURE_ONLY=1`: Skip execution.

Outputs (logs, CSVs) are saved in a structured directory.

## Prerequisites

1. **Environment**:
   
   - Python 3.11 with vLLM.
   - CUDA GPU (e.g., set `CUDA_VISIBLE_DEVICES=0`).

2. **Directory Structure**:

   - Root: `/home/heelim/vllm` (adjust `ROOT` if needed).
   - Files:
      - `supported_methods.json`: Method CLI args.
      - `examples/test_distN.py`: Benchmark script.
      - `configs/test_no_prefetch_logging.json`: Logging config.
      - `benchmark/test_traces/test_best_worst/*.json`: Trace files.

3. **Install Dependencies**:

   ```bash
   pip install vllm
   ```

## Script Configuration

Edit the **CONSTANTS** section:

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_CONFIGURE_LOGGING=1
LOGGING_LEVEL=CRITICAL  # DEBUG, INFO, WARNING, ERROR, CRITICAL
ROOT="/home/heelim/vllm"
FIGURE_ONLY="${1:-0}"  # 0: Run benchmarks, 1: Skip execution

EXP_LIST=(TestBestAndWorst)
METHOD_LIST=(Ours OursUniformSolver DistNSingle)  # Public methods
TRACE_LIST=(test_shortshort_enough)  # Trace basenames
TRACE_CFG_DIR="${ROOT}/benchmark/test_traces/test_best_worst"
SLO_RATIO_LIST=(1.5)  # e.g., 1.5, 2.0
```

- **EXP_LIST**: Experiment names.
- **METHOD_LIST**: Public methods from `supported_methods.json`.
- **TRACE_LIST**: Trace files (without `.json`) in `TRACE_CFG_DIR`.
- **SLO_RATIO_LIST**: SLO ratios.

## Creating Custom Traces

You may create your own trace files to simulate specific workloads or existing traces we made. Place them in `TRACE_CFG_DIR` (e.g., `benchmark/test_traces/test_best_worst/`) and list their basenames (without `.json`) in `TRACE_LIST`.

### Trace Format

Trace files are JSON objects with the following structure:

- **batch_size**: Integer, number of requests processed together.
- **max_model_len**: Integer, maximum sequence length the model supports.
- **num_gpu_blocks_override**: Integer, number of GPU memory blocks to use.
- **arrival_pattern**: String, defines request arrival distribution (e.g., `BimodalArrival(l1=0.16,l2=0.1,p=0.7,max=888)`).
- **vocab**: (You may ignore this one). Array of integers, vocabulary range for token generation.
- **peak_batch_blocks**: Integer, maximum GPU blocks needed for the batch.
- **requests**: Object mapping request IDs to details:
   - **category**: String, request type (e.g., `C2_IN3680-4496_OUT4-32`).
   - **input_length**: Integer, number of input tokens.
   - **output_length**: Integer, number of output tokens.
   - **arrival_time**: Integer, time (in arbitrary units) when the request arrives.
   - **sched_time**: Integer, time when the request is scheduled.
   - **wait_time**: Integer, time difference between arrival and scheduling.

**Example Trace** (`example_trace.json`):

```json
{
  "batch_size": 4,
  "max_model_len": 32384,
  "num_gpu_blocks_override": 1664,
  "arrival_pattern": "BimodalArrival(l1=0.16087516087516088,l2=0.1,p=0.7,max=888)",
  "vocab": [200, 30000],
  "peak_batch_blocks": 4459,
  "requests": {
    "request_0": {
      "category": "C2_IN3680-4496_OUT4-32",
      "input_length": 3705,
      "output_length": 27,
      "arrival_time": 0,
      "sched_time": 0,
      "wait_time": 0
    },
    "request_1": {
      "category": "C2_IN448-560_OUT4-32",
      "input_length": 545,
      "output_length": 22,
      "arrival_time": 4,
      "sched_time": 4,
      "wait_time": 0
    }
  }
}
```

- Create traces manually or with a script to match your workload.
- Validate JSON syntax before running.

## Publicly Disclosed Methods

The following methods from `supported_methods.json` are available for public use:

- **Ours**:

   - **Description**: Uses a solver to dynamically determine the exact number of layers to offload to the CPU for each request, optimizing resource allocation based on workload demands.

   - **Configuration**:

      ```json
      {
        "Ours": [
          "--prefetch-mode", "solver",
          "--prefetch-distance", "1",
          "--flattened-cache", "true",
          "--merge-prefetch-buffer", "true",
          "--pause-and-resume",
          "--enable-deposit"
        ]
      }
      ```

- **OursUniformSolver**:

   - **Description**: Employs a solver but offloads a fixed number of layers to the CPU for each request, ensuring uniform resource allocation across requests.

   - **Configuration**:

      ```json
      {
        "OursUniformSolver": [
          "--prefetch-mode", "solver",
          "--prefetch-distance", "1",
          "--flattened-cache", "true",
          "--merge-prefetch-buffer", "true",
          "--pause-and-resume",
          "--enable-deposit",
          "--uniform-solver"
        ]
      }
      ```

- **DistNSingle**:

   - **Description**: Uses a heuristic approach (no solver) to determine the number of layers to offload to the CPU, providing a simpler, less computationally intensive method.

   - **Configuration**:

      ```json
      {
        "DistNSingle": [
          "--prefetch-mode", "distn_single",
          "--prefetch-distance", "1",
          "--flattened-cache", "true",
          "--merge-prefetch-buffer", "true"
        ]
      }
      ```

Add or modify these methods in `supported_methods.json`, ensuring valid `test_distN.py` arguments.

## Running the Script

1. **Set Permissions**:

   ```bash
   chmod +x run_benchmarks.sh
   ```

2. **Run**:

   - Run benchmarks:

      ```bash
      ./run_benchmarks.sh 0
      ```

   - Skip execution:

      ```bash
      ./run_benchmarks.sh 1
      ```

3. **Output**:
   Saved in:

   ```
   ${ROOT}/outputs/benchmark/${EXP}/slo${SLO}/${METHOD}/${TRACE}/
   ```

   - `outputs.log`: Benchmark log.
   - `vllm_msg.log`: vLLM log.
   - `outputs.csv`: Results (if generated).
   - `logging_cfg.json`: Per-run logging config.

4. **Example**:
   Run `Ours` with SLO=1.5, trace `test_shortshort_enough`:

   - Set `METHOD_LIST=(Ours)`, `SLO_RATIO_LIST=(1.5)`, `TRACE_LIST=(test_shortshort_enough)`.

   - Execute:

      ```bash
      ./run_benchmarks.sh 0
      ```

## Troubleshooting

- **No CSV**: Check `test_distN.py` or trace file validity.
- **JSON Errors**: Verify `supported_methods.json` entries match `METHOD_LIST`.
- **Logs**: Use `LOGGING_LEVEL=DEBUG` for detailed `vllm_msg.log`.

## Adding Experiments

1. Add traces to `benchmark/test_traces/test_best_worst/`.
2. Update `TRACE_LIST`.
3. Add public methods to `supported_methods.json` and `METHOD_LIST`.
4. Modify `SLO_RATIO_LIST` or `EXP_LIST`.

## Notes

- Verify `ROOT` path.
- Backup `outputs/` to avoid overwrites.
- Monitor GPU memory for large traces.

<!-- For support, contact [your contact info]. -->