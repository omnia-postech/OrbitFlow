# OrbitFlow Benchmark Script README

This guide explains how to use `run_orbitflow.sh` to execute vLLM benchmarks for specific prefetch methods, traces, and SLO ratios.

## Overview

The script automates benchmarking with `orbitflow.py`. It loops over:

- **SLO ratios**: Latency scaling factors (e.g., 1.5).
- **Experiments**: Named groups (e.g., `TestBestAndWorst`).
- **Methods**: Prefetch strategies (e.g., `Ours`, `FlexGen`, `DistNSingle`) from `supported_methods.json`.
- **Traces**: Workload inputs (e.g., `test_shortshort_enough`).

## Prerequisites
1. **Environment**:
   
   - Python 3.11 with vLLM 0.6.6.
   - CUDA GPU version 12.1.

2. **Directory Structure**:

   - Root: `$HOME/vllm` (adjust `ROOT` if needed).
   - Files:
      - `benchmark/scripts/supported_methods.json`: Method CLI args.
      - `examples/orbitflow.py`: Benchmark script.
      - `configs/logging_template.json`: Logging config.
      - `benchmark/selected_traces/test_best_worst/*.json`: Trace files.
      - `benchmark/data_analysis/profiling/extract_profiled_results.py`: Profiling script.

3. **Install Dependencies**:

   ```bash
   pip install vllm
   ```

## Script Configuration

Edit the **CONSTANTS** section in `run_orbitflow.sh`:

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_CONFIGURE_LOGGING=1
export NUM_LAYERS=32  # e.g., 32 for LLaMa3-8B, 80 for LLaMa3-70B
LOGGING_LEVEL=CRITICAL  # DEBUG, INFO, WARNING, ERROR, CRITICAL
ROOT="$HOME/vllm"
MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"  # Or use local path: "$HOME/models/llama-3.1-8b-instruct"
profiled_path="$HOME/vllm/benchmark/scripts/profiling_data/profiled_results_A6000.json"
FIGURE_ONLY="${1:-0}"  # 0: Run benchmarks + plot, 1: Plot only

EXP_LIST=(paper_main_exp)
METHOD_LIST=(Ours)
TRACE_LIST=(both_dyn_veryhigh_bs2)  # Trace basenames
TRACE_CFG_DIR="${ROOT}/benchmark/selected_traces"
SLO_RATIO_LIST=(1.5)  # e.g., 1.5, 2.0
```

- **EXP_LIST**: Experiment names.
- **METHOD_LIST**: Public methods from `supported_methods.json`.
- **TRACE_LIST**: Trace files (without `.json`) in `TRACE_CFG_DIR`.
- **SLO_RATIO_LIST**: SLO ratios.
- **MODEL_PATH**: Path to model (e.g., LLaMa3-8B).
- **NUM_LAYERS**: Model layers (e.g., 32 for LLaMa3-8B).
- **profiled_path**: Path to profiling data (see Generating Profiled Data).

## Creating Custom Traces

You may create your own trace files to simulate specific workloads or existing traces we made. Place them in `TRACE_CFG_DIR` (e.g., `benchmark/selected_traces/test_best_worst/`) and list their basenames (without `.json`) in `TRACE_LIST`.

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

## Generating Profiled Data

To generate the `profiled_path` file (e.g., `profiled_results_A6000.json`), use the `extract_profiled_results.py` script located at `${HOME}/vllm/benchmark/data_analysis/profiling/extract_profiled_results.py`. This script processes two CSV files generated from benchmarks using the `NoPrefetch` and `NextLayer` methods to create profiling data tailored to your GPU and model.

### Steps to Generate Profiled Data

1. **Run Benchmarks for `NoPrefetch` and `NextLayer`**:

   - Configure `run_orbitflow.sh` with `METHOD_LIST=(NoPrefetch NextLayer)` and `TRACE_LIST=(profile_trace)`.
   - Ensure the trace file `$HOME/vllm/benchmark/selected_traces/profile/profile_trace.json` exists.

   - Execute the script to generate CSV outputs with profiling trace:

      ```bash
      ./run_orbitflow.sh 0
      ```

   - Find the CSV files in `${ROOT}/outputs/benchmark/${EXP}/slo${SLO}/${METHOD}/${TRACE}/outputs.csv`.

2. **Run the Profiling Script**:

   - Use the CSV files from `NoPrefetch` and `NextLayer` runs as inputs.

   - Example command:

      ```bash
      python $HOME/vllm/benchmark/data_analysis/profiling/extract_profiled_results.py \
      ${ROOT}/outputs/benchmark/paper_main_exp/slo1.5/NoPrefetch/profile_trace/outputs.csv \
      ${ROOT}/outputs/benchmark/paper_main_exp/slo1.5/NextLayer/profile_trace/outputs.csv \
      --out ${ROOT}/benchmark/scripts/profiling_data/profiled_results_A6000.json
      ```

   - The script outputs a JSON file (e.g., `profiled_results_A6000.json`) containing linear fit parameters (`A`, `B`, `R2`) for `NoPrefetch` and `Communication` (NextLayer).

3. **Update `run_orbitflow.sh`**:

   - Set `profiled_path` to the generated JSON file path (e.g., `$HOME/vllm/benchmark/scripts/profiling_data/profiled_results_A6000.json`).

## KV Placement Methods

The following methods from `supported_methods.json` are available:

- **Flexgen**: FlexGen-based prefetching strategy.
- **NoPrefetch**: Disables prefetching entirely.
- **NextLayer**: Prefetches only the immediate next layer.
- **Static1/2/4/8**: Static prefetching with fixed distances (1, 2, 4, or 8 layers ahead).

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

Add or modify these methods in `supported_methods.json`, ensuring valid `orbitflow.py` arguments.

## Running the Script

1. **Set Permissions**:

   ```bash
   chmod +x run_orbitflow.sh
   ```

2. **Run**:

   - Run benchmarks:

      ```bash
      ./run_orbitflow.sh 0
      ```

   - Skip execution:

      ```bash
      ./run_orbitflow.sh 1
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
      ./run_orbitflow.sh 0
      ```