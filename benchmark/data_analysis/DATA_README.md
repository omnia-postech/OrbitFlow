# Data Analysis & Transformation

Tools for processing experiment results and generating paper figures.

## Structure
```bash
benchmark/
└── data_analysis/
├── data_parsing/
│ ├── change_slo_scale.py
│ ├── make_summerize.py
│ ├── sim_slo_violation.py
│ ├── solver_summerize.py
│ └── solver_total_overhead_table2.py
├── update_all_fig.sh
└── data_parse_update_all_fig.sh
```

## Data parsing

### `change_slo_scale.py`

- **Purpose**: Generates `outputs.csv` files for new SLO thresholds based on baseline experiment results (e.g., FlexGen, No prefetching).
- **Inputs**:
  - `outputs.csv.`
- **Inner Inputs**:
  - `old_sc`, `metrics`, `traces`, `methods`: defines the source experiments.
  - `new_sc_list`: specifies the new SLO scales to generate.
- **Output**: New `outputs.csv` files with recalculated `slo_threshold` and `slo_violations`.

### `sim_slo_violation.py`
- **Purpose**: Simulates and calculates SLO violations from `outputs.csv`.
- **Inputs**: `outputs.csv.`
- **Output**: Stores the results in `slo_violation.csv`.



| Stage                 | Action                                                                                                                                                                                  |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Scan**              | Finds every `outputs.csv` produced by the simulator (or only those you explicitly supply).                                                                                              |
| **Reference lookup**  | Opens the matching `*.json` trace description in **`REFERENCE_ROOT`** to know the ground-truth `output_length` for each request.                                                        |
| **Metric load**       | Reads token-level timing columns from `outputs.csv`, converts any stringified lists with `ast.literal_eval`, and coerces numeric columns to floats.                                     |
| **Per-request check** | Calls `compute_slo_violation()` to :<br>1. Reconstruct inter-token generation times.<br>2. Track the SLO “deposit” bucket.<br>3. Count violations plus malformed (negative) TBT events. |
| **Post-processing**   | Adds penalties for **unfinished tokens** (simulator stopped early) and sets a `failed` flag if decoded length ≠ `reference_length-1`.                                                   |
| **Output**            | Writes **one CSV per experiment folder** named `slo_violation.csv` with columns:<br>`request_id, slo_violation, exceptions, failed`.                                                    |


### `make_summerize.py`

- **Purpose**: Computes aggregate metrics from `outputs.csv` files.
- **Inputs**:
  - `outputs.csv.`
  - `slo_violation.csv.`
- **Metrics Calculated**: `tpot`, `tbt`, `throughput`, `slo_threshold`, `p90_ratio`, `p95_ratio`, `p99_ratio`.
- **Output**: Stores results in `summerize.csv` in the same directory as `outputs.csv`.

Quick start
```bash
# Compute violations for *all* experiments under ROOT_DIR
python sim_slo_violation.py
# Re-run only those that do NOT already have slo_violation.csv
python sim_slo_violation.py --missing

# You can point the script at one or more folders instead of scanning everything

# Relative path (resolved against ROOT_DIR)
python sim_slo_violation.py Ours/both_static_high
# Absolute path
python sim_slo_violation.py /home/heelim/.../Flexgen/token_dyn_mid
# Multiple folders in one command
python sim_slo_violation.py OursMinusPause/batch_dyn_low /abs/path/to/NoPrefetch/both_static_mid

```

### `solver_summerize.py`

- **Purpose**: Summarizes solver activity.
- **Inputs**:
  - `outputs.log.`
  - `vllm_msg.log.`
- **Metrics**: Total number of solver calls, fallbacks, and total decoding steps.
- **Output**: Stores results in `solver_summerize.csv`.


## Visualization

### `solver_total_overhead_table2.py`
- **Purpose**: Create data to compare solver overhead per trace for use in table 
- **Inputs**:
  - `outputs.csv.`
  - `solver_summerize.csv`
- **Metrics**: 
  - Mean solver time / mean TBT
  - Total solver time / total end-to-end time
  - Solver call count / number of requests
- **Output**: Stores results in `solver_summary_table_slo_total.csv`.


### Graph Required files
- `summerize.csv`

### Scripts

#### `update_all_fig.sh`
- Generate all figures for the paper

#### `data_parse_update_all_fig.sh`
- Run `sim_slo_violation.py`, `make_summerize.py`  
- Then generate all figures  
- Use `input_dirs` to filter targets
