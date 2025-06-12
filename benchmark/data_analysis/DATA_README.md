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

### `make_summerize.py`

- **Purpose**: Computes aggregate metrics from `outputs.csv` files.
- **Inputs**:
  - `outputs.csv.`
  - `slo_violation.csv.`
- **Metrics Calculated**: `tpot`, `tbt`, `throughput`, `slo_threshold`, `p90_ratio`, `p95_ratio`, `p99_ratio`.
- **Output**: Stores results in `summerize.csv` in the same directory as `outputs.csv`.

### `sim_slo_violation.py`

- **Purpose**: Simulates and calculates SLO violations from `outputs.csv`.
- **Inputs**:
  - `outputs.csv.`
- **Output**: Stores the results in `slo_violation.csv`.

### `solver_summerize.py`

- **Purpose**: Summarizes solver activity.
- **Inputs**:
  - `outputs.log.`
  - `vllm_msg.log.`
- **Metrics**: Total number of solver calls, fallbacks, and total decoding steps.
- **Output**: Stores results in `solver_summerize.csv`.

### `solver_total_overhead_table2.py`

- **Purpose**: Compares solver overhead per trace.
- **Inputs**:
  - `outputs.csv.`
  - `solver_summerize.csv`
- **Metrics**: 
  - Mean solver time / mean TBT
  - Total solver time / total end-to-end time
  - Solver call count / number of requests
- **Output**: Stores results in `solver_summary_table_slo_total.csv`.

## Plotting

### Required files
- `summerize.csv`

### Scripts

#### `update_all_fig.sh`
- Generate all figures for the paper

#### `data_parse_update_all_fig.sh`
- Run `sim_slo_violation.py`, `make_summerize.py`  
- Then generate all figures  
- Use `input_dirs` to filter targets

