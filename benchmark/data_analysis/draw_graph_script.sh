#!/usr/bin/env bash

# ───────────────────────────────────────────────
# Step 3. Calculate SLO Violation and Arrival Rate
# ───────────────────────────────────────────────

'''
python ./draw_graph/batch_size_graph.py \
  {Path where batch size was tested} \
  {Path where batch size 4 was experimented} \
  --output-dir {Path where you want to save the figure}

python ./draw_graph/cv_scale_graph.py \
  {Experimental path according to CV scale} \
  --output-dir {Path where you want to save the figure}

python ./draw_graph/fallback_strategy_graph.py \
  Path where the random selection results are saved \
  Path where the shortest selection results are saved \
  Path where the longest selection results are saved \
  --output-dir {Path where you want to save the figure}

python ./draw_graph/indivisual_component_token_deposit_graph.py \
  Path where design validation results are saved \
  Path where main results are saved \
  --output-dir {Path where you want to save the figure}

python ./draw_graph/p95_tbt_slo_attainment_gpu_utilization_graph.py \
  Path where main results are saved \
  --output-dir {Path where you want to save the figure}

python ./draw_graph/tbt_tpot_slo_attainment_graph.py \
  Path where main results are saved \
  --output-dir {Path where you want to save the figure}

python ./draw_graph/tp2_tp4_tbt_graph.py \
  Path where tp2 results are saved \
  Path where tp4 results are saved \
  --output-dir {Path where you want to save the figure}
'''

echo "Draw the picture you want"

python ./draw_graph/batch_size_graph.py \
  ../../outputs/benchmark/outputs/benchmark/paper_main_exp_bs \
  ../../outputs/benchmark/outputs/benchmark/paper_main_exp_32k \
  --output-dir {Path where you want to save the figure}

python ./draw_graph/cv_scale_graph.py \
  ../../outputs/benchmark/outputs/benchmark/paper_main_exp_CV \
  --output-dir {Path where you want to save the figure}

python ./draw_graph/fallback_strategy_graph.py \
  ../../outputs/benchmark/outputs/benchmark/paper_main_exp_random_xinyue \
  ../../outputs/benchmark/outputs/benchmark/paper_main_exp_shortest_xinyue \
  ../../outputs/benchmark/outputs/benchmark/paper_main_exp_longest_xinyue \
  --output-dir {Path where you want to save the figure}

python ./draw_graph/indivisual_component_token_deposit_graph.py \
  ../../outputs/benchmark/outputs/benchmark/paper_main_exp_design_ablation \
  ../../outputs/benchmark/outputs/benchmark/paper_main_exp_32k \
  --output-dir {Path where you want to save the figure}

python ./draw_graph/p95_tbt_slo_attainment_gpu_utilization_graph.py \
  ../../outputs/benchmark/outputs/benchmark/paper_main_exp_32k \
  --output-dir {Path where you want to save the figure}

python ./draw_graph/tbt_tpot_slo_attainment_graph.py \
  ../../outputs/benchmark/outputs/benchmark/paper_main_exp_32k \
  --output-dir {Path where you want to save the figure}

python ./draw_graph/tp2_tp4_tbt_graph.py \
  ../../outputs/benchmark/outputs/benchmark/paper_main_exp_TP \
  ../../outputs/benchmark/outputs/benchmark/paper_main_exp_TP_4 \
  --output-dir {Path where you want to save the figure}
