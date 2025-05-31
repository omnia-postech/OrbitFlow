#!/bin/bash

# sim_slo_violation.py 실행
echo "Running sim_slo_violation.py ..."
python ./data_parsing/sim_slo_violation.py
if [ $? -ne 0 ]; then
    echo "sim_slo_violation.py 실행 실패"
    exit 1
fi

# make_summerize.py 실행
echo "Running make_summerize.py ..."
python ./data_parsing/make_summerize.py
if [ $? -ne 0 ]; then
    echo "make_summerize.py 실행 실패"
    exit 1
fi

# make_summerize.py 실행
echo "Running design_validation.py ..."
python ./draw_paper_graph/design_validation.py
if [ $? -ne 0 ]; then
    echo "design_validation.py 실행 실패"
    exit 1
fi

# draw_tp_tbt.py 실행
echo "Running draw_tp_tbt.py ..."
python ./draw_paper_graph/draw_tp_tbt.py
if [ $? -ne 0 ]; then
    echo "draw_tp_tbt.py 실행 실패"
    exit 1
fi

# p_slo_scale_bar.py 실행
echo "Running p_slo_scale_bar.py ..."
python ./draw_paper_graph/p_slo_scale_bar.py
if [ $? -ne 0 ]; then
    echo "p_slo_scale_bar.py 실행 실패"
    exit 1
fi

# throughput_total_compare.py 실행
echo "Running throughput_total_compare.py ..."
python ./draw_paper_graph/throughput_total_compare.py
if [ $? -ne 0 ]; then
    echo "throughput_total_compare.py 실행 실패"
    exit 1
fi

# tpot_tbt_total_compare_graph.py 실행
echo "Running tpot_tbt_total_compare_graph.py ..."
python ./draw_paper_graph/tpot_tbt_total_compare_graph.py
if [ $? -ne 0 ]; then
    echo "tpot_tbt_total_compare_graph.py 실행 실패"
    exit 1
fi

