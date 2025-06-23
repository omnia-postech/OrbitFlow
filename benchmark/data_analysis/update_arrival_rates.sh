# input_dir은 리스트 형태로 여러 경로를 받을 수 있음
input_dirs=(
    # "/path/to/dir1"
    # "/path/to/dir2"
    # "/path/to/dir3"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1.5/Ours_TP"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/Ours_TP"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/Ours_TP"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo4.5/Ours_TP"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1/Flexgen"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1.5/Flexgen"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/Flexgen"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/Flexgen"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1.5/SelectN_TP"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/SelectN_TP"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo3.5/SelectN_TP"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo4.5/SelectN_TP"
    "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1/Ours"
    "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1.5/Ours"
    "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/Ours"

    "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1/Flexgen"
    "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1.5/Flexgen"
    "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/Flexgen"

    "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1/SelectN"
    "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1.5/SelectN"
    "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/SelectN"

    "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1/NextLayer"
    "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1.5/NextLayer"
    "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/NextLayer"

    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1/DistNSingle"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo1.5/DistNSingle"
    # "/home/heelim/vllm/outputs/benchmark/paper_main_exp/slo2.5/DistNSingle"
)

# 리스트를 공백으로 이어붙여 하나의 문자열로 만듦
args="${input_dirs[@]}"

# 디렉토리 패턴 매칭이 없을 때 그대로 문자열이 남지 않도록
shopt -s nullglob

all_subdirs=()
for root in "${input_dirs[@]}"; do
    if [ -d "$root" ]; then
        # lambda로 시작하는 하위 디렉토리만 순회
        for subdir in "$root"/lambda*; do
            [ -d "$subdir" ] || continue
            all_subdirs+=("$subdir")
        done
    else
        echo "경로가 유효하지 않음: $root"
        exit 1
    fi
done


# sim_slo_violation.py 실행
echo "Running sim_slo_violation.py ..."
python ./data_parsing/sim_slo_violation.py "${all_subdirs[@]}"
if [ $? -ne 0 ]; then
    echo "❌ sim_slo_violation.py 실행 실패"
fi


# make_arrival_rate_summerize.py 실행
echo "Running make_arrival_rate_summerize.py ..."
python ./data_parsing/make_arrival_rate_summerize.py $args
if [ $? -ne 0 ]; then
    echo "❌ make_arrival_rate_summerize.py 실행 실패"
    exit 1
fi

echo "Running arrival_rate.py ..."
python arrival_rate_tbt_tpot.py 
if [ $? -ne 0 ]; then
    echo "❌ arrival_rate_tbt_tpot.py 실행 실패"
fi

echo "Running arrival_rate_cv.py ..."
python arrival_rate_cv_tbt_tpot.py 
if [ $? -ne 0 ]; then
    echo "❌ arrival_rate_cv_tbt_tpot.py 실행 실패"
    exit 1
fi
