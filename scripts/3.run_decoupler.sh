#!/bin/bash
# set -euo pipefail
# zsh: enable "pipefail" if possible
# [[ -n "${ZSH_VERSION:-}" ]] && setopt localoptions pipefail 2>/dev/null || true

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-6}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-6}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-6}
input_file="./data/cities.txt"   # default: file containing city per line

source ./share/lock_server.sh
LOCK_SERVER="http://rl3.yumeow.site:16699"

max_jobs=${MAX_JOBS:-6}   # max concurrent background jobs (can override via env)
current_jobs=0
cleanup() {
    echo "Cleaning up: killing child processes..."
    local job_pids
    job_pids=$(jobs -p)
    if [[ -n "$job_pids" ]]; then
        kill -INT $job_pids 2>/dev/null || true
    fi
    wait 2>/dev/null || true # wait for children to exit
}
trap 'echo "Interrupted."; cleanup; exit 130' INT TERM
trap 'cleanup' EXIT

run_command() {
    args=(
        "$@"  # Other arguments passed to the script
    )
    echo "[$(date '+%H:%M:%S')] python run_gnnexplainer.py ${args[@]}"
    python run_gnnexplainer.py "${args[@]}"
}

for sample in ./data/augmentation/*/*; do
    star1=$(basename "$(dirname "$sample")")  # 上一级目录名
    star2=$(basename "$sample")               # 文件名
    for explain_for in feature structure; do  # 解释类型

        train_log_path="./logs/train/20251119_with-traffic-assignment_155328_LM2"
        allocate "${LOCK_SERVER}" "with-traffic-assignment-explain-${star1}-${star2}-${explain_for}" && \
        run_command \
            --name "with-traffic-assignment-explain-${star1}-${star2}-${explain_for}" \
            --data_path "${sample}" \
            --explain_for ${explain_for} \
            --model_path "${train_log_path}/best_model.pth" \
            --save_data_dir "${train_log_path}/gnnexplainer-${explain_for}" \
            --edge1_features capacity shortest_route_count free_flow_time volume voc travel_time disrupted_rank \
            --edge2_features diversity weighted_diversity \
            --visualize \
            &
        ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); }

        train_log_path="./logs/train/20251119_wo-traffic-assignment_160956_LM2"
        allocate "${LOCK_SERVER}" "wo-traffic-assignment-explain-${star1}-${star2}-${explain_for}" && \
        run_command \
            --name "wo-traffic-assignment-explain-${star1}-${star2}-${explain_for}" \
            --data_path "${sample}" \
            --explain_for ${explain_for} \
            --model_path "${train_log_path}/best_model.pth" \
            --save_data_dir "${train_log_path}/gnnexplainer-${explain_for}" \
            --edge1_features capacity shortest_route_count free_flow_time \
            --edge2_features \
            --visualize \
            &
        ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); }

    done
done

wait
echo "✅ All jobs completed."
