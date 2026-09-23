#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# set -euo pipefail
# zsh: enable "pipefail" if possible
# [[ -n "${ZSH_VERSION:-}" ]] && setopt localoptions pipefail 2>/dev/null || true

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-6}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-6}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-6}
source ./src/utils/share/lock_server.sh
LOCK_SERVER="${LOCK_SERVER:-}"

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
    echo "[$(date '+%H:%M:%S')] python 3.run_decoupler.py ${args[@]}"
    python 3.run_decoupler.py "${args[@]}"
}

for sample in ./data/augmentation/*/*; do
    star1=$(basename "$(dirname "$sample")")  # Parent directory name
    star2=$(basename "$sample")               # Sample directory name
    for explain_for in feature structure; do  # Explanation type

        train_log_path="./logs/train/paper_us_assignment_informed"
        allocate "${LOCK_SERVER}" "assignment-informed-explain-${star1}-${star2}-${explain_for}" && \
        run_command \
            --name "assignment-informed-explain-${star1}-${star2}-${explain_for}" \
            --data_path "${sample}" \
            --explain_for ${explain_for} \
            --model_path "${train_log_path}/best_model.pth" \
            --save_data_dir "${train_log_path}/gnnexplainer-${explain_for}" \
            --edge1_features capacity shortest_route_count free_flow_time volume voc travel_time disrupted_rank \
            --edge2_features diversity weighted_diversity \
            --visualize \
            &
        ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); }

        train_log_path="./logs/train/paper_us_static"
        allocate "${LOCK_SERVER}" "static-explain-${star1}-${star2}-${explain_for}" && \
        run_command \
            --name "static-explain-${star1}-${star2}-${explain_for}" \
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
