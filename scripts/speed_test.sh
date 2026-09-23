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

max_jobs=${MAX_JOBS:-1}   # max concurrent background jobs (can override via env)
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
        --disrupting_fraction {20..80..20}
        --seed 43
        --disrupting_strategy greedy
        "$@"  # Other arguments passed to the script
    )
    echo "[$(date '+%H:%M:%S')] python speed_test.py ${args[@]}"
    python speed_test.py "${args[@]}"
}

while IFS= read -r city || [[ -n "$city" ]]; do
    [[ -z "${city// /}" ]] && continue  # skip empty/blank lines

    # Generate Traffic Assignment Data
    allocate "${LOCK_SERVER}" "${city}_speed_test3" && {
        run_command \
            --name "${city}_speed_test" \
            --dataset "${city}" \
            --no-augment-OD \
            --save_data_dir ./logs/speed_test/results \
            --skip_existing \
            &
        
        sleep 2
        ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); }
    }

done < <(cat "$input_file")

wait
echo "✅ All jobs completed."
