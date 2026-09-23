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
input_file="./data/cities.txt"   # file containing one city per line

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
        --fix_existing
        # --skip_existing
        --disrupting_fraction {20..80..20}
        --seed 43
        --disrupting_strategy greedy
        "$@"  # Additional arguments passed to the script
    )
    echo "[$(date '+%H:%M:%S')] python scripts/scale_capacity.py ${args[@]}"
    python scripts/scale_capacity.py "${args[@]}"
}

while IFS= read -r city || [[ -n "$city" ]]; do
    [[ -z "${city// /}" ]] && continue # skip empty/blank lines

    scale_ratio=1.0
    select_ratio=1.0
    select_by=all
    allocate "${LOCK_SERVER}" "${city}_${select_by}-${select_ratio}_by_${scale_ratio}_new" && \
    run_command \
        --name "${city}_${select_by}-${select_ratio}_by_${scale_ratio}_new" \
        --raw_data_path "./data/augmentation/${city}/raw" \
        --select_by "${select_by}" \
        --select_ratio "${select_ratio}" \
        --scale_ratio "${scale_ratio}" \
        &
    ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); }

    for scale_ratio in 0.9 1.1 1.2 1.5 2.0 2.5 3.0 4.0 5.0; do
        select_ratio=1.0
        select_by=all
        allocate "${LOCK_SERVER}" "${city}_${select_by}-${select_ratio}_by_${scale_ratio}_new" && \
        run_command \
            --name "${city}_${select_by}-${select_ratio}_by_${scale_ratio}_new" \
            --raw_data_path "./data/augmentation/${city}/raw" \
            --select_by "${select_by}" \
            --select_ratio "${select_ratio}" \
            --scale_ratio "${scale_ratio}" \
            &
        ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); }
    done

    for select_by in random volume travel_time betweenness; do
        select_ratio=0.05
        scale_ratio=1.2
        allocate "${LOCK_SERVER}" "${city}_${select_by}-${select_ratio}_by_${scale_ratio}_new" && \
        run_command \
            --name "${city}_${select_by}-${select_ratio}_by_${scale_ratio}_new" \
            --raw_data_path "./data/augmentation/${city}/raw" \
            --select_by "${select_by}" \
            --select_ratio "${select_ratio}" \
            --scale_ratio "${scale_ratio}" \
            &
        ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); }

        select_ratio=0.1
        scale_ratio=1.1
        allocate "${LOCK_SERVER}" "${city}_${select_by}-${select_ratio}_by_${scale_ratio}_new" && \
        run_command \
            --name "${city}_${select_by}-${select_ratio}_by_${scale_ratio}_new" \
            --raw_data_path "./data/augmentation/${city}/raw" \
            --select_by "${select_by}" \
            --select_ratio "${select_ratio}" \
            --scale_ratio "${scale_ratio}" \
            &
        ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); }

        select_ratio=0.1
        scale_ratio=1.2
        allocate "${LOCK_SERVER}" "${city}_${select_by}-${select_ratio}_by_${scale_ratio}_new" && \
        run_command \
            --name "${city}_${select_by}-${select_ratio}_by_${scale_ratio}_new" \
            --raw_data_path "./data/augmentation/${city}/raw" \
            --select_by "${select_by}" \
            --select_ratio "${select_ratio}" \
            --scale_ratio "${scale_ratio}" \
            &
        ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); }

        select_ratio=0.05
        scale_ratio=1.05
        allocate "${LOCK_SERVER}" "${city}_${select_by}-${select_ratio}_by_${scale_ratio}_new" && \
        run_command \
            --name "${city}_${select_by}-${select_ratio}_by_${scale_ratio}_new" \
            --raw_data_path "./data/augmentation/${city}/raw" \
            --select_by "${select_by}" \
            --select_ratio "${select_ratio}" \
            --scale_ratio "${scale_ratio}" \
            &
        ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); }

        select_ratio=0.05
        scale_ratio=1.1
        allocate "${LOCK_SERVER}" "${city}_${select_by}-${select_ratio}_by_${scale_ratio}_new" && \
        run_command \
            --name "${city}_${select_by}-${select_ratio}_by_${scale_ratio}_new" \
            --raw_data_path "./data/augmentation/${city}/raw" \
            --select_by "${select_by}" \
            --select_ratio "${select_ratio}" \
            --scale_ratio "${scale_ratio}" \
            &
        ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); }

        select_ratio=0.1
        scale_ratio=1.05
        allocate "${LOCK_SERVER}" "${city}_${select_by}-${select_ratio}_by_${scale_ratio}_new" && \
        run_command \
            --name "${city}_${select_by}-${select_ratio}_by_${scale_ratio}_new" \
            --raw_data_path "./data/augmentation/${city}/raw" \
            --select_by "${select_by}" \
            --select_ratio "${select_ratio}" \
            --scale_ratio "${scale_ratio}" \
            &
        ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); }

        select_ratio=0.1
        scale_ratio=1.02
        allocate "${LOCK_SERVER}" "${city}_${select_by}-${select_ratio}_by_${scale_ratio}_new" && \
        run_command \
            --name "${city}_${select_by}-${select_ratio}_by_${scale_ratio}_new" \
            --raw_data_path "./data/augmentation/${city}/raw" \
            --select_by "${select_by}" \
            --select_ratio "${select_ratio}" \
            --scale_ratio "${scale_ratio}" \
            &
        ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); }
    done

done < <(cat "$input_file")

wait
echo "✅ All jobs completed."
