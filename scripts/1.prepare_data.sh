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
input_file="./data/globalsouth_cities.txt"   # default: file containing city per line

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
        --disrupting_fraction {20..80..20}
        --seed 43
        --disrupting_strategy greedy
        "$@"  # Other arguments passed to the script
    )
    echo "[$(date '+%H:%M:%S')] python 1.prepare_data.py ${args[@]}"
    python 1.prepare_data.py "${args[@]}"
}

while IFS= read -r city || [[ -n "$city" ]]; do
    [[ -z "${city// /}" ]] && continue  # skip empty/blank lines

    # Generate Traffic Assignment Data
    allocate "${LOCK_SERVER}" "${city}_no_augment_OD" && {
        run_command \
            --name "${city}_no_augment_OD" \
            --dataset "${city}" \
            --no-augment-OD \
            --save_data_dir ./data/globalsouth \
            &
        
        sleep 2
        ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); }
    }

    # Generate Augmented Data
    for i in {1..19}; do
        allocate "${LOCK_SERVER}" "${city}_augment_OD_${i}" && {
            run_command \
                --name "${city}_augment_OD_${i}" \
                --dataset "${city}" \
                --augment-OD \
                --augment-OD-num 1 \
                --max-sample-num 20 \
                --save_data_dir ./data/globalsouth \
                &
            
            sleep 2
            ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); }
        }
    done

    ## Generate Traffic Assignment Data with Random Failures 
    # allocate "${LOCK_SERVER}" "${city}_error" && sleep 1 && \
    # run_command \
    #     --save_data_dir "./data/error" \
    #     --name "${city}_error" \
    #     --dataset "${city}" \
    #     --no-augment-OD \
    #     --disrupting_strategy random \
    #     &
    # ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); }

    ## Fix Existing Data
    # run_command \
    #     --name "${city}_fix_existing" \
    #     --dataset "${city}" \
    #     --fix-existing \
    #     &
    # ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); }

done < <(cat "$input_file")

wait
echo "✅ All jobs completed."
