#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Seven-feature, X/Y-normalized symbolic search reported in Sections S3.3.3
# and S4.3.1. Seed 8971 matches the archived paper run.
python 5.extract_formula.py \
    --name symbolic-7-features \
    --exp_name paper_symbolic_7_features \
    --data_path ./logs/get_lowdim_functions/paper_proxy_7_features/sr.csv.gz \
    --use_all_features \
    --target output \
    --sample_num 50000 \
    --seed 8971
