#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python scripts/speed_test_surrogate.py \
    --data_path ./data/augmentation/*/raw \
    --name surrogate-runtime \
    --exp_name paper_surrogate_runtime \
    --model_path ./logs/train/paper_us_static/best_model.pth \
    --edge1_features capacity shortest_route_count free_flow_time \
    --edge2_features
