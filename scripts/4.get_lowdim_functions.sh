#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# The four feature sets reported in Fig. 4b and Supplementary Section S3.3.2.
python 4.get_lowdim_functions.py \
    --name proxy-2-features \
    --exp_name paper_proxy_2_features \
    --used_features spec0 shortest_route_count

python 4.get_lowdim_functions.py \
    --name proxy-4-features \
    --exp_name paper_proxy_4_features \
    --used_features spec0 shortest_route_count dist0 capacity

python 4.get_lowdim_functions.py \
    --name proxy-7-features \
    --exp_name paper_proxy_7_features \
    --used_features spec0 shortest_route_count dist0 capacity total_out free_flow_time total_in

python 4.get_lowdim_functions.py \
    --name proxy-83-features \
    --exp_name paper_proxy_83_features \
    --used_features \
    capacity free_flow_time shortest_route_count \
    volume voc travel_time disrupted_rank \
    total_in total_out \
    spec{0..9} \
    dist{0..63}
