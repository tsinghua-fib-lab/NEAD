#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Main U.S. scenario-level surrogates used by the structure decoupler.
python 2.train_surrogate.py \
    --name us-static \
    --exp_name paper_us_static \
    --seed 42 \
    --lr 1e-3 \
    --edge1_features capacity shortest_route_count free_flow_time \
    --edge2_features

python 2.train_surrogate.py \
    --name us-assignment-informed \
    --exp_name paper_us_assignment_informed \
    --seed 42 \
    --lr 1e-3 \
    --edge1_features capacity shortest_route_count free_flow_time volume voc travel_time disrupted_rank \
    --edge2_features diversity weighted_diversity

# Five-fold held-out-city evaluation reported in Supplementary Section S4.1.
for fold in {0..4}; do
    python 2.train_surrogate.py \
        --name "us-static-fold-${fold}" \
        --exp_name "paper_us_static_fold_${fold}" \
        --seed 42 \
        --lr 1e-3 \
        --edge1_features capacity shortest_route_count free_flow_time \
        --edge2_features \
        --fold_by_city "${fold}"
done

# Global South scenario-level surrogates reported in Supplementary Section S7.
GLOBAL_SOUTH_CITIES=(
    "Ahmedabad" "Bandung" "Bengaluru" "Bogotá" "Bucaramanga" "Cartagena"
    "Chennai" "Cúcuta" "Delhi" "Depok" "Ecatepec" "Guadalajara" "Hyderabad"
    "Ibagué" "Jaipur" "Kanpur" "Kolkata" "León" "Lucknow" "Makassar" "Medan"
    "Medellín" "Mérida" "Montería" "Mumbai" "Palembang" "Puebla City" "Pune"
    "Querétaro City" "Santa Marta" "Santiago de Cali" "Semarang" "Surabaya" "Surat"
    "Valledupar"
)

python 2.train_surrogate.py \
    --data_dir ./data/augmentation_globalsouth \
    --datasets "${GLOBAL_SOUTH_CITIES[@]}" \
    --name globalsouth-static \
    --exp_name paper_globalsouth_static \
    --seed 42 \
    --lr 1e-3 \
    --edge1_features capacity shortest_route_count free_flow_time \
    --edge2_features

python 2.train_surrogate.py \
    --data_dir ./data/augmentation_globalsouth \
    --datasets "${GLOBAL_SOUTH_CITIES[@]}" \
    --name globalsouth-assignment-informed \
    --exp_name paper_globalsouth_assignment_informed \
    --seed 42 \
    --lr 1e-3 \
    --edge1_features capacity shortest_route_count free_flow_time volume voc travel_time disrupted_rank \
    --edge2_features diversity weighted_diversity

# Five-fold Global South held-out-city evaluation. The paper uses 96 hidden
# channels for these larger networks (Supplementary Section S7.2).
for fold in {0..4}; do
    python 2.train_surrogate.py \
        --data_dir ./data/augmentation_globalsouth \
        --datasets "${GLOBAL_SOUTH_CITIES[@]}" \
        --name "globalsouth-static-fold-${fold}" \
        --exp_name "paper_globalsouth_static_fold_${fold}" \
        --seed 42 \
        --lr 1e-3 \
        --Df 96 \
        --edge1_features capacity shortest_route_count free_flow_time \
        --edge2_features \
        --fold_by_city "${fold}"

    python 2.train_surrogate.py \
        --data_dir ./data/augmentation_globalsouth \
        --datasets "${GLOBAL_SOUTH_CITIES[@]}" \
        --name "globalsouth-assignment-informed-fold-${fold}" \
        --exp_name "paper_globalsouth_assignment_informed_fold_${fold}" \
        --seed 42 \
        --lr 1e-3 \
        --Df 96 \
        --edge1_features capacity shortest_route_count free_flow_time volume voc travel_time disrupted_rank \
        --edge2_features diversity weighted_diversity \
        --fold_by_city "${fold}"
done
