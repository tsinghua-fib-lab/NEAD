python 2.train_surrogate \
    --name "wo-traffic-assignment" \
    --seed 42 \
    --lr 1e-3 \
    --edge1_features capacity shortest_route_count free_flow_time \
    --edge2_features
    # --split_by_city

python 2.train_surrogate \
    --name "with-traffic-assignment" \
    --seed 42 \
    --lr 1e-3 \
    --edge1_features capacity shortest_route_count free_flow_time volume voc travel_time disrupted_rank \
    --edge2_features diversity weighted_diversity
    # --split_by_city

python 2.train_surrogate \
    --name "wo-traffic-assignment-split-by-city" \
    --seed 42 \
    --lr 1e-3 \
    --edge1_features capacity shortest_route_count free_flow_time \
    --edge2_features \
    --split_by_city

FOLD=0 python 2.train_surrogate \
    --name "wo-traffic-assignment-split-by-city_fold$FOLD" \
    --seed 42 \
    --lr 1e-3 \
    --edge1_features capacity shortest_route_count free_flow_time \
    --edge2_features \
    --fold_by_city $FOLD

python 2.train_surrogate \
    --data_path ./data/globalsouth \
    --datasets "Ahmedabad" "Bandung" "Bengaluru" "Bogotá" "Bucaramanga" "Cartagena" "Chennai" "Cúcuta" "Delhi" "Depok" "Ecatepec" "Guadalajara" "Hyderabad" "Ibagué" "Jaipur" "Kanpur" "Kolkata" "León" "Lucknow" "Makassar" "Medan" "Medellín" "Mérida" "Montería" "Mumbai" "Palembang" "Puebla City" "Pune" "Querétaro City" "Santa Marta" "Santiago de Cali" "Semarang" "Surabaya" "Surat" "Valledupar" \
    --name "globalsouth-wo-traffic-assignment" \
    --seed 42 \
    --lr 1e-3 \
    --edge1_features capacity shortest_route_count free_flow_time \
    --edge2_features

python 2.train_surrogate \
    --data_path ./data/globalsouth \
    --datasets "Ahmedabad" "Bandung" "Bengaluru" "Bogotá" "Bucaramanga" "Cartagena" "Chennai" "Cúcuta" "Delhi" "Depok" "Ecatepec" "Guadalajara" "Hyderabad" "Ibagué" "Jaipur" "Kanpur" "Kolkata" "León" "Lucknow" "Makassar" "Medan" "Medellín" "Mérida" "Montería" "Mumbai" "Palembang" "Puebla City" "Pune" "Querétaro City" "Santa Marta" "Santiago de Cali" "Semarang" "Surabaya" "Surat" "Valledupar" \
    --name "globalsouth-with-traffic-assignment" \
    --seed 42 \
    --lr 1e-3 \
    --edge1_features capacity shortest_route_count free_flow_time volume voc travel_time disrupted_rank \
    --edge2_features diversity weighted_diversity
