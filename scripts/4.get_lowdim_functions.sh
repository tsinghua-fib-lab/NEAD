python get_lowdim_functions.py \
    --name "get_lowdim_functions_allfeatures" \
    --used_features \
    capacity free_flow_time shortest_route_count \
    volume voc travel_time disrupted_rank \
    total_in total_out \
    spec{0..9} \
    dist{0..63}

python get_lowdim_functions.py \
    --name "get_lowdim_functions_edge_and_node_features" \
    --used_features \
    capacity free_flow_time shortest_route_count \
    volume voc travel_time disrupted_rank \
    total_in total_out

python get_lowdim_functions.py \
    --name "get_lowdim_functions_edge_features" \
    --used_features \
    capacity free_flow_time shortest_route_count \
    volume voc travel_time disrupted_rank

python get_lowdim_functions.py \
    --name "get_lowdim_functions_8features" \
    --used_features capacity free_flow_time volume voc shortest_route_count travel_time disrupted_rank spec0

python get_lowdim_functions.py \
    --name "get_lowdim_functions_7features" \
    --used_features capacity free_flow_time volume voc shortest_route_count travel_time spec0

python get_lowdim_functions.py \
    --name "get_lowdim_functions_4features" \
    --used_features capacity free_flow_time shortest_route_count spec0

python get_lowdim_functions.py \
    --name "get_lowdim_functions_4features_and_volume" \
    --used_features capacity free_flow_time shortest_route_count volume spec0

python get_lowdim_functions.py \
    --name "get_lowdim_functions_4features_and_travel_time" \
    --used_features capacity free_flow_time shortest_route_count travel_time spec0

python get_lowdim_functions.py \
    --name "get_lowdim_functions_4features_and_disrupted_rank" \
    --used_features capacity free_flow_time shortest_route_count disrupted_rank spec0

python get_lowdim_functions.py \
    --name "get_lowdim_functions_spec_and_betweenness" \
    --used_features spec0 shortest_route_count

python get_lowdim_functions.py \
    --name "get_lowdim_functions_spec_betweenness_dist_capacity" \
    --used_features spec0 shortest_route_count dist0 capacity

python get_lowdim_functions.py \
    --name "get_lowdim_functions_all_7_features" \
    --used_features spec0 shortest_route_count dist0 capacity total_out free_flow_time total_in

python get_lowdim_functions.py \
    --name "get_lowdim_functions_with_ta_3features" \
    --used_features volume spec0 capacity

python get_lowdim_functions.py \
    --name "get_lowdim_functions_with_ta_4features" \
    --used_features volume spec0 capacity free_flow_time

python get_lowdim_functions.py \
    --name "get_lowdim_functions_with_ta_5features" \
    --used_features volume spec0 capacity free_flow_time shortest_route_count

python get_lowdim_functions.py \
    --data_path ./data/globalsouth \
    --datasets "Ahmedabad" "Bandung" "Bengaluru" "Bogotá" "Bucaramanga" "Cartagena" "Chennai" "Cúcuta" "Delhi" "Depok" "Ecatepec" "Guadalajara" "Hyderabad" "Ibagué" "Jaipur" "Kanpur" "Kolkata" "León" "Lucknow" "Makassar" "Medan" "Medellín" "Mérida" "Montería" "Mumbai" "Palembang" "Puebla City" "Pune" "Querétaro City" "Santa Marta" "Santiago de Cali" "Semarang" "Surabaya" "Surat" "Valledupar" \
    --name "globalsouth_get_lowdim_functions_7features" \
    --used_features capacity free_flow_time volume voc shortest_route_count travel_time spec0

python get_lowdim_functions.py \
    --data_path ./data/globalsouth \
    --datasets "Ahmedabad" "Bandung" "Bengaluru" "Bogotá" "Bucaramanga" "Cartagena" "Chennai" "Cúcuta" "Delhi" "Depok" "Ecatepec" "Guadalajara" "Hyderabad" "Ibagué" "Jaipur" "Kanpur" "Kolkata" "León" "Lucknow" "Makassar" "Medan" "Medellín" "Mérida" "Montería" "Mumbai" "Palembang" "Puebla City" "Pune" "Querétaro City" "Santa Marta" "Santiago de Cali" "Semarang" "Surabaya" "Surat" "Valledupar" \
    --name "globalsouth_get_lowdim_functions_4features" \
    --used_features capacity free_flow_time shortest_route_count spec0
