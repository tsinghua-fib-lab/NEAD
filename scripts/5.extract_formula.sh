python search.py \
    --name "get_lowdim_functions_6features-normalize_x" \
    --data_path "./logs/get_lowdim_functions/get_lowdim_functions_6features-split/sr.csv.gz" \
    --use_all_features --target output \
    --sample_num 50000 \
    --no_normalize_y

python search.py \
    --name "get_lowdim_functions_all_7_features-normalize_xy" \
    --data_path "./logs/get_lowdim_functions/20251124_get_lowdim_functions_all_7_features_144849_rl2/sr.csv.gz" \
    --use_all_features --target output \
    --sample_num 50000
