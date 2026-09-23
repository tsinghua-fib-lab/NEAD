python speed_test2.py \
    --data_path ./data/augmentation/*/raw \
    --model_path "./logs/train/20251119_wo-traffic-assignment_160956_LM2/best_model.pth" \
    --edge1_features capacity shortest_route_count free_flow_time \
    --edge2_features