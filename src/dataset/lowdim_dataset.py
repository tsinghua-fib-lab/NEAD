import json
import torch
import logging
import torch.utils.data as D
import numpy as np
import pandas as pd
from pathlib import Path
from .dataset import ResilienceDataset

_logger = logging.getLogger(__name__)


class LowdimDataset(ResilienceDataset):
    def __init__(self, args, keep_mean_std=False, cache_to='./data/.cache'):
        root_path = Path(args.data_dir)
        path_list = []
        for dataname in args.datasets:
            for path in sorted((root_path / dataname).glob('*')):
                path_list.append(path)
        super().__init__(
            path_list=path_list,
            node_features=args.node_features,
            edge1_features=args.edge1_features,
            edge2_features=args.edge2_features,
            graph_features=args.graph_features,
            od_spec_dim=args.od_spec_dim,
            od_dist_dim=args.od_dist_dim,
            shortest_path_num=args.shortest_path_num,
            consider_delay_factor=args.consider_delay_factor,
            cache_to=cache_to
        )
        if not keep_mean_std:
            self.set_mean_std(num_workers=args.num_workers)
        self.args = args

    def __getitem__(self, idx):
        if not hasattr(self, 'args'):
            return super().__getitem__(idx)

        features = self.calc_features_with_cache(idx)
        for f in [
            *self.node_features,
            *self.edge1_features,
            *self.edge2_features,
            *self.graph_features,
            'disrupted_volume',
            'disrupted_travel_time',
            'resilience',
        ]:
            mean, std = self.mean_std[f]
            features[f] = (features[f] - mean) / np.clip(std, 1e-6, None)
        
        dataset, path = self.datasets[idx]

        df_dict = {}
        for f in self.args.used_features:
            if f in self.node_features:
                df_dict[f] = features[f].mean()
            elif f in self.edge1_features:
                df_dict[f] = features[f]
            elif f in self.edge2_features:
                df_dict[f] = features[f].mean()
            elif f.startswith('spec'):
                n = int(f.removeprefix('spec'))
                df_dict[f] = features['od_spec'][n]
            elif f.startswith('dist'):
                n = int(f.removeprefix('dist'))
                df_dict[f] = features['od_dist'][n]
            else:
                raise ValueError(f"Unknown feature: {f}")
        df = pd.DataFrame(df_dict)
        df = df[self.args.used_features]
        return df.values, features['resilience'], path
