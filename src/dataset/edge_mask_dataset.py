import json
import torch
import logging
import torch.utils.data as D
import numpy as np
import pandas as pd
from pathlib import Path

_logger = logging.getLogger(__name__)


class EdgemaskDataset(D.Dataset):
    def __init__(self, args, sort_roads=False):
        super().__init__()
        self.args = args
        self.sort_roads = sort_roads

        resilience_map = pd.read_csv("data/resilience.csv")
        resilience_map = resilience_map.set_index(["dataset", "dirname"])
        self.resilience_map = resilience_map

        mean_std_path = Path(args.data_dir).parent.parent / 'mean_std.json'
        with open(mean_std_path, 'r') as f:
            mean_std = json.load(f)
        _logger.debug(f"mean_std: {mean_std}")
        self.mean_std = mean_std
        
        root_path = Path(args.data_dir)
        loader = []
        for dataname in args.datasets:
            paths = list((root_path / dataname).glob('*'))
            loader.extend(paths)
        valid_loader = []
        for path in loader:
            if not (
                (path / "node.csv.gz").exists()
                and (path / "edge.csv.gz").exists()
                and (path / "graph.csv.gz").exists()
            ):
                _logger.warning(f"Missing File in {path}")
                continue
            valid_loader.append(path)
        self.loader = valid_loader

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, idx):
        path = self.loader[idx]
        dataname = path.parent.name
        dirname = path.name

        df_node = pd.read_csv(path / "node.csv.gz", compression="gzip")
        # for feature in self.args.node_features:
        #     mean, std = self.mean_std[feature]
        #     df_node[feature] = df_node[feature] * std + mean

        df_edge = pd.read_csv(path / "edge.csv.gz", compression="gzip")
        # for feature in self.args.edge1_features:
        #     mean, std = self.mean_std[feature]
        #     df_edge[feature] = df_edge[feature] * std + mean

        df_graph = pd.read_csv(path / "graph.csv.gz", compression="gzip")
        df_spec = df_graph[:self.args.od_spec_dim] 
        # mean, std = self.mean_std["od_spec"]
        # df_spec = df_spec * std + mean
        df_dist = df_graph[self.args.od_spec_dim:].reset_index(drop=True)
        # mean, std = self.mean_std["od_dist"]
        # df_dist = df_dist * std + mean

        if self.args.important_edge_threshold is not None:
            edge_mask = df_edge["mask"] >= self.args.important_edge_threshold
        elif self.args.important_edge_ratio is not None:
            edge_mask = df_edge["mask"] >= df_edge["mask"].quantile(1 - self.args.important_edge_ratio)
        else:
            edge_mask = torch.ones(len(df_edge), dtype=bool)
        if self.args.reverse_important_edges:
            edge_mask = ~edge_mask
        if edge_mask.any():
            df_edge = df_edge[edge_mask]
        else:
            _logger.debug(f"No edge left in {path} with threshold {self.args.threshold}")

        # if not hasattr(_logger, '_debug_shown'):
        #     _logger.debug(f"Select {len(df_edge)}/{len(edge_mask)} edges in {path} (threshold={self.args.threshold}, by_ratio={self.args.by_ratio}, reverse={self.args.reverse})")
        #     _logger._debug_shown = True

        more_features = {}
        for x in self.args.node_features:
            more_features[x] = df_node[x].mean()
        for x in self.args.edge1_features:
            more_features[x] = df_edge[x].mean()
        for spec_idx in self.args.spec_indices:
            more_features[f'spec{spec_idx}'] = df_spec.loc[spec_idx, "graph_attr"]
        more_features['spec_avg'] = df_spec["graph_attr"].mean()
        for dist_idx in self.args.dist_indices:
            more_features[f'dist{dist_idx}'] = df_dist.loc[dist_idx, "graph_attr"]
        more_features['dist_avg'] = df_dist["graph_attr"].mean()

        for x in self.args.used_features:
            if x not in df_edge.columns:
                df_edge[x] = more_features[x]
        if self.sort_roads:
            df_edge = df_edge.sort_values('mask', ascending=False)
        df_edge = df_edge[self.args.used_features]

        try:
            resilience = self.resilience_map.loc[dataname, dirname]['resilience']
        except Exception as e:
            _logger.warning(e)
            tmp_path = f'./data/augmentation/{dataname}/{dirname}/result_obs.csv.gz'
            assig_results = pd.read_csv(tmp_path, compression='gzip')
            factor = (1-1/assig_results['Delay_factor_AB']) if self.args.consider_delay_factor else 1
            total_travel_time = (assig_results['matrix_ab'] * assig_results['Congested_Time_AB'] * factor).sum()

            disrupted_travel_time = []
            for ratio in [20, 40, 60, 80]:
                tmp_path = f'./data/augmentation/{dataname}/{dirname}/result_disrupt{ratio:03d}.csv.gz'
                assig_results = pd.read_csv(tmp_path, compression='gzip')
                factor = (1-1/assig_results['Delay_factor_AB']) if self.args.consider_delay_factor else 1
                disrupted_travel_time.append((assig_results['matrix_ab'] * assig_results['Congested_Time_AB'] * factor).sum())
            disrupted_travel_time = np.array(disrupted_travel_time) / total_travel_time
            resilience = (1 + sum(disrupted_travel_time) + 50) / 6
            _logger.info(f">>> {dataname},{dirname},{resilience}")
        finally:
            mean, std = self.mean_std['resilience']
            resilience = (resilience - mean) / std
        
        return df_edge.values.astype(np.float32), np.array(resilience).astype(np.float32), path
