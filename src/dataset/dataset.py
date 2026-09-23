import torch
import joblib
import logging
import numpy as np
import pandas as pd
import torch.utils.data as DATA
from pathlib import Path
from tqdm.rich import tqdm
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from sklearn.decomposition import TruncatedSVD
from ..data.io import infer_raw_data_dir, read_data

_logger = logging.getLogger(__name__)


class ResilienceDataset(DATA.Dataset):
    def __init__(
            self, 
            path_list=[], 
            node_features=['total_in', 'total_out'],
            edge1_features=['capacity', 'free_flow_time', 'shortest_route_count', 'volume', 'voc', 'travel_time', 'disrupted_rank'],
            edge2_features=['diversity', 'weighted_diversity'],
            graph_features=['od_spec', 'od_dist'],
            od_spec_dim=10,
            od_dist_dim=64,
            shortest_path_num=5,
            consider_delay_factor=True,
            cache_to='./data/.cache',
        ):
        """
        Args:
        - datasets: List of dataset names or paths, e.g., ['0_New York city', '1_San Francisco city'].
        - ignore_scale: Whether to ignore datasets that are used to test different OD scales (e.g., raw_10PCT )
        - node_features: List of node features to use
        - edge1_features: List of edge features to use
        - edge2_features: List of edge features to use for OD topology
        - graph_features: List of graph features to use
        - od_spec_dim: Dimension of the OD flow spectrum
        - od_dist_dim: Dimension of the OD flow distribution
        - consider_delay_factor: Whether to consider the delay factor in disrupted travel time calculation (When set to true, use time delay rather than time consume as the efficiency)
        """
        self.path_list = path_list
        self.node_features = node_features
        self.edge1_features = edge1_features
        self.edge2_features = edge2_features
        self.graph_features = graph_features
        self.od_spec_dim = od_spec_dim
        self.od_dist_dim = od_dist_dim
        self.shortest_path_num = shortest_path_num
        self.consider_delay_factor = consider_delay_factor
        self.cache_to = cache_to
        self.mean_std = defaultdict(lambda: (0, 1))  # Default mean and std for normalization

        def is_valid(path: Path) -> bool:
            if (path / 'result_obs.csv.gz').exists() and not (path / 'free_assignment.csv.gz').exists():
                _logger.warning(f'Please rename {path}/result_obs.csv.gz to free_assignment.csv.gz')
                return False

            # Exclude directories with incomplete data files.
            for file in [
                'od.csv.gz', 
                'free_assignment.csv.gz',
                'result_disrupt020.csv.gz', 
                'result_disrupt040.csv.gz', 
                'result_disrupt060.csv.gz', 
                'result_disrupt080.csv.gz',
                'shortest_routes.csv.gz',
                'resilience.txt',
            ]:
                if not (path / file).exists():
                    _logger.warning(f'Missing file {file} in {path}, skipping this dataset.')
                    return False
            # Optional resilience sanity check (disabled because it is too slow).
            if False:
                travel_time = lambda df: (df['matrix_ab'] * df['Congested_Time_AB']).sum()
                df = pd.read_csv(path / 'free_assignment.csv.gz', compression='gzip')
                t0 = travel_time(df)
                for ratio in [20, 40, 60, 80]:
                    df = pd.read_csv(path / f'result_disrupt{ratio:03d}.csv.gz', compression='gzip')
                    t = travel_time(df)
                    if t / t0 > 100:
                        _logger.warning(f'Abnormal resilience in {path}/result_disrupt{ratio:03d}.csv.gz: {t}/{t0}={t/t0}')
                        return False

            return True
        self.datasets = [(path.parent.name, path) for path in path_list if is_valid(path)]

    def set_mean_std(self, mean_std=None, num_workers=0):
        if mean_std is None:
            torch.multiprocessing.set_sharing_strategy('file_system')
            loader = torch.utils.data.DataLoader(
                self, batch_size=1, collate_fn=self.collate_fn, shuffle=False,
                num_workers=num_workers, drop_last=False
            )
            features = defaultdict(list)
            for data in tqdm(loader, desc='Calculating mean and std', total=len(self)):
                for k, v in zip(self.node_features, data.x.T):
                    features[k].append(v.numpy())
                for k, v in zip(self.edge1_features, data.edge_attr1.T):
                    features[k].append(v.numpy())
                for k, v in zip(self.edge2_features, data.edge_attr2.T):
                    features[k].append(v.numpy())
                features['od_spec'].append(data.graph_attr[0, 0:self.od_spec_dim].numpy())
                features['od_dist'].append(data.graph_attr[0, self.od_spec_dim:].numpy())
                features['disrupted_volume'].append(data.disrupted_volume[0, :].numpy())
                features['disrupted_travel_time'].append(data.disrupted_travel_time[0, :].numpy())
                features['resilience'].append(data.y[0, :].numpy())
            mean_std = {}
            for k, v in features.items():
                v = np.concatenate(v, axis=0)
                mean_std[k] = (v.mean().item(), v.std().item())
        self.mean_std = mean_std

    def __len__(self):
        return len(self.datasets)

    def calc_features_with_cache(self, index):
        dataset, path = self.datasets[index]
        if self.cache_to is None: # Do not use a cache.
            return self.calc_features(index)
        else:
            cache_file = Path(self.cache_to) / dataset / f"{path.name}.pkl"
            features = None
            if cache_file.exists():
                try:
                    features = joblib.load(cache_file)
                except:
                    _logger.warning(f"Failed to load cache from {cache_file}, recalculating...")
            if features is None:
                features = self.calc_features(index)
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(features, cache_file)
            return features

    def calc_features(self, index):
        """
        Returns:
            # Topology
            - topology: Urban road newtork topology, shape = (2, num_links)
            - od_topo: Sparsed Origin-Destination topology, shape = (2, num_od_pairs)
            
            # Node-Features
            - total_in: Total-in Flow (only "zone" node has non-zero values), shape = (num_nodes,)
            - total_out: Total-out Flow (only "zone" node has non-zero values), shape = (num_nodes,)
            
            # Edge-Features (topology)
            - capacity: Road capacity, shape = (num_links,)
            - free_flow_time: Free flow travel time, shape = (num_links,)
            - volume: Traffic volume, shape = (num_links,)
            - voc: Volume-over-Capacity, shape = (num_links,)
            - travel_time: Observed travel time, shape = (num_links,)
            - disrupted_rank: Disrupted rank, shape = (num_links,)
            - shortest_route_count: Number of shortest routes, shape = (num_links,)

            # Edge-Fetures (od_topo)
            - diversity: diversity of od_topo pairs, shape = (num_od_pairs,)
            - weighted_diversity: diversity of od_topo pairs, with length as weight, shape = (num_od_pairs,)
                        
            # Garph-Features
            - od_spec: OD flow spectrum, shape = (10,)
            - od_dist: OD flow distribution, shape = (64,)
            
            # Targets
            - disrupted_volume: Normalized disrupted traffic volume, shape = (num_links, 4)
                (20%, 40%, 60%, 80% disrupted) / free_volume
            - disrupted_travel_time: Normalized disrupted travel time, shape = (4,)
                (20%, 40%, 60%, 80% disrupted) / free_flow_time
            - resilience: Urban road newtork resilience, shape = (1,)
                mean(0, 20%, 40%, 60%, 80%, 100% disrupted) / free_flow_time
        """
        dataset, path = self.datasets[index]
        raw_data_dir = infer_raw_data_dir(path)
        _, network, _ = read_data(dataset, data_root_path=raw_data_dir)
        topology = np.stack([network['a_node'].values, network['b_node'].values], axis=0) - 1
        capacity = network['capacity'].values
        free_flow_time = network['free_flow_time'].values

        try:
            od = pd.read_csv(path / 'od.csv.gz', header=None).values
        except Exception as e:
            raise RuntimeError(f"Failed to read OD matrix from {path / 'od.csv.gz'}: {e}")
        num_zone = od.shape[0]
        num_node = topology.max() + 1
        num_link = topology.shape[1]
        total_in = np.zeros(num_node)
        total_in[:num_zone] = od.sum(axis=0)
        total_out = np.zeros(num_node)
        total_out[:num_zone] = od.sum(axis=1)

        if od.size > 100:
            topk = np.argpartition(np.reshape(od, -1), -100)[-100:]
        else:
            topk = np.arange(od.size)
        row = topk // num_zone
        col = topk % num_zone
        od_topo = np.stack([row, col], axis=0)

        if num_zone >= self.od_spec_dim:
            svd = TruncatedSVD(n_components=self.od_spec_dim, algorithm='randomized')
            svd.fit(od)
            od_spec = svd.singular_values_
        else:
            svd = TruncatedSVD(n_components=num_zone, algorithm='randomized')
            svd.fit(od)
            od_spec = np.zeros(self.od_spec_dim)
            od_spec[:num_zone] = svd.singular_values_
        od_dist = np.histogram(od.reshape(-1), bins=self.od_dist_dim)[0] / od.size

        try:
            shortest_paths = pd.read_csv(path / 'shortest_routes.csv.gz')
        except Exception as e:
            raise RuntimeError(f"Failed to read shortest routes from {path / 'shortest_routes.csv.gz'}: {e}")
        shortest_paths['OD'] = shortest_paths['OD'].apply(eval)
        od_topo = np.array(shortest_paths['OD'].tolist()).T - 1
        shortest_paths['Paths'] = shortest_paths['Paths'].apply(eval)
        diversity = []
        weighted_diversity = []
        for paths in shortest_paths['Paths']:
            if 0 < len(paths) < self.shortest_path_num: 
                paths = paths + [paths[-1]] * (self.shortest_path_num - len(paths))
            idx, cnt = np.unique(sum(paths, []), return_counts=True)
            length = network.loc[idx-1, 'length'].values
            diversity.append(idx.size / np.sum(cnt).clip(1e-6))
            weighted_diversity.append(np.sum(length) / np.sum(length * cnt).clip(1e-6))
        diversity = np.array(diversity)
        weighted_diversity = np.array(weighted_diversity)

        network['cnt'] = 0
        idx, cnt = np.unique(sum(sum(shortest_paths['Paths'].tolist(), []), []), return_counts=True)
        network.loc[idx-1, 'cnt'] = cnt
        shortest_route_count = network['cnt'].values

        try:
            assig_results = pd.read_csv(path / 'free_assignment.csv.gz')
        except Exception as e:
            raise RuntimeError(f"Failed to read assignment results from {path / 'free_assignment.csv.gz'}: {e}")
        volume = assig_results['matrix_ab'].values
        voc = assig_results['VOC_AB'].values
        travel_time = assig_results['Congested_Time_AB'].values
        disrupted_rank = assig_results['Disrupted_Rank'] / assig_results['Disrupted_Rank'].max()
        factor = (1-1/assig_results['Delay_factor_AB']) if self.consider_delay_factor else 1
        total_travel_time = (assig_results['matrix_ab'] * assig_results['Congested_Time_AB'] * factor).sum()

        disrupted_volume = []
        disrupted_travel_time = []
        for ratio in [20, 40, 60, 80]:
            try:
                assig_results = pd.read_csv(path / f'result_disrupt{ratio:03d}.csv.gz')
            except Exception as e:
                raise RuntimeError(f"Failed to read disrupted assignment results from {path / f'result_disrupt{ratio:03d}.csv.gz'}: {e}")
            disrupted_volume.append(assig_results['matrix_ab'].values)
            factor = (1-1/assig_results['Delay_factor_AB']) if self.consider_delay_factor else 1
            disrupted_travel_time.append((assig_results['matrix_ab'] * assig_results['Congested_Time_AB'] * factor).sum())
        disrupted_volume = np.log10(np.stack(disrupted_volume, axis=-1).clip(1e-6)) - np.log10(volume[:, np.newaxis].clip(1e-6))
        disrupted_travel_time = np.array(disrupted_travel_time) / total_travel_time
        resilience = (1 + sum(disrupted_travel_time) + 50) / 6
        
        resilience_saved = float((path / 'resilience.txt').read_text().strip())
        if abs(resilience / resilience_saved - 1) > 0.01:
            _logger.warning(f"Resilience mismatch for {path}: calculated {resilience}, saved {resilience_saved}")

        return dict(
            topology=topology,
            od_topo=od_topo,
            total_in=total_in,
            total_out=total_out,
            capacity=capacity,
            free_flow_time=free_flow_time,
            volume=volume,
            voc=voc,
            travel_time=travel_time,
            disrupted_rank=disrupted_rank,
            shortest_route_count=shortest_route_count,
            diversity=diversity,
            weighted_diversity=weighted_diversity,
            od_spec=od_spec,
            od_dist=od_dist,
            disrupted_volume=disrupted_volume,
            disrupted_travel_time=disrupted_travel_time,
            resilience=resilience
        )

    def __getitem__(self, index):
        features = self.calc_features_with_cache(index)
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
            features[f] = (features[f] - mean) / std
        
        dataset, path = self.datasets[index]
        x = [features[i] for i in self.node_features] or [np.zeros_like(features['total_in'])]
        edge_attr1 = [features[i] for i in self.edge1_features] or [np.zeros_like(features['capacity'])]
        edge_attr2 = [features[i] for i in self.edge2_features] or [np.zeros_like(features['diversity'])]
        graph_attr = [features[i] for i in self.graph_features] or [0.0]
        data = Data(
            edge_index1=torch.from_numpy(features['topology']).to(torch.long),
            edge_index2=torch.from_numpy(features['od_topo']).to(torch.long),
            x=torch.from_numpy(np.stack(x, axis=-1)).to(torch.float32),
            edge_attr1=torch.from_numpy(np.stack(edge_attr1, axis=-1)).to(torch.float32),
            edge_attr2=torch.from_numpy(np.stack(edge_attr2, axis=-1)).to(torch.float32),
            graph_attr=torch.from_numpy(np.concatenate(graph_attr, axis=-1)).to(torch.float32).unsqueeze(0),
            disrupted_volume=torch.from_numpy(features['disrupted_volume']).to(torch.float32),
            disrupted_travel_time=torch.from_numpy(features['disrupted_travel_time']).to(torch.float32).unsqueeze(0),
            y=torch.from_numpy(np.array([[features['resilience']]])).to(torch.float32),
            dataset=dataset,
            path=path
        )
        return data

    @staticmethod
    def collate_fn(batch):
        batch = [data for data in batch if data is not None]
        if len(batch) == 0: 
            return None
        return Batch.from_data_list(batch)
