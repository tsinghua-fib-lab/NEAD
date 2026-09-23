import sys
import json
import time
import torch
import random
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from shapely import wkt
from pathlib import Path
from datetime import datetime
from socket import gethostname
from setproctitle import setproctitle
from torch_geometric.explain import Explainer
from argparse import ArgumentParser, Namespace
from src.utils.plot import get_fig
from src.utils.random import set_seed
from src.utils.auto_gpu import AutoGPU
from src.utils.logger import init_logger
from src.dataset.dataset import ResilienceDataset
from src.model.gnn import GNN

_logger = logging.getLogger("src")


def main(args):
    dataset = ResilienceDataset(
        path_list=[Path(p) for p in args.data_path],
        node_features=args.node_features,
        edge1_features=args.edge1_features,
        edge2_features=args.edge2_features,
        graph_features=args.graph_features,
        od_spec_dim=args.od_spec_dim,
        od_dist_dim=args.od_dist_dim,
        consider_delay_factor=args.consider_delay_factor,
        cache_to='./data/.cache',
    )
    mean_std_path = Path(args.model_path).parent / "mean_std.json"
    with open(mean_std_path, 'r') as f:
        mean_std = json.load(f)
    dataset.set_mean_std(mean_std)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, collate_fn=ResilienceDataset.collate_fn,
        shuffle=False, drop_last=False
    )

    data = dataset[0]
    model = GNN(
        node_channels=data.x.shape[-1],
        edge_channels1=data.edge_attr1.shape[-1],
        edge_channels2=data.edge_attr2.shape[-1],
        graph_channels=data.graph_attr.shape[-1],
        hidden_channels=args.Df,
        dropout=args.dropout,
        num_layers=args.num_layers
    )
    state_dict = torch.load(args.model_path, map_location=args.device, weights_only=False)
    saved_args = Namespace(**state_dict['args'])
    for k in set(vars(saved_args).keys()) & set(vars(args).keys()):
        if k in ['name', 'exp_name', 'save_dir', 'save_path', 'command', 'device', 'GPU_memory', 'num_epochs', 'seed']:
            continue
        if getattr(saved_args, k) != getattr(args, k):
            _logger.warning(f"Argument mismatch for '{k}': model saved with {getattr(saved_args, k)}, now is {getattr(args, k)}")
    model.load_state_dict(state_dict['model'])
    model.to(args.device)
    model.eval()

    for data in tqdm(loader):
        data_path = Path(data.path[0])
        cityname = data_path.parent.name
        dirname = data_path.name

        _start_time = time.time()
        save_path = Path(args.save_path) / 'results2'
        save_path.mkdir(parents=True, exist_ok=True)

        data = data.to(args.device)
        output = model(data)[0]
        resilience = output.item() * mean_std['resilience']['std'] + mean_std['resilience']['mean']
        duration = time.time() - _start_time
        (save_path / f'{cityname}_duration.txt').write_text(str(duration)) 
        (save_path / f'{cityname}_resilience.txt').write_text(str(resilience)) 
        _logger.info(f"Took {duration} seconds to calculate the {cityname}'s resilience of {resilience}")
                

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='speed_test2')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--Df', type=int, default=128)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--data_path', type=str, nargs='+', default=[
        './data/augmentation/0_New York city/raw'
    ], help='List of samples to explain.')
    parser.add_argument('--GPU_memory', type=int, default=5000)
    parser.add_argument('--node_features', type=str, nargs='*', default=[
        'total_in', 'total_out',
    ])
    parser.add_argument('--edge1_features', type=str, nargs='*', default=[
        'capacity', 'volume', 'voc', 'travel_time', 'disrupted_rank', 'shortest_route_count', 
    ])
    parser.add_argument('--edge2_features', type=str, nargs='*', default=[
        'diversity', 'weighted_diversity',
    ])
    parser.add_argument('--graph_features', type=str, nargs='*', default=[
        'od_spec', 'od_dist',
    ])
    parser.add_argument('--od_spec_dim', type=int, default=10)
    parser.add_argument('--od_dist_dim', type=int, default=64)
    parser.add_argument('--consider_delay_factor', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./logs/speed_test2')
    parser.add_argument('--skip_existing', action='store_true', default=False)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--explain_for', type=str, choices=['feature', 'structure', 'cross'], default='structure')
    parser.add_argument('--visualize', action='store_true')
    args, unknown = parser.parse_known_args()

    ## Pre-init logger `src`
    init_logger("src", exp_name=args.name, info_level="debug")

    ## Set device
    if args.device == "auto":
        args.device = AutoGPU().choice_gpu(args.GPU_memory, interval=15, force=True)

    ## Build Save Path
    if args.exp_name is None:
        now = datetime.now()
        date = now.strftime("%Y%m%d")
        hour = now.strftime("%H%M%S")
        host = gethostname()
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in invalid_chars:
            args.name = args.name.replace(char, '_')
        args.exp_name = f'{date}_{args.name}_{hour}_{host}'
    name = args.exp_name.split('_', 1)[1].rsplit('_', 2)[0]
    save_path = Path(args.save_dir) / args.exp_name
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        _logger.warning(f"Save path {save_path} already exists.")
    args.save_path = str(save_path)

    ## Set Seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    set_seed(args.seed)

    ## Save Command
    args.command = ' '.join([sys.executable, *sys.argv])

    ## Init logger `src`
    init_logger(
        "src",
        exp_name=args.name,
        log_file=save_path / "info.log",
        info_level="info",
    )

    ## Warn Unknown Args
    if unknown:
        _logger.warning(f"Unknown args: {unknown}")

    ## Save Args
    args_path = save_path / "args.json"
    if args_path.exists():
        i = 1
        while args_path.with_suffix(f".json.{i}").exists(): i += 1
        args_path.rename(args_path.with_suffix(f".json.{i}"))
        _logger.warning(f"args.json already exists, backup to args.json.{i}")
    _logger.note(f"Args: {args}")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

    setproctitle(f"{args.exp_name}@ZihanYu")
    main(args)

"""
python run_gnnexplainer.py \
    --name "run_gnnexplainer" \
    --save_dir "./logs/不使用交通分配的特征-不按城市分" \
    --model_path "./logs/不使用交通分配的特征-不按城市分/model_best.pth" \
    --edge1_features capacity shortest_route_count free_flow_time \
    --edge2_features

python run_gnnexplainer.py \
    --name "run_gnnexplainer" \
    --save_dir "./logs/只使用100cities训练-new2" \
    --model_path "./logs/只使用100cities训练-new2/model_best.pth" \

    """
