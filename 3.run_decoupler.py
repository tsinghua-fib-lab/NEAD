import sys
import json
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
from src.data.io import infer_raw_data_dir
from src.dataset.dataset import ResilienceDataset
from src.model.gnn_explainer import MyGNNExplainer, Wrapper

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

    for data in tqdm(dataset):
        data_path = Path(data.path)
        cityname = data_path.parent.name
        dirname = data_path.name
        save_path = Path(args.save_data_dir) / cityname / dirname
        save_path.mkdir(parents=True, exist_ok=True)

        node_exists = (save_path / "node.csv.gz").exists()
        edge_exists = (save_path / "edge.csv.gz").exists()
        graph_exists = (save_path / "graph.csv.gz").exists()
        if not (node_exists and edge_exists and graph_exists):
            wrapper = Wrapper(
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
            wrapper.load_state_dict(state_dict['model'])
            wrapper.to(args.device)
            wrapper.eval()

            if args.explain_for == 'feature':
                edge_feat_mask_type = 'common_attributes'
                node_mask_type = 'common_attributes'
            elif args.explain_for == 'structure':
                edge_feat_mask_type = 'object'
                node_mask_type = 'object'
            elif args.explain_for == 'cross':
                raise NotImplementedError("cross mask not implemented yet.")
                edge_feat_mask_type = 'attributes'
                node_mask_type = 'attributes'
            else:
                raise ValueError(f"Unknown --explain_for: {args.explain_for}")
            explainer = Explainer(
                wrapper,
                algorithm=MyGNNExplainer(
                    epochs=args.num_epochs,
                    edge_feat_mask_type=edge_feat_mask_type,
                    graph_mask_type='common_attributes',
                ),
                explanation_type='phenomenon',
                model_config=dict(
                    mode='regression',
                    task_level='graph',
                    return_type='raw',
                ),
                node_mask_type=node_mask_type,
                edge_mask_type=None,
            )

            data = data.to(args.device)
            explanation = explainer(
                x=data.x, 
                target=data.y, 
                edge_index=data.edge_index1, 
                batch=data.batch, 
                edge_attr=data.edge_attr1, 
                graph_attr=data.graph_attr
            )

            if args.explain_for == 'structure':
                df_node = pd.DataFrame(data.x.cpu().numpy(), columns=args.node_features)
                df_node['mask'] = explanation.node_mask.squeeze().detach().cpu().numpy()

                df_edge = pd.DataFrame(data.edge_attr1.cpu().numpy(), columns=args.edge1_features)
                df_edge['mask'] = explanation.edge_feat_mask.squeeze().detach().cpu().numpy()

                df_graph = pd.DataFrame(data.graph_attr.cpu().numpy().reshape(-1), columns=['graph_attr'])
                df_graph['mask'] = explanation.graph_mask.squeeze().detach().cpu().numpy()
            elif args.explain_for == 'feature':
                df_node = pd.DataFrame([explanation.node_mask.squeeze().detach().cpu().numpy()], columns=args.node_features)
                df_edge = pd.DataFrame([explanation.edge_feat_mask.squeeze().detach().cpu().numpy()], columns=args.edge1_features)
                graph_features = [f'spec{i}' for i in range(args.od_spec_dim)] + [f'dist{i}' for i in range(args.od_dist_dim)]
                df_graph = pd.DataFrame([explanation.graph_mask.squeeze().detach().cpu().numpy()], columns=graph_features)
            elif args.explain_for == 'cross':
                raise NotImplementedError("cross mask not implemented yet.")
            else:
                raise ValueError(f"Unknown --explain_for: {args.explain_for}")

            df_node.to_csv(save_path / 'node.csv.gz', index=False, compression='gzip')
            df_edge.to_csv(save_path / 'edge.csv.gz', index=False, compression='gzip')
            df_graph.to_csv(save_path / 'graph.csv.gz', index=False, compression='gzip')
            _logger.info(f'Processed {cityname}/{dirname} to {save_path}')
        else:
            df_node = pd.read_csv(save_path / 'node.csv.gz')
            df_edge = pd.read_csv(save_path / 'edge.csv.gz')
            df_graph = pd.read_csv(save_path / 'graph.csv.gz')
            _logger.info(f'Exists {cityname}/{dirname} at {save_path}, skip processing.')

        pdf_exists = (save_path / 'edge.pdf').exists()
        png_exists = (save_path / 'edge.png').exists()
        if args.visualize and not (pdf_exists and png_exists):
            if args.explain_for != 'structure':
                _logger.warning("Visualization only implemented for structure mask.")
                continue
            try:
                mask = df_edge['mask'].values
                cmap = plt.get_cmap('RdBu_r', 256)
                # norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
                norm = mcolors.Normalize(vmin=mask.min(), vmax=mask.max())
                fi, fig, axes = get_fig(1, 1, AW=8, AH=8, dpi=100, fontsize=8, lw=0.5)
                ax = axes[0]
                
                raw_data_dir = infer_raw_data_dir(data_path)
                gdf = pd.read_csv(raw_data_dir / cityname / 'link.csv')
                gdf['geometry'] = gdf['geometry'].apply(wkt.loads)
                gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs='EPSG:4326').to_crs(epsg=4326)
                gdf.plot(ax=ax, linewidth=0.3, color=cmap(norm(mask)), aspect=None, rasterized=False)
                try:
                    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.HOT, crs="EPSG:4326", attribution=False, alpha=0.2)
                except Exception as e:
                    _logger.warning(f"Failed to add basemap since [{type(e)}] {e}")
                ax.set_title(f'{cityname.split('_', 1)[1]}')

                x, y, w, h = fi['right_box']
                cax = fig.add_axes([x + 0.2 * fi['r_HS'], y, 0.2 * fi['r_HS'], h])
                cax.imshow(np.linspace(0, 1, 100)[:, None], aspect='auto', origin='lower', cmap=cmap, norm=norm, extent=[0, 1, 0, 1], rasterized=True)
                cax.yaxis.tick_right()
                cax.xaxis.set_visible(False)
                cax.yaxis.set_label_position('right')
                cax.set_ylabel('Edge Mask')

                fig.savefig(save_path / 'edge.pdf', bbox_inches='tight', transparent=True)
                fig.savefig(save_path / 'edge.png', bbox_inches='tight', transparent=True, dpi=600)
                _logger.info(f"Visualized {cityname}/{dirname} at {save_path}/edge.pdf&png")
            except Exception as e:
                _logger.error(f"Failed to visualize {cityname}/{dirname}: [{type(e)}] {e}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
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
    parser.add_argument('--save_dir', type=str, default='./logs/run_gnnexplainer')
    parser.add_argument('--skip_existing', action='store_true', default=False)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--save_data_dir', type=str, required=True) # ./{model_path}/gnnexplainer/
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
