import sys
import time
import json
import shlex
import random
import logging
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from tqdm.rich import tqdm
from itertools import islice
from datetime import datetime
from socket import gethostname
from argparse import ArgumentParser
from setproctitle import setproctitle
from absl import logging as absl_logging
from src.utils.random import set_seed
from src.utils.logger import init_logger
from src.data.io import infer_raw_data_dir, read_data
from src.data.analyze import get_disrupting_links
from src.data.assignment import traffic_assignment
from src.pipeline.pipeline import load_od, load_free_assign, load_disrupt_assign, load_resilience, load_shortest_routes

_logger = logging.getLogger('src')


def main(args):
    exp_start_time = time.time()
    save_path = Path(args.save_path)

    # Read Network & OD data
    raw_data_path = Path(args.raw_data_path)
    dataset = raw_data_path.parent.name # e.g., '0_New York city'
    sample = raw_data_path.name # e.g., 'raw'
    raw_data_dir = infer_raw_data_dir(raw_data_path)
    aem, original_network, index = read_data(
        dataset, data_root_path=raw_data_dir
    )

    # Scale OD
    raw_od = aem.matrix['matrix'].copy()
    aem.matrix['matrix'][:, :] *= args.scale_ratio
    _logger.info(
        f"Scale OD matrix with ratio {args.scale_ratio}. "
        f"Total demand changes from {raw_od.sum()} to {aem.matrix['matrix'].sum()}."
    )
        
    ## Noisy OD matrix
    aem = load_od(args, aem, sample, save_path)

    ## Free assignment
    total_travel_time, free_assig_results, all_disrupting_links = load_free_assign(args, aem, original_network, index, save_path)

    ## Disrupted assignment
    disrupted_travel_time = load_disrupt_assign(args, aem, original_network, index, free_assig_results, all_disrupting_links, save_path)

    ## Calculate Resilience Metrics
    load_resilience(args, total_travel_time, disrupted_travel_time, save_path)

    ## Shortest Routes
    if args.with_shortest_route_count:
        load_shortest_routes(args, aem, original_network, free_assig_results, save_path)

    _logger.info(f"--- Exp time cost: {time.time() - exp_start_time} seconds ---")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default='scale_od')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--raw_data_path', type=str, help='data/argumentation/0_New York city/raw')
    parser.add_argument("--scale_ratio", type=float, default=1.0, help="Scaling ratio for OD.")
    parser.add_argument("--disrupting_term", type=str, default="free_flow_time", help="Strategy for disrupting road networks.")
    parser.add_argument("--disrupting_strategy", type=str, default="greedy", help="Strategy for disrupting road networks.")
    parser.add_argument(
        "--disrupting_indicator",
        type=str,
        choices=["nil", "matrix_ab", "Delay_factor_AB", "VOC_AB", "degree", "betweenness", "betweenness_od", "closeness"],
        default="matrix_ab",
        help="Indicator for selecting links to disrupt."
    )
    parser.add_argument("--disrupting_fraction", type=int, nargs='+', default=[10], help="Fraction of links to disrupt.")
    parser.add_argument("--ta_algorithm", type=str, default="bfw", help="Traffic assignment algorithm.")
    parser.add_argument("--vdf", type=str, default="BPR", help="Volume delay function.")
    parser.add_argument("--rgap_target", type=float, default=1e-5, help="Relative gap target for traffic assignment.")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations for traffic assignment.")
    parser.add_argument("--consider_delay_factor", action='store_true', help="Whether to consider delay factor in travel time calculation.")
    parser.add_argument("--with_shortest_route_count", action='store_true', help="Whether to compute Shortest Route Count.")
    parser.add_argument("--od_pairs_num", type=int, default=100, help="Number of OD pairs to analyze.")
    parser.add_argument("--shortest_path_num", type=int, default=1, help="Number of shortest paths to analyze.")
    parser.add_argument("--save_dir", type=str, default="./logs/scale_od", help="Directory to save the results.")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name. If None, will be generated automatically.")
    parser.add_argument("--skip_existing", action='store_true', help="Whether to skip the experiment if the save path already exists.")
    parser.add_argument("--fix_existing", action='store_true', help="Whether to fix the existing experiment if the save path already exists.")
    args, unknown = parser.parse_known_args()
    
    # Pre-init logger `src`
    init_logger("src", exp_name=args.name, info_level="debug")

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
    existing_running = list(Path(args.save_dir).glob(f'*_{name}_*'))
    if any(existing_running):
        if args.skip_existing:
            _logger.warning(f'Skip running since {existing_running} has been existing.')
            exit(0)
        elif args.fix_existing:
            args.exp_name = existing_running[0].name
            if len(existing_running) > 1:
                _logger.warning(f'Multiple existing runnings found: {existing_running}. Using the first one: {args.exp_name}.')
        else:
            _logger.warning(f'Found existing running: {existing_running}. Use --skip_existing or --fix_existing to avoid overwriting.')

    save_path = Path(args.save_dir) / args.exp_name
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    elif args.fix_existing:
        _logger.info(f"Fix existing save path {save_path}.")
    else:
        _logger.warning(f"Save path {save_path} already exists.")
    args.save_path = str(save_path)

    ## Set Seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    set_seed(args.seed)

    ## Save Command
    args.command = ' '.join(map(shlex.quote, [sys.executable, *sys.argv]))
    ## Set other args
    args.return_centrality = args.disrupting_indicator in ["degree", "betweenness", "betweenness_od", "closeness"]

    ## Init logger `src`
    init_logger(
        "src",
        exp_name=args.exp_name,
        log_file=save_path / "info.log",
        info_level="debug",
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

    # Start Running
    setproctitle(f"{args.exp_name}@ZihanYu")
    main(args)
