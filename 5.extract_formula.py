import os
import sys
import json
import shlex
import random
import logging
import numpy as np
import sympy as sp
import pandas as pd
from pathlib import Path
from datetime import datetime
from pysr import PySRRegressor
from socket import gethostname
from argparse import ArgumentParser
from setproctitle import setproctitle
from sklearn.model_selection import train_test_split
from src.utils.random import set_seed
from src.utils.logger import init_logger
from src.utils.save_dir import set_save_dir
from src.utils.metrics import r2, rmse, mae, p_spearman, p_pearson, mape, smape

_logger = logging.getLogger("src")


def main(args):
    save_path = Path(args.save_path)

    # Prepare Data
    df = pd.read_csv(args.data_path)
    if args.subset == 'lt30':
        df = df[df[args.target] < 30]
    elif args.subset == 'gt30':
        df = df[df[args.target] >= 30]
    elif args.subset == 'full':
        pass
    else:
        raise ValueError(f"Unknown subset: {args.subset}")

    if args.raw_only:
        raw_len = len(df)
        df = df.loc[df['dirname'].eq('raw')]
        _logger.info(f"Using only raw data, total {len(df)}/{raw_len} samples.")

    if args.sample_num is not None and len(df) > args.sample_num:
        df = df.sample(args.sample_num, random_state=args.seed)
    if args.features is None:
        args.features = [col for col in df.columns if col != args.target and df[col].dtype in [np.float32, np.float64, np.int32, np.int64]]
    _logger.note(f"Data shape: {df.shape}, features: {args.features}, target: {args.target}")

    X = df[args.features]
    y = df[args.target]
    if args.normalize:
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X = (X - X_mean) / X_std
    else:
        X_mean = np.zeros_like(X.mean(axis=0))
        X_std = np.ones_like(X.std(axis=0))
    if args.normalize_y:
        y_mean = y.mean()
        y_std = y.std()
        y = (y - y_mean) / y_std
    else:
        y_mean = 0
        y_std = 1
    for i, col in enumerate(X.columns):
        _logger.info(f"Feature {col}: use mean={X_mean[i]:.4f} and std={X_std[i]:.4f} to normalize")
    _logger.info(f"Target {args.target} use mean={y_mean:.4f} and std={y_std:.4f} to normalize")

    if args.dataset_split is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-args.dataset_split, random_state=args.seed)
    else:
        X_train = X_test = X
        y_train = y_test = y

    # Prepare Params
    complexity_of_variables = {
        "volume": 1, "voc": 1, "disrupted_rank": 1, 
        "capacity": 1, "shortest_route_count": 1, "free_flow_time": 1, 
        "spec0": 1, "spec_avg": 2, "dist0": 1, "dist1": 2, "dist63": 2, "dist_avg": 2,
    }
    params = dict(
        maxsize=args.maxsize,
        niterations=args.niterations,  # < Increase me for better results
        binary_operators=["+", "*", "-", "/"],
        unary_operators=[ "exp", "log", "tanh", "sqrt", "inv"],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        complexity_of_variables=[complexity_of_variables.get(col, 1) for col in X.columns],
        random_state=args.seed,
        # complexity_of_operators={"tanh": 2},
        nested_constraints={
            "tanh": {"tanh": 0},
            "exp": {"exp": 0, "tanh": 0},
        },
        batching=True,
    )
    _logger.info(", ".join(f"{k}={v}" for k, v in params.items()))
    with open(save_path / "pysr_params.json", "w") as f:
        def serialize(obj):
            if callable(obj):
                return obj.__name__
            raise TypeError(f"Type {type(obj)} not serializable")
        f.write(json.dumps(params, indent=4, ensure_ascii=False, default=serialize))

    # Prepare Model
    model = PySRRegressor(**params)
    if args.load_checkpoint:
        model = model.from_file(run_directory=args.load_checkpoint)

    # Search Equations
    try:
        model.fit(X_train, y_train)
    except KeyboardInterrupt:
        _logger.warning("KeyboardInterrupt: Stopping training and saving current results...")

    # Evaluate & Save Results
    results = model.equations_.copy()
    for index in results.index:
        rescaled_equation = results.loc[index, 'sympy_format']
        if args.normalize:
            for i, col in enumerate(X.columns):
                rescaled_equation = rescaled_equation.subs(sp.Symbol(col), (sp.Symbol(col) - X_mean[i]) / X_std[i])
            rescaled_equation = rescaled_equation * y_std + y_mean
        results.loc[index, 'rescaled_equation'] = str(rescaled_equation)
        results.loc[index, 'r2'] = r2(y_test, model.predict(X_test, index=index))
        results.loc[index, 'p_spearman'] = p_spearman(y_test, model.predict(X_test, index=index))
        results.loc[index, 'p_pearson'] = p_pearson(y_test, model.predict(X_test, index=index))
        results.loc[index, 'mape'] = mape(y_test, model.predict(X_test, index=index))
        results.loc[index, 'smape'] = smape(y_test, model.predict(X_test, index=index))
        results.loc[index, 'rmse'] = rmse(y_test, model.predict(X_test, index=index))
        results.loc[index, 'mae'] = mae(y_test, model.predict(X_test, index=index))

    results = results.drop(columns=['sympy_format', 'lambda_format'])
    save_file = save_path / "results.csv"
    results.to_csv(save_file, index=False)
    _logger.info(f"Results saved to {save_file}")

    logs = []
    for idx, (idx, row) in enumerate(results.iterrows()):
        logs.append(f'[{idx}] \033[1;32m{row["equation"]}\033[0m')
        logs.append(f'Rescaled: \033[1;33m{row["rescaled_equation"]}\033[0m')
        logs.append(f'\033[1;4mR2:\033[0m{row["r2"]:.4f}, '
            f'\033[1;4mP-Spearman:\033[0m{row["p_spearman"]:.4f}, '
            f'\033[1;4mP-Pearson:\033[0m{row["p_pearson"]:.4f}, '
            f'\033[1;4mMAPE:\033[0m{row["mape"]:.2%}, '
            f'\033[1;4msMAPE:\033[0m{row["smape"]:.2%}, '
            f'\033[1;4mRMSE:\033[0m{row["rmse"]:.4f}, '
            f'\033[1;4mMAE:\033[0m{row["mae"]:.4f}, '
            f'\033[1;4mComplexity:\033[0m{row['complexity']}, '
            f'\033[1;4mScore:\033[0m{row['score']}, ')
        logs.append('')
    _logger.info("Pareto Front\n" + "\n".join(logs))

    if True:
        import torch.utils.data as D
        from tqdm import tqdm
        from argparse import Namespace
        from src.dataset.lowdim_dataset import LowdimDataset
        with open(Path(args.data_path).parent / 'args.json', 'r') as f:
            saved_args = Namespace(**json.load(f))
        dataset = LowdimDataset(saved_args, keep_mean_std=True)
        loader = D.DataLoader(
            dataset, batch_size=None, shuffle=False,
            num_workers=saved_args.num_workers, drop_last=False,
            collate_fn=lambda x: x,
        )
        records = {'path': [], 'true': [], **{f'pred-{idx}': [] for idx in range(len(results))}}
        for data, true, path in tqdm(loader, dynamic_ncols=True, disable=False):
            records['path'].append(path.parent.name + '-' + path.name)

            # mean, std = dataset.mean_std['resilience']
            # true = true.item() * std + mean
            records['true'].append(true)

            X = pd.DataFrame(data, columns=saved_args.used_features)
            # for k in args.features:
            #     if k in saved_args.node_features + saved_args.edge1_features + saved_args.edge2_features:
            #         mean, std = dataset.mean_std[k]
            #     elif k.startswith('spec'):
            #         mean, std = dataset.mean_std['od_spec']
            #     elif k.startswith('dist'):
            #         mean, std = dataset.mean_std['od_dist']
            #     else:
            #         raise ValueError(f'Unknown feature: {k}')
            #     X[k] = X[k] * std + mean
            for idx, row in results.iterrows():
                symbols = sp.symbols(args.features)
                f = sp.lambdify(symbols, sp.sympify(row['rescaled_equation']), modules='numpy')
                output = f(*(X[col].to_numpy() for col in args.features))
                if isinstance(output, (float, int)):
                    pred = output
                else:
                    pred = output.mean()
                records[f'pred-{idx}'].append(pred)
        df_records = pd.DataFrame(records)
        for idx in range(len(results)):
            results.loc[idx, 'r2'] = r2(df_records['true'], df_records[f'pred-{idx}'])
            results.loc[idx, 'p_spearman'] = p_spearman(df_records['true'], df_records[f'pred-{idx}'])
            results.loc[idx, 'p_pearson'] = p_pearson(df_records['true'], df_records[f'pred-{idx}'])
            results.loc[idx, 'mape'] = mape(df_records['true'], df_records[f'pred-{idx}'])
            results.loc[idx, 'smape'] = smape(df_records['true'], df_records[f'pred-{idx}'])
            results.loc[idx, 'rmse'] = rmse(df_records['true'], df_records[f'pred-{idx}'])
            results.loc[idx, 'mae'] = mae(df_records['true'], df_records[f'pred-{idx}'])
        
        results = results.sort_values(by=['r2'], ascending=True)
        save_file = save_path / "results_on_rawdata.csv"
        results.to_csv(save_file, index=False)
        _logger.info(f"Results on raw data saved to {save_file}")

        logs = []
        for idx, row in results.iterrows():
            logs.append(f'[{idx}] \033[1;32m{row["equation"]}\033[0m')
            logs.append(f'Rescaled: \033[1;33m{row["rescaled_equation"]}\033[0m')
            logs.append(f'\033[1;4mR2:\033[0m{row["r2"]:.4f}, '
                f'\033[1;4mP-Spearman:\033[0m{row["p_spearman"]:.4f}, '
                f'\033[1;4mP-Pearson:\033[0m{row["p_pearson"]:.4f}, '
                f'\033[1;4mMAPE:\033[0m{row["mape"]:.2%}, '
                f'\033[1;4msMAPE:\033[0m{row["smape"]:.2%}, '
                f'\033[1;4mRMSE:\033[0m{row["rmse"]:.4f}, '
                f'\033[1;4mMAE:\033[0m{row["mae"]:.4f}, '
                f'\033[1;4mComplexity:\033[0m{row['complexity']}, ')
            logs.append('')
        _logger.info("Pareto Front on Raw Data\n" + "\n".join(logs))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./logs/search')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--features', type=str, nargs='+', default=[
        'total_in', 'total_out', 
        'capacity', 'shortest_route_count', 'free_flow_time', 
        'spec0', 'spec_avg', 'dist0', 'dist1', 'dist63', 'dist_avg',
    ])
    parser.add_argument('--use_all_features', action='store_const', dest='features', const=None)
    parser.add_argument('--target', type=str, default='resilience') 
    parser.add_argument('--maxsize', type=int, default=40)
    parser.add_argument('--niterations', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--load_checkpoint', type=str, default=None, help="Path to load checkpoint, e.g., outputs/20250409_211518_dXB0Ii")
    parser.add_argument('--dataset_split', type=float, default=0.8)
    parser.add_argument('--sample_num', type=int, default=None)
    parser.add_argument('--no_dataset_split', dest='dataset_split', action='store_const', const=None)
    parser.add_argument('--no_normalize', dest='normalize', action='store_false', default=True)
    parser.add_argument('--no_normalize_y', dest='normalize_y', action='store_false', default=True)
    parser.add_argument('--subset', type=str, default='full', choices=['full', 'lt30', 'gt30'])
    parser.add_argument('--skip_existing', action='store_true', help="Skip running if existing experiment found.")
    parser.add_argument('--fix_existing', action='store_true', help="Fix existing experiment if found.")
    parser.add_argument('--exp_name', type=str, default=None, help="Experiment name. If not provided, will be generated.")
    parser.add_argument('--raw_only', action='store_true', help="Only use raw data for searching.")
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

    # Init logger `src`
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

    # Start Running
    setproctitle(f"{args.exp_name}@ZihanYu")
    main(args)
