import sys
import json
import torch
import shlex
import random
import logging
import torch.nn as nn
import torch.utils.data as D
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from socket import gethostname
from argparse import ArgumentParser
from setproctitle import setproctitle
from src.model.mlp import MLP
from src.utils.random import set_seed
from src.utils.auto_gpu import AutoGPU
from src.utils.timer import NamedTimer
from src.utils.logger import init_logger
from src.dataset.lowdim_dataset import LowdimDataset

_logger = logging.getLogger('src')


def main(args):
    save_path = Path(args.save_path)

    ## dataset
    dataset = LowdimDataset(args, cache_to='./data/.cache')
    _logger.info(dataset.mean_std)
    valid_indices = []
    if args.subset == 'full':
        pass
    elif args.subset == 'lt30':
        for i in tqdm(range(len(dataset))):
            data, true, path = dataset[i]
            mean, std = dataset.mean_std['resilience']
            if true * std + mean < 30:
                valid_indices.append(i)
        dataset.datasets = [dataset.datasets[i] for i in valid_indices]
    elif args.subset == 'gt30':
        for i in tqdm(range(len(dataset))):
            data, true, path = dataset[i]
            mean, std = dataset.mean_std['resilience']
            if true * std + mean > 30:
                valid_indices.append(i)
        dataset.datasets = [dataset.datasets[i] for i in valid_indices]
    else:
        raise ValueError(f"Unknown subset: {args.subset}")

    if args.dataset_split is None:
        train_set, test_set = dataset, dataset
    elif args.split_by_city:
        tmp = list(set(cityname for cityname, path in dataset.datasets))
        rng = np.random.default_rng(args.seed)
        rng.shuffle(tmp)
        train_num = int(len(tmp) * args.dataset_split)
        train_set = deepcopy(dataset)
        test_set = deepcopy(dataset)
        train_set.datasets = [(cityname, path) for cityname, path in dataset.datasets if cityname in tmp[:train_num]]
        test_set.datasets = [(cityname, path) for cityname, path in dataset.datasets if cityname in tmp[train_num:]]
    else:
        train_num = int(len(dataset) * args.dataset_split)
        test_num = len(dataset) - train_num
        train_set = deepcopy(dataset)
        test_set = deepcopy(dataset)
        datasets = deepcopy(dataset.datasets)
        rng = np.random.default_rng(args.seed)
        rng.shuffle(datasets)
        train_set.datasets = datasets[:train_num]
        test_set.datasets = datasets[train_num:]

    train_loader = D.DataLoader(
        train_set, 
        batch_size=None,
        shuffle=True, 
        num_workers=args.num_workers, 
        drop_last=False
    )
    test_loader = D.DataLoader(
        test_set, 
        batch_size=None, 
        shuffle=False, 
        num_workers=args.num_workers, 
        drop_last=False
    )
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        count = pd.merge(
            pd.DataFrame(np.array(np.unique([cityname for cityname, path in train_set.datasets], return_counts=True)).T, columns=['Dataset', 'Train']), 
            pd.DataFrame(np.array(np.unique([cityname for cityname, path in test_set.datasets], return_counts=True)).T, columns=['Dataset', 'Test']),
            on='Dataset', how='outer'
        ).fillna('')
        count['Total'] = count['Train'].apply(lambda x: 0 if x == '' else int(x)) + count['Test'].apply(lambda x: 0 if x == '' else int(x))
        _logger.info(f"Train: {len(train_set)} ({len(train_set)/len(dataset):.0%}), "
                     f"Test: {len(test_set)} ({len(test_set)/len(dataset):.0%})\n"
                     f"{count}")

    ## model
    in_dim = len(args.used_features)
    model = MLP(in_dim, 128, 128, 128, 1, activation=nn.ELU, out_activation=None, dropout=args.dropout)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    criterion = nn.MSELoss()

    ## Reload checkpoint
    if args.reload_checkpoint is not None:
        checkpoint = torch.load(args.reload_checkpoint, map_location=args.device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            _logger.info(f"Model reloaded from {args.reload_checkpoint}.")
        else:
            _logger.warning(f"No model found in checkpoint {args.reload_checkpoint}.")
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            _logger.info(f"Optimizer reloaded from {args.reload_checkpoint}.")
        else:
            _logger.warning(f"No optimizer found in checkpoint {args.reload_checkpoint}.")
        if 'args' in checkpoint:
            for key in vars(args) | vars(checkpoint['args']):
                if (current_val := getattr(args, key, None)) != (saved_val := getattr(checkpoint['args'], key, None)):
                    _logger.warning(f"Arg {key} mismatch: current {current_val} vs checkpoint {saved_val}")
        else:
            _logger.warning(f"No args found in checkpoint {args.reload_checkpoint}.")
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            _logger.info(f"Resuming training from epoch {start_epoch}.")
        else:
            start_epoch = 1
            _logger.info(f"Starting training from epoch {start_epoch}.")
    else:
        start_epoch = 1
        _logger.info(f"Starting training from epoch {start_epoch}.")


    ## train
    timer = NamedTimer()
    for epoch in range(start_epoch, args.num_epochs+1):
        record = {'epoch': epoch}

        ## Train
        model.train()
        train_loss = 0
        true_list = []
        pred_list = []
        path_list = []
        pbar = tqdm(train_loader, dynamic_ncols=True, disable=False, leave=False, desc=f"Epoch {epoch}/{args.num_epochs} Training")
        timer.add('preprocess')
        for sample_idx, (data, true, path) in enumerate(pbar):
            timer.add('train-data')
            data, true = data.to(args.device).float(), true.to(args.device).float()
            output = model(data).squeeze(-1)
            pred = output.mean()
            l1 = (output - pred).abs().mean()
            l2 = output.var() # ((output - pred)**2).mean()
            loss = criterion(pred, true) - args.alpha * l1 - args.alpha2 * l2
            timer.add('train-forward')
            loss.backward()
            timer.add('train-backward')
            if sample_idx % args.batch_size == 0 or \
               sample_idx == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()
            train_loss += loss.item()
            true_list.append(true.cpu().item())
            pred_list.append(pred.detach().cpu().item())
            path_list.append(path.parent.name + '-' + path.name)
            timer.add('train-step')
        train_loss /= len(train_loader.dataset)
        true_list = np.array(true_list)
        pred_list = np.array(pred_list)
        train_r2 = 1 - np.mean((true_list - pred_list)**2) / np.var(true_list)
        record['train_loss'] = train_loss
        record['train_r2'] = train_r2
        record['train_true'] = true_list.tolist()
        record['train_pred'] = pred_list.tolist()
        record['train_path'] = path_list

        ## Test
        model.eval()
        test_loss = 0
        true_list = []
        pred_list = []
        path_list = []
        with torch.no_grad():
            pbar = tqdm(test_loader, dynamic_ncols=True, disable=False, leave=False, desc=f"Epoch {epoch}/{args.num_epochs} Testing")
            timer.add('test-preprocess')
            for data, true, path in pbar:
                timer.add('test-data')
                data, true = data.to(args.device).float(), true.to(args.device).float()
                output = model(data).squeeze(-1)
                pred = output.mean()
                loss = criterion(pred, true) - args.alpha * output.abs().mean() - args.alpha2 * (output**2).mean()
                test_loss += loss.item()
                timer.add('test-forward')
                true_list.append(true.cpu().item())
                pred_list.append(pred.cpu().item())
                path_list.append(path.parent.name + '-' + path.name)
        test_loss /= len(test_loader.dataset)
        true_list = np.array(true_list)
        pred_list = np.array(pred_list)
        test_r2 = 1 - np.mean((true_list - pred_list)**2) / np.var(true_list)
        record['test_loss'] = test_loss
        record['test_r2'] = test_r2
        record['test_true'] = true_list.tolist()
        record['test_pred'] = pred_list.tolist()
        record['test_path'] = path_list

        if 'best_record' not in locals() or record['test_r2'] > best_record['test_r2']:
            best_record = record
            patience = args.patience
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": args,
            }, save_path / 'model_best.pth')
            _logger.info(f"New best model saved with test record {best_record['test_r2']:.4f}")
        else:
            patience -= 1
            _logger.info(
                f"No improvement, patience left {patience}/{args.patience} "
                f"(best {best_record['test_r2']:.4f} at Epoch {best_record['epoch']})"
            )
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": args,
        }, save_path / 'model_final.pth')

        with open(save_path / 'records.jsonl', 'a') as f:
            f.write(json.dumps(record) + '\n')

        log = {}
        log['Epoch'] = f'{epoch}/{args.num_epochs}'
        log['Train Loss'] = f'{train_loss:.4f}'
        log['Train R2'] = f'{train_r2:.4f}'
        log['Test Loss'] = f'{test_loss:.4f}'
        log['Test R2'] = f'{test_r2:.4f}'
        log['Best Test R2'] = f'{best_record['test_r2']:.4f}'
        log['Patience'] = f'{patience}'
        log['Time Usage'] = str(timer)
        _logger.info(', '.join(f'\033[4m{k}\033[0m={v}' for k, v in log.items()))

        if patience == 0:
            _logger.info(
                f"Early stopping at epoch {epoch}, "
                f"best test R2={best_record['test_r2']:.4f} at Epoch {best_record['epoch']}"
            )
            break
    _logger.info(
        f"Training finished with best test R2={best_record['test_r2']:.4f} at Epoch {best_record['epoch']}"
    )
    
    if args.save_data:
        loader = D.DataLoader(
            dataset, 
            batch_size=None, 
            shuffle=False, 
            num_workers=args.num_workers, 
            drop_last=False
        )
        df_list = []
        df_detail_list = []
        for idx, (data, true, path) in enumerate(tqdm(loader, desc='Preparing lowdim function data', dynamic_ncols=True)):
            data, true = data.to(args.device).float(), true.to(args.device).float()
            path = dataset.path_list[idx]
            dataname = path.parent.name
            dirname = path.name
            output = model(data).squeeze(-1)
            pred = output.mean()
            true = true.cpu().item()
            pred = pred.detach().cpu().item()
            output = output.detach().cpu().numpy()

            mean, std = dataset.mean_std['resilience']
            true = true * std + mean
            pred = pred * std + mean
            output = output * std + mean
            df_list.append({
                'dataname': dataname, 'dirname': dirname, 'true': true, 'pred': pred,
            })

            df = pd.DataFrame(data.cpu(), columns=args.used_features)
            df['output'] = output
            df['dataname'] = dataname
            df['dirname'] = dirname
            for col in args.used_features:
                if col.startswith('spec'):
                    mean, std = dataset.mean_std['od_spec']
                elif col.startswith('dist'):
                    mean, std = dataset.mean_std['od_dist']
                else:
                    mean, std = dataset.mean_std[col]
                df[col] = df[col] * std + mean
            if len(df) > args.sample_num:
                # Randomly subsample to keep the exported road-level dataset manageable.
                df = df.sample(n=args.sample_num, random_state=args.seed)
            df_detail_list.append(df)
        df_mean = pd.DataFrame(df_list)
        df_detail = pd.concat(df_detail_list, ignore_index=True)

        df_detail.to_csv(save_path / 'sr.csv.gz', compression='gzip', index=False)
        _logger.note(f"Lowdim function data saved to {save_path / 'sr.csv.gz'}")

        r = df_mean['pred'].corr(df_mean['true'], method='pearson')
        r2 = 1 - np.mean((df_mean['true'] - df_mean['pred'])**2) / np.var(df_mean['true'])
        _logger.note(f"Overall R2 between true and pred: {r2:.4f} (Pearson R: {r:.4f})")

        if args.predictable_test:
            from sklearn.svm import SVR
            from sklearn.metrics import r2_score
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split, RandomizedSearchCV

            # sample and prepare data
            sample_n = min(10000, len(df_detail))
            data = df_detail.sample(n=sample_n, random_state=42)
            X = data.drop(columns=['dataname', 'dirname', 'output'])
            y = data['output']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # pipeline
            svm_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVR())
            ])
            # hyperparameter search space
            param_dist = {
                'svm__kernel': ['rbf'],
                'svm__C': [0.1, 1, 10, 100, 1000],
                'svm__gamma': ['scale', 'auto', 1e-3, 1e-2, 1e-1, 1],
                'svm__epsilon': [1e-3, 1e-2, 1e-1, 0.2],
            }
            # random search
            rs = RandomizedSearchCV(
                svm_pipeline,
                param_distributions=param_dist,
                n_iter=30,
                scoring='r2',
                cv=3,
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
            rs.fit(X_train, y_train)
            best_cv_score = rs.best_score_
            best_params = rs.best_params_
            best_model = rs.best_estimator_
            r2_test = r2_score(y_test, best_model.predict(X_test))
            _logger.note(f"Predictable Test: SVR best CV R2={best_cv_score:.4f}, Test R2={r2_test:.4f}, params={best_params}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='get_lowdim_functions')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--datasets', type=str, nargs='+', default=[
        "0_New York city","1_Los Angeles city","2_Chicago city","3_Houston city","4_Phoenix city",
        "5_Philadelphia city","6_San Antonio city","7_San Diego city","8_Dallas city","9_San Jose city",
        "10_Austin city","11_Jacksonville city","12_Fort Worth city","13_Columbus city","14_Charlotte city",
        "15_Indianapolis city (balance)","16_San Francisco city","17_Seattle city","18_Denver city","19_Oklahoma City city",
        "100_Hialeah city","101_San Bernardino city","102_Tacoma city","103_Port St. Lucie city","104_Spring Valley CDP",
        "105_Huntsville city","106_Modesto city","107_Des Moines city","108_Fontana city","109_Moreno Valley city",
        "110_Frisco city","111_Rochester city","112_Fayetteville city","113_Yonkers city","114_Cape Coral city",
        "115_Worcester city","116_Columbus city","117_Salt Lake City city","118_Little Rock city","119_McKinney city",
        "200_Orange city","201_Warren city","202_Columbia city","203_West Valley City city","204_Hampton city",
        "205_Cedar Rapids city","206_Dayton city","207_Pasadena city","208_Miramar city","209_Victorville city",
        "210_Elizabeth city","211_Stamford city","212_Kent city","213_Midland city","214_Coral Springs city",
        "215_Sterling Heights city","216_New Haven city","217_Carrollton city","218_Santa Clara city","219_Fargo city",
        "300_Sandy Springs city","301_El Monte city","302_Hillsboro city","303_Menifee city","304_Green Bay city",
        "305_Rio Rancho city","306_Concord city","307_Davie town","308_Nampa city","309_Boulder city",
        "310_Jurupa Valley city","311_Columbia CDP","312_Inglewood city","313_Spokane Valley city","314_Renton city",
        "315_San Tan Valley CDP","316_Burbank city","317_Brockton city","318_El Cajon city","319_Rialto city",
        "400_Fort Smith city","401_Clifton city","402_Waukegan city","403_Bloomington city","404_Champaign city",
        "405_Greenville city", "406_San Leandro city","407_Newton city","408_Lawrence city","409_Santa Fe city",
        "410_Santa Barbara city","411_Springdale city","412_Troy city","413_Citrus Heights city","414_Ogden city",
        "415_Duluth city","416_Deerfield Beach city","417_Town _n_ Country CDP","418_Manteca city","419_Temple city",
    ], help='List of datasets to train and test.')
    parser.add_argument('--node_features', type=str, nargs='*', default=[
        'total_in', 'total_out',
    ])
    parser.add_argument('--edge1_features', type=str, nargs='*', default=[
        'capacity', 'shortest_route_count', 'free_flow_time', 'volume', 'voc', 'travel_time', 'disrupted_rank', 
    ])
    parser.add_argument('--edge2_features', type=str, nargs='*', default=[
        'diversity', 'weighted_diversity',
    ])
    parser.add_argument('--graph_features', type=str, nargs='*', default=[
        'od_spec', 'od_dist',
    ])
    parser.add_argument('--used_features', type=str, nargs='+', default=[
        'capacity', 'shortest_route_count', 'free_flow_time', 'spec0', 'dist0',
    ])
    parser.add_argument('--consider_delay_factor', action='store_true')
    parser.add_argument('--od_spec_dim', type=int, default=10)
    parser.add_argument('--od_dist_dim', type=int, default=64)
    parser.add_argument('--save_dir', type=str, default='./logs/get_lowdim_functions')
    parser.add_argument('--data_dir', type=str, default='./data/augmentation')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--alpha', type=float, default=0.0, help='Coefficient for the mean absolute output as a regularization term.')
    parser.add_argument('--alpha2', type=float, default=0.0, help='Coefficient for the mean absolute output as a regularization term.')
    parser.add_argument('--split_by_city', action='store_true')
    parser.add_argument('--important_edge_threshold', type=float, default=None)
    parser.add_argument('--important_edge_ratio', type=float, default=None)
    parser.add_argument('--reverse_important_edges', action='store_true')
    parser.add_argument('--dataset_split', type=float, default=0.8)
    parser.add_argument('--spec_indices', type=int, nargs='*', default=[0])
    parser.add_argument('--dist_indices', type=int, nargs='*', default=[0, 1, 63])
    parser.add_argument('--no_dataset_split', dest='dataset_split', action='store_const', const=None)
    parser.add_argument('--subset', type=str, default='full', choices=['full', 'lt30', 'gt30'])
    parser.add_argument('--no-save_data', action='store_false', dest='save_data', default=True, help='Whether to save the lowdim function data for each road.')
    parser.add_argument('--sample_num', type=int, default=1000, help='Number of samples to save for each road\'s lowdim function data.')
    parser.add_argument('--no-predictable_test', action='store_false', dest='predictable_test', default=True, help='Whether to test the predictability of the lowdim function data using sklearn SVR.')
    parser.add_argument('--shortest_path_num', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='./data/augmentation')
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name. If None, will be generated automatically.")
    parser.add_argument('--skip_existing', action='store_true', help='Skip the experiment if an existing experiment with the same name is found.')
    parser.add_argument('--fix_existing', action='store_true', help='Use the existing experiment with the same name if found, instead of creating a new one.')
    parser.add_argument('--reload_checkpoint', type=str, default=None, help='Path to a checkpoint to reload the model and optimizer state.')
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
    # Set device
    if args.device == "auto":
        args.device = AutoGPU().choice_gpu(1000, interval=15, force=True)

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
