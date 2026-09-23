import sys
import json
import time
import torch
import shlex
import random
import psutil
import shutil
import logging
import traceback
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import absl.logging as absl_logging
from pathlib import Path
from copy import deepcopy
from tqdm.rich import tqdm
from datetime import datetime
from socket import gethostname
from argparse import ArgumentParser
from setproctitle import setproctitle
from src.model.gnn import GNN
from src.dataset.dataset import ResilienceDataset
from src.utils.plot import get_fig
from src.utils.random import set_seed
from src.utils.auto_gpu import AutoGPU
from src.utils.logger import init_logger
from src.utils.tag2ansi import tag2ansi
from src.utils.timer import NamedTimer
from src.utils.metrics import r2, p_spearman, p_pearson, mae, rmse, smape, mape

import warnings
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

_logger = logging.getLogger("src")


def main(args):
    save_path = Path(args.save_path)

    ## Load Data
    path_list = []
    data_dir = Path(args.data_dir)
    for dataset in args.datasets:
        for path in (data_dir / dataset).iterdir():
            if path.is_dir():
                path_list.append(path)
    dataset = ResilienceDataset(
        path_list=path_list,
        node_features=args.node_features,
        edge1_features=args.edge1_features,
        edge2_features=args.edge2_features,
        graph_features=args.graph_features,
        od_spec_dim=args.od_spec_dim,
        od_dist_dim=args.od_dist_dim,
        shortest_path_num=args.shortest_path_num,
        consider_delay_factor=args.consider_delay_factor,
        cache_to='./data/.cache/'
    )
    dataset.set_mean_std()
    _logger.info(dataset.mean_std)
    with open(save_path / "mean_std.json", "w") as f:
        f.write(json.dumps(dataset.mean_std, indent=4))
    if args.dataset_split is None:
        train_dataset, test_dataset = dataset, dataset
    elif args.fold_by_city is not None:
        tmp = sorted(list(set(d for d, _ in dataset.datasets)))
        rng = np.random.default_rng(args.seed)
        rng.shuffle(tmp)
        assert args.dataset_split == 0.8 # 5 Fold
        fold_num = int(len(tmp) * 0.2)
        test_fold = tmp[fold_num * args.fold_by_city : fold_num * (args.fold_by_city + 1)]
        train_dataset = deepcopy(dataset)
        test_dataset = deepcopy(dataset)
        train_dataset.datasets = [(d, p) for d, p in dataset.datasets if d not in test_fold]
        test_dataset.datasets = [(d, p) for d, p in dataset.datasets if d in test_fold]
        train_dataset.set_mean_std(dataset.mean_std)
        test_dataset.set_mean_std(dataset.mean_std)
    elif args.split_by_city:
        tmp = list(set(d for d, _ in dataset.datasets))
        rng = np.random.default_rng(args.seed)
        rng.shuffle(tmp)
        train_num = int(len(tmp) * args.dataset_split)
        train_dataset = deepcopy(dataset)
        test_dataset = deepcopy(dataset)
        train_dataset.datasets = [(d, p) for d, p in dataset.datasets if d in tmp[:train_num]]
        test_dataset.datasets = [(d, p) for d, p in dataset.datasets if d in tmp[train_num:]]
        train_dataset.set_mean_std(dataset.mean_std)
        test_dataset.set_mean_std(dataset.mean_std)
    else:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(dataset.datasets)
        train_num = int(len(dataset) * args.dataset_split)
        train_dataset = deepcopy(dataset)
        test_dataset = deepcopy(dataset)
        train_dataset.datasets=dataset.datasets[:train_num]
        test_dataset.datasets=dataset.datasets[train_num:]
        train_dataset.set_mean_std(dataset.mean_std)
        test_dataset.set_mean_std(dataset.mean_std)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, collate_fn=ResilienceDataset.collate_fn,
        shuffle=True, num_workers=args.num_workers, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, collate_fn=ResilienceDataset.collate_fn,
        shuffle=False, num_workers=args.num_workers, drop_last=False
    )
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        df_train = pd.DataFrame(np.array(np.unique([d for d, _ in train_dataset.datasets], return_counts=True)).T, columns=['Dataset', 'Train'])
        df_test = pd.DataFrame(np.array(np.unique([d for d, _ in test_dataset.datasets], return_counts=True)).T, columns=['Dataset', 'Test'])
        count = pd.merge(df_train, df_test, on='Dataset', how='outer').fillna('')
        count['Total'] = count['Train'].apply(lambda x: 0 if x == '' else int(x)) + count['Test'].apply(lambda x: 0 if x == '' else int(x))
        _logger.info(
            f"Train: {len(train_dataset)} ({len(train_dataset)/len(dataset):.0%}), "
            f"Test: {len(test_dataset)} ({len(test_dataset)/len(dataset):.0%})\n"
            f"{count}"
        )

    ## Init Model
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
    model.to(args.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    ## Reload Checkpoint
    if args.reload_checkpoint is not None:
        # 如果指定了 checkpoint 路径，则从该路径加载
        checkpoint_path = Path(args.reload_checkpoint)
    elif (save_path / "checkpoint.pth").exists():
        # 如果当前保存路径下存在 checkpoint，则从该路径加载
        checkpoint_path = save_path / "checkpoint.pth"
    else:
        checkpoint_path = None
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        start_epoch = checkpoint["epoch"] + 1
        if 'args' in checkpoint:
            saved_args = checkpoint['args']
            for key in sorted(set(saved_args.keys()) | set(vars(args).keys())):
                val1 = saved_args.get(key, None)
                val2 = getattr(args, key, None)
                if val1 != val2:
                    _logger.warning(
                        f"Argument '{key}' differs from the saved checkpoint: "
                        f"saved_args={val1} vs. current_args={val2}"
                    )
        model.load_state_dict(checkpoint["model"])
        _logger.note(tag2ansi(f"Checkpoint loaded from [underline green]{checkpoint_path}[reset], resume from epoch [underline green]{start_epoch}[reset]."))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            _logger.warning("Optimizer state not found in checkpoint, optimizer re-initialized.")
    else:
        start_epoch = 1

    ## Train Model
    timer = NamedTimer()
    for epoch in range(start_epoch, args.num_epochs+1):
        ## Train an epoch
        model.train()
        metrics = {'epoch': epoch, 'time': timer.time}
        loader = tqdm(train_loader, desc=f'Epoch {epoch} (Train)', leave=False, disable=False)
        for data_idx, data in enumerate(loader):
            data = data.to(args.device)
            pred_resilience, pred_time, pred_volume = model(data)
            loss1 = criterion(pred_resilience, data.y)
            loss2 = criterion(pred_time, data.disrupted_travel_time)
            loss3 = criterion(pred_volume, data.disrupted_volume) 
            if args.mix_loss:
                loss = 1.0 * loss1 + 1e-3 * loss2 + 1e-3 * loss3
            else:
                loss = loss1
            metrics.setdefault('loss', []).append(loss.item())
            metrics.setdefault('loss1', []).append(loss1.item())
            metrics.setdefault('loss2', []).append(loss2.item())
            metrics.setdefault('loss3', []).append(loss3.item())
            if epoch > 0 or not args.test_before_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()
            loader.set_postfix_str(f'loss={loss.item():.4f}')
        metrics['loss'] = np.mean(metrics['loss'])
        metrics['loss1'] = np.mean(metrics['loss1'])
        metrics['loss2'] = np.mean(metrics['loss2'])
        metrics['loss3'] = np.mean(metrics['loss3'])
        timer.add('train')

        ## Test an epoch
        model.eval()
        loader = tqdm(test_loader, desc=f'Epoch {epoch} (Test)', leave=False, disable=False)
        for data in loader:
            data = data.to(args.device)
            with torch.no_grad():
                pred, _, _ = model(data)
            true = data.y[:, 0] * dataset.mean_std['resilience'][1] + dataset.mean_std['resilience'][0]
            pred = pred[:, 0] * dataset.mean_std['resilience'][1] + dataset.mean_std['resilience'][0]
            metrics.setdefault('true', []).extend(true.detach().cpu().numpy().round(4).tolist())
            metrics.setdefault('pred', []).extend(pred.detach().cpu().numpy().round(4).tolist())
            metrics.setdefault('dataset', []).extend(data.dataset)
            metrics.setdefault('path', []).extend(data.path)
        metrics['R2'] = r2(metrics['true'], metrics['pred'], clip=True)
        metrics['p_spearman'] = p_spearman(metrics['true'], metrics['pred'])
        metrics['p_pearson'] = p_pearson(metrics['true'], metrics['pred'])
        metrics['MAE'] = mae(metrics['true'], metrics['pred'])
        metrics['RMSE'] = rmse(metrics['true'], metrics['pred'])
        metrics['sMAPE'] = smape(metrics['true'], metrics['pred'])
        metrics['MAPE'] = mape(metrics['true'], metrics['pred'])
        timer.add('test')

        ## Print Metrics
        n_gpus = torch.cuda.device_count()
        metrics['time_usage'] = (timer._time, timer._count)
        metrics['mem_alloc'] = sum(torch.cuda.memory_allocated(idx) for idx in range(n_gpus)) / 1024 ** 3
        metrics['mem_reserv'] = sum(torch.cuda.memory_reserved(idx) for idx in range(n_gpus)) / 1024 ** 3
        metrics['mem_max_alloc'] = sum(torch.cuda.max_memory_allocated(idx) for idx in range(n_gpus)) / 1024 ** 3
        metrics['mem_max_reserv'] = sum(torch.cuda.max_memory_reserved(idx) for idx in range(n_gpus)) / 1024 ** 3
        metrics['main_memory'] = psutil.Process().memory_info().rss / 1024 ** 3
        log = dict(
            Epoch=str(epoch),
            Loss=f'{metrics["loss"]:.4f}',
            **({f'Loss{i}': f'{metrics[f"loss{i}"]:.4f}' for i in range(1, 4)} if args.mix_loss else {}),
            R2=f'{metrics["R2"]:.4f}',
            p_spearman=f'{metrics["p_spearman"]:.4f}',
            p_pearson=f'{metrics["p_pearson"]:.4f}',
            MAE=f'{metrics["MAE"]:.4f}',
            RMSE=f'{metrics["RMSE"]:.4f}',
            sMAPE=f'{metrics["sMAPE"]:.2%}',
            MAPE=f'{metrics["MAPE"]:.2%}',
            TimeUsage=str(timer),
            CUDAUsage=(
                f"({args.device}) "
                f"allocated={metrics['mem_alloc']:.1f}GiB, "
                f"peak={metrics['mem_max_alloc']:.1f}GiB, "
                f"reserved={metrics['mem_reserv']:.1f}GiB"
            ),
        )
        msg = ', '.join(f'[#66CCFF bold]{k}[reset]={v}' for k, v in log.items())
        _logger.note(tag2ansi(msg))
        timer.add('log_metrics')

        ## Save Metrics
        with open(save_path / "metrics.jsonl", 'a') as f:
            f.write(json.dumps(metrics) + '\n')

        ## Plot Figure
        fig = plot(metrics)
        absl_logging.set_verbosity(absl_logging.WARNING)
        fig.savefig(save_path / "plot.pdf", dpi=600, bbox_inches='tight')
        absl_logging.set_verbosity(absl_logging.INFO)
        plt.close(fig)
        if epoch % 50 == 0:
            (save_path / "plot").mkdir(exist_ok=True, parents=True)
            shutil.copy(save_path / "plot.pdf", save_path / "plot" / f"plot_{epoch}.pdf")
        timer.add('plot')

        ## Save Checkpoint
        torch.save({
            "epoch": epoch,
            "args": vars(args),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, save_path / "checkpoint.pth")
        _logger.info(tag2ansi(f"Checkpoint saved to [underline green]{save_path / "checkpoint.pth"}[reset]."))
        if epoch % 50 == 0:
            (save_path / "checkpoints").mkdir(exist_ok=True, parents=True)
            shutil.copy(save_path / "checkpoint.pth", save_path / "checkpoints" / f"checkpoint_{epoch}.pth")
        timer.add('save_checkpoint')

        ## Free Extra GPU Memory
        if False:
            peak = torch.cuda.max_memory_allocated(args.device) / 1024 / 1024
            reserved_raw = torch.cuda.memory_reserved(args.device) / 1024 / 1024
            torch.cuda.empty_cache() # 释放 reserved 但是未被 allocated 的 block
            reserved_new = torch.cuda.memory_reserved(args.device) / 1024 / 1024
            if reserved_new < peak: # 释放了过多的显存，之后可能会 OOM
                allocated = torch.cuda.memory_allocated(args.device) / 1024 / 1024
                if (keep_MB := int(np.ceil(peak - allocated))) > 0: # 把需要的显存再占回来
                    try:
                        keep_cuda = AutoGPU.fuck_gpu(device=args.device, memory_MB=keep_MB, block_MB=None)
                        del keep_cuda
                    except torch.OurOfMemoryError as e:
                        pass
                reserved_new = torch.cuda.memory_reserved(args.device) / 1024 / 1024
            _logger.info(tag2ansi(f"[pink]Adjust reserved memory from {reserved_raw/1024:.1f}GiB to {reserved_new/1024:.1f}GiB.[reset]"))
            timer.add('free_gpu_memory')

        ## Update Best & Early Stopping
        if (
            'best_metrics' not in locals() or 
            np.mean(metrics['R2']) > np.mean(best_metrics['R2'])
        ):
            patience = args.patience
            best_metrics = metrics
            shutil.copy(save_path / "checkpoint.pth", save_path / "best_model.pth")
            shutil.copy(save_path / "plot.pdf", save_path / "best_result.pdf")
            _logger.note(tag2ansi(f"Best model saved to [underline green]{save_path / "best_model.pth"}[reset]"))
        elif patience > 0:
            patience -= 1
            _logger.info(tag2ansi(
                f"Patience left: [brightred]{patience}/{args.patience}[reset] ("
                f"[bold underline orange]best R2={best_metrics['R2']:.2%}[reset] "
                f"at epoch [#66CCFF]{best_metrics['epoch']}[reset]. "
                f"[#66CCFF]p_pearson={np.mean(best_metrics['p_pearson']):.4f}, "
                f"[#66CCFF]p_spearman={np.mean(best_metrics['p_spearman']):.4f}, "
                f"[#66CCFF]MAE={np.mean(best_metrics['MAE']):.4f}, "
                f"[#66CCFF]RMSE={np.mean(best_metrics['RMSE']):.1f}, "
                f"[#66CCFF]MAPE={np.mean(best_metrics['MAPE']):.1f})"
                f"[#66CCFF]sMAPE={np.mean(best_metrics['sMAPE']):.1f})"
            ))
        else:
            _logger.warning(tag2ansi(
                f"Early stopping at epoch [lightred]{epoch}/{args.num_epochs}[reset], "
                f"[bold underline orange]best R2={best_metrics['R2']:.2%}[reset] "
                f"at epoch [#66CCFF]{best_metrics['epoch']}[reset]. "
                f"[#66CCFF]p_pearson={np.mean(best_metrics['p_pearson']):.4f}, "
                f"[#66CCFF]p_spearman={np.mean(best_metrics['p_spearman']):.4f}, "
                f"[#66CCFF]MAE={np.mean(best_metrics['MAE']):.4f}, "
                f"[#66CCFF]RMSE={np.mean(best_metrics['RMSE']):.1f}, "
                f"[#66CCFF]MAPE={np.mean(best_metrics['MAPE']):.1f})"
                f"[#66CCFF]sMAPE={np.mean(best_metrics['sMAPE']):.1f})"
            ))
            break
        timer.add('update_best_model')
    _logger.note(f"Training finished. Time elapsed: {timer.time:.1f}s.")


def plot(metrics):
    df = pd.DataFrame({
        "True": metrics["true"],
        "Pred": metrics["pred"],
        "Dataset": metrics["dataset"],
    })
    try:
        # 按 all_data 的顺序排序
        all_data = pd.read_csv("./data/all_data.csv", sep="\t")
        all_data = all_data[all_data.Dataset.isin(df.Dataset)]
        sortby = all_data.Dataset[all_data.Dataset]
        df = df.set_index("Dataset").loc[sortby].reset_index()
    except:
        pass
    df["Rank_True"] = df["True"].argsort().argsort()
    df["Rank_Pred"] = df["Pred"].argsort().argsort()
    fi, fig, axes = get_fig(1, 3, AW=5, AH=5, dpi=600, LM=5, TM=5, RM=5, BM=5)
    sns.scatterplot(data=df, x="True", y="Pred", hue="Dataset", ax=axes[0])
    axes[0].set_xlabel("True")
    axes[0].set_ylabel("Pred")
    axes[0].set_title(rf'$\rho_{{Pearson}}={metrics["p_pearson"]:.5f}$')
    sns.scatterplot(data=df, x="Rank_True", y="Rank_Pred", hue="Dataset", ax=axes[1])
    axes[1].set_xlabel("True (Rank)")
    axes[1].set_ylabel("Pred (Rank)")
    axes[1].set_title(rf'$\rho_{{Spearman}}={metrics["p_spearman"]:.5f}$')
    axes[2].legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper left",
        fontsize="small",
        handlelength=1,
        columnspacing=0.5,
        handletextpad=0.5,
        borderpad=0.0,
        borderaxespad=0.0,
        frameon=False,
        ncols=3,
    )
    axes[2].axis("off")
    axes[0].get_legend().set_visible(False)
    axes[1].get_legend().set_visible(False)
    fig.suptitle(f'Epoch {metrics["epoch"]}')
    return fig


# fmt: off
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--Df', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--mix_loss', action='store_true')
    parser.add_argument('--no_test_before_train', dest='test_before_train', action='store_false', default=True)  # default: True
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

        # Remove Github 数据集和 GlobalSouth 数据集：做可解释性分析时发现有 outlier，分布也和 100 cities 差别较大
        # "Ahmedabad","Bandung","Bengaluru","Bogotá","Bucaramanga",
        # "Cartagena","Chennai","Cúcuta","Delhi","Depok",
        # "Ecatepec","Guadalajara","Hyderabad","Ibagué",
        # "Jaipur","Kanpur","Kolkata","León","Lucknow",
        # "Makassar","Medan","Medellín","Mérida","Montería",
        # "Mumbai","Palembang","Puebla City","Pune",
        # "Querétaro City","Santa Marta","Santiago de Cali","Semarang",
        # "Surabaya","Surat","Valledupar",

        # "Anaheim","Austin","Barcelona","Berlin-Center","Berlin-Friedrichshain",
        # "Berlin-Mitte-Center","Berlin-Mitte-Prenzlauerberg-Friedrichshain-Center",
        # "Berlin-Prenzlauerberg-Center","Berlin-Tiergarten","Birmingham-England",
        # "Braess-Example","chicago-regional","Chicago-Sketch",
        # "Eastern-Massachusetts","GoldCoast","Hessen-Asymmetric","Munich",
        # "Philadelphia","SiouxFalls","Sydney","Terrassa-Asymmetric","Winnipeg",
        # "Winnipeg-Asymmetric",
    ], help='List of datasets to train and test.')
    parser.add_argument('--dataset_split', type=float, default=0.8)
    parser.add_argument('--no_dataset_split', dest='dataset_split', action='store_const', const=None)
    parser.add_argument('--split_by_city', action='store_true')  # default: False
    parser.add_argument('--fold_by_city', type=int, default=None)
    parser.add_argument('--GPU_memory', type=int, default=40000)
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
    parser.add_argument('--save_dir', type=str, default='./logs/train')
    parser.add_argument('--data_dir', type=str, default='./data/augmentation')
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name. If None, will be generated automatically.")
    parser.add_argument('--shortest_path_num', type=int, default=1)
    parser.add_argument('--reload_checkpoint', type=str, default=None, help="Path to a checkpoint to reload.")
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
    args.command = ' '.join(map(shlex.quote, [sys.executable, *sys.argv]))

    ## Init logger `src`
    init_logger(
        "src",
        exp_name=args.exp_name,
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
    try:
        main(args)
    except Exception as e:
        _logger.error(
            f"Exception occurred during training: [{type(e)}] {e}\n"
            f"{traceback.format_exc()}"
        )
    finally:
        _logger.note(tag2ansi(f"Training process ended. Re-run with [pink bold]{args.command}[reset]"))
