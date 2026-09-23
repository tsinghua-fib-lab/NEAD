import sys
import time
import json
import random
import logging
from pathlib import Path
from tqdm.rich import tqdm
from datetime import datetime
from socket import gethostname
from argparse import ArgumentParser
from setproctitle import setproctitle
from src.utils.random import set_seed
from src.utils.logger import init_logger
from src.utils.fix_parser import add_minus_flags, add_negation_flags
from src.data.io import read_data
from src.pipeline.pipeline import add_noise_to_od, load_od, load_free_assign, load_disrupt_assign, load_resilience, load_shortest_routes

_logger = logging.getLogger('src')


def main(args):
    exp_start_time = time.time()

    save_data_dir = Path(args.save_data_dir)
    for dataset in args.datasets:
        ## Read Network & OD data
        aem, original_network, index = read_data(dataset, data_root_path=args.raw_data_dir)
        num_links = len(original_network)

        ## Prepare samples to generate
        if args.fix_existing: # Repair missing results in an existing sample.
            loader = [i.name for i in sorted((save_data_dir / dataset).iterdir()) if i.is_dir()]
        elif not args.augment_OD: # Generate one sample from the raw OD matrix.
            loader = ['raw']
        elif args.augment_OD_num: # Generate augment_OD_num samples from augmented OD matrices.
            loader = [None] * args.augment_OD_num
        else:
            raise NotImplementedError("Either --fix-existing, --no-augment-OD or --augment-OD-num should be set.")
        
        ## Iterate samples
        for sample in tqdm(loader, disable=True):
            ## Prepare save_path
            if sample is None: 
                existing_samples = [i.name for i in (save_data_dir / dataset).iterdir() if i.is_dir()] if (save_data_dir / dataset).exists() else []
                if args.max_sample_num is not None and len(existing_samples) >= args.max_sample_num:
                    _logger.warning(f"Reach max sample num {len(existing_samples)} >= {args.max_sample_num} for dataset {dataset}, stop.")
                    break
                sample = time.strftime("%Y%m%d-%H%M%S")
            save_path = save_data_dir / dataset / sample
            if args.skip_existing and save_path.exists():
                _logger.info(f"Skip existing sample {save_path}.")
                continue
            else:
                save_path.mkdir(parents=True, exist_ok=True)

            ## Noisy OD matrix
            aem = load_od(args, aem, sample, save_path)

            ## Free assignment
            total_travel_time, free_assig_results, all_disrupting_links = load_free_assign(args, aem, original_network, index, save_path)
            
            ## Shortest Routes
            if args.calc_shortest_route_count:
                shortest_paths = load_shortest_routes(args, aem, original_network, free_assig_results, save_path)
            else:
                shortest_paths = None
            
            ## Disrupted assignment
            disrupted_travel_time = load_disrupt_assign(args, aem, original_network, index, free_assig_results, all_disrupting_links, save_path)

            ## Calculate Resilience Metrics
            load_resilience(args, total_travel_time, disrupted_travel_time, save_path)

                        
    _logger.info(f"--- Exp time cost: {time.time() - exp_start_time} seconds ---")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default='generate_data')
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument("--fix_existing", action='store_true', help="Fix the existing samples.")
    parser.add_argument('--skip_existing', action='store_true', help="Skip the existing samples.")
    parser.add_argument("--augment_OD", action='store_true', default=True, help='Use raw OD instead of augmentation OD to generate samples.')
    parser.add_argument("--augment_OD_num", type=int, default=19, help='Generate {augment-num} samples with augmentation OD.')
    parser.add_argument("--max_sample_num", type=int, default=20, help='Maximum number of samples to generate for each dataset.')
    parser.add_argument("--raw_od", action='store_true', help="use raw OD matrix.")
    parser.add_argument("--sample_num", type=int, default=10, help="Sample number.")
    parser.add_argument("--new_sample_num", action='store_true', help="Wherether --sample_num refers to additional number.")
    parser.add_argument("--od_pairs_num", type=int, default=100, help="Number of OD pairs to analyze.")
    parser.add_argument("--shortest_path_num", type=int, default=1, help="Number of shortest paths to analyze.")
    parser.add_argument("--save_data_dir", type=str, default='./data/augmentation', help="Root path to save the data.")
    parser.add_argument("--raw_data_dir", type=str, default='./data/raw', help="Root path containing the raw networks.")
    parser.add_argument("--save_log_dir", type=str, default="./logs/generate_data", help="Directory to save the results.")
    parser.add_argument("--datasets", type=str, nargs='+', default=[
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

        # Exclude the GitHub and Global South datasets: they contain outliers and
        # differ substantially from the 100-city distribution.
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
    ], help="List of datasets to disrupt.")
    parser.add_argument("--disrupting_term", type=str, default="free_flow_time", help="Strategy for disrupting road networks.")
    parser.add_argument("--disrupting_strategy", type=str, default="greedy", help="Strategy for disrupting road networks.")
    parser.add_argument(
        "--disrupting_indicator", type=str, default="matrix_ab", help="Indicator for selecting links to disrupt.",
        choices=["nil", "matrix_ab", "Delay_factor_AB", "VOC_AB", "degree", "betweenness", "betweenness_od", "closeness"],
    )
    parser.add_argument("--disrupting_fraction", type=int, nargs='+', default=[10], help="Fraction of links to disrupt.")
    parser.add_argument("--ta_algorithm", type=str, default="bfw", help="Traffic assignment algorithm.")
    parser.add_argument("--vdf", type=str, default="BPR", help="Volume delay function.")
    parser.add_argument("--rgap_target", type=float, default=1e-5, help="Relative gap target for traffic assignment.")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations for traffic assignment.")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name. If None, will be generated automatically.")
    parser.add_argument("--consider_delay_factor", action='store_true', help="Consider delay factor in travel time calculation.")
    parser.add_argument('--K', type=float, default=50)
    parser.add_argument('--calc_shortest_route_count', action='store_true', default=True)
    parser = add_negation_flags(parser)
    parser = add_minus_flags(parser)
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

    save_log_path = Path(args.save_log_dir) / args.exp_name
    if not save_log_path.exists():
        save_log_path.mkdir(parents=True, exist_ok=True)
    else:
        _logger.warning(f"Save path {save_log_path} already exists.")
    args.save_log_path = str(save_log_path)

    ## Set Seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    set_seed(args.seed)

    ## Save Command
    args.command = ' '.join([sys.executable, *sys.argv])
    ## Set other args
    args.return_centrality = args.disrupting_indicator in ["degree", "betweenness", "betweenness_od", "closeness"]

    ## Init logger `src`
    init_logger(
        "src",
        exp_name=args.exp_name,
        log_file=f"{args.save_log_path}/info.log",
        info_level="debug",
    )

    ## Warn Unknown Args
    if unknown:
        _logger.warning(f"Unknown args: {unknown}")

    ## Save Args
    args_path = save_log_path / "args.json"
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
