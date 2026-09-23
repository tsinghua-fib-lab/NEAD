import logging
import numpy as np
import pandas as pd
import networkx as nx
from tqdm.rich import tqdm
from itertools import islice
from absl import logging as absl_logging
from src.utils.my_hash import my_hash
from src.data.analyze import get_disrupting_links
from src.data.assignment import traffic_assignment

_logger = logging.getLogger(__name__)


def add_noise_to_od(od, random_state=None):
    """ Add noise to the OD matrix to create more data. """
    if random_state is not None:
        _logger.debug(f"Add noise to OD with random state {random_state}.")
    rng = np.random.default_rng(random_state)
    noisy_od = od.copy()
    # Shuffle
    shuffle = rng.permutation(noisy_od.shape[0])
    noisy_od = noisy_od[shuffle, :][:, shuffle]
    # Average
    ratio = rng.choice([0, 1.0])
    alpha = rng.choice([-0.1, 0.1])
    flat_od = noisy_od.flatten()
    nonzero = np.nonzero(flat_od > 0)[0]
    for a, b in rng.choice(nonzero, (int(nonzero.size * ratio), 2)):
        flat_od[a] = (1 - alpha / 2) * flat_od[a] + alpha / 2 * flat_od[b]
        flat_od[b] = (1 - alpha / 2) * flat_od[b] + alpha / 2 * flat_od[a]
    noisy_od = flat_od.reshape(noisy_od.shape)
    # UnSparsify
    ratio = rng.choice([0, 1.0])
    beta = 0.2
    flat_od = noisy_od.flatten()
    for a, b in rng.integers(0, noisy_od.size, size=(int(noisy_od.size * ratio), 2)):
        flat_od[a] = (1 - beta / 2) * flat_od[a] + beta / 2 * flat_od[b]
        flat_od[b] = (1 - beta / 2) * flat_od[b] + beta / 2 * flat_od[a]
    noisy_od = flat_od.reshape(noisy_od.shape)
    # Add Noise
    strength = rng.choice([0.0, 1.0])
    noisy_od = noisy_od + rng.standard_normal(noisy_od.shape) * strength * np.std(noisy_od)
    # Clip
    noisy_od = noisy_od.clip(0.)
    _logger.debug(
        f"Add noise to OD: sum before {od.sum():.3g}, sum after {noisy_od.sum():.3g}, "
        f"max before {od.max():.3g}, max after {noisy_od.max():.3g}"
    )
    return noisy_od


def load_od(args, aem, sample, save_path):
    save_file = save_path / 'od.csv.gz'
    if not save_file.exists():
        od = aem.matrix['matrix'].copy()
        if sample != 'raw': 
            od = add_noise_to_od(od, random_state=my_hash(f"{args.seed}_{sample}"))
        aem.matrix['matrix'][:, :] = od

        pd.DataFrame(aem.matrix['matrix']).to_csv(
            save_file, 
            index=False, 
            header=False, 
            compression="gzip", 
            float_format='%.6f'
        )
        _logger.info(f"Saved OD matrix to {save_file}")
    else:
        _logger.debug(f"Exist OD matrix in {save_file}")
        aem.matrix['matrix'][:, :] = pd.read_csv(save_file, header=None).values
    return aem


def load_free_assign(args, aem, original_network, index, save_path):
    save_file = save_path / 'free_assignment.csv.gz'
    num_links = len(original_network)
    if not save_file.exists():
        network = original_network.copy(deep=True)
        free_assig_results, _, _, centrality_results = traffic_assignment(
            aem, network, index, 
            vdf=args.vdf, 
            ta_algorithm=args.ta_algorithm, 
            max_iter=args.max_iter, 
            rgap_target=args.rgap_target,
            return_centrality=args.return_centrality,
            K=args.K,
        )
        all_disrupting_links = get_disrupting_links(
            free_assig_results, 
            centrality_results, 
            num_links, # disrupting_count[-1]
            args.disrupting_strategy, 
            args.disrupting_indicator
        )
        free_assig_results = free_assig_results[['matrix_ab', 'Congested_Time_AB', 'Delay_factor_AB', 'VOC_AB']]
        free_assig_results['Disrupted_Rank'] = 0
        for i, k in enumerate(all_disrupting_links, start=1):
            free_assig_results.loc[k, 'Disrupted_Rank'] = i
        free_assig_results.to_csv(save_file, index=False, compression='gzip')
        _logger.info(f"Saved free assignment results to {save_file}")
    else:
        _logger.debug(f"Exist free assignment results in {save_file}")
        free_assig_results = pd.read_csv(save_file)
        free_assig_results['link_id'] = free_assig_results.index + 1
        free_assig_results = free_assig_results.set_index('link_id')
        all_disrupting_links = np.zeros(num_links, dtype=int)
        for k, i in enumerate(free_assig_results['Disrupted_Rank']):
            all_disrupting_links[i-1] = k+1
        if args.disrupting_strategy != 'random':
            for fraction in args.disrupting_fraction:
                count = np.ceil(num_links * fraction / 100).astype(int)
                all_disrupting_links2 = get_disrupting_links(
                    free_assig_results, 
                    None, 
                    num_links,  # disrupting_count[-1],
                    args.disrupting_strategy, 
                    args.disrupting_indicator
                )
                set1 = set(all_disrupting_links[:count])
                set2 = set(all_disrupting_links2[:count])
                similarity = len(set1 & set2) / len(set1 | set2)
                assert similarity > 0.95, f"{save_file}, {fraction}, {count}, {similarity}"
    factor = (1-1/free_assig_results['Delay_factor_AB']) if args.consider_delay_factor else 1
    total_travel_time = (free_assig_results['matrix_ab'] * free_assig_results['Congested_Time_AB'] * factor).sum()
    return total_travel_time, free_assig_results, all_disrupting_links


def load_disrupt_assign(args, aem, original_network, index, free_assig_results, all_disrupting_links, save_path):
    num_links = len(original_network)
    disrupting_fraction = np.array(args.disrupting_fraction)
    disrupting_count = np.ceil(num_links * disrupting_fraction / 100).astype(int)
    disrupted_travel_time = []
    for i, count in enumerate(tqdm(disrupting_count, desc="Disruption analysis", disable=True)):
        save_file = save_path / f'result_disrupt{disrupting_fraction[i]:03d}.csv.gz'
        if save_file.exists():
            _logger.debug(f"Exist disrupted assignment results in {save_file}")
            assig_results = pd.read_csv(save_file)
        else:
            network = original_network.copy(deep=True)
            disrupting_links = all_disrupting_links[:count]
            absl_logging.set_verbosity(absl_logging.WARNING)
            assig_results, _, _, _ = traffic_assignment(
                aem, 
                network, 
                index, 
                disrupting_term=args.disrupting_term, 
                vdf=args.vdf, 
                ta_algorithm=args.ta_algorithm, 
                max_iter=args.max_iter, 
                rgap_target=args.rgap_target,
                disrupting_links=disrupting_links,
                return_centrality=args.return_centrality,
                K=args.K,
            )
            absl_logging.set_verbosity(absl_logging.INFO)
            assig_results = assig_results[['matrix_ab', 'Congested_Time_AB', 'Delay_factor_AB', 'VOC_AB']]
            assig_results['Original_Time'] = free_assig_results['Congested_Time_AB']
            assig_results['Disrupted'] = 0
            assig_results.loc[disrupting_links, 'Disrupted'] = 1
            assig_results.to_csv(save_file, index=False, compression='gzip')
            _logger.info(f"Saved disrupted assignment results to {save_file}")
        factor = (1-1/assig_results['Delay_factor_AB']) if args.consider_delay_factor else 1
        disrupted_travel_time.append((assig_results['matrix_ab'] * assig_results['Congested_Time_AB'] * factor).sum())
    return disrupted_travel_time


def load_resilience(args, total_travel_time, disrupted_travel_time, save_path):
    save_file = save_path / 'resilience.txt'
    if save_file.exists():
        _logger.debug(f"Exist resilience results in {save_file}")
    else:
        disrupted_travel_time = np.array([total_travel_time, *disrupted_travel_time, args.K * total_travel_time])
        resilience = np.mean(disrupted_travel_time / total_travel_time)
        with open(save_file, 'w') as f:
            f.write(str(resilience))
        _logger.info(f"Saved resilience {resilience} to {save_file}")
    return


def load_shortest_routes(args, aem, original_network, free_assig_results, save_path):
    save_file = save_path / 'shortest_routes.csv.gz'
    if not save_file.exists():
        ## Find the most important OD pairs.
        tmp = np.copy(aem.matrix['matrix'])
        tmp -= np.diag(np.diag(tmp))
        if tmp.size > args.od_pairs_num:
            topk = np.argpartition(np.reshape(tmp, -1), -args.od_pairs_num)[-args.od_pairs_num:]
        else:
            topk = np.arange(tmp.size)
        row = topk // aem.matrix['matrix'].shape[1] + 1
        col = topk % aem.matrix['matrix'].shape[1] + 1
        value = aem.matrix['matrix'][row-1, col-1]

        ## Find shortest routes.
        shortest_paths = {}
        graph = nx.from_pandas_edgelist(pd.concat([
            original_network[['a_node', 'b_node', 'link_id']], 
            free_assig_results.reset_index()[['Congested_Time_AB']]
        ], axis=1), 'a_node', 'b_node', edge_attr=['Congested_Time_AB', 'link_id'], create_using=nx.DiGraph)
        for r, c, v in tqdm(zip(row, col, value), total=len(row), disable=True):
            # print(f'From {r} to {c}, demand is {v}')
            shortest_paths[(r, c)] = []
            paths = []
            try:
                for path in islice(nx.shortest_simple_paths(graph, source=r, target=c, weight='Congested_Time_AB'), args.shortest_path_num):
                    paths.append([graph[u][v]['link_id'] for u, v in zip(path[:-1], path[1:])])
                    # print(path)
            except nx.NetworkXNoPath:
                pass
            except nx.NodeNotFound:
                pass
            shortest_paths[(r, c)] = paths
        
        # Save results.
        shortest_paths = pd.DataFrame(shortest_paths.items(), columns=['OD', 'Paths'])
        shortest_paths.to_csv(save_file, index=False, compression='gzip')
        _logger.info(f"Saved shortest routes to {save_file}")
    else:
        _logger.debug(f"Exist shortest routes in {save_file}")
        shortest_paths = pd.read_csv(save_file)
        shortest_paths['OD'] = shortest_paths['OD'].apply(eval)
        shortest_paths['Paths'] = shortest_paths['Paths'].apply(eval)
    return shortest_paths

