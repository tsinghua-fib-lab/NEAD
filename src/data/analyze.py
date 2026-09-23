from absl import app, flags, logging
import gc
from setproctitle import setproctitle
import sys
import time
from tqdm.rich import tqdm

import math
import numpy as np
import pandas as pd

from .assignment import traffic_assignment
from .io import read_data, save_result_to_db, test_postgres_connection
from ..utils.random import set_seed

flags.DEFINE_boolean("slurm_array", False, "Whether to run in SLURM array mode.")
flags.DEFINE_integer("task_id", 0, "Task ID for SLURM array mode.")
flags.DEFINE_integer("num_tasks", 0, "Number of tasks for SLURM array mode.")
flags.DEFINE_enum("task_division", "dataset", ["dataset", "disruption"], "Task division for SLURM array mode.")
flags.DEFINE_string("dataset_name", "", "Name of the dataset.")
flags.DEFINE_string("disrupting_term", "free_flow_time", "Strategy for disrupting road networks.")
flags.DEFINE_string("disrupting_strategy", "random", "Strategy for disrupting road networks.")
flags.DEFINE_enum(
    "disrupting_indicator",
    "matrix_ab",
    ["nil", "matrix_ab", "Delay_factor_AB", "VOC_AB", "degree", "betweenness", "betweenness_od", "closeness"],
    "Indicator for selecting links to disrupt.")
flags.DEFINE_boolean("debug", False, "Whether to run in debug mode.")
flags.DEFINE_boolean("batch_analyze", False, "Whether to run batch analysis of stepped disruptions.")
flags.DEFINE_integer("disrupting_fraction", 10, "Fraction of links to disrupt.")
flags.DEFINE_integer("batch_step", 1, "Step size for batch analysis.")
flags.DEFINE_integer("batch_min", 1, "Minimum number of disruptions for batch analysis.")
flags.DEFINE_integer("batch_max", 10, "Maximum number of disruptions for batch analysis.")
flags.DEFINE_string("ta_algorithm", "bfw", "Traffic assignment algorithm.")
flags.DEFINE_string("vdf", "BPR", "Volume delay function.")
flags.DEFINE_float("rgap_target", 1e-5, "Relative gap target for traffic assignment.")
flags.DEFINE_integer("max_iter", 100, "Maximum number of iterations for traffic assignment.")
flags.DEFINE_integer("global_seed", 42, "Global seed for random number generators.")
flags.DEFINE_enum("save_to_db", "nil", ["nil", "aggregated", "intermediate", "all"], "Save results to database.")
flags.DEFINE_integer("save_interval", 20, "Interval for saving results to database.")
flags.DEFINE_boolean("skip_first_row_save", False, "Whether to skip saving the first row of results.")
FLAGS = flags.FLAGS


def get_disrupting_links(
    assig_results: pd.DataFrame, 
    centrality_results: pd.DataFrame, 
    disrupting_count: int,
    disrupting_strategy: str, 
    disrupting_indicator: str
) -> np.ndarray:
    if disrupting_strategy == "random":
        disrupting_links = np.random.choice(assig_results.index, size=disrupting_count, replace=False)
    elif disrupting_strategy == "greedy":
        table = pd.concat([assig_results, centrality_results], axis=1)
        disrupting_links = table.sort_values(disrupting_indicator, ascending=False, kind='mergesort').head(disrupting_count).index
    else:
        raise ValueError(f"Disrupting strategy {disrupting_strategy} not supported!")
    return disrupting_links


def add_flags_to_df(df: pd.DataFrame) -> pd.DataFrame:
    df["dataset_name"] = FLAGS.dataset_name
    df["disrupting_term"] = FLAGS.disrupting_term
    df["disrupting_strategy"] = FLAGS.disrupting_strategy
    df["disrupting_indicator"] = FLAGS.disrupting_indicator
    df["ta_algorithm"] = FLAGS.ta_algorithm
    df["vdf"] = FLAGS.vdf
    df["rgap_target"] = FLAGS.rgap_target
    df["max_iter"] = FLAGS.max_iter
    df["global_seed"] = FLAGS.global_seed
    df["timestamp"] = pd.Timestamp.now(tz="EST")
    return df


def main(_):
    exp_start_time = time.time()
    set_seed()

    if FLAGS.save_to_db != "nil":
        test_postgres_connection()

    if FLAGS.disrupting_strategy == "random":
        FLAGS.disrupting_indicator = "nil"

    if not FLAGS.batch_analyze:
        disrupting_fraction = np.array([FLAGS.disrupting_fraction])
    else:
        disrupting_fraction = np.arange(FLAGS.batch_min, FLAGS.batch_max+1, FLAGS.batch_step)

    all_datasets = [
        "SiouxFalls",
        "Eastern-Massachusetts",
        "Berlin-Friedrichshain",
        "Berlin-Prenzlauerberg-Center",
        "Berlin-Tiergarten",
        "Berlin-Mitte-Center",
        "Anaheim",
        "Berlin-Mitte-Prenzlauerberg-Friedrichshain-Center",
        "Barcelona",
        "Winnipeg-Asymmetric",
        "Winnipeg",
        "Chicago-Sketch",
        "Terrassa-Asymmetric",
        "Hessen-Asymmetric",
        "GoldCoast",
        "Berlin-Center",
        "Birmingham-England",
        "chicago-regional",
        "Philadelphia",
        "Sydney",
        "Valledupar",
        "SantaMarta",
        "Monteria",
        "Cartagena",
        "Palembang",
        "Merida",
    ]
    
    if FLAGS.slurm_array:
        if FLAGS.task_division == "dataset":
            FLAGS.dataset_name = all_datasets[FLAGS.task_id]
        elif FLAGS.task_division == "disruption":
            assert FLAGS.skip_first_row_save, "Skip first row save must be enabled for task division disruption."
            disrupting_fraction = disrupting_fraction[FLAGS.task_id::FLAGS.num_tasks]
        else:
            raise ValueError(f"Task division {FLAGS.task_division} not supported!")

    num_records = len(disrupting_fraction) + 1
    result_disruption = np.insert(disrupting_fraction, 0, 0)
    result_travel_time = np.zeros(num_records)
    if FLAGS.save_to_db != "nil":
        batch_disruption = []
        batch_travel_time = []
        batch_assig_results = []

    aem, original_network, index = read_data(FLAGS.dataset_name)
    network = original_network.copy(deep=True)
    num_links = len(network)
    disrupting_count = np.ceil(num_links * disrupting_fraction / 100).astype(int)
    logging.info("Network information:")
    logging.info(f"\tNumber of zones: {len(index)}")
    logging.info(f"\tNumber of nodes: {max(network.a_node.max(), network.b_node.max())}")
    logging.info(f"\tNumber of links: {len(network)}")

    if FLAGS.disrupting_indicator in ["degree", "betweenness", "betweenness_od", "closeness"]:
        return_centrality = True
    else:
        return_centrality = False
    assig_results, original_travel_time, free_flow_time, centrality_results = traffic_assignment(
        aem, network, index, 
        vdf=FLAGS.vdf, ta_algorithm=FLAGS.ta_algorithm, max_iter=FLAGS.max_iter, rgap_target=FLAGS.rgap_target,
        return_centrality=return_centrality)

    all_disrupting_links = get_disrupting_links(
        assig_results, centrality_results, disrupting_count[-1],
        FLAGS.disrupting_strategy, FLAGS.disrupting_indicator)

    def wrap_result(disruption: np.ndarray, travel_time: np.ndarray):
        delay = travel_time - free_flow_time
        additional_delay = travel_time - original_travel_time
        relative_additional_delay = 100.0 * additional_delay / original_travel_time
        slow_down = travel_time / original_travel_time
        result = pd.DataFrame({
            "Disruption": disruption,
            "Travel Time": travel_time,
            "Delay": delay,
            "Additional Delay": additional_delay,
            "Relative Additional Delay (%)": relative_additional_delay,
            "Travel Time Slowdown": slow_down
        })
        return result

    result_travel_time[0] = original_travel_time
    if FLAGS.save_to_db != "nil" and not FLAGS.skip_first_row_save:
        batch_disruption.append(0)
        batch_travel_time.append(original_travel_time)
        assig_results["Disruption"] = 0
        assig_results["Disrupted"] = False
        assig_results.reset_index(inplace=True)
        assig_results["link_id"] = assig_results["link_id"].astype("int64")
        batch_assig_results.append(assig_results)

    for i, count in enumerate(tqdm(disrupting_count, desc="Disruption analysis")):
        logging.info(f"Batch {i+1}/{len(disrupting_count)}: disrupting {count} ({disrupting_fraction[i]}%) links.")
        network = original_network.copy(deep=True)
        disrupting_links = all_disrupting_links[:count]
        assig_results, disrupted_travel_time, _, _ = traffic_assignment(
            aem, network, index, disrupting_term=FLAGS.disrupting_term, 
            vdf=FLAGS.vdf, ta_algorithm=FLAGS.ta_algorithm, max_iter=FLAGS.max_iter, rgap_target=FLAGS.rgap_target,
            disrupting_links=disrupting_links)
        result_travel_time[i+1] = disrupted_travel_time
        if FLAGS.save_to_db != "nil":
            batch_disruption.append(result_disruption[i+1])
            batch_travel_time.append(disrupted_travel_time)
            assig_results["Disruption"] = result_disruption[i+1]
            assig_results["Disrupted"] = False
            assig_results.loc[disrupting_links, "Disrupted"] = True
            assig_results.reset_index(inplace=True)
            assig_results["link_id"] = assig_results["link_id"].astype("int64")
            batch_assig_results.append(assig_results)
            if (i+1) % FLAGS.save_interval == 0 or i+1 == len(disrupting_count):
                if FLAGS.save_to_db in ["all", "aggregated"]:
                    batch_agg_result = wrap_result(np.array(batch_disruption), np.array(batch_travel_time))
                    batch_agg_result = add_flags_to_df(batch_agg_result)
                    table_name = "disruption_analysis" if not FLAGS.debug else "disruption_analysis_debug"
                    save_result_to_db(batch_agg_result, table_name)
                    logging.info(
                        f"Saved batch aggregated {math.ceil((i+1)/FLAGS.save_interval)} to database: disruption {batch_disruption}")
                if FLAGS.save_to_db in ["all", "intermediate"]:
                    batch_intermediate_result = pd.concat(batch_assig_results, axis=0)
                    batch_intermediate_result = add_flags_to_df(batch_intermediate_result)
                    if FLAGS.disrupting_strategy == "greedy" and FLAGS.disrupting_indicator == "matrix_ab":
                        table_name = f"inter_{FLAGS.dataset_name}" if not FLAGS.debug else "inter_debug"
                    elif FLAGS.disrupting_strategy == "random" and FLAGS.disrupting_indicator == "nil":
                        table_name = f"inter_{FLAGS.dataset_name}_r" if not FLAGS.debug else "inter_debug_r"
                    else:
                        raise ValueError(f"Intermediate not supported: {FLAGS.disrupting_strategy}, {FLAGS.disrupting_indicator}")
                    save_result_to_db(batch_intermediate_result, table_name)
                    logging.info(
                        f"Saved batch intermediate {math.ceil((i+1)/FLAGS.save_interval)} to database: disruption {batch_disruption}")
                batch_disruption = []
                batch_travel_time = []
                batch_assig_results = []
        del network
        gc.collect()

    logging.info(f"Free flow travel time: {free_flow_time}")
    df_result = wrap_result(result_disruption, result_travel_time)
    logging.info(f"\n{df_result.to_markdown(floatfmt='.2f')}")

    logging.info(f"--- Exp time cost: {time.time() - exp_start_time} seconds ---")


if __name__ == "__main__":
    setproctitle("debug@yu_zheng")
    logging.set_verbosity(logging.INFO)
    logging.get_absl_handler().python_handler.stream = sys.stdout
    app.run(main)
