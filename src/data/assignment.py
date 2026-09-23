from absl import logging
import time

import numpy as np

from aequilibrae.paths import Graph, PathResults
from aequilibrae.paths import TrafficAssignment
from aequilibrae.paths.traffic_class import TrafficClass

from .topology import compute_centrality


def disrupt_network(network, disrupting_term, disrupting_links, K=50):
    if disrupting_term == "free_flow_time":
        network.loc[disrupting_links-1, "free_flow_time"] = network.loc[disrupting_links-1, "free_flow_time"]*K
    elif disrupting_term == "capacity":
        network.loc[disrupting_links-1, "capacity"] = network.loc[disrupting_links-1, "capacity"]*0.1
    else:
        raise ValueError(f"Disrupting term {disrupting_term} not supported!")
    return network


def traffic_assignment(
    aem, 
    network, 
    index, 
    disrupting_term="free_flow_time",
    disrupting_links=[], 
    vdf="BPR",
    ta_algorithm="bfw",
    max_iter=100,
    rgap_target=1e-5,
    travel_time_computation='batch', 
    return_centrality=False,
    K=50,
):
    if len(disrupting_links) > 0:
        network = disrupt_network(network, disrupting_term, disrupting_links, K)
    g = Graph()
    g.cost = network["free_flow_time"].values
    g.capacity = network["capacity"].values
    g.free_flow_time = network["free_flow_time"].values
    g.network = network
    g.prepare_graph(index)
    g.set_graph("free_flow_time")
    g.cost = np.array(g.cost, copy=True)
    g.set_skimming(["free_flow_time"])
    g.set_blocked_centroid_flows(False)
    g.network["id"] = g.network.link_id

    aem.computational_view(["matrix"])
    assigclass = TrafficClass("car", g, aem)
    assig = TrafficAssignment()
    assig.set_classes([assigclass])
    assig.set_vdf(vdf)
    assig.set_vdf_parameters({"alpha": "b", "beta": "power"})
    assig.set_capacity_field("capacity")
    assig.set_time_field("free_flow_time")
    assig.set_algorithm(ta_algorithm)
    assig.max_iter = max_iter
    assig.rgap_target = rgap_target

    start_time = time.time()
    import logging
    logging.getLogger('aequilibrae').setLevel(logging.WARNING)
    assig.execute()
    assig_results = assig.results()
    logging.info(f"--- TA cost: {time.time() - start_time} seconds ---")

    flow = assig_results["matrix_ab"].to_numpy()
    congested_time = assig_results["Congested_Time_AB"].to_numpy()
    free_flow_time = network["free_flow_time"].to_numpy()
    free_flow_travel_time = (flow*free_flow_time).sum()

    start_time = time.time()
    if travel_time_computation == 'batch':
        travel_time = (flow*congested_time).sum()
    else:
        network["congested_time"] = congested_time
        g.network = network
        g.prepare_graph(index)
        g.set_graph("congested_time")
        g.set_skimming(["congested_time"])
        g.set_blocked_centroid_flows(False)

        path_res = PathResults()
        path_res.prepare(g)

        travel_time = 0
        demand_mat = aem.get_matrix(core="matrix")
        all_time = []
        for i in range(demand_mat.shape[0]):
            for j in range(demand_mat.shape[1]):
                demand_ij = demand_mat[i, j]
                travel_time_ij = 0
                if (i != j) and (demand_ij != 0):
                    path_res.compute_path(i + 1, j + 1)
                    travel_path = path_res.path
                    for k in travel_path:
                        travel_time_ij += congested_time[k-1]
                    all_time.append(travel_time_ij)
                    travel_time += demand_ij * travel_time_ij
    logging.info(f"--- Delay computation cost: {time.time() - start_time} seconds ---")

    centrality_results = None
    if return_centrality:
        centrality_results = compute_centrality(network, index)

    return assig_results, travel_time, free_flow_travel_time, centrality_results
