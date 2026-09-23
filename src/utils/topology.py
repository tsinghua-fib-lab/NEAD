from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd


def construct_topology(network: pd.DataFrame) -> Tuple[nx.DiGraph, nx.DiGraph]:
    primal_topology = nx.from_pandas_edgelist(
        network, "a_node", "b_node", ["link_id", "free_flow_time", "capacity"], create_using=nx.DiGraph)
    dual_topology = nx.line_graph(primal_topology)
    dual_topology.add_nodes_from((n, primal_topology.edges[n]) for n in dual_topology)
    dual_topology = nx.relabel_nodes(dual_topology, lambda x: dual_topology.nodes[x]["link_id"])
    return primal_topology, dual_topology


def compute_centrality(network: pd.DataFrame, index: np.array) -> pd.DataFrame:
    primal_topology, dual_topology = construct_topology(network)

    edge_betweenness = nx.edge_betweenness_centrality(primal_topology, normalized=False, weight="free_flow_time")
    link_id_betweenness = {link_id: edge_betweenness[(u ,v)] for (u, v, link_id) in primal_topology.edges.data("link_id")}
    edge_betweenness_od = nx.edge_betweenness_centrality_subset(
        primal_topology, sources=index, targets=index, normalized=False, weight="free_flow_time")
    link_id_betweenness_od = {link_id: edge_betweenness_od[(u ,v)] for (u, v, link_id) in primal_topology.edges.data("link_id")}
    
    degree = nx.degree(dual_topology)
    closeness = nx.closeness_centrality(dual_topology, distance="free_flow_time")
    centrality_results = pd.DataFrame({
        "link_id": list(dual_topology.nodes),
        "degree": [degree[link_id] for link_id in dual_topology.nodes],
        "betweenness": [link_id_betweenness[link_id] for link_id in dual_topology.nodes],
        "betweenness_od": [link_id_betweenness_od[link_id] for link_id in dual_topology.nodes],
        "closeness": [closeness[link_id] for link_id in dual_topology.nodes],
    })
    centrality_results = centrality_results.sort_values("link_id").set_index("link_id")
    return centrality_results
