import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from torch_geometric.nn import MessagePassing, global_mean_pool

__all__ = [
    "global_mean_pool",  # node -> graph
    "global_mean_pool_edge",  # edge -> graph
    "broadcast_to_node",  # graph -> node
    "broadcast_to_edge",  # graph -> edge
    "MP",
]


def global_mean_pool_edge(edge_index, edge_attr, batch=None):
    dim = -1 if isinstance(edge_attr, torch.Tensor) and edge_attr.dim() == 1 else -2

    if batch is None:
        return edge_attr.mean(dim=dim, keepdim=edge_attr.dim() <= 2)
    return scatter(edge_attr, batch[edge_index[0]], dim, dim_size=None, reduce="mean")


def broadcast_to_node(graph_attr, batch=None):
    if batch is None:
        return graph_attr
    return graph_attr[batch, :]


def broadcast_to_edge(edge_index, graph_attr, batch=None):
    if batch is None:
        return graph_attr
    return graph_attr[batch[edge_index[0]], :]


class MP(MessagePassing):
    def __init__(self, dimension, dropout=0.0, normalize=True):
        super(MP, self).__init__(aggr="mean")
        self.phi_e = nn.Sequential(
            # nn.LayerNorm(dimension) if normalize else nn.Identity(),
            nn.Linear(dimension, 4 * dimension),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(4 * dimension, dimension),
        )
        self.phi_n = nn.Sequential(
            # nn.LayerNorm(dimension) if normalize else nn.Identity(),
            nn.Linear(dimension, 4 * dimension),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(4 * dimension, dimension),
        )

    def forward(self, x, edge_index, edge_attr, graph_attr, batch):
        graph_attr = broadcast_to_edge(edge_index, graph_attr, batch)
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr, graph_attr=graph_attr)
        edge_attr = self.edge_updater(
            edge_index, x=x, edge_attr=edge_attr, graph_attr=graph_attr
        )
        return x, edge_attr

    def message(self, x_i, x_j, edge_attr, graph_attr):
        return self.phi_n(x_i + x_j + edge_attr + graph_attr)

    def update(self, aggr_out):
        return aggr_out

    def edge_update(self, x_i, x_j, edge_attr, graph_attr):
        return self.phi_e(x_i + x_j + edge_attr + graph_attr)
