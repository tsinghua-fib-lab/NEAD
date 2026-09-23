import torch.nn as nn
from .mp import *


class GNN(nn.Module):
    def __init__(
        self,
        node_channels,
        edge_channels1,
        edge_channels2,
        graph_channels,
        hidden_channels,
        num_layers=3,
        dropout=0.0,
    ):
        super(GNN, self).__init__()

        # embedding
        self.emb_x = nn.Linear(node_channels, hidden_channels)
        self.emb_edge_attr1 = nn.Linear(edge_channels1, hidden_channels)
        self.emb_edge_attr2 = nn.Linear(edge_channels2, hidden_channels)
        self.emb_graph_attr = nn.Linear(graph_channels, hidden_channels)

        self.conv1 = nn.ModuleList(
            [MP(hidden_channels, dropout=dropout) for _ in range(num_layers)]
        )
        self.conv2 = nn.ModuleList(
            [MP(hidden_channels, dropout=dropout) for _ in range(num_layers)]
        )
        self.conv3 = nn.ModuleList([
            nn.Sequential(
                # nn.LayerNorm(hidden_channels),
                nn.Linear(hidden_channels, 4 * hidden_channels),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(4 * hidden_channels, hidden_channels),
            )
            for _ in range(num_layers)
        ])

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_channels, 4 * hidden_channels),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(4 * hidden_channels, 1),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_channels, 4 * hidden_channels),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(4 * hidden_channels, 4),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(hidden_channels, 4 * hidden_channels),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(4 * hidden_channels, 4),
        )

    def forward(self, data):
        x = data.x
        edge_index1 = data.edge_index1
        edge_attr1 = data.edge_attr1
        graph_attr = data.graph_attr
        batch = data.batch

        x = self.emb_x(x)
        edge_attr1 = self.emb_edge_attr1(edge_attr1)
        graph_attr = self.emb_graph_attr(graph_attr)
        for conv1, conv2, conv3 in zip(self.conv1, self.conv2, self.conv3):
            delta_x1, delta_edge_attr1 = conv1(
                x, edge_index1, edge_attr1, graph_attr, batch
            )
            delta_graph_attr = conv3(
                global_mean_pool(x, batch)
                + global_mean_pool_edge(edge_index1, edge_attr1, batch)
                + graph_attr
            )
            x = x + delta_x1
            edge_attr1 = edge_attr1 + delta_edge_attr1
            graph_attr = graph_attr + delta_graph_attr

        avg_x = global_mean_pool(x, batch)
        avg_edge_attr1 = global_mean_pool_edge(edge_index1, edge_attr1, batch)
        out = self.fc1(avg_x + avg_edge_attr1 + graph_attr)
        pred_time = self.fc2(avg_x + avg_edge_attr1 + graph_attr)
        pred_volume = self.fc3(
            edge_attr1 + broadcast_to_edge(edge_index1, graph_attr, batch)
        )
        return out, pred_time, pred_volume
