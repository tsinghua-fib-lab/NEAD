from .mp import *
from .gnn import GNN


class GNN_w_edgeattr2(GNN):
    def forward(self, data):
        x = data.x
        edge_index1 = data.edge_index1
        edge_index2 = data.edge_index2
        edge_attr1 = data.edge_attr1
        edge_attr2 = data.edge_attr2
        graph_attr = data.graph_attr
        batch = data.batch

        x = self.emb_x(x)
        edge_attr1 = self.emb_edge_attr1(edge_attr1)
        edge_attr2 = self.emb_edge_attr2(edge_attr2)
        graph_attr = self.emb_graph_attr(graph_attr)
        for conv1, conv2, conv3 in zip(self.conv1, self.conv2, self.conv3):
            delta_x1, delta_edge_attr1 = conv1(
                x, edge_index1, edge_attr1, graph_attr, batch
            )
            delta_x2, delta_edge_attr2 = conv2(
                x, edge_index2, edge_attr2, graph_attr, batch
            )
            delta_graph_attr = conv3(
                global_mean_pool(x, batch)
                + global_mean_pool_edge(edge_index1, edge_attr1, batch)
                + global_mean_pool_edge(edge_index2, edge_attr2, batch)
                + graph_attr
            )
            x = x + delta_x1 + delta_x2
            edge_attr1 = edge_attr1 + delta_edge_attr1
            edge_attr2 = edge_attr2 + delta_edge_attr2
            graph_attr = graph_attr + delta_graph_attr

        avg_x = global_mean_pool(x, batch)
        avg_edge_attr1 = global_mean_pool_edge(edge_index1, edge_attr1, batch)
        avg_edge_attr2 = global_mean_pool_edge(edge_index2, edge_attr2, batch)
        out = self.fc1(avg_x + avg_edge_attr1 + avg_edge_attr2 + graph_attr)
        pred_time = self.fc2(avg_x + avg_edge_attr1 + avg_edge_attr2 + graph_attr)
        pred_volume = self.fc3(
            edge_attr1 + broadcast_to_edge(edge_index1, graph_attr, batch)
        )
        return out, pred_time, pred_volume
