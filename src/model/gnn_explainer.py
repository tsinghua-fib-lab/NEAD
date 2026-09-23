from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch_geometric.explain import Explanation
from torch_geometric.explain.config import MaskType
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.algorithm.utils import set_masks
from .gnn import GNN, global_mean_pool, global_mean_pool_edge


class MyGNNExplainer(GNNExplainer):
    coeffs = {
        "edge_size": 0.005,
        "edge_reduction": "sum",
        "node_feat_size": 1.0,
        "node_feat_reduction": "mean",
        "edge_feat_size": 1.0,
        "edge_feat_reduction": "mean",
        "graph_feat_size": 1.0,
        "graph_feat_reduction": "mean",
        "edge_ent": 1.0,
        "node_feat_ent": 0.1,
        "edge_feat_ent": 0.1,
        "graph_feat_ent": 0.1,
        "EPS": 1e-15,
        "edge_feat_mask_type": None,
        "graph_mask_type": None,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_feat_mask = self.hard_edge_feat_mask = None
        self.graph_mask = self.hard_graph_mask = None
        if self.coeffs["edge_feat_mask_type"] is not None:
            self.coeffs["edge_feat_mask_type"] = MaskType(
                self.coeffs["edge_feat_mask_type"]
            )
        if self.coeffs["graph_mask_type"] is not None:
            self.coeffs["graph_mask_type"] = MaskType(self.coeffs["graph_mask_type"])

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        graph_attr: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(
                f"Heterogeneous graphs not yet supported in "
                f"'{self.__class__.__name__}'"
            )

        self._train(
            model,
            x,
            edge_index,
            target=target,
            index=index,
            edge_attr=edge_attr,
            graph_attr=graph_attr,
            **kwargs,
        )

        node_mask = self._post_process_mask(
            self.node_mask,
            self.hard_node_mask,
            apply_sigmoid=True,
        )
        edge_mask = self._post_process_mask(
            self.edge_mask,
            self.hard_edge_mask,
            apply_sigmoid=True,
        )
        edge_feat_mask = self._post_process_mask(
            self.edge_feat_mask,
            self.hard_edge_feat_mask,
            apply_sigmoid=True,
        )
        graph_mask = self._post_process_mask(
            self.graph_mask,
            self.hard_graph_mask,
            apply_sigmoid=True,
        )

        self._clean_model(model)

        return Explanation(
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_feat_mask=edge_feat_mask,
            graph_mask=graph_mask,
        )

    def _train(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        graph_attr: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        self._initialize_masks(x, edge_index, edge_attr, graph_attr)

        parameters = []
        if self.node_mask is not None:
            parameters.append(self.node_mask)
        if self.edge_mask is not None:
            set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            parameters.append(self.edge_mask)
        if self.edge_feat_mask is not None:
            parameters.append(self.edge_feat_mask)
        if self.graph_mask is not None:
            parameters.append(self.graph_mask)

        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for i in range(self.epochs):
            optimizer.zero_grad()

            h = x if self.node_mask is None else x * self.node_mask.sigmoid()
            e = (
                edge_attr
                if self.edge_feat_mask is None
                else edge_attr * self.edge_feat_mask.sigmoid()
            )
            g = (
                graph_attr
                if self.graph_mask is None
                else graph_attr * self.graph_mask.sigmoid()
            )
            y_hat, y = (
                model(x=h, edge_index=edge_index, edge_attr=e, graph_attr=g, **kwargs),
                target,
            )

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y)

            loss.backward()
            optimizer.step()

            # In the first iteration, we collect the nodes and edges that are
            # involved into making the prediction. These are all the nodes and
            # edges with gradient != 0 (without regularization applied).
            if i == 0 and self.node_mask is not None:
                if self.node_mask.grad is None:
                    raise ValueError(
                        "Could not compute gradients for node "
                        "features. Please make sure that node "
                        "features are used inside the model or "
                        "disable it via `node_mask_type=None`."
                    )
                self.hard_node_mask = self.node_mask.grad != 0.0
            if i == 0 and self.edge_mask is not None:
                if self.edge_mask.grad is None:
                    raise ValueError(
                        "Could not compute gradients for edges. "
                        "Please make sure that edges are used "
                        "via message passing inside the model or "
                        "disable it via `edge_mask_type=None`."
                    )
                self.hard_edge_mask = self.edge_mask.grad != 0.0
            if i == 0 and self.edge_feat_mask is not None:
                if self.edge_feat_mask.grad is None:
                    raise ValueError(
                        "Could not compute gradients for edge "
                        "features. Please make sure that edge "
                        "features are used inside the model or "
                        "disable it via `edge_feat_mask_type=None`."
                    )
                self.hard_edge_feat_mask = self.edge_feat_mask.grad != 0.0
            if i == 0 and self.graph_mask is not None:
                if self.graph_mask.grad is None:
                    raise ValueError(
                        "Could not compute gradients for graph "
                        "features. Please make sure that graph "
                        "features are used inside the model or "
                        "disable it via `graph_mask_type=None`."
                    )
                self.hard_graph_mask = self.graph_mask.grad != 0.0

    def _initialize_masks(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, graph_attr: Tensor
    ):
        super()._initialize_masks(x, edge_index)
        edge_feat_mask_type = self.coeffs["edge_feat_mask_type"]
        graph_mask_type = self.coeffs["graph_mask_type"]

        device = x.device
        (E, F), G = edge_attr.size(), graph_attr.size(1)

        std = 0.1
        if edge_feat_mask_type is None:
            self.edge_feat_mask = None
        elif edge_feat_mask_type == MaskType.object:
            self.edge_feat_mask = Parameter(torch.randn(E, 1, device=device) * std)
        elif edge_feat_mask_type == MaskType.attributes:
            self.edge_feat_mask = Parameter(torch.randn(E, F, device=device) * std)
        elif edge_feat_mask_type == MaskType.common_attributes:
            self.edge_feat_mask = Parameter(torch.randn(1, F, device=device) * std)
        else:
            assert False

        std = 0.1
        if graph_mask_type is None:
            self.graph_mask = None
        elif graph_mask_type == MaskType.common_attributes:
            self.graph_mask = Parameter(torch.randn(1, G, device=device) * std)
        else:
            assert False

    def _loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        loss = super()._loss(y_hat, y)

        if self.hard_edge_feat_mask is not None:
            assert self.edge_feat_mask is not None
            m = self.edge_feat_mask[self.hard_edge_feat_mask].sigmoid()
            edge_feat_reduce = getattr(torch, self.coeffs["edge_feat_reduction"])
            loss = loss + self.coeffs["edge_feat_size"] * edge_feat_reduce(m)
            ent = -m * torch.log(m + self.coeffs["EPS"]) - (1 - m) * torch.log(
                1 - m + self.coeffs["EPS"]
            )
            loss = loss + self.coeffs["edge_feat_ent"] * ent.mean()

        if self.hard_graph_mask is not None:
            assert self.graph_mask is not None
            m = self.graph_mask[self.hard_graph_mask].sigmoid()
            graph_reduce = getattr(torch, self.coeffs["graph_feat_reduction"])
            loss = loss + self.coeffs["graph_feat_size"] * graph_reduce(m)
            ent = -m * torch.log(m + self.coeffs["EPS"]) - (1 - m) * torch.log(
                1 - m + self.coeffs["EPS"]
            )
            loss = loss + self.coeffs["graph_feat_ent"] * ent.mean()

        return loss

    def _clean_model(self, model):
        super()._clean_model(model)
        self.edge_feat_mask = self.hard_edge_feat_mask = None
        self.graph_mask = self.hard_graph_mask = None


class Wrapper(GNN):
    def forward(self, x, edge_index, edge_attr, graph_attr, batch):
        x = self.emb_x(x)
        edge_attr = self.emb_edge_attr1(edge_attr)
        graph_attr = self.emb_graph_attr(graph_attr)
        for conv1, conv3 in zip(self.conv1, self.conv3):
            delta_x1, delta_edge_attr1 = conv1(
                x, edge_index, edge_attr, graph_attr, batch
            )
            delta_graph_attr = conv3(
                global_mean_pool(x, batch)
                + global_mean_pool_edge(edge_index, edge_attr, batch)
                + graph_attr
            )
            x = x + delta_x1
            edge_attr = edge_attr + delta_edge_attr1
            graph_attr = graph_attr + delta_graph_attr

        avg_x = global_mean_pool(x, batch)
        avg_edge_attr = global_mean_pool_edge(edge_index, edge_attr, batch)
        out = self.fc1(avg_x + avg_edge_attr + graph_attr)
        return out
