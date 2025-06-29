from typing import Union

import torch
from torch import nn
from torch_geometric.data import Data

from model.graphormer.functional import shortest_path_distance, batched_shortest_path_distance
from model.graphormer.layers import GraphormerEncoderLayer, CentralityEncoding, SpatialEncoding

import time



class Graphormer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 input_node_dim: int,
                 node_dim: int,
                 input_edge_dim: int,
                 edge_dim: int,
                 output_dim: int,
                 n_heads: int,
                 ff_dim: int,
                 max_in_degree: int,
                 max_out_degree: int,
                 max_path_distance: int):
        """
        :param num_layers: number of Graphormer layers
        :param input_node_dim: input dimension of node features
        :param node_dim: hidden dimensions of node features
        :param input_edge_dim: input dimension of edge features
        :param edge_dim: hidden dimensions of edge features
        :param output_dim: number of output node features
        :param n_heads: number of attention heads
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max in degree of nodes
        :param max_path_distance: max pairwise distance between two nodes
        """
        super().__init__()

        self.num_layers = num_layers
        self.input_node_dim = input_node_dim
        self.node_dim = node_dim
        self.input_edge_dim = input_edge_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.max_path_distance = max_path_distance

        self.node_in_lin = nn.Linear(self.input_node_dim, self.node_dim)
        self.edge_in_lin = nn.Linear(self.input_edge_dim, self.edge_dim)

        self.centrality_encoding = CentralityEncoding(
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
            node_dim=self.node_dim
        )

        self.spatial_encoding = SpatialEncoding(
            max_path_distance=max_path_distance,
        )

        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                n_heads=self.n_heads,
                ff_dim=self.ff_dim,
                max_path_distance=self.max_path_distance) for _ in range(self.num_layers)
        ])

        self.node_out_lin = nn.Linear(self.node_dim, self.output_dim)

    def forward(self, data: Union[Data]) -> torch.Tensor:
        """
        :param data: input graph of batch of graphs
        :return: torch.Tensor, output node embeddings
        """
        start_time = time.time()

        x = data.x.float()
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.float()

        if type(data) == Data:
            ptr = None
            node_paths, edge_paths = shortest_path_distance(data)
        else:
            ptr = data.ptr
            node_paths, edge_paths = batched_shortest_path_distance(data)

        linear_transformation_start = time.time()
        x = self.node_in_lin(x)
        edge_attr = self.edge_in_lin(edge_attr)
        linear_transformation_time = time.time() - linear_transformation_start

        centrality_encoding_start = time.time()
        x = self.centrality_encoding(x, edge_index)
        centrality_encoding_time = time.time() - centrality_encoding_start

        spatial_encoding_start = time.time()
        b = self.spatial_encoding(x, node_paths)
        spatial_encoding_time = time.time() - spatial_encoding_start

        attention_layers_start = time.time()
        for layer in self.layers:
            x = layer(x, edge_attr, b, edge_paths, ptr)
        attention_layers_time = time.time() - attention_layers_start

        final_transformation_start = time.time()
        x = self.node_out_lin(x)
        final_transformation_time = time.time() - final_transformation_start

        total_time = time.time() - start_time

        timing_info = {
            "Linear Transformation Time": linear_transformation_time,
            "Centrality Encoding Time": centrality_encoding_time,
            "Spatial Encoding Time": spatial_encoding_time,
            "Attention Layers Time": attention_layers_time,
            "Final Transformation Time": final_transformation_time,
            "Total Forward Pass Time": total_time
        }

        return x, timing_info