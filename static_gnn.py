# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 14:34:04 2025

@author: ryuse
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class StaticGraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges]
        """
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x  # [num_nodes, out_channels]
