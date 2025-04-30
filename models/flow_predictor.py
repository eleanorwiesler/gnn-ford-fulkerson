import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv

class FlowPredictorGNN(torch.nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim))
        self.convs.append(GATConv(hidden_dim, 1))  # output = predicted flow per edge

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        return self.convs[-1](x, edge_index).squeeze(-1)