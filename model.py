import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv

class RGATModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_relations: int, out_channels: int, 
                 num_heads: int = 4, num_layers: int = 2, edge_dim: int = None):  # <-- ADDED edge_dim
        """
        Step 7: Train the Relational Graph Attention Network (RGAT) - Architecture definition
        """
        super(RGATModel, self).__init__()
        
        self.num_layers = num_layers
        
        # RGAT Layers
        self.convs = nn.ModuleList()
        # First layer (Pass edge_dim here)
        self.convs.append(RGATConv(in_channels, hidden_channels, num_relations=num_relations, 
                                   heads=num_heads, concat=False, edge_dim=edge_dim))
        
        # Hidden layers (Pass edge_dim here as well)
        for _ in range(num_layers - 1):
            self.convs.append(RGATConv(hidden_channels, hidden_channels, num_relations=num_relations, 
                                       heads=num_heads, concat=False, edge_dim=edge_dim))
            
        # Multi-head feed-forward neural network projection layer for multi-horizon targets
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(self, x, edge_index, edge_type, edge_attr=None):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_type, edge_attr=edge_attr)
            if i < self.num_layers - 1:
                x = F.elu(x)
                
        out = self.ffn(x)
        return out