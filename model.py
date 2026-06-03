import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv

class TCNNodeEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(TCNNodeEncoder, self).__init__()
        # 1D Convolution over time axis (Seq_Len)
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x input shape: (Num_Nodes, Seq_Len, Features)
        x = x.transpose(1, 2) # Transpose to (Num_Nodes, Features, Seq_Len) for Conv1d engine
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Temporal pooling over the lookback horizon window steps
        x_out = x.mean(dim=-1) # Output matrix shape: (Num_Nodes, Hidden_Channels)
        return x_out

class RGATModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_relations: int, out_channels: int, 
                 num_heads: int = 4, num_layers: int = 2, edge_dim: int = 10): 
        super(RGATModel, self).__init__()
        
        self.num_layers = num_layers
        
        # 1. Temporal Memory Encoder (TCN)
        self.tcn = TCNNodeEncoder(in_channels, hidden_channels)
        
        # 2. Input Projection & Normalization
        self.input_proj = nn.Linear(hidden_channels, hidden_channels)
        self.feature_dropout = nn.Dropout(0.2)
        
        # 3. Dynamic RGAT Layers with Residuals and LayerNorm
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # FIX: By setting concat=False, the multi-head outputs are averaged.
            # This forces output node geometries to stay strictly at hidden_channels dimension width
            self.convs.append(RGATConv(hidden_channels, hidden_channels, num_relations=num_relations, 
                                       heads=num_heads, concat=False, edge_dim=edge_dim))
            self.norms.append(nn.LayerNorm(hidden_channels))
            
        # 4. Multi-horizon prediction projection layer
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(self, x_seq, edge_index, edge_type, edge_attr=None):
        """
        x_seq: Node feature history vector sequence (Num_Nodes, Seq_Len, Features)
        edge_index: Graph connectivity matrix topology coordinates (2, Edges_Count)
        edge_type: 1D identifier mapping array for edges (Edges_Count,)
        edge_attr: 10-dimensional signed alpha feature edge attributes tensor (Edges_Count, 10)
        """
        # 1. Compress Temporal sequence dimension using the TCN node encoder
        h = self.tcn(x_seq) # Output shape matrix: (Num_Nodes, Hidden_Channels)
        
        # 2. Apply Linear layer projections and feature regularizations
        h = self.input_proj(h)
        h = self.feature_dropout(h)
        
        # 3. Message Passing Loops
        # FIX: Keep h as a pure 2D matrix shape (Num_Nodes, Hidden_Channels) to satisfy PyG's indexing check.
        for i in range(self.num_layers):
            h_out = self.convs[i](h, edge_index, edge_type, edge_attr=edge_attr)
            
            # LayerNorm and Residual accumulation on flat 2D structures
            h = self.norms[i](h + F.elu(h_out))
                
        # 4. Final univariate horizon regression calculation
        out = self.ffn(h)
        return out