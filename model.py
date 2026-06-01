import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv

class RGATModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_relations: int, out_channels: int, num_heads: int = 4, num_layers: int = 2):
        """
        Step 7: Train the Relational Graph Attention Network (RGAT) - Architecture definition
        
        in_channels: Number of input features per node
        hidden_channels: Hidden dimension
        num_relations: Number of discrete edge types (e.g. 3: Sector, Lead-Lag, Correlation)
        out_channels: Number of prediction targets (e.g. 7 for the 7 forward horizons)
        num_heads: Number of attention heads
        """
        super(RGATModel, self).__init__()
        
        self.num_layers = num_layers
        
        # RGAT Layers
        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(RGATConv(in_channels, hidden_channels, num_relations=num_relations, heads=num_heads, concat=False))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(RGATConv(hidden_channels, hidden_channels, num_relations=num_relations, heads=num_heads, concat=False))
            
        # Multi-head feed-forward neural network projection layer for multi-horizon targets
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(self, x, edge_index, edge_type, edge_attr=None):
        """
        x: Node feature matrix (N, F)
        edge_index: Graph connectivity (2, E)
        edge_type: 1D tensor indicating the type of edge (E,)
        edge_attr: Edge weights (E,)
        """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_type, edge_attr=edge_attr)
            if i < self.num_layers - 1:
                x = F.elu(x)
                
        # Final projection to multi-horizon targets
        out = self.ffn(x)
        return out

if __name__ == "__main__":
    # Test RGAT Model shape
    N = 10
    F_in = 16
    out_dim = 7
    relations = 3
    
    x = torch.randn((N, F_in))
    edge_index = torch.randint(0, N, (2, 20))
    edge_type = torch.randint(0, relations, (20,))
    edge_attr = torch.rand(20)
    
    model = RGATModel(in_channels=F_in, hidden_channels=32, num_relations=relations, out_channels=out_dim)
    
    out = model(x, edge_index, edge_type, edge_attr)
    print("Output shape:", out.shape)
    assert out.shape == (N, out_dim)
