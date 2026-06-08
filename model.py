import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class CausalConv1d(nn.Module):
    """Strictly causal 1D convolution (no future leakage)."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding] if self.padding > 0 else x


class TCNNodeEncoder(nn.Module):
    """Dilated causal TCN.

    The receptive field must cover the input sequence or the lookback is
    wasted. With kernel=3 and dilations (1,2,4,8) the RF is
    1 + 2*(1+2+4+8) = 31 timesteps, matching a ~30s window. Residual
    connections + dropout stabilize the deeper stack.
    """

    def __init__(self, in_channels, hidden_channels, dilations=(1, 2, 4, 8), dropout=0.1):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList([
            CausalConv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=d) for d in dilations
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in dilations])
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (N, Seq_Len, Features)
        h = self.proj(x.transpose(1, 2))            # (N, H, Seq_Len)
        for conv, norm in zip(self.blocks, self.norms):
            h = h + self.drop(F.relu(norm(conv(h))))  # residual dilated block
        return h[:, :, -1]                          # latest causal state -> (N, H)


class RGATModel(nn.Module):
    """Spatio-temporal GNN.

    Architectural lightening: instead of a stack of heavy RGATConv layers (one
    attention mechanism per relation), we fold the discrete relation type into a
    one-hot appended to the continuous edge attributes and use a single
    relation-aware GATv2Conv per layer. Same relational expressiveness at a
    fraction of the parameters / FLOPs.
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_relations: int,
                 out_channels: int = 7, num_heads: int = 4, num_layers: int = 2,
                 edge_dim: int = 11):
        super().__init__()
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.edge_in_dim = edge_dim + num_relations   # continuous + one-hot relation

        self.tcn = TCNNodeEncoder(in_channels, hidden_channels)
        self.input_proj = nn.Linear(hidden_channels, hidden_channels)
        self.feature_dropout = nn.Dropout(0.2)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=num_heads,
                                        concat=False, edge_dim=self.edge_in_dim,
                                        add_self_loops=False, dropout=0.1))
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Per-horizon heads. Short horizons are driven by microstructure mean
        # reversion, long horizons by factor/macro momentum; a shared trunk with
        # one head each lets every horizon weight the representation differently
        # (and carry its own sign) instead of being forced into one compromise.
        self.trunk = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_channels, hidden_channels // 2), nn.GELU(),
                          nn.Linear(hidden_channels // 2, 1))
            for _ in range(out_channels)
        ])

    def _edge_features(self, edge_attr, edge_type):
        onehot = F.one_hot(edge_type, num_classes=self.num_relations).to(edge_attr.dtype)
        return torch.cat([edge_attr, onehot], dim=1)

    def forward(self, x_seq, edge_index, edge_type, edge_attr=None):
        edge_feat = self._edge_features(edge_attr, edge_type)

        h = self.tcn(x_seq)              # (N, H)
        h = self.input_proj(h)
        h = self.feature_dropout(h)

        for i in range(self.num_layers):
            h_out = self.convs[i](h, edge_index, edge_attr=edge_feat)
            h = self.norms[i](h + F.elu(h_out))

        h = self.trunk(h)
        return torch.cat([head(h) for head in self.heads], dim=1)   # (N, out_channels)