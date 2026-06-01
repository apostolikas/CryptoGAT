import numpy as np
import polars as pl
import pandas as pd
import torch
from torch_geometric.data import Data

def compute_correlation_edges(returns_df: pl.DataFrame | pd.DataFrame, threshold: float = 0.7):
    """
    Compute correlation edges based on a rolling or static historical window.
    returns_df: DataFrame where columns are assets and rows are timestamps.
    """
    if isinstance(returns_df, pl.DataFrame):
        returns_df = returns_df.to_pandas()
        
    corr_matrix = returns_df.corr().abs()
    edges = []
    edge_weights = []
    
    assets = returns_df.columns
    for i, a1 in enumerate(assets):
        for j, a2 in enumerate(assets):
            if i != j:
                r = corr_matrix.loc[a1, a2]
                if pd.notna(r) and r > threshold:
                    edges.append([i, j])
                    edge_weights.append(r)
                    
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty(0, dtype=torch.float)
        
    return edge_index, edge_weight

def build_graph(assets: list, macro_anchors: list, sector_map: dict, returns_df: pl.DataFrame | pd.DataFrame = None):
    """
    Step 6: Define the Relational Graph Schema
    assets: list of equity tickers (e.g. ['AAPL', 'NVDA', 'KO', 'PEP'])
    macro_anchors: list of macro tickers (e.g. ['NQ', 'GC'])
    sector_map: dict mapping ticker to sector string
    returns_df: DataFrame of returns for correlation computation
    """
    all_nodes = assets + macro_anchors
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    
    edge_types = {}
    
    # 1. Sector/Industry relation (Equities only)
    sector_edges = []
    for i, a1 in enumerate(assets):
        for j, a2 in enumerate(assets):
            if i != j and sector_map.get(a1) == sector_map.get(a2):
                sector_edges.append([node_to_idx[a1], node_to_idx[a2]])
                
    if sector_edges:
        edge_types['sector'] = (
            torch.tensor(sector_edges, dtype=torch.long).t().contiguous(),
            torch.ones(len(sector_edges), dtype=torch.float)
        )
        
    # 2. Lead-Lag relation (Macro -> Equities)
    # Index-Beta Edges (NQ -> Equities) and Commodity-Hedge Edges (GC -> Equities)
    lead_lag_edges = []
    for macro in macro_anchors:
        idx_m = node_to_idx[macro]
        for asset in assets:
            idx_a = node_to_idx[asset]
            lead_lag_edges.append([idx_m, idx_a])
            
    if lead_lag_edges:
        edge_types['lead_lag'] = (
            torch.tensor(lead_lag_edges, dtype=torch.long).t().contiguous(),
            torch.ones(len(lead_lag_edges), dtype=torch.float) # Weights could be dynamic (beta)
        )
        
    # 3. Correlation relation
    if returns_df is not None:
        if isinstance(returns_df, pl.DataFrame):
            assets_df = returns_df.select(assets)
        else:
            assets_df = returns_df[assets]
            
        corr_edge_index, corr_edge_weight = compute_correlation_edges(assets_df, threshold=0.7)
        # Remap indices to global graph indices
        if corr_edge_index.numel() > 0:
            for i in range(corr_edge_index.shape[1]):
                u = assets[corr_edge_index[0, i].item()]
                v = assets[corr_edge_index[1, i].item()]
                corr_edge_index[0, i] = node_to_idx[u]
                corr_edge_index[1, i] = node_to_idx[v]
            edge_types['correlation'] = (corr_edge_index, corr_edge_weight)
            
    return all_nodes, node_to_idx, edge_types

if __name__ == "__main__":
    pass
