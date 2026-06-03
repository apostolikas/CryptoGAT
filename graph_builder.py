import pandas as pd
import numpy as np
import torch

def _compute_cross_correlation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes the asymmetric cross-correlation matrix between two time-series matrices.
    A and B must have shape (T_steps, N_assets).
    Returns shape (N_assets, N_assets) where entry [i, j] is corr(A_i, B_j).
    """
    if len(A) < 2:
        return np.zeros((A.shape[1], B.shape[1]))
        
    # Mean center
    A_c = A - np.nanmean(A, axis=0)
    B_c = B - np.nanmean(B, axis=0)
    
    # Covariance
    cov = (A_c.T @ B_c) / (len(A) - 1)
    
    # Standard deviations
    std_A = np.nanstd(A, axis=0)
    std_B = np.nanstd(B, axis=0)
    
    # Cross-correlation (handling div by zero)
    denominator = np.outer(std_A, std_B)
    corr = np.divide(cov, denominator, out=np.zeros_like(cov), where=(denominator != 0))
    
    return np.nan_to_num(corr)


def compute_rich_dynamic_edges(
    returns_5m: pd.DataFrame, 
    returns_30m: pd.DataFrame = None, 
    node_features_t = None, 
    threshold: float = 0.1
):
    """
    Constructs the dynamic, signed graph for a specific timestamp (A_fast_rolling_t + A_slow_rolling_t).
    
    Args:
        returns_5m: pd.DataFrame of shape (T_5m, N) containing returns for the last 5 minutes.
        returns_30m: pd.DataFrame of shape (T_30m, N) for the last 30 minutes (optional).
        node_features_t: pl.DataFrame or pd.DataFrame containing cross-sectional LOB attributes at time t.
        threshold: Minimum absolute 5m correlation to instantiate an edge.
    """
    symbols = returns_5m.columns.tolist()
    N = len(symbols)
    
    # -------------------------------------------------------------------------
    # 1. 5-Minute Fast Dynamics (Base Correlation & Beta)
    # -------------------------------------------------------------------------
    cov_5m = returns_5m.cov().fillna(0).to_numpy()
    var_5m = returns_5m.var().fillna(1e-12).to_numpy()
    
    # Signed and Absolute Correlation
    corr_signed_5m = returns_5m.corr(method='pearson').fillna(0).to_numpy()
    corr_abs_5m = np.abs(corr_signed_5m)
    
    # Asymmetric Beta: beta[i, j] = Cov(i,j) / Var(j) (Asset i's beta to Asset j)
    beta_5m = cov_5m / var_5m[None, :] 
    
    # Volatility Ratio: vol[i] / vol[j]
    vol_5m = np.sqrt(var_5m)
    vol_ratio = vol_5m[:, None] / (vol_5m[None, :] + 1e-12)

    # -------------------------------------------------------------------------
    # 2. 5-Second Lead-Lag Cross-Correlation (Micro-structural leading)
    # -------------------------------------------------------------------------
    if len(returns_5m) > 5:
        ret_t = returns_5m.to_numpy()[5:]
        ret_t_minus_5 = returns_5m.shift(5).to_numpy()[5:]
        lead_lag_5s = _compute_cross_correlation(ret_t, ret_t_minus_5)
    else:
        lead_lag_5s = np.zeros((N, N))

    # -------------------------------------------------------------------------
    # 3. 30-Minute Slow Dynamics (Macro Regimes)
    # -------------------------------------------------------------------------
    if returns_30m is not None and len(returns_30m) > 30:
        cov_30m = returns_30m.cov().fillna(0).to_numpy()
        var_30m = returns_30m.var().fillna(1e-12).to_numpy()
        beta_30m = cov_30m / var_30m[None, :]
        
        ret_30m_t = returns_30m.to_numpy()[30:]
        ret_30m_minus_30 = returns_30m.shift(30).to_numpy()[30:]
        lead_lag_30s = _compute_cross_correlation(ret_30m_t, ret_30m_minus_30)
    else:
        beta_30m = beta_5m
        lead_lag_30s = lead_lag_5s

    # -------------------------------------------------------------------------
    # 4. Cross-Sectional LOB & Residual State Ratios
    # -------------------------------------------------------------------------
    spread_z = np.ones((N, N))
    liq_ratio = np.ones((N, N))
    res_z = np.zeros((N, N))
    
    if node_features_t is not None:
        # Handles extraction gracefully whether input is Pandas or Polars
        if "spread_bps" in node_features_t.columns:
            spreads = node_features_t["spread_bps"].to_numpy() if hasattr(node_features_t["spread_bps"], "to_numpy") else node_features_t["spread_bps"].values
            spread_z = spreads[:, None] / (spreads[None, :] + 1e-9)
            
        if "top5_book_size" in node_features_t.columns:
            liqs = node_features_t["top5_book_size"].to_numpy() if hasattr(node_features_t["top5_book_size"], "to_numpy") else node_features_t["top5_book_size"].values
            liq_ratio = liqs[:, None] / (liqs[None, :] + 1e-9)
            
        res_cols = [c for c in node_features_t.columns if "kalman_z_score" in c]
        if res_cols:
            res_vals = node_features_t[res_cols[0]].to_numpy() if hasattr(node_features_t[res_cols[0]], "to_numpy") else node_features_t[res_cols[0]].values
            res_z = res_vals[:, None] - res_vals[None, :]

    # -------------------------------------------------------------------------
    # 5. Graph Assembly & Filtering
    # -------------------------------------------------------------------------
    edge_index = []
    edge_attr = []
    
    for i in range(N):
        for j in range(N):
            if i != j:
                if corr_abs_5m[i, j] >= threshold:
                    attr = [
                        corr_signed_5m[i, j],
                        corr_abs_5m[i, j],
                        beta_5m[i, j],
                        beta_30m[i, j],
                        lead_lag_5s[i, j],
                        lead_lag_30s[i, j],
                        vol_ratio[i, j],
                        spread_z[i, j],
                        liq_ratio[i, j],
                        res_z[i, j]
                    ]
                    edge_index.append([i, j])
                    edge_attr.append(attr)

    if not edge_index:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 10), dtype=torch.float)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=10.0, neginf=-10.0)

    return edge_index, edge_attr

def build_static_edges(symbols: list, sector_map: dict):
    edge_index = []
    for i, sym_i in enumerate(symbols):
        for j, sym_j in enumerate(symbols):
            if i != j:
                if sector_map.get(sym_i) == sector_map.get(sym_j):
                    edge_index.append([i, j])
                    
    if not edge_index:
        return torch.empty((2, 0), dtype=torch.long)
        
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()