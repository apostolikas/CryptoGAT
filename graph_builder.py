import numpy as np
import torch

EPS = 1e-12

# Relation taxonomy (keep in sync with NUM_RELATIONS in train_evaluate.py)
REL_SELF = 0      # self-loop
REL_MACRO = 1     # macro future -> equity
REL_POS = 2       # positive contemporaneous correlation
REL_NEG = 3       # negative contemporaneous correlation
REL_LEADLAG = 4   # equity lead-lag
REL_SECTOR = 5    # same-sector structural edge
REL_MARKET = 6    # market/index (e.g. QQQ) -> equity
NUM_RELATIONS = 7
EDGE_DIM = 11     # number of continuous edge attributes


def _demean(x):
    return x - x.mean(axis=0, keepdims=True)


def _xcorr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson correlation between every column of A and every column of B.

    Returns (Na, Nb): out[i, j] = corr(A[:, i], B[:, j]).
    """
    n = A.shape[0]
    if n < 2:
        return np.zeros((A.shape[1], B.shape[1]))
    Ac, Bc = _demean(A), _demean(B)
    cov = (Ac.T @ Bc) / (n - 1)
    sa = A.std(axis=0)
    sb = B.std(axis=0)
    denom = np.outer(sa, sb)
    return np.divide(cov, denom, out=np.zeros_like(cov), where=(denom > 0))


def compute_rich_dynamic_edges(
    returns_5m: np.ndarray,
    returns_30m: np.ndarray = None,
    spread_vec: np.ndarray = None,
    liq_vec: np.ndarray = None,
    res_vec: np.ndarray = None,
    num_equities: int = 0,
    sector_mask: np.ndarray = None,
    market_mask: np.ndarray = None,
    threshold: float = 0.2,
    top_k: int = 8,
    macro_lag: int = 5,
):
    """Fully vectorized dynamic edge builder.

    All matrices are indexed [target i, source j]; an edge means j -> i.
    `returns_*` are dense numpy arrays (T, N) in canonical node order
    (equities first, then macros). Structural masks are (N, N) booleans.
    Returns torch tensors (edge_index[2,E], edge_attr[E,EDGE_DIM], edge_type[E]).
    """
    R5 = np.asarray(returns_5m, dtype=np.float64)
    T5, N = R5.shape

    # ---- contemporaneous second-moment statistics (vectorized) -------------
    var_5m = R5.var(axis=0)
    var_safe = np.where(var_5m > 0, var_5m, 1e-12)
    Rc = _demean(R5)
    cov_5m = (Rc.T @ Rc) / max(T5 - 1, 1)
    std_5m = np.sqrt(var_safe)
    denom = np.outer(std_5m, std_5m)
    corr_signed_5m = np.divide(cov_5m, denom, out=np.zeros_like(cov_5m), where=(denom > 0))
    corr_abs_5m = np.abs(corr_signed_5m)

    # beta[i, j] = cov(i, j) / var(j): j drives i (clipped: betas can blow up)
    beta_5m = np.clip(cov_5m / var_safe[None, :], -5.0, 5.0)
    vol_5m = std_5m
    # log volatility ratio: symmetric around 0, no heavy right tail
    vol_ratio = np.log((vol_5m[:, None] + EPS) / (vol_5m[None, :] + EPS))

    # equity lead-lag: corr(ret_i(t), ret_j(t-lag))
    if T5 > macro_lag + 1:
        ret_t = R5[macro_lag:]
        ret_lag = R5[:-macro_lag]
        lead_lag_5s = _xcorr(ret_t, ret_lag)
    else:
        lead_lag_5s = np.zeros((N, N))

    # lagged macro cross-correlation (lead-lag with liquid proxies):
    # corr(equity_i(t), macro_j(t-lag)) -- same array, we slice macro cols below
    macro_leadlag = lead_lag_5s

    # ---- 30m horizon stats -------------------------------------------------
    if returns_30m is not None:
        R30 = np.asarray(returns_30m, dtype=np.float64)
        if R30.shape[0] > 30:
            var_30 = R30.var(axis=0)
            var_30s = np.where(var_30 > 0, var_30, 1e-12)
            R30c = _demean(R30)
            cov_30 = (R30c.T @ R30c) / (R30.shape[0] - 1)
            beta_30m = np.clip(cov_30 / var_30s[None, :], -5.0, 5.0)
            lead_lag_30s = _xcorr(R30[30:], R30[:-30])
        else:
            beta_30m, lead_lag_30s = beta_5m, lead_lag_5s
    else:
        beta_30m, lead_lag_30s = beta_5m, lead_lag_5s

    # ---- cross-sectional node modifiers ------------------------------------
    spread_vec = np.ones(N) if spread_vec is None else np.asarray(spread_vec, dtype=np.float64)
    liq_vec = np.ones(N) if liq_vec is None else np.asarray(liq_vec, dtype=np.float64)
    res_vec = np.zeros(N) if res_vec is None else np.asarray(res_vec, dtype=np.float64)

    # log-ratios keep these symmetric around 0 instead of a long right tail.
    # Clamp to non-negative first: a crossed/locked book or a stale forward-fill
    # can make a raw spread negative, and log of a negative ratio is NaN.
    sv = np.clip(spread_vec, 0.0, None) + 1e-9
    lv = np.clip(liq_vec, 0.0, None) + 1e-9
    spread_z = np.log(sv[:, None] / sv[None, :])
    liq_ratio = np.log(lv[:, None] / lv[None, :])
    res_z = res_vec[:, None] - res_vec[None, :]

    # ---- relation assignment (vectorized; later writes win) ----------------
    rel = np.full((N, N), -1, dtype=np.int64)
    rel[corr_signed_5m >= threshold] = REL_POS
    rel[corr_signed_5m <= -threshold] = REL_NEG
    # lead-lag correlations are structurally smaller than contemporaneous ones,
    # so they get a lower threshold or they almost never form an edge.
    ll_threshold = max(0.08, threshold * 0.5)
    rel[np.abs(lead_lag_5s) >= ll_threshold] = REL_LEADLAG

    if sector_mask is not None:
        rel[sector_mask] = REL_SECTOR
    if market_mask is not None:
        rel[market_mask] = REL_MARKET

    # macro -> equity (source col is macro, target row is equity)
    col_is_macro = np.arange(N)[None, :] >= num_equities
    row_is_equity = np.arange(N)[:, None] < num_equities
    macro_src = col_is_macro & row_is_equity
    macro_qualify = macro_src & (
        (corr_abs_5m >= threshold / 2.0) | (np.abs(macro_leadlag) >= threshold / 2.0)
    )
    rel[macro_qualify] = REL_MACRO

    np.fill_diagonal(rel, REL_SELF)

    # ---- top-k sparsification of "dense" statistical relations -------------
    # Keep structural edges (self/macro/sector/market) always; prune only the
    # correlation / lead-lag edges to the strongest top_k sources per target.
    if top_k is not None and top_k > 0:
        score = np.maximum(corr_abs_5m, np.abs(lead_lag_5s))
        dense_mask = np.isin(rel, (REL_POS, REL_NEG, REL_LEADLAG))
        for i in range(N):
            cols = np.flatnonzero(dense_mask[i])
            if cols.size > top_k:
                worst = cols[np.argsort(score[i, cols])[:-top_k]]
                rel[i, worst] = -1

    # ---- materialize edges -------------------------------------------------
    i_idx, j_idx = np.nonzero(rel >= 0)        # i = target, j = source
    if i_idx.size == 0:                        # safety: fall back to self loops
        i_idx = j_idx = np.arange(N)
        rel[np.arange(N), np.arange(N)] = REL_SELF

    edge_index = np.stack([j_idx, i_idx], axis=0)   # [source, target]
    edge_type = rel[i_idx, j_idx]

    edge_attr = np.stack([
        corr_signed_5m[i_idx, j_idx],
        corr_abs_5m[i_idx, j_idx],
        beta_5m[i_idx, j_idx],
        beta_30m[i_idx, j_idx],
        lead_lag_5s[i_idx, j_idx],
        lead_lag_30s[i_idx, j_idx],
        vol_ratio[i_idx, j_idx],
        spread_z[i_idx, j_idx],
        liq_ratio[i_idx, j_idx],
        res_z[i_idx, j_idx],
        macro_leadlag[i_idx, j_idx],
    ], axis=1)

    edge_index = torch.from_numpy(edge_index).long().contiguous()
    edge_attr = torch.from_numpy(edge_attr).float()
    edge_type = torch.from_numpy(edge_type).long()
    edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=10.0, neginf=-10.0)
    return edge_index, edge_attr, edge_type


def build_static_masks(num_equities: int, num_macros: int, sector_groups, market_idx=None):
    """Precompute the (N, N) sector and market boolean masks once.

    sector_groups: list of lists of equity node indices that share a sector.
    market_idx: node index of the market/index node (e.g. QQQ), or None.
    Masks are indexed [target i, source j]; True means an edge j -> i.
    """
    n = num_equities + num_macros
    sector_mask = np.zeros((n, n), dtype=bool)
    for grp in sector_groups:
        for i in grp:
            for j in grp:
                if i != j:
                    sector_mask[i, j] = True   # j -> i within sector

    market_mask = np.zeros((n, n), dtype=bool)
    if market_idx is not None:
        for i in range(num_equities):
            if i != market_idx:
                market_mask[i, market_idx] = True   # market -> equity
    return sector_mask, market_mask