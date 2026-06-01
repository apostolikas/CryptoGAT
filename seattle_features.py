"""
qstrat_v15_10_formulas.py
=========================

Source-of-truth reference for every formula in the QStrat v15_10 alpha research
pipeline. Every function here is what the notebook actually executes.

Self-contained: depends only on numpy, pandas, and (optionally) numba.

Layout (sections):
    1. Group A — book microstructure
    2. Group B — book-state raw inputs (OFI, top-K book size, quote update count)
    3. Group C — trade tape (volume, CVD)
    4. Group D — book-event rates (add/remove rates, net_book_pressure)
    5. Group E — absorption (buy/sell, multiple windows)
    6. Group F — returns derivatives, vol state, variance ratios, trade-side ratio
    7. EWMA z-score (H1-fixed) and signed-z sweep (H2-fixed)
    8. Trajectory descriptors (slope, net_change, curvature, monotonicity, efficiency, drift_vol)
    9. Forward return computation and event classification
   10. Lift over noise (Stage 1)
   11. Day-block bootstrap CI (Stage 2 / Stage 3)
   12. Stability checks A, B, C, D, F (Cell 14L)

Conventions:
    - All "rates" are per-bar (50ms bar grid)
    - Times in seconds (s), milliseconds (ms), or bars (b) — disambiguated by suffix
    - bps = basis points = 10⁴ × log_return (or relative price change)
    - All formulas use closed-interval arithmetic; NaN denotes "insufficient data" not "missing"

Run this file directly to execute the unit tests at the bottom (no third-party test framework).
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd

# numba is optional. If absent, the @njit decorator is a no-op (pure Python fallback works
# but is much slower). In production we always use numba.
try:
    from numba import njit
except ImportError:                                                       # pragma: no cover
    def njit(f, *args, **kwargs):                                         # noqa: D401
        """Identity decorator if numba unavailable."""
        return f


GRID_MS = 50                       # 50ms per bar
BARS_PER_SEC = 1000 // GRID_MS     # 20 bars per second
EPS = 1e-12


# ════════════════════════════════════════════════════════════════════════════
# 1. GROUP A — BOOK MICROSTRUCTURE
# ════════════════════════════════════════════════════════════════════════════

def spread_bps(bid0: np.ndarray, ask0: np.ndarray) -> np.ndarray:
    """Bid-ask spread in basis points.

        spread_bps = (ask0 - bid0) / mid * 10⁴
        mid = (bid0 + ask0) / 2

    Returns NaN when mid <= 0 (e.g. before book initialization).
    """
    mid = 0.5 * (bid0 + ask0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mid > 0, (ask0 - bid0) / mid * 1e4, np.nan).astype(np.float32)


def microprice(bid0: np.ndarray, ask0: np.ndarray,
               bid_sz0: np.ndarray, ask_sz0: np.ndarray) -> np.ndarray:
    """Volume-weighted "true" price between best bid and ask.

        microprice = (ask0 * bid_sz0 + bid0 * ask_sz0) / (bid_sz0 + ask_sz0)

    Note the swap (ask weighted by bid size, bid by ask size): this reflects which
    side is more likely to clear next given the resting depth.
    Falls back to mid when both sizes are zero.
    """
    denom = bid_sz0 + ask_sz0
    mid = 0.5 * (bid0 + ask0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0,
                        (ask0 * bid_sz0 + bid0 * ask_sz0) / denom,
                        mid).astype(np.float32)


def microprice_dev_bps(bid0: np.ndarray, ask0: np.ndarray,
                       bid_sz0: np.ndarray, ask_sz0: np.ndarray) -> np.ndarray:
    """Microprice deviation from mid in basis points.

        microprice_dev_bps = (microprice - mid) / mid * 10⁴

    Positive => order book skewed toward upward clearing.
    """
    mid = 0.5 * (bid0 + ask0)
    mp = microprice(bid0, ask0, bid_sz0, ask_sz0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mid > 0, (mp - mid) / mid * 1e4, np.nan).astype(np.float32)


def obi_level(bid_szs: np.ndarray, ask_szs: np.ndarray) -> np.ndarray:
    """Order book imbalance for given side sizes (1D each).

        obi = (bid - ask) / (bid + ask) ∈ [-1, +1]

    Works for any aggregation (L1, L5, etc.) — caller computes the sums.
    """
    denom = bid_szs + ask_szs
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, (bid_szs - ask_szs) / denom, np.nan).astype(np.float32)


def depth_ratio(this_level: np.ndarray, level_one: np.ndarray) -> np.ndarray:
    """Ratio of size at a deeper level to size at level 1 (best).

        depth_ratio = level_K_size / level_1_size

    Stays raw (no z-scoring): naturally bounded above 0, captures depth profile shape.
    Used at K ∈ {2, 3, 5} for both bid and ask sides.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(level_one > 0, this_level / level_one, np.nan).astype(np.float32)


def mid_log_return_bps(mid: np.ndarray) -> np.ndarray:
    """Per-bar log return of mid in basis points.

        ret_bps[t] = (log(mid[t]) - log(mid[t-1])) * 10⁴

    First element is 0 (no prior bar). NaN where mid <= 0.

    *** IMPORTANT: this feature is computed but DELIBERATELY EXCLUDED from trajectory
    analysis. Including it would be tautological — events are themselves defined by
    forward log returns of mid. ***
    """
    valid = mid > 0
    log_mid = np.where(valid, np.log(mid), np.nan)
    log_ret = np.diff(log_mid, prepend=log_mid[0])
    log_ret[0] = 0.0
    return (log_ret * 1e4).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# 2. GROUP B — BOOK STATE RAW (top-K size, OFI, quote update count)
# ════════════════════════════════════════════════════════════════════════════

def topk_book_size(bid_sizes_by_lv: List[np.ndarray],
                   ask_sizes_by_lv: List[np.ndarray],
                   K: int) -> np.ndarray:
    """Sum of bid AND ask depth across top K price levels.

        topK = Σ_{lv=0..K-1} bid_sz[lv] + Σ_{lv=0..K-1} ask_sz[lv]

    Used for K ∈ {5, 10}. Z-scored downstream (log1p first).

    bid_sizes_by_lv, ask_sizes_by_lv: lists of 1D arrays, one per level.
    """
    assert len(bid_sizes_by_lv) >= K and len(ask_sizes_by_lv) >= K, \
        f"need at least {K} levels"
    total = np.zeros_like(bid_sizes_by_lv[0], dtype=np.float64)
    for lv in range(K):
        total += bid_sizes_by_lv[lv].astype(np.float64)
        total += ask_sizes_by_lv[lv].astype(np.float64)
    return total.astype(np.float32)


def shifted(arr: np.ndarray, k: int) -> np.ndarray:
    """Shift by k. Forward fills initial with arr[0]."""
    out = np.roll(arr, k)
    if k > 0:
        out[:k] = arr[0]
    elif k < 0:
        out[k:] = arr[-1]
    return out


def ofi_multilevel(bid_pxs: List[np.ndarray], ask_pxs: List[np.ndarray],
                   bid_szs: List[np.ndarray], ask_szs: List[np.ndarray],
                   K: int = 5) -> np.ndarray:
    """Cont-Kukanov multi-level Order Flow Imbalance.

    For each price level lv ∈ [0, K):
        bid_event[lv] = + bid_sz[lv]            if bid_px[lv] > bid_px_prev[lv]   (better bid posted)
                        - bid_sz_prev[lv]        if bid_px[lv] < bid_px_prev[lv]   (bid pulled back)
                        bid_sz[lv] - bid_sz_prev[lv]   otherwise                  (size adjusted at same px)
        ask_event[lv] = + ask_sz[lv]            if ask_px[lv] < ask_px_prev[lv]   (better ask posted)
                        - ask_sz_prev[lv]        if ask_px[lv] > ask_px_prev[lv]   (ask pulled back)
                        ask_sz[lv] - ask_sz_prev[lv]   otherwise

        ofi = Σ_{lv} (1 / (1 + lv)) * (bid_event[lv] - ask_event[lv])

    Positive => buy pressure (bids added or asks pulled). Signed quantity, mean-zero
    in expectation. After H2 fix: z-scored directly on this signed series.
    """
    n = len(bid_pxs[0])
    ofi = np.zeros(n, dtype=np.float64)
    for lv in range(K):
        if lv >= len(bid_pxs):
            break
        bp = bid_pxs[lv].astype(np.float64)
        ap = ask_pxs[lv].astype(np.float64)
        bs = bid_szs[lv].astype(np.float64)
        a_sz = ask_szs[lv].astype(np.float64)
        bp_prev = shifted(bp, 1)
        ap_prev = shifted(ap, 1)
        bs_prev = shifted(bs, 1)
        as_prev = shifted(a_sz, 1)
        bid_e = np.where(bp > bp_prev,  bs,
                np.where(bp < bp_prev, -bs_prev, bs - bs_prev))
        ask_e = np.where(ap < ap_prev,  a_sz,
                np.where(ap > ap_prev, -as_prev, a_sz - as_prev))
        w = 1.0 / (1.0 + lv)
        ofi += w * (bid_e - ask_e)
    return np.nan_to_num(ofi, nan=0.0).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# 3. GROUP C — TRADE TAPE
# ════════════════════════════════════════════════════════════════════════════

def cvd_window(signed_trade_size: np.ndarray, window_s: int) -> np.ndarray:
    """Cumulative Volume Delta over a rolling window.

        cvd_W[t] = Σ_{i=t-W+1..t} signed_trade_size[i]

    signed_trade_size[t] = + size  if trade was a buy
                          - size  if trade was a sell
                          0       if no trade in bar t

    Signed series, can be negative. Z-scored directly (H2 fix).
    """
    w_b = window_s * BARS_PER_SEC
    return rolling_sum(signed_trade_size.astype(np.float64), w_b).astype(np.float32)


def rolling_sum(arr: np.ndarray, w_bars: int) -> np.ndarray:
    """Rolling sum over w_bars window. Right-aligned (sums [t-w+1..t])."""
    return pd.Series(arr).rolling(w_bars, min_periods=1).sum().to_numpy()


# ════════════════════════════════════════════════════════════════════════════
# 4. GROUP D — BOOK EVENT RATES
# ════════════════════════════════════════════════════════════════════════════

def net_book_pressure(bid_add: np.ndarray, ask_add: np.ndarray,
                      bid_remove: np.ndarray, ask_remove: np.ndarray) -> np.ndarray:
    """Net pressure exerted by book event flow.

        net_book_pressure = (ask_remove + bid_add) - (bid_remove + ask_add)

    Positive => upward book pressure (asks being lifted, bids being added).
    Negative => downward (bids being lifted, asks being added).
    Signed series; z-scored directly (H2 fix).
    """
    return (ask_remove.astype(np.float64) + bid_add.astype(np.float64)
            - bid_remove.astype(np.float64) - ask_add.astype(np.float64)).astype(np.float32)


def bid_to_ask_ratio(bid_count: np.ndarray, ask_count: np.ndarray,
                     fallback: float = 0.5) -> np.ndarray:
    """Bid share of total bid+ask event count.

        ratio = bid / (bid + ask)   ∈ [0, 1]

    Falls back to `fallback` (default 0.5 = neutral) when total is zero.
    Bounded in [0, 1] → no z-scoring needed.
    """
    denom = bid_count.astype(np.float64) + ask_count.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0,
                        bid_count.astype(np.float64) / denom,
                        np.float64(fallback)).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# 5. GROUP E — ABSORPTION
# ════════════════════════════════════════════════════════════════════════════

def absorption(side_volume_per_bar: np.ndarray, ret_bps: np.ndarray,
               window_s: int, side: str) -> np.ndarray:
    """Absorption: how much volume per basis-point of price movement.

        buy_absorption_W  = Σ(buy_vol over W bars) / Σ(max(ret_bps, 0) over W bars)
        sell_absorption_W = Σ(sell_vol over W bars) / Σ(max(-ret_bps, 0) over W bars)

    Intuition: high absorption ⇒ market is "eating" the volume without moving price
    ⇒ liquidity is strong on that side ⇒ price exhaustion. Eventually fails →
    reversal. Computed at multiple intrinsic windows {30s, 180s, 600s} per the
    `absorption_tails` research.

    Non-negative, right-skewed → log1p then EWMA-z'd downstream.

    side ∈ {"buy", "sell"}. Volume side determines numerator; move-direction
    determines denominator.
    """
    assert side in ("buy", "sell")
    w_b = window_s * BARS_PER_SEC
    vol_w = rolling_sum(side_volume_per_bar.astype(np.float64), w_b)
    if side == "buy":
        moves = np.maximum(ret_bps.astype(np.float64), 0.0)
    else:
        moves = np.maximum(-ret_bps.astype(np.float64), 0.0)
    moves_w = rolling_sum(moves, w_b)
    EPS_LOC = 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(moves_w > EPS_LOC, vol_w / (moves_w + EPS_LOC), np.nan)
    return out.astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# 6. GROUP F — RETURNS DERIVATIVES, VOL STATE, VARIANCE RATIOS, TRADE SIDE
# ════════════════════════════════════════════════════════════════════════════

def realized_vol(ret_bps_per_bar: np.ndarray, window_s: int) -> np.ndarray:
    """Rolling std of per-bar log returns (in bps units).

        rvol_W[t] = std(ret_bps[t-W+1..t])     ddof=1

    Non-negative, right-skewed → log1p then EWMA-z'd.
    """
    w_b = window_s * BARS_PER_SEC
    s = pd.Series(ret_bps_per_bar.astype(np.float64))
    min_p = max(20, w_b // 4)
    return s.rolling(w_b, min_periods=min_p).std().to_numpy(dtype=np.float32)


def vol_expand_ratio(rvol_short: np.ndarray, rvol_long: np.ndarray) -> np.ndarray:
    """Vol expansion ratio: short-window vol over long-window vol.

        vol_expand = rvol_short / (rvol_long + ε)

    > 1 ⇒ recent vol increasing relative to longer baseline.
    < 1 ⇒ recent vol contracting.
    Bounded near 0+ → kept raw (no z-scoring).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        out = rvol_short / (rvol_long + EPS)
    return np.where(np.isfinite(out), out, np.nan).astype(np.float32)


def trade_side_ratio(buy_vol: np.ndarray, sell_vol: np.ndarray, window_s: int) -> np.ndarray:
    """Buy share of total trade volume over rolling window.

        trade_side_ratio_W[t] = Σ(buy_vol over W) / (Σ(buy_vol) + Σ(sell_vol))

    ∈ [0, 1]. Falls back to 0.5 (neutral) when total volume is zero.
    Bounded → kept raw.
    """
    w_b = window_s * BARS_PER_SEC
    bv_w = pd.Series(buy_vol.astype(np.float64)
                    ).rolling(w_b, min_periods=max(20, w_b // 4)).sum().to_numpy()
    sv_w = pd.Series(sell_vol.astype(np.float64)
                    ).rolling(w_b, min_periods=max(20, w_b // 4)).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = bv_w / (bv_w + sv_w)
    return np.where(np.isfinite(out), out, 0.5).astype(np.float32)


def variance_ratio(mid: np.ndarray, K: int, W_s: int) -> np.ndarray:
    """Lo-MacKinlay variance ratio.

        K-bar log return: ret_K[t] = log(mid[t]) - log(mid[t-K])
        1-bar log return: ret_1[t] = log(mid[t]) - log(mid[t-1])

        var_K = rolling Var of ret_K over W_s seconds  (overlapping K-bar returns)
        var_1 = rolling Var of ret_1 over W_s seconds

        VR(K, W) = var_K / (K * var_1)

    Interpretation:
        > 1 ⇒ trending (K-step variance grows faster than linear)
        < 1 ⇒ mean-reverting
        ≈ 1 ⇒ random walk

    Used at (K=5, W=60s) and (K=15, W=180s). Non-negative, right-skewed → log1p + z.
    """
    W_b = W_s * BARS_PER_SEC
    valid = mid > 0
    log_mid = np.where(valid, np.log(mid), np.nan)
    log_mid_shifted = np.roll(log_mid, K)
    log_mid_shifted[:K] = np.nan
    ret_K = log_mid - log_mid_shifted
    ret_1 = np.diff(log_mid, prepend=log_mid[0])
    ret_1[0] = np.nan
    s_K = pd.Series(ret_K)
    s_1 = pd.Series(ret_1)
    var_K = s_K.rolling(W_b, min_periods=max(K * 4, W_b // 4)).var().to_numpy(dtype=np.float64)
    var_1 = s_1.rolling(W_b, min_periods=max(20, W_b // 4)).var().to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        vr = var_K / (K * var_1 + 1e-30)
    return np.where(np.isfinite(vr) & (var_1 > 0), vr, np.nan).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# 7. EWMA Z-SCORE (H1-FIXED) and signed-z sweep (H2-FIXED)
# ════════════════════════════════════════════════════════════════════════════

@njit
def _ewma_z_with_state(x: np.ndarray, halflife: float,
                        m0: float, v0: float, started: bool) -> tuple:
    """EWMA-driven streaming z-score. State-passing version (per-instrument
    persistence across days).

    Decay parameter from halflife (bars):
        a = 1 - 0.5^(1/halflife)

    For each finite x[i]:
        delta = x[i] - m
        # H1 FIX (v15_10): compute z FIRST using pre-update m, v
        z[i] = (x[i] - m) / sqrt(v)        if v > 1e-12 else NaN
        m ← m + a * delta
        v ← (1 - a) * v + a * delta²

    Returns (out, m_final, v_final, started_final).
    """
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    a = 1.0 - 0.5 ** (1.0 / halflife)
    one_minus_a = 1.0 - a
    m = m0; v = v0; s = started
    for i in range(n):
        xi = x[i]
        if xi != xi or xi == np.inf or xi == -np.inf:
            continue
        if not s:
            m = xi; v = 0.0; s = True
            continue
        delta = xi - m
        if v > 1e-12:
            out[i] = (xi - m) / np.sqrt(v)
        m = m + a * delta
        v = one_minus_a * v + a * delta * delta
    return out, m, v, s


def ewma_zscore(arr: np.ndarray, halflife_bars: float,
                state: Optional[Tuple[float, float, bool]] = None
                ) -> Tuple[np.ndarray, Tuple[float, float, bool]]:
    """Convenience wrapper around _ewma_z_with_state. Defaults to fresh state."""
    if state is None:
        m0, v0, s0 = 0.0, 0.0, False
    else:
        m0, v0, s0 = state
    z, m, v, s = _ewma_z_with_state(np.ascontiguousarray(arr, dtype=np.float64),
                                    float(halflife_bars), m0, v0, s0)
    return z.astype(np.float32), (m, v, s)


def z_feature_sweep(raw_arr: np.ndarray, sign_preserve: bool, log1p_flag: bool,
                    halflives_bars: List[float],
                    ewma_states: Dict[float, Tuple[float, float, bool]]
                    ) -> Tuple[Dict[float, np.ndarray], Dict[float, Tuple[float, float, bool]]]:
    """Apply EWMA z-score across a list of halflives.

    H2 FIX (v15_10): for sign_preserve features, compute z directly on the SIGNED
    series. Old (broken) behavior was z(|x|) * sign(x), which is not a z-score of x.

    log1p_flag is incompatible with sign_preserve=True (cannot log1p signed data).

    Returns:
        z_outputs: dict halflife_bars → z-scored array (same length as input)
        new_states: updated EWMA states
    """
    if sign_preserve:
        if log1p_flag:
            raise ValueError("sign_preserve=True is incompatible with log1p_flag=True")
        pre_z = np.nan_to_num(raw_arr.astype(np.float64), nan=0.0)  # signed, no abs, no log1p
    else:
        if log1p_flag:
            pre_z = np.log1p(np.nan_to_num(np.maximum(raw_arr.astype(np.float64), 0.0),
                                            nan=0.0))
        else:
            pre_z = np.nan_to_num(raw_arr.astype(np.float64), nan=0.0)
    z_out: Dict[float, np.ndarray] = {}
    new_states: Dict[float, Tuple[float, float, bool]] = {}
    for hl_b in halflives_bars:
        z, st = ewma_zscore(pre_z, hl_b, state=ewma_states.get(hl_b))
        z_out[hl_b] = z
        new_states[hl_b] = st
    return z_out, new_states


# Z_FEATURES_SPEC — exact tuple list as used in Cell 4
# Each entry: (base_feature_name, log1p_flag, sign_preserve_flag)
Z_FEATURES_SPEC = [
    # Group B
    ("top5_book_size",         True,  False),
    ("top10_book_size",        True,  False),
    ("quote_update_count",     True,  False),
    ("ofi_raw",                False, True),     # signed — H2: true signed z
    # Group C
    ("buy_volume_per_bar",     True,  False),
    ("sell_volume_per_bar",    True,  False),
    ("cvd_30s",                False, True),     # signed
    ("cvd_60s",                False, True),     # signed
    ("cvd_180s",               False, True),     # signed
    # Group D
    ("bid_add_rate",           True,  False),
    ("ask_add_rate",           True,  False),
    ("bid_remove_rate",        True,  False),
    ("ask_remove_rate",        True,  False),
    ("net_book_pressure",      False, True),     # signed
    # Group E
    ("buy_absorption_30s",     True,  False),
    ("buy_absorption_180s",    True,  False),
    ("buy_absorption_600s",    True,  False),
    ("sell_absorption_30s",    True,  False),
    ("sell_absorption_180s",   True,  False),
    ("sell_absorption_600s",   True,  False),
    # Group F
    ("rvol_5s",                True,  False),
    ("rvol_30s",               True,  False),
    ("vr_k5_w60s",             True,  False),
    ("vr_k15_w180s",           True,  False),
]


# ════════════════════════════════════════════════════════════════════════════
# 8. TRAJECTORY DESCRIPTORS
# ════════════════════════════════════════════════════════════════════════════
# Each takes a 2D array `traj` of shape (n_anchors, W_bars). Returns 1D array
# of length n_anchors with one descriptor value per anchor.

@njit
def _slope_numba(traj: np.ndarray, dt_per_bar_s: float) -> np.ndarray:
    """OLS slope of v vs x = j * dt_per_bar_s, per second.

        slope = (n * Σxv - Σx Σv) / (n * Σx² - (Σx)²)

    Returns NaN if fewer than 4 finite values or denom ≤ 1e-12.
    """
    n_events, n_bars = traj.shape
    out = np.full(n_events, np.nan, dtype=np.float64)
    for i in range(n_events):
        sx = 0.0; sy = 0.0; sxy = 0.0; sxx = 0.0; cnt = 0
        for j in range(n_bars):
            v = traj[i, j]
            if v != v:
                continue
            x = j * dt_per_bar_s
            sx += x; sy += v
            sxy += x * v; sxx += x * x
            cnt += 1
        if cnt < 4:
            continue
        denom = cnt * sxx - sx * sx
        if denom <= 1e-12:
            continue
        out[i] = (cnt * sxy - sx * sy) / denom
    return out


@njit
def _curvature_numba(traj: np.ndarray, dt_per_bar_s: float) -> np.ndarray:
    """Quadratic fit 'a' coefficient (per s²). Solved via Cramer's rule on 3×3
    normal equations:

        [Σx⁴ Σx³ Σx²] [a]   [Σx²v]
        [Σx³ Σx² Σx¹] [b] = [Σxv ]
        [Σx² Σx¹ Σx⁰] [c]   [Σv  ]

    Returns NaN if fewer than 5 finite values or determinant ≈ 0.
    Less stable than slope at small n.
    """
    n_events, n_bars = traj.shape
    out = np.full(n_events, np.nan, dtype=np.float64)
    for i in range(n_events):
        sx0 = 0.0; sx1 = 0.0; sx2 = 0.0; sx3 = 0.0; sx4 = 0.0
        sv = 0.0;  sxv = 0.0; sx2v = 0.0
        cnt = 0
        for j in range(n_bars):
            v = traj[i, j]
            if v != v:
                continue
            x = j * dt_per_bar_s
            x2 = x * x
            sx0 += 1.0; sx1 += x; sx2 += x2; sx3 += x2 * x; sx4 += x2 * x2
            sv += v; sxv += x * v; sx2v += x2 * v
            cnt += 1
        if cnt < 5:
            continue
        D = (sx4 * (sx2 * sx0 - sx1 * sx1)
             - sx3 * (sx3 * sx0 - sx1 * sx2)
             + sx2 * (sx3 * sx1 - sx2 * sx2))
        if abs(D) < 1e-18:
            continue
        Da = (sx2v * (sx2 * sx0 - sx1 * sx1)
              - sx3 * (sxv * sx0 - sx1 * sv)
              + sx2 * (sxv * sx1 - sx2 * sv))
        out[i] = Da / D
    return out


@njit
def _net_change_numba(traj: np.ndarray) -> np.ndarray:
    """Last finite value minus first finite value.

    NaN if fewer than 2 finite values.
    """
    n_events, n_bars = traj.shape
    out = np.full(n_events, np.nan, dtype=np.float64)
    for i in range(n_events):
        first = np.nan
        for j in range(n_bars):
            v = traj[i, j]
            if v == v:
                first = v
                break
        if first != first:
            continue
        last = np.nan
        for j in range(n_bars - 1, -1, -1):
            v = traj[i, j]
            if v == v:
                last = v
                break
        if last != last:
            continue
        out[i] = last - first
    return out


@njit
def _monotonicity_numba(traj: np.ndarray) -> np.ndarray:
    """Fraction of bar-to-bar steps that are increases.

        monotonicity = (count of pairs where v[j] > v[j-1]) / (count of valid pairs)

    0.5 = random walk
    1.0 = monotonically rising
    0.0 = monotonically falling

    NaN if fewer than 5 valid pairs.
    """
    n_events, n_bars = traj.shape
    out = np.full(n_events, np.nan, dtype=np.float64)
    for i in range(n_events):
        ups = 0
        pairs = 0
        prev = np.nan
        prev_set = False
        for j in range(n_bars):
            v = traj[i, j]
            if v != v:
                continue
            if not prev_set:
                prev = v; prev_set = True
                continue
            if v > prev:
                ups += 1
            pairs += 1
            prev = v
        if pairs < 5:
            continue
        out[i] = ups / pairs
    return out


@njit
def _efficiency_numba(traj: np.ndarray) -> np.ndarray:
    """Path efficiency.

        efficiency = |Σ Δv| / Σ|Δv|   ∈ [0, 1]

    1.0 = monotonic (every step in same direction)
    0.0 = pure chop (zero net displacement despite path length)

    Distinct from monotonicity: monotonicity counts step directions; efficiency
    weights by step magnitude. A trajectory with one large countering step has
    low efficiency even if most other steps were monotonic.

    NaN if fewer than 5 pairs or zero total path length.
    """
    n_events, n_bars = traj.shape
    out = np.full(n_events, np.nan, dtype=np.float64)
    for i in range(n_events):
        net = 0.0
        total = 0.0
        pairs = 0
        prev = np.nan
        prev_set = False
        for j in range(n_bars):
            v = traj[i, j]
            if v != v:
                continue
            if not prev_set:
                prev = v; prev_set = True
                continue
            d = v - prev
            net += d
            total += abs(d)
            pairs += 1
            prev = v
        if pairs < 5 or total <= 0.0:
            continue
        out[i] = abs(net) / total
    return out


@njit
def _drift_vol_numba(traj: np.ndarray) -> np.ndarray:
    """Drift-to-volatility (t-statistic style).

        drift_vol = |Σ Δv| / σ(Δv)

    Net displacement relative to step-noise. High value = large move relative
    to its own step variability (significant drift).

    Different from slope (rate of change) and from monotonicity (direction consistency).

    NaN if fewer than 5 pairs or zero step std.
    """
    n_events, n_bars = traj.shape
    out = np.full(n_events, np.nan, dtype=np.float64)
    for i in range(n_events):
        sum_d = 0.0
        sum_dsq = 0.0
        pairs = 0
        prev = np.nan
        prev_set = False
        for j in range(n_bars):
            v = traj[i, j]
            if v != v:
                continue
            if not prev_set:
                prev = v; prev_set = True
                continue
            d = v - prev
            sum_d += d
            sum_dsq += d * d
            pairs += 1
            prev = v
        if pairs < 5:
            continue
        mean_d = sum_d / pairs
        var_d = (sum_dsq / pairs) - mean_d * mean_d
        if var_d <= 0.0:
            continue
        out[i] = abs(sum_d) / np.sqrt(var_d)
    return out


def compute_descriptors(traj: np.ndarray, dt_per_bar_s: float = 0.050) -> Dict[str, np.ndarray]:
    """Compute all six descriptors for a 2D trajectory array (n_anchors, W_bars).

    Returns dict { "slope", "net_change", "curvature", "monotonicity_score",
                   "efficiency", "drift_vol" } each of shape (n_anchors,) float32.
    """
    traj_c = np.ascontiguousarray(traj, dtype=np.float32)
    return {
        "slope":              _slope_numba(traj_c, dt_per_bar_s).astype(np.float32),
        "net_change":         _net_change_numba(traj_c).astype(np.float32),
        "curvature":          _curvature_numba(traj_c, dt_per_bar_s).astype(np.float32),
        "monotonicity_score": _monotonicity_numba(traj_c).astype(np.float32),
        "efficiency":         _efficiency_numba(traj_c).astype(np.float32),
        "drift_vol":          _drift_vol_numba(traj_c).astype(np.float32),
    }


@njit
def extract_trajectory(feature_arr: np.ndarray, anchor_bars: np.ndarray,
                       window_bars: int, onset_bars: int) -> np.ndarray:
    """Extract trajectories ending at each anchor + onset.

    For anchor a_i at row i:
        end   = anchor_bars[i] + onset_bars
        start = end - window_bars + 1
        out[i, :] = feature_arr[start..end+1]   # length window_bars

    NaN where index is out of [0, n_feat).

    *** TIMING CONSTRAINT (v15_10): onset_bars must be ≤ 0 to keep trajectory
    strictly past-only relative to anchor. With onset > 0, the trajectory would
    extend into the future, leaking forward-return data. ***
    """
    n_feat = len(feature_arr)
    n_anchors = len(anchor_bars)
    out = np.full((n_anchors, window_bars), np.nan, dtype=np.float32)
    for i in range(n_anchors):
        end = anchor_bars[i] + onset_bars
        start = end - window_bars + 1
        for j in range(window_bars):
            idx = start + j
            if 0 <= idx < n_feat:
                out[i, j] = feature_arr[idx]
    return out


# ════════════════════════════════════════════════════════════════════════════
# 9. FORWARD RETURNS AND EVENT CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════

def _shift_neg(arr: np.ndarray, k: int) -> np.ndarray:
    """Shift forward by k (look ahead). Last k bars become NaN."""
    out = np.full_like(arr, np.nan, dtype=np.float64)
    out[:-k] = arr[k:]
    return out


def _forward_window_max(arr: np.ndarray, h_bars: int) -> np.ndarray:
    """Rolling max over forward window of h_bars (NOT including current bar +1)."""
    return pd.Series(arr).rolling(h_bars, min_periods=1).max().shift(-h_bars).to_numpy()


def _forward_window_min(arr: np.ndarray, h_bars: int) -> np.ndarray:
    return pd.Series(arr).rolling(h_bars, min_periods=1).min().shift(-h_bars).to_numpy()


def _forward_window_sum(arr: np.ndarray, h_bars: int) -> np.ndarray:
    return pd.Series(arr).rolling(h_bars, min_periods=1).sum().shift(-h_bars).to_numpy()


def forward_path_stats(mid: np.ndarray, h_bars: int) -> Dict[str, np.ndarray]:
    """Forward-path statistics for event detection and forward-return calculation.

    For each bar A:
        R_end  = (mid[A+h_bars] / mid[A] - 1) * 10⁴    bps point-in-time return
        R_high = (max(mid[A+1..A+h]) / mid[A] - 1) * 10⁴
        R_low  = (min(mid[A+1..A+h]) / mid[A] - 1) * 10⁴
        RV_fwd = sqrt(Σ log_ret² over forward h bars) * 10⁴
        RV_trail = sqrt(Σ log_ret² over trailing h bars) * 10⁴
    """
    n = len(mid)
    mid_fwd = _shift_neg(mid, h_bars)
    with np.errstate(divide="ignore", invalid="ignore"):
        R_end = np.where(mid > 0, (mid_fwd / mid - 1.0) * 1e4, np.nan)
    fwd_max = _forward_window_max(mid, h_bars)
    fwd_min = _forward_window_min(mid, h_bars)
    with np.errstate(divide="ignore", invalid="ignore"):
        R_high = np.where(mid > 0, (fwd_max / mid - 1.0) * 1e4, np.nan)
        R_low  = np.where(mid > 0, (fwd_min / mid - 1.0) * 1e4, np.nan)
    valid_mid = mid > 0
    log_mid = np.where(valid_mid, np.log(mid), np.nan)
    log_ret = np.diff(log_mid, prepend=log_mid[0]); log_ret[0] = 0.0
    sq = log_ret ** 2
    rv_fwd_sq = _forward_window_sum(sq, h_bars)
    rv_trail_sq = rolling_sum(sq, h_bars)
    with np.errstate(invalid="ignore"):
        RV_fwd   = np.sqrt(np.maximum(rv_fwd_sq, 0.0)) * 1e4
        RV_trail = np.sqrt(np.maximum(rv_trail_sq, 0.0)) * 1e4
    return {"R_end": R_end, "R_high": R_high, "R_low": R_low,
            "RV_fwd": RV_fwd, "RV_trail": RV_trail}


def classify_events(stats: Dict[str, np.ndarray], theta: float, rho: float,
                    in_session: np.ndarray) -> np.ndarray:
    """5-class event detection per bar.

    Classes (mutually exclusive, evaluated in order):
        1 strong_up:        R_high ≥ θ AND R_low ≥ -θ/2 AND R_end ≥ 0.5·θ
        2 strong_down:      R_low ≤ -θ AND R_high ≤ θ/2 AND R_end ≤ -0.5·θ
        3 reversion:        max(R_high, -R_low) ≥ θ AND |R_end| ≤ 0.3·θ
        4 vol_expansion:    RV_fwd / RV_trail ≥ ρ
        5 vol_contraction:  RV_fwd / RV_trail ≤ 1/ρ

    θ = 97.5th percentile of |R_end| at event-detection horizon (calibrated per inst).
    ρ = 2.0 default.

    Returns int8 array of class labels (0 = no event).
    """
    R_end   = stats["R_end"]
    R_high  = stats["R_high"]
    R_low   = stats["R_low"]
    RV_fwd  = stats["RV_fwd"]
    RV_trail= stats["RV_trail"]
    n = len(R_end)
    out = np.zeros(n, dtype=np.int8)
    elig = in_session & np.isfinite(R_end) & np.isfinite(R_high) & np.isfinite(R_low)
    m_up = elig & (R_high >= theta) & (R_low >= -theta * 0.5) & (R_end >= 0.5 * theta)
    out[m_up] = 1
    not_set = out == 0
    m_dn = not_set & elig & (R_low <= -theta) & (R_high <= theta * 0.5) & (R_end <= -0.5 * theta)
    out[m_dn] = 2
    not_set = out == 0
    excursion = np.maximum(R_high, -R_low)
    m_rv = not_set & elig & (excursion >= theta) & (np.abs(R_end) <= 0.3 * theta)
    out[m_rv] = 3
    not_set = out == 0
    elig_v = in_session & np.isfinite(RV_fwd) & np.isfinite(RV_trail) & (RV_trail > 0)
    m_ve = not_set & elig_v & (RV_fwd / np.maximum(RV_trail, 1e-12) >= rho)
    out[m_ve] = 4
    not_set = out == 0
    m_vc = not_set & elig_v & (RV_fwd / np.maximum(RV_trail, 1e-12) <= 1.0 / rho)
    out[m_vc] = 5
    return out


# ════════════════════════════════════════════════════════════════════════════
# 10. LIFT OVER NOISE (Stage 1)
# ════════════════════════════════════════════════════════════════════════════

def lift_over_noise(descriptor: np.ndarray, fwd_return: np.ndarray,
                    direction: str, tail_pct: float = 0.10,
                    min_baseline: int = 500, min_trig: int = 50,
                    ) -> Optional[Dict[str, float]]:
    """Single-feature lift over noise.

    For one feature at one horizon, on a pool of bars where both descriptor
    and forward return are defined:

        top_thr = (1 - tail_pct) quantile of descriptor
        bot_thr =       tail_pct quantile of descriptor

        if direction == "long":
            trigger = descriptor ≥ top_thr
        else:
            trigger = descriptor ≤ bot_thr

        baseline_mean = mean(fwd_return)            over pool
        baseline_std  = std(fwd_return, ddof=1)     over pool
        trig_mean     = mean(fwd_return[trigger])

        lift_std_units = (trig_mean - baseline_mean) / baseline_std

    Returns None if min_baseline or min_trig not met.

    Interpretation: lift = +0.10 means triggered forward returns average 0.10
    standard deviations above the unconditional mean. NOT a t-statistic — adjacent
    baseline bars have overlapping forward windows.
    """
    finite = np.isfinite(descriptor) & np.isfinite(fwd_return)
    if int(finite.sum()) < min_baseline:
        return None
    d_g = descriptor[finite]
    f_g = fwd_return[finite]
    base_mean = float(f_g.mean())
    base_std = float(f_g.std(ddof=1))
    if base_std <= 0:
        return None
    top_thr, bot_thr = np.quantile(d_g, [1 - tail_pct, tail_pct])
    if direction == "long":
        trig_mask = descriptor >= top_thr
    else:
        trig_mask = descriptor <= bot_thr
    trig_mask = trig_mask & finite
    n_trig = int(trig_mask.sum())
    if n_trig < min_trig:
        return None
    trig_mean = float(fwd_return[trig_mask].mean())
    lift = (trig_mean - base_mean) / base_std
    implied = +1 if direction == "long" else -1
    hit_rate = float((np.sign(fwd_return[trig_mask]) == implied).mean())
    return {
        "n_baseline":     int(finite.sum()),
        "n_triggers":     n_trig,
        "trig_pct":       100.0 * n_trig / int(finite.sum()),
        "baseline_mean":  base_mean,
        "baseline_std":   base_std,
        "trigger_mean":   trig_mean,
        "lift_std_units": lift,
        "hit_rate_sign":  hit_rate,
    }


# ════════════════════════════════════════════════════════════════════════════
# 11. DAY-BLOCK BOOTSTRAP CI (Stage 2 / Stage 3, C3 fix)
# ════════════════════════════════════════════════════════════════════════════

def bootstrap_lift_ci(triggered_fwd: np.ndarray, triggered_day: np.ndarray,
                       base_mean: float, base_std: float,
                       train_day_indices: np.ndarray,
                       n_boot: int = 200, ci_lo: float = 2.5, ci_hi: float = 97.5,
                       seed: int = 42) -> Tuple[float, float, float]:
    """Day-block bootstrap 95% CI on lift.

    Preserves intra-day autocorrelation (resamples DAYS with replacement; within each
    sampled day, takes ALL triggered bars for that day).

    Algorithm:
        Pre-bucket:
            by_day[d] = triggered_fwd[triggered_day == d]

        For b = 1..n_boot:
            chosen_days = random.choice(train_day_indices, size=n_train_days, replace=True)
            boot_fwd = concatenate([by_day[d] for d in chosen_days if d in by_day])
            lift_b = (mean(boot_fwd) - base_mean) / base_std

        Return (percentile(lifts, ci_lo), median, percentile(lifts, ci_hi))

    Returns (NaN, NaN, NaN) if insufficient triggered data or base_std ≤ 0.
    """
    if triggered_fwd.size == 0 or base_std <= 0:
        return float("nan"), float("nan"), float("nan")
    train_idx_arr = np.asarray(train_day_indices, dtype=np.int32)
    n_train = train_idx_arr.size
    by_day = {}
    for di in train_idx_arr:
        sel = (triggered_day == di)
        if sel.any():
            by_day[int(di)] = triggered_fwd[sel]
    if not by_day:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    lifts = np.full(n_boot, np.nan)
    for b in range(n_boot):
        chosen = rng.choice(train_idx_arr, n_train, replace=True)
        pieces = [by_day[int(di)] for di in chosen if int(di) in by_day]
        if not pieces:
            continue
        boot_fwd = np.concatenate(pieces)
        if boot_fwd.size > 0:
            lifts[b] = (boot_fwd.mean() - base_mean) / base_std
    finite_lifts = lifts[np.isfinite(lifts)]
    if finite_lifts.size < 10:
        return float("nan"), float("nan"), float("nan")
    return (float(np.percentile(finite_lifts, ci_lo)),
            float(np.percentile(finite_lifts, 50)),
            float(np.percentile(finite_lifts, ci_hi)))


# ════════════════════════════════════════════════════════════════════════════
# 12. STABILITY CHECKS A, B, C, D, F (Cell 14L)
# ════════════════════════════════════════════════════════════════════════════

# Default thresholds. Override per call site.
STAB_C_MIN_DAY_FRAC     = 0.60
STAB_D_MAX_LIFT_DROP    = 0.50
STAB_F_MIN_TRIG_PER_DAY = 3
STAB_B_MIN_FOLD_DAYS    = 6
STAB_B_MAG_RATIO_MIN    = 0.40


def stability_checks(per_day_n_trig: np.ndarray,
                     per_day_trig_sum: np.ndarray,
                     base_mean: float, base_std: float,
                     aggregate_lift: float,
                     n_train_days: int,
                     c_min_frac: float = STAB_C_MIN_DAY_FRAC,
                     d_max_drop: float = STAB_D_MAX_LIFT_DROP,
                     f_min_trig: int = STAB_F_MIN_TRIG_PER_DAY,
                     b_min_days: int = STAB_B_MIN_FOLD_DAYS,
                     b_mag_min: float = STAB_B_MAG_RATIO_MIN
                     ) -> Dict[str, float]:
    """All five stability checks for a single (feature, horizon, direction) row.

    Inputs are per-day aggregates over the fold's training days, in training-day order.

    A — Sign across halves:
        Split training days in half. Compute aggregate per-half lift relative to
        full-fold (base_mean, base_std). Require sign(half1_lift) == sign(half2_lift).

    B — Half magnitude non-collapse (only when n_train_days ≥ b_min_days):
        |half2_lift| / |half1_lift| ≥ b_mag_min, AND same sign as A.
        For folds with < b_min_days training days, returns pass_B = True automatically.

    C — Day-by-day sign consistency:
        For each training day with ≥1 trigger:
            day_lift = (per_day_trig_sum[d] / per_day_n_trig[d] - base_mean) / base_std
        Fraction of days-with-triggers where sign(day_lift) == sign(aggregate_lift) ≥ c_min_frac.

    D — Jackknife (baseline held constant):
        For each training day d with triggers:
            n_excl = total_n - per_day_n_trig[d]
            sum_excl = total_sum - per_day_trig_sum[d]
            lift_excl = (sum_excl/n_excl - base_mean) / base_std
            drop_d = 1 - |lift_excl| / |aggregate_lift|
        Require max(drop_d) ≤ d_max_drop.

    F — Coverage:
        min(per_day_n_trig) ≥ f_min_trig. No dead days allowed.

    Returns dict with pass_A, pass_B, pass_C, pass_D, pass_F, pass_all, and numeric
    diagnostics.
    """
    out = {"pass_A": False, "pass_B": True, "pass_C": False, "pass_D": False,
           "pass_F": False, "pass_all": False,
           "stab_half1_lift": float("nan"), "stab_half2_lift": float("nan"),
           "stab_magnitude_ratio": float("nan"),
           "stab_frac_days_correct": float("nan"),
           "stab_max_jackknife_drop": float("nan")}
    if base_std <= 0:
        return out
    sign_agg = 1 if aggregate_lift >= 0 else -1

    # F
    out["pass_F"] = bool(np.all(per_day_n_trig >= f_min_trig))

    # per-day signed lifts
    day_lifts = np.full(n_train_days, np.nan, dtype=np.float64)
    for i in range(n_train_days):
        if per_day_n_trig[i] > 0:
            day_lifts[i] = (per_day_trig_sum[i] / per_day_n_trig[i] - base_mean) / base_std
    days_with_any = per_day_n_trig > 0

    # C
    if days_with_any.any():
        valid_lifts = day_lifts[days_with_any]
        frac = float((np.sign(valid_lifts) == sign_agg).mean())
        out["stab_frac_days_correct"] = frac
        out["pass_C"] = frac >= c_min_frac

    # A / B (half-split)
    mid = n_train_days // 2
    if mid >= 1 and (n_train_days - mid) >= 1:
        h1_n = int(per_day_n_trig[:mid].sum()); h1_s = float(per_day_trig_sum[:mid].sum())
        h2_n = int(per_day_n_trig[mid:].sum()); h2_s = float(per_day_trig_sum[mid:].sum())
        if h1_n > 0 and h2_n > 0:
            h1_lift = (h1_s / h1_n - base_mean) / base_std
            h2_lift = (h2_s / h2_n - base_mean) / base_std
            out["stab_half1_lift"] = float(h1_lift)
            out["stab_half2_lift"] = float(h2_lift)
            out["pass_A"] = (np.sign(h1_lift) == np.sign(h2_lift))
            if n_train_days >= b_min_days:
                if abs(h1_lift) > 1e-9 and out["pass_A"]:
                    mag = abs(h2_lift) / abs(h1_lift)
                    out["stab_magnitude_ratio"] = float(mag)
                    out["pass_B"] = mag >= b_mag_min
                else:
                    out["pass_B"] = False

    # D — jackknife
    if abs(aggregate_lift) > 1e-9:
        total_n = int(per_day_n_trig.sum())
        total_s = float(per_day_trig_sum.sum())
        max_drop = 0.0
        for d in range(n_train_days):
            if per_day_n_trig[d] == 0:
                continue
            n_excl = total_n - int(per_day_n_trig[d])
            s_excl = total_s - float(per_day_trig_sum[d])
            if n_excl <= 0:
                continue
            mean_excl = s_excl / n_excl
            lift_excl = (mean_excl - base_mean) / base_std
            drop = 1.0 - abs(lift_excl) / abs(aggregate_lift)
            if drop > max_drop:
                max_drop = drop
        out["stab_max_jackknife_drop"] = float(max_drop)
        out["pass_D"] = max_drop <= d_max_drop

    out["pass_all"] = bool(out["pass_A"] and out["pass_B"] and out["pass_C"]
                            and out["pass_D"] and out["pass_F"])
    return out


# ════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — run this file directly to verify each formula
# ════════════════════════════════════════════════════════════════════════════

def _run_tests():
    """Sanity checks. Synthetic inputs with known expected outputs."""
    np.random.seed(0)
    n_pass = 0; n_fail = 0

    def check(name, cond):
        nonlocal n_pass, n_fail
        if cond:
            print(f"  PASS  {name}")
            n_pass += 1
        else:
            print(f"  FAIL  {name}")
            n_fail += 1

    # --- Group A ---
    print("\n[Group A]")
    bid0 = np.array([99.50, 99.50, 99.49]); ask0 = np.array([99.51, 99.52, 99.51])
    bs   = np.array([100,   50,    200]);    a_sz = np.array([100,   200,   50])
    spr = spread_bps(bid0, ask0)
    expected = (np.array([0.01, 0.02, 0.02]) / np.array([99.505, 99.51, 99.50])) * 1e4
    check("spread_bps within 0.1bps", np.allclose(spr, expected, atol=0.1))
    mp = microprice(bid0, ask0, bs, a_sz)
    check("microprice bounded [bid, ask]", np.all((mp >= bid0) & (mp <= ask0)))
    obi = obi_level(bs, a_sz)
    check("obi at imbalance 100/100 is 0", abs(obi[0]) < 1e-6)
    check("obi at 50/200 is -0.6", abs(obi[1] - (-0.6)) < 1e-6)

    # --- EWMA z-score ---
    print("\n[EWMA z (H1)]")
    x_norm = np.random.randn(10000).astype(np.float32)
    z, _ = ewma_zscore(x_norm, halflife_bars=1000)
    valid = np.isfinite(z[5000:])
    check("z mean ~0 in tail", abs(z[5000:][valid].mean()) < 0.3)
    check("z std ~1 in tail",  abs(z[5000:][valid].std() - 1.0) < 0.5)

    # --- Signed z (H2 fix) ---
    print("\n[Signed-z H2 fix]")
    x_signed = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0] * 100, dtype=np.float32)
    z_out, _ = z_feature_sweep(x_signed, sign_preserve=True, log1p_flag=False,
                                halflives_bars=[100.0], ewma_states={})
    z_signed = z_out[100.0]
    valid = np.isfinite(z_signed[200:])
    check("z of bipolar series has both signs", (z_signed[200:][valid] > 0).any()
                                              and (z_signed[200:][valid] < 0).any())

    # --- Trajectory descriptors ---
    print("\n[Trajectory descriptors]")
    # Constant trajectory
    const_traj = np.ones((1, 50), dtype=np.float32)
    d = compute_descriptors(const_traj)
    check("slope of constant traj is 0",       abs(d["slope"][0]) < 1e-6)
    check("net_change of constant is 0",       abs(d["net_change"][0]) < 1e-6)
    check("monotonicity of constant is 0",     abs(d["monotonicity_score"][0]) < 1e-6)
    check("efficiency of constant is NaN",     np.isnan(d["efficiency"][0]))
    check("drift_vol of constant is NaN",      np.isnan(d["drift_vol"][0]))

    # Monotonic rise
    rise_traj = np.arange(50, dtype=np.float32).reshape(1, 50)
    d = compute_descriptors(rise_traj)
    check("slope of linear rise > 0",          d["slope"][0] > 0)
    check("net_change of rise == 49",          abs(d["net_change"][0] - 49.0) < 1e-3)
    check("monotonicity of rise == 1.0",       abs(d["monotonicity_score"][0] - 1.0) < 1e-6)
    check("efficiency of rise == 1.0",         abs(d["efficiency"][0] - 1.0) < 1e-6)

    # Monotonic fall
    fall_traj = (50 - np.arange(50, dtype=np.float32)).reshape(1, 50)
    d = compute_descriptors(fall_traj)
    check("monotonicity of fall ~ 0",          d["monotonicity_score"][0] < 0.05)

    # Curvature on a parabola
    x = np.arange(50, dtype=np.float32)
    parab = (0.5 * x**2).reshape(1, 50)
    d = compute_descriptors(parab)
    check("curvature of parabola > 0",         d["curvature"][0] > 0)

    # --- Trajectory extraction (no look-ahead constraint) ---
    print("\n[Trajectory extraction]")
    feat = np.arange(100, dtype=np.float32)
    anchors = np.array([50], dtype=np.int64)
    # Onset = 0 means trajectory ends AT anchor
    traj = extract_trajectory(feat, anchors, window_bars=10, onset_bars=0)
    check("onset=0: last value of traj is feat[anchor]", traj[0, -1] == feat[50])
    check("onset=0: first value is feat[anchor - 9]",    traj[0, 0]  == feat[41])
    # Onset = -10 means trajectory ends 10 bars before anchor
    traj_pre = extract_trajectory(feat, anchors, window_bars=10, onset_bars=-10)
    check("onset=-10: last value is feat[anchor - 10]",  traj_pre[0, -1] == feat[40])

    # --- Lift over noise ---
    print("\n[Lift over noise]")
    rng = np.random.default_rng(0)
    n = 20000
    desc = rng.standard_normal(n).astype(np.float32)
    fwd  = (0.10 * desc + rng.standard_normal(n) * 0.5).astype(np.float32)
    res = lift_over_noise(desc, fwd, direction="long", tail_pct=0.10)
    check("long lift positive (desc + corr fwd)", res is not None and res["lift_std_units"] > 0)
    res2 = lift_over_noise(desc, fwd, direction="short", tail_pct=0.10)
    check("short lift negative (desc + corr fwd)", res2 is not None and res2["lift_std_units"] < 0)

    # --- Day-block bootstrap CI ---
    print("\n[Day-block bootstrap CI]")
    train_days = np.array([0, 1, 2, 3, 4])
    trig_fwd = np.array([1.0]*20 + [2.0]*20 + [1.5]*20 + [1.0]*20 + [1.0]*20)
    trig_day = np.array([0]*20 + [1]*20 + [2]*20 + [3]*20 + [4]*20)
    lo, med, hi = bootstrap_lift_ci(trig_fwd, trig_day, base_mean=0.0, base_std=1.0,
                                      train_day_indices=train_days, n_boot=200)
    check("bootstrap CI finite", np.isfinite(lo) and np.isfinite(med) and np.isfinite(hi))
    check("CI brackets median",   lo <= med <= hi)
    check("CI positive (positive trig_fwd)", lo > 0)

    # --- Stability checks ---
    print("\n[Stability checks]")
    # Synthetic case: 8 training days, half/half, all positive lifts, no dead days
    n_train = 8
    per_day_n = np.array([10]*n_train)
    per_day_sum = np.array([5.0]*n_train)  # mean 0.5 per trigger, half_lift > 0
    out = stability_checks(per_day_n, per_day_sum, base_mean=0.0, base_std=1.0,
                            aggregate_lift=0.5, n_train_days=n_train)
    check("A passes (same sign halves)",   out["pass_A"])
    check("B passes (mag ratio 1.0)",      out["pass_B"])
    check("C passes (all days correct)",   out["pass_C"])
    check("D passes (jackknife stable)",   out["pass_D"])
    check("F passes (all days ≥ 3 trig)",  out["pass_F"])
    check("pass_all",                       out["pass_all"])
    # Failure case: half-flip sign
    per_day_sum_flip = np.array([5.0, 5.0, 5.0, 5.0, -5.0, -5.0, -5.0, -5.0])
    out = stability_checks(per_day_n, per_day_sum_flip, base_mean=0.0, base_std=1.0,
                            aggregate_lift=0.0, n_train_days=n_train)
    check("A fails on sign flip",          not out["pass_A"])
    # Failure case: dead day
    per_day_n_dead = np.array([10, 10, 10, 0, 10, 10, 10, 10])
    out = stability_checks(per_day_n_dead, np.array([5.0]*n_train),
                            base_mean=0.0, base_std=1.0,
                            aggregate_lift=0.5, n_train_days=n_train)
    check("F fails on dead day",           not out["pass_F"])

    print(f"\n{n_pass} passed, {n_fail} failed")
    return n_fail == 0


if __name__ == "__main__":
    ok = _run_tests()
    if not ok:
        exit(1)
