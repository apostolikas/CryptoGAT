import argparse
import glob
import os
import re
from datetime import timedelta
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, mean_squared_error
from tqdm import tqdm

from data_processor import load_equity_data, process_raw_data, engineer_targets, list_days
from features import extract_features
from graph_builder import compute_rich_dynamic_edges, build_static_masks, NUM_RELATIONS, EDGE_DIM
from model import RGATModel
from backtest import run_backtests

# Extra per-node channels appended to the base features:
#   [stale_flag, tod_sin, tod_cos, frac_since_open, frac_to_close, auction_flag]
# (age_ms was dropped: equities and macros share an aligned 1s grid, so the
#  as-of age was always 0 -> a dead, all-zero channel.)
EXTRA_DIM = 6

# Sector partition for the working universe (market/index node = QQQ).
TICKER_SECTORS = {
    "AVGO": "SEMI", "AMD": "SEMI", "NVDA": "SEMI", "QCOM": "SEMI",
    "MRVL": "SEMI", "MU": "SEMI", "INTC": "SEMI", "LITE": "SEMI",
    "AAPL": "MEGA", "MSFT": "MEGA", "META": "MEGA", "GOOGL": "MEGA", "AMZN": "MEGA",
    "PLTR": "SOFT", "MSTR": "SOFT",
    "PAYP": "FINCRYPTO", "HOOD": "FINCRYPTO", "CRCL": "FINCRYPTO", "COIN": "FINCRYPTO",
    "TSLA": "AUTO", "CSCO": "NET", "USAR": "MAT", "QQQ": "INDEX",
}
MARKET_TICKER = "QQQ"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def newey_west_tstat(x: np.ndarray, lag: int) -> float:
    """t-stat of mean(x) with a Newey-West HAC variance to account for the
    serial correlation induced by overlapping (rolling) observations."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 3:
        return 0.0
    e = x - x.mean()
    s = (e @ e) / n
    L = int(min(max(lag, 0), n - 1))
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        s += 2.0 * w * (e[l:] @ e[:-l]) / n
    se = np.sqrt(max(s, 1e-18) / n)
    return float(x.mean() / (se + 1e-18))


def cross_sectional_ic(per_window_pred, per_window_true, min_names=5, overlap_lag=1):
    """Cross-sectional rank IC: at each timestamp, Spearman-correlate the
    model's ranking of the names against the realized-return ranking, then
    average over time. This is the metric a cross-sectional long/short book
    actually monetizes, and it separates 'which name' from 'which moment'.

    Returns (mean_ic, ic_tstat, n_used). The t-stat uses a Newey-West HAC
    variance with lag=`overlap_lag` (the horizon in seconds) because the
    per-timestamp ICs are hugely autocorrelated under overlapping targets --
    a naive mean/std*sqrt(n) would massively overstate significance.
    """
    ics = []
    n_seen = 0
    for p, t in zip(per_window_pred, per_window_true):
        if p.shape[0] < min_names:
            continue
        n_seen += 1
        # Skip windows where either side is constant or non-finite: Spearman is
        # undefined there (this is the ConstantInputWarning source). A constant
        # PREDICTION vector means the model emitted the same value for every name
        # at that instant -- watch n_used vs n_seen below; if many are skipped the
        # model is collapsing, not just hitting flat-market timestamps.
        if (np.ptp(p) == 0 or np.ptp(t) == 0
                or not np.isfinite(p).all() or not np.isfinite(t).all()):
            continue
        rho, _ = spearmanr(p, t)
        if np.isfinite(rho):
            ics.append(rho)
    if len(ics) < 3:
        return 0.0, 0.0, len(ics)
    ics = np.asarray(ics)
    return float(ics.mean()), newey_west_tstat(ics, overlap_lag), len(ics)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, overlap_lag: int = 1):
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_corr, p_pearson = pearsonr(y_pred, y_true)
        spearman_corr, p_spearman = spearmanr(y_pred, y_true)
    else:
        pearson_corr, p_pearson = 0.0, 1.0
        spearman_corr, p_spearman = 0.0, 1.0

    y_true_bin = (y_true > 0).astype(int)
    auc = roc_auc_score(y_true_bin, y_pred) if len(np.unique(y_true_bin)) > 1 else 0.5
    mse = mean_squared_error(y_true, y_pred)

    # Hit rate excluding exactly-zero true returns: sign(0) never matches a
    # nonzero prediction, so at short horizons (many flat 1s moves) zeros would
    # mechanically drag hit rate well below 50%. We score direction only where
    # there is a direction to get right.
    nz = y_true != 0
    if nz.sum() > 2:
        hits = (np.sign(y_pred[nz]) == np.sign(y_true[nz])).astype(float)
        hit_rate = float(np.mean(hits))
        t_hit = newey_west_tstat(hits - 0.5, overlap_lag)
    else:
        hit_rate, t_hit = 0.5, 0.0

    # FIX #7: strategy returns are heavily overlapping, so the previous
    # sqrt(252*23400) annualization wildly overstated the IR/t-stat. Report a
    # per-observation information ratio plus a HAC t-stat that deflates for the
    # overlap instead of pretending every second is an independent bet.
    strat = np.sign(y_pred) * y_true
    std_ret = np.std(strat)
    ir = float(np.mean(strat) / std_ret) if std_ret > 0 else 0.0
    ir_t = newey_west_tstat(strat, overlap_lag)

    return {
        'Pearson': (pearson_corr, p_pearson),
        'Spearman': (spearman_corr, p_spearman),
        'AUC': auc, 'MSE': mse,
        'Hit_Rate': (hit_rate, t_hit),
        'IR_per_obs': ir, 'IR_HAC_tstat': ir_t,
    }


# ---------------------------------------------------------------------------
# Dense split preparation (fixes missing-asset dropouts and seq discontinuity)
# ---------------------------------------------------------------------------
def _ffill_2d(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs down axis 0 (vectorized)."""
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(arr.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = np.take_along_axis(np.nan_to_num(arr, nan=0.0), idx, axis=0)
    # rows before the first valid value stay 0 (already nan->0)
    first_valid = (~mask).any(axis=0)
    out[:, ~first_valid] = 0.0
    return out


def _seasonality(ts_series: pl.Series) -> np.ndarray:
    """(T, 5): [tod_sin, tod_cos, frac_since_open, frac_to_close, auction_flag].
    Computed from TIME-OF-DAY so it resets every session. The previous version
    used elapsed-seconds over the whole span, which across multiple days encoded
    'which day' instead of 'what time of day' -- useless as a seasonality signal."""
    hh = ts_series.dt.hour().to_numpy().astype(np.float64)
    mm = ts_series.dt.minute().to_numpy().astype(np.float64)
    ss = ts_series.dt.second().to_numpy().astype(np.float64)
    tod = hh * 3600.0 + mm * 60.0 + ss                                   # seconds into the day
    lo, hi = float(tod.min()), float(tod.max())                          # ~session bounds (same each day)
    span = max(hi - lo, 1.0)
    frac = np.clip((tod - lo) / span, 0.0, 1.0)
    sin = np.sin(2 * np.pi * frac)
    cos = np.cos(2 * np.pi * frac)
    auction = (((tod - lo) < 300) | ((hi - tod) < 300)).astype(np.float64)
    return np.stack([sin, cos, frac, 1.0 - frac, auction], axis=1)


def prepare_split(df: pl.DataFrame, base_cols, target_cols, assets, macro_keys, target_std=None, feat_stats=None):
    n_eq = len(assets)
    n_mac = len(macro_keys)
    N = n_eq + n_mac
    F = len(base_cols)
    nt = len(target_cols)

    ts_df = df.select("ts_event").unique().sort("ts_event")
    ts_series = ts_df["ts_event"]
    ts_list = ts_series.to_list()
    T = len(ts_list)
    # run-length day id (timestamps are sorted, so this increments at each new
    # session) -> used to keep TCN windows and correlation windows day-local.
    day_code = ts_df.select(pl.col("ts_event").dt.date()).to_series().rle_id().to_numpy()
    ts_index = ts_df.with_row_index("ti")
    sym_index = pl.DataFrame({"symbol": assets}).with_row_index("si")

    # ----- equity base features, returns, targets --------------------------
    dfe = (df.select(["ts_event", "symbol", "ret_bps"] + base_cols + target_cols)
             .join(ts_index, on="ts_event").join(sym_index, on="symbol"))
    ti = dfe["ti"].to_numpy()
    si = dfe["si"].to_numpy()

    X_eq = np.full((T, n_eq, F), np.nan, dtype=np.float64)
    X_eq[ti, si] = dfe.select(base_cols).to_numpy()
    present = np.zeros((T, n_eq), dtype=bool)
    present[ti, si] = True

    R = np.zeros((T, N), dtype=np.float64)
    R[ti, si] = dfe["ret_bps"].to_numpy()

    Y_raw = np.full((T, n_eq, nt), np.nan, dtype=np.float64)
    Y_raw[ti, si] = dfe.select(target_cols).to_numpy()
    target_valid = present & np.isfinite(Y_raw).all(axis=2)

    # forward-fill features down time per (asset, feature); track staleness
    X_eq = _ffill_2d(X_eq.reshape(T, n_eq * F)).reshape(T, n_eq, F)
    stale_eq = (~present).astype(np.float64)

    # ----- macro nodes (one row per ts; already as-of joined upstream) ------
    X_mac = np.zeros((T, n_mac, F), dtype=np.float64)
    age_mac = np.zeros((T, n_mac), dtype=np.float64)
    stale_mac = np.zeros((T, n_mac), dtype=np.float64)
    if n_mac:
        dfm = df.group_by("ts_event").first().join(ts_index, on="ts_event").sort("ti")
        mti = dfm["ti"].to_numpy()
        for k, p in enumerate(macro_keys):
            mac_cols = [f"{p}_{c}" for c in base_cols]
            X_mac[mti, k, :] = dfm.select(mac_cols).to_numpy()
            R[mti, n_eq + k] = dfm[f"{p}_ret_bps"].to_numpy()
            age_mac[mti, k] = dfm[f"{p}_age_ms"].to_numpy()
            stale_mac[mti, k] = dfm[f"{p}_stale_flag"].to_numpy()

    # ----- assemble (T, N, F+EXTRA) ----------------------------------------
    seas = _seasonality(ts_series)                     # (T, 5)
    stale_all = np.concatenate([stale_eq, stale_mac], axis=1)
    base_all = np.concatenate([X_eq, X_mac], axis=1)   # (T, N, F)  -- RAW

    # Graph node modifiers must use the RAW (pre-standardization) book/spread
    # values so the log-ratios in the edge builder are meaningful.
    idx = {c: i for i, c in enumerate(base_cols)}
    res_name = next((c for c in base_cols if "kalman_z_score" in c), None)
    spread_all = base_all[:, :, idx["spread_bps"]] if "spread_bps" in idx else np.ones((T, N))
    liq_all = base_all[:, :, idx["top5_book_size"]] if "top5_book_size" in idx else np.ones((T, N))
    res_all = base_all[:, :, idx[res_name]] if res_name else np.zeros((T, N))

    # ----- global per-channel feature standardization (leakage-safe) -------
    # Computed on TRAIN only and reused on test (like target_std). This is the
    # backstop that prevents any single un-bounded channel from dominating the
    # input scale (the std=7541 problem).
    if feat_stats is None:
        flat = base_all.reshape(-1, F)
        fm = np.nanmean(flat, axis=0)
        fs = np.nanstd(flat, axis=0)
        fs = np.where(fs > 1e-8, fs, 1.0)
        feat_stats = (fm, fs)
    fm, fs = feat_stats
    base_std = np.clip((base_all - fm) / fs, -10.0, 10.0)

    extra = np.empty((T, N, EXTRA_DIM), dtype=np.float64)
    extra[:, :, 0] = stale_all
    extra[:, :, 1:6] = seas[:, None, :]                # broadcast seasonality to all nodes

    X_all = np.concatenate([base_std, extra], axis=2).astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    # ----- per-horizon target scale normalization (FIX #2) -----------------
    if target_std is None:
        flat = Y_raw[target_valid]                      # (M, nt)
        target_std = np.std(flat, axis=0)
        target_std = np.where(target_std > 0, target_std, 1.0)
    Y_raw_filled = np.nan_to_num(Y_raw, nan=0.0)
    # clip in normalized units to tame the fat tails before the loss sees them
    Y_norm = np.clip(Y_raw_filled / target_std, -8.0, 8.0).astype(np.float32)

    return {
        "X_all": X_all, "R": R, "Y_norm": Y_norm, "Y_raw": Y_raw_filled,
        "target_valid": target_valid, "ts_list": ts_list, "T": T, "N": N, "n_eq": n_eq,
        "spread_all": np.nan_to_num(spread_all), "liq_all": np.nan_to_num(liq_all),
        "res_all": np.nan_to_num(res_all), "target_std": target_std, "feat_stats": feat_stats,
        "day_code": day_code,
    }


def precompute_graphs(split, seq_len, sector_mask, market_mask, threshold, top_k, ablate=False):
    """Build (and cache) one graph per valid window end. Done once per split,
    not once per epoch (FIX: Perf A).

    ablate=True replaces every dynamic graph with self-loops only -- an A/B test
    of whether the cross-asset edges are contributing any alpha at all. If IC is
    unchanged with the graph ablated, the GNN's expensive relational machinery
    is dead weight for your data and a per-node model would do just as well.
    """
    R = split["R"]; T = split["T"]; n_eq = split["n_eq"]; Nn = split["N"]
    spread_all, liq_all, res_all = split["spread_all"], split["liq_all"], split["res_all"]
    valid = split["target_valid"]

    # Day-local windowing: the dense grid concatenates multiple sessions, so a
    # window or correlation lookback must never straddle the overnight gap.
    day_code = split.get("day_code", np.zeros(T, dtype=np.int64))
    bnds = np.r_[0, np.flatnonzero(np.diff(day_code)) + 1]
    day_start = bnds[np.searchsorted(bnds, np.arange(T), side="right") - 1]

    self_ei = torch.arange(Nn).repeat(2, 1)
    self_ea = torch.zeros(Nn, EDGE_DIM, dtype=torch.float32)
    self_et = torch.zeros(Nn, dtype=torch.long)

    graphs = {}
    windows = []
    for w in range(0, T - seq_len + 1):
        te = w + seq_len - 1
        if not valid[te].any():
            continue
        if w < day_start[te]:          # window would cross a day boundary -> skip
            continue
        if ablate:
            graphs[w] = (self_ei, self_ea, self_et)
            windows.append(w)
            continue
        ds = day_start[te]
        r5 = R[max(ds, te - 300):te + 1]
        r30 = R[max(ds, te - 1800):te + 1]
        ei, ea, et = compute_rich_dynamic_edges(
            returns_5m=r5, returns_30m=r30,
            spread_vec=spread_all[te], liq_vec=liq_all[te], res_vec=res_all[te],
            num_equities=n_eq, sector_mask=sector_mask, market_mask=market_mask,
            threshold=threshold, top_k=top_k,
        )
        graphs[w] = (ei, ea, et)
        windows.append(w)
    return graphs, windows


# ---------------------------------------------------------------------------
# Batched forward (mini-batching + GPU + AMP: Perf E)
# ---------------------------------------------------------------------------
def run_batch(model, X_all_t, Y_t, valid_t, graphs, batch_windows, seq_len, N, n_eq, device):
    xs, eis, eas, ets, ys, ms = [], [], [], [], [], []
    for bi, w in enumerate(batch_windows):
        te = w + seq_len - 1
        xs.append(X_all_t[te - seq_len + 1:te + 1].permute(1, 0, 2))   # (N, seq, F)
        ei, ea, et = graphs[w]
        eis.append(ei + bi * N)
        eas.append(ea)
        ets.append(et)
        ys.append(Y_t[te])
        ms.append(valid_t[te])
    x = torch.cat(xs, 0)
    edge_index = torch.cat(eis, 1).to(device)
    edge_attr = torch.cat(eas, 0).to(device)
    edge_type = torch.cat(ets, 0).to(device)
    out = model(x, edge_index, edge_type, edge_attr=edge_attr)        # (B*N, nt)
    out_eq = out.view(len(batch_windows), N, -1)[:, :n_eq, :]         # (B, n_eq, nt)
    y = torch.stack(ys, 0)                                            # (B, n_eq, nt)
    m = torch.stack(ms, 0)                                            # (B, n_eq)
    return out_eq, y, m


# ---------------------------------------------------------------------------
# Shared data pipeline (used by both the GNN and the LGBM baseline)
# ---------------------------------------------------------------------------
# Whole-month globs (the YYYYMMDD is a wildcard so list_days can stream day by
# day). Override the equity glob with --equity_glob.
EQUITY_PATTERN = "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/XNAS.ITCH/*/2026/xnas-itch-202605*.mbp-10.*.parquet"
MACRO_CONFIGS = {
    "ES": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/ES.n.0/2026/glbx-mdp3-202605*.mbp-10.ES.n.0.parquet",
    "NQ": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/NQ.n.0/2026/glbx-mdp3-202605*.mbp-10.NQ.n.0.parquet",
    "CL": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/CL.n.0/2026/glbx-mdp3-202605*.mbp-10.CL.n.0.parquet",
    "BZ": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/BZ.n.0/2026/glbx-mdp3-202605*.mbp-10.BZ.n.0.parquet",
    "NG": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/NG.n.0/2026/glbx-mdp3-202605*.mbp-10.NG.n.0.parquet",
    "GC": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/GC.n.0/2026/glbx-mdp3-202605*.mbp-10.GC.n.0.parquet",
    "SI": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/SI.n.0/2026/glbx-mdp3-202605*.mbp-10.SI.n.0.parquet",
    "HG": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/HG.n.0/2026/glbx-mdp3-202605*.mbp-10.HG.n.0.parquet",
}
TARGET_COLS = ["ret_0_5s", "ret_5_10s", "ret_10_30s", "ret_30_60s", "ret_60_300s", "ret_300_600s", "ret_600_900s"]
HORIZON_END_S = [5, 10, 30, 60, 300, 600, 900]


def _files_for_days(pattern, days):
    """Filter a glob to only files whose YYYYMMDD is in `days` (None = all)."""
    files = glob.glob(pattern)
    if days is None:
        return files
    dayset = set(days)
    out = []
    for f in files:
        m = re.search(r"(20\d{6})", os.path.basename(f))
        if m and m.group(1) in dayset:
            out.append(f)
    return out


def _select_base_feature_cols(all_cols):
    """Rule-based selection of the model's base (EQUITY) features from a frame's
    columns. Factored out so the streaming cache can recompute it from a cached
    schema without re-running feature engineering.

    IMPORTANT: exclude macro-prefixed columns (ES_*, NQ_*, ...). The cached
    per-day parquet is POST macro-join, so without this guard the name rules
    below also match e.g. 'ES_rvol_60s_z180' -- inflating the LGBM baseline from
    127 to ~967 features (it must stay equity-only) and corrupting the GNN, whose
    macro nodes are built as '{prefix}_{base_col}' from these equity names."""
    macro_prefixes = tuple(p + "_" for p in MACRO_CONFIGS)      # ('ES_', 'NQ_', ...)
    cols = [c for c in all_cols if not c.startswith(macro_prefixes)]
    return [
        col for col in cols
        if col.endswith("_z180") or col.endswith("_z600") or col in [
            "microprice_dev_bps", "obi_L1", "obi_L5", "depth_entropy_bid", "depth_entropy_ask",
            "distance_weighted_imbalance", "cancel_burst_bid", "cancel_burst_ask", "large_trade_flag",
            "spread_bps", "top5_book_size", "vpin_30s", "vpin_60s", "vpin_300s", "jump_ratio", "mid_dev_z300",
        ] or col.startswith("rank_") or col.startswith("kalman_z_score_")
    ]


def _tod_seconds_expr():
    """Seconds-into-day (UTC) as Int32 to avoid Int8 overflow on hour*3600."""
    return (pl.col("ts_event").dt.hour().cast(pl.Int32) * 3600
            + pl.col("ts_event").dt.minute().cast(pl.Int32) * 60
            + pl.col("ts_event").dt.second().cast(pl.Int32))


def _hhmm_to_seconds(s):
    h, m = s.split(":")
    return int(h) * 3600 + int(m) * 60


def build_dataset(args, days=None):
    """Shared pipeline for BOTH models. With `days` set, only files for those
    dates are loaded -- the unit the streaming cache processes one block at a
    time. Order matters for extended-hours data:

      load -> features (over the FULL extended session, so EWMA/rolling windows
      are warm) -> per-(symbol,day) warm-up drop -> SESSION FILTER (keep only the
      tradeable window, RTH by default) -> targets (forward returns stay inside
      the session) -> macro as-of join -> base-feature selection -> dropna.
    """
    eq_glob = getattr(args, "equity_glob", None) or EQUITY_PATTERN
    print(f"1. Loading raw data & Extracting Equities (full extended session){'' if days is None else ' for ' + ','.join(days)}...")
    try:
        df_bars = process_raw_data(load_equity_data(_files_for_days(eq_glob, days)))
    except Exception as e:
        print(f"   equity load failed: {e}")
        return None, None, None, None, None
    df_feat = extract_features(df_bars)

    df_feat = df_feat.with_columns(
        (pl.col("ts_event") - pl.col("ts_event").min().over(["symbol", "date"])).dt.total_seconds().alias("_warm_s")
    ).filter(pl.col("_warm_s") >= args.warmup_seconds).drop("_warm_s")

    if not getattr(args, "include_extended", False):
        before = len(df_feat)
        lo, hi = _hhmm_to_seconds(args.session_start), _hhmm_to_seconds(args.session_end)
        df_feat = df_feat.filter((_tod_seconds_expr() >= lo) & (_tod_seconds_expr() <= hi))
        print(f"   Session filter [{args.session_start}-{args.session_end} UTC]: {before} -> {len(df_feat)} rows "
              f"(pre/post-market kept only as feature warm-up)")

    df_tgt = engineer_targets(df_feat)

    print("2. Loading and Extracting FULL LOB Macro Futures Nodes...")
    df_final = df_tgt.sort("ts_event")
    macro_keys_active = []
    for prefix, file_path in MACRO_CONFIGS.items():
        try:
            df_macro_feat = extract_features(process_raw_data(load_equity_data(_files_for_days(file_path, days))))
            ts_macro_col = f"ts_{prefix}"
            df_macro_feat = df_macro_feat.select(
                [pl.col("ts_event"), pl.col("ts_event").alias(ts_macro_col)] +
                [pl.col(c).alias(f"{prefix}_{c}") for c in df_macro_feat.columns if c != "ts_event"]
            ).sort("ts_event")
            df_final = df_final.join_asof(df_macro_feat, on="ts_event", strategy="backward")
            age_col, stale_col = f"{prefix}_age_ms", f"{prefix}_stale_flag"
            df_final = df_final.with_columns([
                (pl.col("ts_event") - pl.col(ts_macro_col)).dt.total_milliseconds().alias(age_col).fill_null(999999),
            ]).with_columns([(pl.col(age_col) > 5000).cast(pl.Float32).alias(stale_col)])
            macro_keys_active.append(prefix)
        except Exception as e:
            print(f"   Warning: Macro ticker {prefix} skipped: {e}")

    base_feature_cols = _select_base_feature_cols(df_feat.columns)

    macro_cols = ([f"{p}_{c}" for p in macro_keys_active for c in base_feature_cols] +
                  [f"{p}_{c}" for p in macro_keys_active for c in ["age_ms", "stale_flag"]] +
                  [f"{p}_ret_bps" for p in macro_keys_active])
    df_final = df_final.with_columns([
        pl.col(c).fill_nan(0.0).fill_null(0.0) for c in macro_cols if c in df_final.columns
    ])

    df_final = df_final.drop_nulls(subset=base_feature_cols + TARGET_COLS)
    print(f"3. Block dataset: {len(df_final)} rows, "
          f"{df_final.select(pl.col('ts_event').dt.date()).n_unique()} day(s), "
          f"{df_final['symbol'].n_unique()} symbols")
    return df_final, base_feature_cols, macro_keys_active, TARGET_COLS, HORIZON_END_S


def precompute_daily_cache(args):
    """PHASE 1 of streaming: process each trading day ONCE (load->features->
    targets->macro join, all day-local) and write a small per-day processed
    parquet to args.cache_dir. Peak memory = one day. Returns
    (cached_days, base_feature_cols, macro_keys_active, assets)."""
    eq_glob = getattr(args, "equity_glob", None) or EQUITY_PATTERN
    os.makedirs(args.cache_dir, exist_ok=True)
    all_days = sorted(list_days(eq_glob).keys())
    if not all_days:
        raise FileNotFoundError(f"No day files for {eq_glob}")
    print(f"=== PHASE 1: caching {len(all_days)} processed days to {args.cache_dir} ===")
    cached = []
    for d in all_days:
        out = os.path.join(args.cache_dir, f"proc_{d}.parquet")
        if os.path.exists(out) and not args.rebuild_cache:
            cached.append(d)
            continue
        dfd, _, _, _, _ = build_dataset(args, days=[d])
        if dfd is None or len(dfd) == 0:
            print(f"   {d}: empty after processing, skipped")
            continue
        dfd.write_parquet(out)
        cached.append(d)
        del dfd
    if not cached:
        raise RuntimeError("No days cached.")
    # derive feature cols / macro keys / asset universe from the cached schema
    schema_cols = pl.scan_parquet(os.path.join(args.cache_dir, f"proc_{cached[0]}.parquet")).collect_schema().names()
    base_feature_cols = _select_base_feature_cols(schema_cols)
    macro_keys_active = [p for p in MACRO_CONFIGS if f"{p}_ret_bps" in schema_cols]
    syms = set()
    for d in cached:
        syms |= set(pl.scan_parquet(os.path.join(args.cache_dir, f"proc_{d}.parquet"))
                    .select("symbol").unique().collect()["symbol"].to_list())
    assets = sorted(syms)
    print(f"=== PHASE 1 done: {len(cached)} days cached, {len(assets)} symbols, "
          f"{len(macro_keys_active)} macros ===")
    return cached, base_feature_cols, macro_keys_active, assets


def stream_fold_frames(cached_days, args):
    """PHASE 2 of streaming: walk forward over cached days, materializing only
    the train block + test block per fold. Streaming is inherently ROLLING --
    an expanding window would reload every prior day on the last fold and
    re-create the very OOM streaming exists to avoid -- so train_days<=0 falls
    back to a sensible rolling default. Day-level splits are leakage-safe by
    construction (different sessions; overnight gap >> purge; targets day-local).
    Yields (fold_id, df_train, df_test)."""
    td = args.train_days if args.train_days > 0 else max(1, min(5, len(cached_days) - 1))
    if args.train_days <= 0:
        print(f"   [stream] train_days not set; using rolling window of {td} days "
              f"(expanding+streaming would defeat the memory bound).")
    tst = max(1, args.test_days)
    k = 0
    pos = td
    while pos < len(cached_days):
        test_block = cached_days[pos:pos + tst]
        train_block = cached_days[max(0, pos - td):pos]
        if not train_block or not test_block:
            break
        k += 1
        df_tr = pl.concat([pl.read_parquet(os.path.join(args.cache_dir, f"proc_{d}.parquet")) for d in train_block])
        df_te = pl.concat([pl.read_parquet(os.path.join(args.cache_dir, f"proc_{d}.parquet")) for d in test_block])
        print(f"   [stream fold {k}] train {train_block[0]}..{train_block[-1]} "
              f"({len(df_tr)} rows) -> test {test_block[0]}..{test_block[-1]} ({len(df_te)} rows)")
        yield k, df_tr, df_te
        del df_tr, df_te
        pos += tst


def make_folds(df_final, n_splits, purge_seconds, train_days=0):
    """Walk-forward folds with a purge gap. Identical splitter for both models.

    train_days=0 (default): EXPANDING window -- each fold trains on everything
    before its test segment. Train size grows to ~the whole dataset on the last
    fold, which is what blows up memory on a full month.

    train_days>0: ROLLING window -- each fold trains only on the last `train_days`
    trading sessions before the purge cutoff. Train size is then bounded
    regardless of how long the total span is (so a month no longer crashes), and
    you typically run more splits -> many independent test segments -> a far
    better read on whether an edge is stable across regimes. More appropriate for
    intraday data anyway, where the recent regime matters most.
    """
    ts_all = df_final.select("ts_event").unique().sort("ts_event")["ts_event"]
    Ttot = len(ts_all)
    edges = np.linspace(0, Ttot, n_splits + 2, dtype=int)
    folds = []
    for k in range(1, n_splits + 1):
        test_start = ts_all[int(edges[k])]
        test_end = ts_all[int(edges[k + 1]) - 1]
        cutoff = test_start - timedelta(seconds=purge_seconds)   # purge >= max horizon
        df_te = df_final.filter((pl.col("ts_event") >= test_start) & (pl.col("ts_event") <= test_end))
        if train_days and train_days > 0:
            tr_dates = (df_final.filter(pl.col("ts_event") < cutoff)
                        .select(pl.col("ts_event").dt.date().alias("d")).unique().sort("d")["d"].to_list())
            if len(tr_dates) > train_days:
                start_date = tr_dates[-train_days]
                df_tr = df_final.filter((pl.col("ts_event") < cutoff)
                                        & (pl.col("ts_event").dt.date() >= start_date))
            else:
                df_tr = df_final.filter(pl.col("ts_event") < cutoff)
        else:
            df_tr = df_final.filter(pl.col("ts_event") < cutoff)
        folds.append((k, df_tr, df_te))
    return folds, Ttot


# ---------------------------------------------------------------------------
def train_and_evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"   Device: {device} | AMP: {use_amp}")

    if args.stream:
        cached_days, base_feature_cols, macro_keys_active, assets = precompute_daily_cache(args)
        target_cols, horizon_end_s = TARGET_COLS, HORIZON_END_S
        df_final = None
    else:
        df_final, base_feature_cols, macro_keys_active, target_cols, horizon_end_s = build_dataset(args)
        if df_final is None:
            return
        assets = df_final["symbol"].unique().sort().to_list()

    # ----- fixed node universe + static masks (constant across all folds) ---
    num_equities = len(assets)
    n_mac = len(macro_keys_active)
    N = num_equities + n_mac
    print(f"   Nodes: {num_equities} equities + {n_mac} macros")
    sector_to_idx = {}
    for i, t in enumerate(assets):
        sector_to_idx.setdefault(TICKER_SECTORS.get(t, f"_solo_{t}"), []).append(i)
    sector_groups = [g for g in sector_to_idx.values() if len(g) > 1]
    market_idx = assets.index(MARKET_TICKER) if MARKET_TICKER in assets else None
    sector_mask, market_mask = build_static_masks(num_equities, n_mac, sector_groups, market_idx)

    seq_len = args.seq_len
    horizon_w = torch.tensor([0.4, 0.4, 0.7, 1.0, 1.0, 1.0, 1.0], device=device)
    horizon_w = horizon_w / horizon_w.mean()
    huber = nn.SmoothL1Loss(beta=1.0, reduction="none")
    ic_weight = 0.5

    def xs_loss(out_eq, y, m):
        """Per-timestamp CROSS-SECTIONAL loss. For each window we demean
        predictions and targets across the names present at that instant, then
        score (Huber + rank-correlation). This optimizes relative ranking within
        a moment -- exactly what cross-sectional IC measures -- instead of the
        pooled fit, which is dominated by the common market move the model
        cannot rank away. Demeaning per timestamp also strips that common factor
        so gradients focus on stock-specific alpha."""
        terms = []
        for b in range(out_eq.shape[0]):
            sel = m[b]
            if int(sel.sum()) < 3:
                continue
            p = out_eq[b][sel]
            t = y[b][sel]
            p = p - p.mean(0, keepdim=True)
            t = t - t.mean(0, keepdim=True)
            reg = (huber(p, t).mean(0) * horizon_w).mean()
            num = (p * t).sum(0)
            den = p.norm(dim=0) * t.norm(dim=0) + 1e-8
            ic = (-(num / den) * horizon_w).mean()
            terms.append(reg + ic_weight * ic)
        if not terms:
            return None
        return torch.stack(terms).mean()

    base_idx = {c: i for i, c in enumerate(base_feature_cols)}
    micro_ix = base_idx.get("microprice_dev_bps", None)
    mom_ix = base_idx.get("mom_60s_z180", base_idx.get("rank_mom_60s", None))

    def run_fold(df_tr, df_te, fold_id):
        """Train a fresh model on df_tr, evaluate on df_te. All normalization
        stats (feat_stats, target_std) are fit on THIS fold's train only and
        reused on its test, so each fold is leakage-free."""
        model = RGATModel(in_channels=len(base_feature_cols) + EXTRA_DIM, hidden_channels=args.hidden_channels,
                          num_relations=NUM_RELATIONS, out_channels=len(target_cols), num_layers=2, edge_dim=EDGE_DIM).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
        scaler = torch.amp.GradScaler(enabled=use_amp)

        tr = prepare_split(df_tr, base_feature_cols, target_cols, assets, macro_keys_active)
        tr_graphs, tr_windows = precompute_graphs(tr, seq_len, sector_mask, market_mask, args.threshold, args.top_k, ablate=args.ablate_graph)
        if len(tr_windows) < args.batch_size:
            print(f"   [Fold {fold_id}] too few train windows ({len(tr_windows)}); skipping.")
            return None
        Xtr = torch.from_numpy(tr["X_all"]).to(device)
        Ytr = torch.from_numpy(tr["Y_norm"]).to(device)
        Vtr = torch.from_numpy(tr["target_valid"]).to(device)
        rng = np.random.default_rng(fold_id)
        last_loss = 0.0
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = rng.permutation(tr_windows)
            batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
            ep_loss, n_obs = 0.0, 0
            for batch in tqdm(batches, desc=f"Fold {fold_id} Ep {epoch}/{args.epochs}", leave=False):
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    out_eq, y, m = run_batch(model, Xtr, Ytr, Vtr, tr_graphs, list(batch), seq_len, N, num_equities, device)
                    if m.sum() < 3:
                        continue
                    loss = xs_loss(out_eq, y, m)
                if loss is None:
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                ep_loss += loss.item() * int(m.sum())
                n_obs += int(m.sum())
            scheduler.step()
            last_loss = ep_loss / max(1, n_obs)
        print(f"   [Fold {fold_id}] trained on {len(tr_windows)} windows, final loss {last_loss:.5f}")

        te = prepare_split(df_te, base_feature_cols, target_cols, assets, macro_keys_active,
                           target_std=tr["target_std"], feat_stats=tr["feat_stats"])
        te_graphs, te_windows = precompute_graphs(te, seq_len, sector_mask, market_mask, args.threshold, args.top_k, ablate=args.ablate_graph)
        if not te_windows:
            print(f"   [Fold {fold_id}] no valid test windows; skipping.")
            return None
        Xte = torch.from_numpy(te["X_all"]).to(device)
        Yte = torch.from_numpy(te["Y_norm"]).to(device)
        Vte = torch.from_numpy(te["target_valid"]).to(device)
        Yte_raw = te["Y_raw"]
        ts_std = tr["target_std"]

        model.eval()
        pw_pred, pw_true, pw_micro, pw_mom, all_p, all_t = [], [], [], [], [], []
        bt_times, bt_scores, bt_valid = [], [], []   # full per-name scores for the backtester
        with torch.no_grad():
            batches = [te_windows[i:i + args.batch_size] for i in range(0, len(te_windows), args.batch_size)]
            for batch in batches:
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    out_eq, _, m = run_batch(model, Xte, Yte, Vte, te_graphs, list(batch), seq_len, N, num_equities, device)
                out_eq = out_eq.float().cpu().numpy()
                m_np = m.cpu().numpy()
                for bi, w in enumerate(batch):
                    tei = w + seq_len - 1
                    # backtester needs the FULL cross-section (all names) at every
                    # evaluated time, with the tradeable mask -- not the IC subset.
                    bt_times.append(tei)
                    bt_scores.append(out_eq[bi])          # (n_eq, nt)
                    bt_valid.append(m_np[bi])             # (n_eq,)
                    sel = m_np[bi]
                    if sel.sum() < 1:
                        continue
                    p = out_eq[bi][sel]
                    t = Yte_raw[tei][sel]
                    pw_pred.append(p); pw_true.append(t); all_p.append(p); all_t.append(t)
                    node_row = te["X_all"][tei, :num_equities][sel]
                    pw_micro.append(node_row[:, micro_ix] if micro_ix is not None else np.zeros(int(sel.sum())))
                    pw_mom.append(node_row[:, mom_ix] if mom_ix is not None else np.zeros(int(sel.sum())))
        if not pw_pred:
            return None
        preds_np = np.vstack(all_p) * ts_std[None, :]
        targets_np = np.vstack(all_t)
        res = {}
        for hi, col in enumerate(target_cols):
            ph_p = [p[:, hi] for p in pw_pred]
            ph_t = [t[:, hi] for t in pw_true]
            res[col] = {
                "model": cross_sectional_ic(ph_p, ph_t, overlap_lag=horizon_end_s[hi]),
                "microprice_dev": cross_sectional_ic(pw_micro, ph_t, overlap_lag=horizon_end_s[hi]),
                "mom_60s": cross_sectional_ic(pw_mom, ph_t, overlap_lag=horizon_end_s[hi]),
                "pooled": calculate_metrics(targets_np[:, hi], preds_np[:, hi], overlap_lag=horizon_end_s[hi]),
            }

        # ----- portfolio backtest (net-of-cost tradeability) ------------------
        order = np.argsort(bt_times)
        bt_times = np.asarray(bt_times)[order]
        bt_scores = np.stack(bt_scores)[order]              # (M, n_eq, nt)
        bt_valid = np.stack(bt_valid)[order]                # (M, n_eq)
        res["backtest"] = run_backtests(
            bt_times, bt_scores, bt_valid,
            ret1s=te["R"][:, :num_equities] / 1e4,          # bps -> fraction
            spread_bps=te["spread_all"][:, :num_equities],
            day_code=te["day_code"], n_eq=num_equities,
            horizon_end_s=horizon_end_s, k=args.bt_k,
            buffer_m=(2 * args.bt_k if args.bt_buffer_m <= 0 else args.bt_buffer_m),
            band=args.bt_band, edge_gate=args.bt_edge_gate,
            score_scale=tr["target_std"],                   # raw score -> fractional return
            extra_cost_bps=args.cost_bps,
        )
        return res

    # ----- walk-forward fold construction -----------------------------------
    if args.stream:
        print(f"4. Streaming walk-forward: rolling {args.train_days or 'all'}d train, "
              f"{args.test_days}d test, over {len(cached_days)} cached days")
        folds_iter = stream_fold_frames(cached_days, args)
    else:
        folds, Ttot = make_folds(df_final, args.n_splits, args.purge_seconds, train_days=args.train_days)
        print(f"4. Walk-forward: {args.n_splits} folds "
              f"({'rolling ' + str(args.train_days) + 'd' if args.train_days > 0 else 'expanding'}), "
              f"purge={args.purge_seconds}s, ~{Ttot // (args.n_splits + 1)}s per segment")
        folds_iter = ((k, dtr, dte) for k, dtr, dte in folds)

    fold_results = []
    for k, df_tr, df_te in folds_iter:
        print(f"   Fold {k}: train rows={len(df_tr)}  test rows={len(df_te)}")
        r = run_fold(df_tr, df_te, k)
        if r is not None:
            fold_results.append(r)
        del df_tr, df_te

    if not fold_results:
        print("   No valid folds completed.")
        return

    # ----- aggregate cross-sectional IC across folds (mean +/- std) --------
    output_path = "output.txt"
    with open(output_path, "w") as f:
        f.write("========================================================\n")
        f.write("   WALK-FORWARD SPATIO-TEMPORAL GNN RUN REPORT\n")
        f.write("========================================================\n\n")
        f.write(f"Epochs/fold      : {args.epochs}\n")
        f.write(f"Hidden channels  : {args.hidden_channels}\n")
        f.write(f"TCN lookback     : {seq_len}s    Seq purge: {args.purge_seconds}s\n")
        f.write(f"Warm-up dropped  : {args.warmup_seconds}s/symbol\n")
        _mode = ("streaming roll " + str(args.train_days) + "d" if args.stream
                 else ("rolling " + str(args.train_days) + "d" if args.train_days > 0 else "expanding"))
        f.write(f"Walk-forward     : {len(fold_results)} folds completed ({_mode})\n")
        f.write(f"Edge thr / top-k : {args.threshold} / {args.top_k}\n")
        f.write(f"Nodes            : {num_equities} equities + {n_mac} macros ({', '.join(macro_keys_active)})\n\n")

        f.write("=== CROSS-SECTIONAL RANK IC, mean +/- std ACROSS FOLDS (headline) ===\n")
        f.write("    A signal is only credible if model_IC is positive and stable across\n")
        f.write("    folds AND beats the micro/mom baselines. std across folds tells you\n")
        f.write("    how regime-dependent it is.\n")
        f.write(f"    {'horizon':<12}{'model_IC':>16}{'IC_tstat':>10}{'micro_IC':>10}{'mom_IC':>10}\n")
        for col in target_cols:
            mic = np.array([r[col]["model"][0] for r in fold_results])
            mir = np.array([r[col]["model"][1] for r in fold_results])
            mcc = np.array([r[col]["microprice_dev"][0] for r in fold_results])
            mmm = np.array([r[col]["mom_60s"][0] for r in fold_results])
            f.write(f"    {col:<12}{mic.mean():>8.4f}+/-{mic.std():<5.4f}{mir.mean():>10.2f}"
                    f"{mcc.mean():>10.4f}{mmm.mean():>10.4f}\n")

        f.write("\n=== PER-FOLD MODEL CROSS-SECTIONAL IC ===\n")
        for col in target_cols:
            vals = "  ".join(f"{r[col]['model'][0]:+.4f}" for r in fold_results)
            f.write(f"    {col:<12}: {vals}\n")

        # ----- portfolio backtest: the net-of-cost tradeability verdict -------
        f.write("\n=== PORTFOLIO BACKTEST (market-neutral top/bottom-"
                f"{args.bt_k}, hold=horizon, costs ON) ===\n")
        bm = 2 * args.bt_k if args.bt_buffer_m <= 0 else args.bt_buffer_m
        f.write(f"    turnover controls: hysteresis exit rank m={bm}, no-trade band={args.bt_band}, "
                f"cost gate={args.bt_edge_gate}x round-trip spread, extra cost={args.cost_bps}bp/side\n")
        f.write("    ONE book carried through time; trade only the delta each rebalance.\n")
        f.write("    net_Sharpe is the decision metric. gross_Sharpe = same book, costs OFF.\n")
        f.write("    held = avg names in book (-> 0 means the cost gate refused ~all trades).\n")
        f.write(f"    {'horizon':<12}{'net_Shrp':>9}{'+/-':>7}{'gross_Shrp':>11}"
                f"{'netRet/yr':>10}{'turn/yr':>9}{'costDrag':>9}{'maxDD':>8}{'hit':>6}{'held':>6}\n")
        for hi, col in enumerate(target_cols):
            ns = np.array([r["backtest"][hi]["net_sharpe"] for r in fold_results])
            gs = np.array([r["backtest"][hi]["gross_sharpe"] for r in fold_results])
            nr = np.array([r["backtest"][hi]["net_ann_return"] for r in fold_results])
            tu = np.array([r["backtest"][hi]["turnover_ann"] for r in fold_results])
            cd = np.array([r["backtest"][hi]["cost_drag_ann"] for r in fold_results])
            dd = np.array([r["backtest"][hi]["max_drawdown"] for r in fold_results])
            ht = np.array([r["backtest"][hi]["hit_rate"] for r in fold_results])
            hp = np.array([r["backtest"][hi]["avg_n_pos"] for r in fold_results])
            f.write(f"    {col:<12}{ns.mean():>9.2f}{ns.std():>7.2f}{gs.mean():>11.2f}"
                    f"{nr.mean():>10.3f}{tu.mean():>9.0f}{cd.mean():>9.3f}{dd.mean():>8.3f}"
                    f"{ht.mean():>6.2f}{hp.mean():>6.1f}\n")
        lastbt = fold_results[-1]["backtest"]
        f.write("    exposure check (last fold): "
                f"gross~{lastbt[0]['avg_gross_exp']:.2f} (want 1.0), "
                f"net~{lastbt[0]['avg_net_exp']:+.3f} (want 0.0)\n")
        f.write("    NOTE: high gross_Sharpe with negative net_Sharpe = real signal, untradeable\n"
                "    at this horizon because turnover/spread eats it. Look for the horizon where\n"
                "    net_Sharpe first turns positive and stable -- that is the tradeable frontier.\n")

        f.write("\n=== POOLED METRICS (last fold, raw return units) ===\n")
        last = fold_results[-1]
        for col in target_cols:
            f.write(f"\n[{col.upper()}]\n")
            for k_, v in last[col]["pooled"].items():
                if isinstance(v, tuple):
                    f.write(f"  {k_}: {v[0]:.5f} (stat/p: {v[1]:.4e})\n")
                else:
                    f.write(f"  {k_}: {v:.5f}\n")
        f.write("\n========================================================\n")
    print(f"   Report written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Number of walk-forward folds (test segments).")
    parser.add_argument("--train_days", type=int, default=0,
                        help="Rolling-window train size in trading days (0 = expanding). Use e.g. 5 on a "
                             "full month to bound memory and get more folds.")
    parser.add_argument("--stream", action="store_true",
                        help="Stream the month day-by-day: cache each processed day to disk (phase 1), "
                             "then walk forward reading only the days each fold needs (phase 2). Use this "
                             "for a full month to avoid loading everything into memory at once.")
    parser.add_argument("--cache_dir", type=str, default="/tmp/obp_cache",
                        help="Where the streaming phase-1 per-day processed parquets are written.")
    parser.add_argument("--rebuild_cache", action="store_true",
                        help="Force phase-1 reprocessing even if cached day files already exist.")
    parser.add_argument("--test_days", type=int, default=1,
                        help="Streaming: number of trading days per test segment (fold step).")
    parser.add_argument("--equity_glob", type=str, default=None,
                        help="Override the equity file glob (e.g. a whole-month '...202605*...' pattern).")
    parser.add_argument("--purge_seconds", type=int, default=900,
                        help="Gap between train end and test start; must be >= max horizon to avoid target leakage.")
    parser.add_argument("--warmup_seconds", type=int, default=900,
                        help="Per-symbol-day leading burn-in dropped (feature warm-up).")
    parser.add_argument("--session_start", type=str, default="13:30",
                        help="Tradeable session start, UTC HH:MM (default 13:30 = 09:30 ET RTH open).")
    parser.add_argument("--session_end", type=str, default="20:00",
                        help="Tradeable session end, UTC HH:MM (default 20:00 = 16:00 ET RTH close).")
    parser.add_argument("--include_extended", action="store_true",
                        help="Skip the session filter and train/eval on the full extended session.")
    parser.add_argument("--ablate_graph", action="store_true",
                        help="Replace dynamic edges with self-loops only: A/B test whether the graph adds alpha.")
    parser.add_argument("--bt_k", type=int, default=5,
                        help="Backtest: number of names held long and short (top-k / bottom-k).")
    parser.add_argument("--bt_buffer_m", type=int, default=0,
                        help="Hysteresis exit rank m (>k): hold a name until it leaves top-m. 0 => 2*bt_k.")
    parser.add_argument("--bt_band", type=float, default=0.0,
                        help="No-trade band: skip a rebalance if total L1 weight change < band (0 disables).")
    parser.add_argument("--bt_edge_gate", type=float, default=0.0,
                        help="Cost gate: trade a name only if |predicted move| >= gate * round-trip spread (0 disables).")
    parser.add_argument("--cost_bps", type=float, default=0.0,
                        help="Backtest: extra per-side commission/slippage in bps, on top of the half-spread.")
    args = parser.parse_args()
    train_and_evaluate(args)
