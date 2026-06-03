import argparse
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, mean_squared_error
from tqdm import tqdm

from data_processor import load_equity_data, process_raw_data, engineer_targets
from features import extract_features
from graph_builder import compute_rich_dynamic_edges, build_static_masks, NUM_RELATIONS, EDGE_DIM
from model import RGATModel

# Extra per-node channels appended to the base features:
#   [log1p(age_ms), stale_flag, tod_sin, tod_cos, frac_since_open, frac_to_close, auction_flag]
EXTRA_DIM = 7

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

    hits = (np.sign(y_pred) == np.sign(y_true)).astype(float)
    hit_rate = float(np.mean(hits))
    # HAC t-stat on (hit - 0.5); overlap_lag reflects the horizon overlap.
    t_hit = newey_west_tstat(hits - 0.5, overlap_lag)

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
    Uses polars duration arithmetic so it is robust to the column time unit."""
    secs = (ts_series - ts_series.min()).dt.total_seconds().to_numpy().astype(np.float64)
    total = max(secs[-1], 1.0)
    frac = np.clip(secs / total, 0.0, 1.0)
    sin = np.sin(2 * np.pi * frac)
    cos = np.cos(2 * np.pi * frac)
    auction = ((secs < 300) | ((total - secs) < 300)).astype(np.float64)
    return np.stack([sin, cos, frac, 1.0 - frac, auction], axis=1)


def prepare_split(df: pl.DataFrame, base_cols, target_cols, assets, macro_keys, target_std=None):
    n_eq = len(assets)
    n_mac = len(macro_keys)
    N = n_eq + n_mac
    F = len(base_cols)
    nt = len(target_cols)

    ts_df = df.select("ts_event").unique().sort("ts_event")
    ts_series = ts_df["ts_event"]
    ts_list = ts_series.to_list()
    T = len(ts_list)
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
    age_eq = np.zeros((T, n_eq))
    age_all = np.concatenate([age_eq, age_mac], axis=1)
    stale_all = np.concatenate([stale_eq, stale_mac], axis=1)
    base_all = np.concatenate([X_eq, X_mac], axis=1)   # (T, N, F)

    extra = np.empty((T, N, EXTRA_DIM), dtype=np.float64)
    extra[:, :, 0] = np.log1p(np.clip(age_all, 0, None))
    extra[:, :, 1] = stale_all
    extra[:, :, 2:7] = seas[:, None, :]                # broadcast seasonality to all nodes

    X_all = np.concatenate([base_all, extra], axis=2).astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    # ----- per-horizon target scale normalization (FIX #2) -----------------
    if target_std is None:
        vt = target_valid
        flat = Y_raw[vt]                                # (M, nt)
        target_std = np.std(flat, axis=0)
        target_std = np.where(target_std > 0, target_std, 1.0)
    Y_raw_filled = np.nan_to_num(Y_raw, nan=0.0)
    Y_norm = (Y_raw_filled / target_std).astype(np.float32)

    # column indices for graph node modifiers
    idx = {c: i for i, c in enumerate(base_cols)}
    res_name = next((c for c in base_cols if "kalman_z_score" in c), None)
    spread_all = base_all[:, :, idx["spread_bps"]] if "spread_bps" in idx else np.ones((T, N))
    liq_all = base_all[:, :, idx["top5_book_size"]] if "top5_book_size" in idx else np.ones((T, N))
    res_all = base_all[:, :, idx[res_name]] if res_name else np.zeros((T, N))

    return {
        "X_all": X_all, "R": R, "Y_norm": Y_norm, "Y_raw": Y_raw_filled,
        "target_valid": target_valid, "ts_list": ts_list, "T": T, "N": N, "n_eq": n_eq,
        "spread_all": np.nan_to_num(spread_all), "liq_all": np.nan_to_num(liq_all),
        "res_all": np.nan_to_num(res_all), "target_std": target_std,
    }


def precompute_graphs(split, seq_len, sector_mask, market_mask, threshold, top_k):
    """Build (and cache) one graph per valid window end. Done once per split,
    not once per epoch (FIX: Perf A)."""
    R = split["R"]; T = split["T"]; n_eq = split["n_eq"]
    spread_all, liq_all, res_all = split["spread_all"], split["liq_all"], split["res_all"]
    valid = split["target_valid"]

    graphs = {}
    windows = []
    for w in range(0, T - seq_len + 1):
        te = w + seq_len - 1
        if not valid[te].any():
            continue
        r5 = R[max(0, te - 300):te + 1]
        r30 = R[max(0, te - 1800):te + 1]
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
def train_and_evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"   Device: {device} | AMP: {use_amp}")

    pattern = "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/XNAS.ITCH/*/2026/xnas-itch-20260504.mbp-10.*.parquet"
    print("1. Loading raw data & Extracting Equities...")
    try:
        df_bars = process_raw_data(load_equity_data(pattern))
    except Exception:
        return
    df_feat = extract_features(df_bars)
    df_tgt = engineer_targets(df_feat)
    df_tgt = df_tgt.with_columns(pl.col("ts_event").cast(pl.Datetime("ns")))
    
    print("2. Loading and Extracting FULL LOB Macro Futures Nodes...")
    macro_configs = {
        "ES": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/ES.n.0/2026/glbx-mdp3-20260504.mbp-10.ES.n.0.parquet",
        "NQ": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/NQ.n.0/2026/glbx-mdp3-20260504.mbp-10.NQ.n.0.parquet",
        "CL": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/CL.n.0/2026/glbx-mdp3-20260504.mbp-10.CL.n.0.parquet",
        "BZ": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/BZ.n.0/2026/glbx-mdp3-20260504.mbp-10.BZ.n.0.parquet",
        "NG": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/NG.n.0/2026/glbx-mdp3-20260504.mbp-10.NG.n.0.parquet",
        "GC": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/GC.n.0/2026/glbx-mdp3-20260504.mbp-10.GC.n.0.parquet",
        "SI": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/SI.n.0/2026/glbx-mdp3-20260504.mbp-10.SI.n.0.parquet",
        "HG": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/HG.n.0/2026/glbx-mdp3-20260504.mbp-10.HG.n.0.parquet",
    }

    df_final = df_tgt.sort("ts_event")
    macro_keys_active = []
    for prefix, file_path in macro_configs.items():
        try:
            print(f"   Processing and Feature Engineering macro feed: {prefix}")
            df_macro_bars = process_raw_data(load_equity_data(file_path))
            df_macro_feat = extract_features(df_macro_bars)
            df_macro_feat = df_macro_feat.with_columns(pl.col("ts_event").cast(pl.Datetime("ns")))
            ts_macro_col = f"ts_{prefix}"
            df_macro_feat = df_macro_feat.select(
                [pl.col("ts_event"), pl.col("ts_event").alias(ts_macro_col)] +
                [pl.col(c).alias(f"{prefix}_{c}") for c in df_macro_feat.columns if c != "ts_event"]
            ).sort("ts_event")
            df_final = df_final.join_asof(df_macro_feat, on="ts_event", strategy="backward")
            age_col, stale_col = f"{prefix}_age_ms", f"{prefix}_stale_flag"
            df_final = df_final.with_columns([
                (pl.col("ts_event") - pl.col(ts_macro_col)).dt.total_milliseconds().alias(age_col).fill_null(999999),
            ]).with_columns([
                (pl.col(age_col) > 5000).cast(pl.Float32).alias(stale_col)
            ])
            macro_keys_active.append(prefix)
        except Exception as e:
            print(f"   Warning: Macro ticker {prefix} skipped: {e}")

    base_feature_cols = [
        col for col in df_feat.columns
        if col.endswith("_z180") or col.endswith("_z600") or col in [
            "microprice_dev_bps", "obi_L1", "obi_L5", "depth_entropy_bid", "depth_entropy_ask",
            "distance_weighted_imbalance", "cancel_burst_bid", "cancel_burst_ask", "large_trade_flag",
            "trade_count_buy", "trade_count_sell", "aggressor_streak_buy", "aggressor_streak_sell",
            "spread_duration", "quote_stability", "price_level_flip_count", "depth_slope_bid",
            "depth_slope_ask", "depth_curvature_bid", "depth_curvature_ask", "spread_bps", "top5_book_size",
            "vpin_60s", "jump_ratio", "msg_intensity_ratio", "trade_intensity_ratio",
        ] or col.startswith("rank_") or col.startswith("kalman_z_score_")
    ]

    macro_feature_cols = [f"{p}_{c}" for p in macro_keys_active for c in base_feature_cols]
    macro_age_stale_cols = [f"{p}_{c}" for p in macro_keys_active for c in ["age_ms", "stale_flag"]]
    macro_ret_cols = [f"{p}_ret_bps" for p in macro_keys_active]
    df_final = df_final.with_columns([
        pl.col(c).fill_nan(0.0).fill_null(0.0)
        for c in macro_feature_cols + macro_age_stale_cols + macro_ret_cols
        if c in df_final.columns
    ])

    target_cols = ["ret_0_5s", "ret_5_10s", "ret_10_30s", "ret_30_60s", "ret_60_300s", "ret_300_600s", "ret_600_900s"]
    horizon_end_s = [5, 10, 30, 60, 300, 600, 900]
    df_final = df_final.drop_nulls(subset=base_feature_cols + target_cols)

    print("3. Chronological Train/Test Data Split with Purge Buffering...")
    min_time, max_time = df_final["ts_event"].min(), df_final["ts_event"].max()
    train_end = min_time + (max_time - min_time) * 0.8
    test_start = train_end + pl.duration(seconds=900)
    df_train = df_final.filter(pl.col("ts_event") <= train_end)
    df_test = df_final.filter(pl.col("ts_event") >= test_start)

    assets = df_train["symbol"].unique().sort().to_list()
    num_equities = len(assets)
    print(f"   Nodes count: {num_equities} Equities + {len(macro_keys_active)} Macros")

    # ----- static sector / market masks (predictive structural edges) ------
    sector_to_idx = {}
    for i, t in enumerate(assets):
        sector_to_idx.setdefault(TICKER_SECTORS.get(t, f"_solo_{t}"), []).append(i)
    sector_groups = [g for g in sector_to_idx.values() if len(g) > 1]
    market_idx = assets.index(MARKET_TICKER) if MARKET_TICKER in assets else None
    sector_mask, market_mask = build_static_masks(num_equities, len(macro_keys_active), sector_groups, market_idx)

    print("4. Instantiating Spatio-Temporal Graph Model Architecture...")
    seq_len = args.seq_len
    model = RGATModel(
        in_channels=len(base_feature_cols) + EXTRA_DIM,
        hidden_channels=args.hidden_channels,
        num_relations=NUM_RELATIONS,
        out_channels=len(target_cols),
        num_layers=2,
        edge_dim=EDGE_DIM,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(enabled=use_amp)

    print("   Preparing dense train tensors & precomputing graphs (once)...")
    train_split = prepare_split(df_train, base_feature_cols, target_cols, assets, macro_keys_active)
    target_std = train_split["target_std"]
    train_graphs, train_windows = precompute_graphs(train_split, seq_len, sector_mask, market_mask, args.threshold, args.top_k)

    Xtr = torch.from_numpy(train_split["X_all"]).to(device)
    Ytr = torch.from_numpy(train_split["Y_norm"]).to(device)
    Vtr = torch.from_numpy(train_split["target_valid"]).to(device)
    N = train_split["N"]

    print(f"5. Beginning Training Loop ({args.epochs} Epochs, {len(train_windows)} windows)...")
    rng = np.random.default_rng(0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_windows)
        batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        epoch_loss, n_obs = 0.0, 0
        for batch in tqdm(batches, desc=f"Epoch {epoch}/{args.epochs}"):
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out_eq, y, m = run_batch(model, Xtr, Ytr, Vtr, train_graphs, list(batch), seq_len, N, num_equities, device)
                if m.sum() == 0:
                    continue
                loss = criterion(out_eq[m], y[m])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * int(m.sum())
            n_obs += int(m.sum())
        print(f"   Epoch {epoch} Complete. Train MSE (norm units): {epoch_loss / max(1, n_obs):.6f}")

    print("6. Generating Out-of-Sample Test Evaluation Pass...")
    test_split = prepare_split(df_test, base_feature_cols, target_cols, assets, macro_keys_active, target_std=target_std)
    test_graphs, test_windows = precompute_graphs(test_split, seq_len, sector_mask, market_mask, args.threshold, args.top_k)
    Xte = torch.from_numpy(test_split["X_all"]).to(device)
    Yte = torch.from_numpy(test_split["Y_norm"]).to(device)
    Vte = torch.from_numpy(test_split["target_valid"]).to(device)
    Yte_raw = test_split["Y_raw"]

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        batches = [test_windows[i:i + args.batch_size] for i in range(0, len(test_windows), args.batch_size)]
        for batch in tqdm(batches, desc="Evaluating Test Set"):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out_eq, _, m = run_batch(model, Xte, Yte, Vte, test_graphs, list(batch), seq_len, N, num_equities, device)
            out_eq = out_eq.float().cpu().numpy()
            m_np = m.cpu().numpy()
            for bi, w in enumerate(batch):
                te = w + seq_len - 1
                sel = m_np[bi]
                if sel.any():
                    all_preds.append(out_eq[bi][sel])
                    all_targets.append(Yte_raw[te][sel])

    if not all_preds:
        print("   No valid test windows.")
        return

    preds_np = np.vstack(all_preds) * target_std[None, :]   # un-normalize to raw return units
    targets_np = np.vstack(all_targets)

    horizon_metrics = {}
    for idx, col in enumerate(target_cols):
        horizon_metrics[col] = calculate_metrics(targets_np[:, idx], preds_np[:, idx], overlap_lag=horizon_end_s[idx])

    output_path = "output.txt"
    with open(output_path, "w") as f:
        f.write("========================================================\n")
        f.write("        MULTI-TASK SPATIO-TEMPORAL GNN RUN REPORT       \n")
        f.write("========================================================\n\n")
        f.write(f"Total Backpropagation Epochs   : {args.epochs}\n")
        f.write(f"Hidden Dim Channel Size Layer  : {args.hidden_channels}\n")
        f.write(f"TCN Causal Lookback Sequence   : {seq_len} seconds\n")
        f.write(f"Edge threshold / top-k         : {args.threshold} / {args.top_k}\n")
        f.write(f"Equity Nodes                   : {num_equities}\n")
        f.write(f"Macro Nodes Interconnected     : {len(macro_keys_active)} ({', '.join(macro_keys_active)})\n\n")
        f.write("--- OUT-OF-SAMPLE RESULTS BY HORIZON (raw return units) ---\n")
        for horizon, metrics in horizon_metrics.items():
            f.write(f"\n[{horizon.upper()}]\n")
            for k, v in metrics.items():
                if isinstance(v, tuple):
                    f.write(f"  {k}: {v[0]:.5f} (stat/p: {v[1]:.4e})\n")
                else:
                    f.write(f"  {k}: {v:.5f}\n")
        f.write("\n========================================================\n")
    print(f"   Report written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=8)
    args = parser.parse_args()
    train_and_evaluate(args)