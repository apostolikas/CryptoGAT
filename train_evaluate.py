import os
import argparse
import numpy as np
import polars as pl
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr, ttest_1samp
from sklearn.metrics import roc_auc_score, mean_squared_error
from tqdm import tqdm
from collections import deque

from data_processor import load_equity_data, process_raw_data, engineer_targets
from features import extract_features
from graph_builder import compute_rich_dynamic_edges
from model import RGATModel

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculate Backtest & Statistical Performance Metrics for a 1D target horizon array.
    """
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_corr, p_pearson = pearsonr(y_pred, y_true)
        spearman_corr, p_spearman = spearmanr(y_pred, y_true)
    else:
        pearson_corr, p_pearson = 0.0, 1.0
        spearman_corr, p_spearman = 0.0, 1.0

    y_true_bin = (y_true > 0).astype(int)
    if len(np.unique(y_true_bin)) > 1:
        auc = roc_auc_score(y_true_bin, y_pred)
    else:
        auc = 0.5
        
    mse = mean_squared_error(y_true, y_pred)
    hit_rate = np.mean(np.sign(y_pred) == np.sign(y_true))
    
    strategy_returns = np.sign(y_pred) * y_true
    mean_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns)
    ir = (mean_ret / std_ret) * np.sqrt(252 * 23400) if std_ret > 0 else 0.0 
    
    t_stat_hr, p_hr = ttest_1samp((np.sign(y_pred) == np.sign(y_true)).astype(float), 0.5)
    
    return {
        'Pearson': (pearson_corr, p_pearson),
        'Spearman': (spearman_corr, p_spearman),
        'AUC': auc,
        'MSE': mse,
        'Hit_Rate': (hit_rate, p_hr),
        'IR': ir
    }

def build_temporal_batches(df: pl.DataFrame, feature_cols: list, target_col: str, assets: list, seq_len: int = 60):
    """
    Converts a flat chronological dataframe into a sequence of aligned 3D Node matrices.
    """
    returns_pd = df.select(["ts_event", "symbol", target_col]).pivot(
        index="ts_event", columns="symbol", values=target_col
    ).to_pandas().set_index("ts_event").fillna(0.0)

    df_sorted = df.sort(["ts_event", "symbol"])
    grouped = df_sorted.group_by("ts_event", maintain_order=True)
    
    history_buffer = deque(maxlen=seq_len)
    
    for ts, group in grouped:
        x_flat = group.select(feature_cols).to_numpy()
        y_flat = group.select([target_col]).to_numpy()
        
        if x_flat.shape[0] != len(assets):
            continue
            
        history_buffer.append(x_flat)
        
        if len(history_buffer) == seq_len:
            x_3d = np.stack(history_buffer, axis=0).transpose(1, 0, 2)
            
            x_tensor = torch.tensor(x_3d.astype(np.float32), dtype=torch.float32)
            y_tensor = torch.tensor(y_flat.astype(np.float32), dtype=torch.float32)
            
            # Unpack the tuple key safely
            ts_value = ts[0] if isinstance(ts, tuple) else ts
            ts_scalar = pd.Timestamp(ts_value)
            ts_idx = returns_pd.index.get_loc(ts_scalar)
            
            start_5m_idx = max(0, ts_idx - 300)
            start_30m_idx = max(0, ts_idx - 1800)
            
            returns_5m = returns_pd.iloc[start_5m_idx:ts_idx + 1]
            returns_30m = returns_pd.iloc[start_30m_idx:ts_idx + 1]
            
            yield ts_value, x_tensor, y_tensor, returns_5m, returns_30m, group

def train_and_evaluate(args):
    """
    Institutional Multi-Horizon Dynamic TCN-RGAT Pipeline Orchestration
    """
    pattern = "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/XNAS.ITCH/[ABC]*/2026/xnas-itch-2026050[45].mbp-10.[ABC]*.parquet"

    print("1. Loading raw data & Extracting Equities...")
    try:
        lf_raw = load_equity_data(pattern)
        df_bars = process_raw_data(lf_raw)
    except FileNotFoundError:
        print(f"No files matched pattern {pattern}. Exiting.")
        return

    print("2. Computing stationary features for Equities...")
    df_feat = extract_features(df_bars)

    print("3. Engineering target variables...")
    df_tgt = engineer_targets(df_feat)

    print("4. Loading and Extracting FULL LOB Macro Futures...")
    macro_configs = {
        "ES": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/ES.n.0/2026/glbx-mdp3-2026050[45].mbp-10.ES.n.0.parquet",
        "NQ": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/NQ.n.0/2026/glbx-mdp3-2026050[45].mbp-10.NQ.n.0.parquet",
        "CL": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/CL.n.0/2026/glbx-mdp3-2026050[45].mbp-10.CL.n.0.parquet",
        "BZ": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/BZ.n.0/2026/glbx-mdp3-2026050[45].mbp-10.BZ.n.0.parquet",
        "NG": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/NG.n.0/2026/glbx-mdp3-2026050[45].mbp-10.NG.n.0.parquet",
        "GC": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/GC.n.0/2026/glbx-mdp3-2026050[45].mbp-10.GC.n.0.parquet",
        "SI": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/SI.n.0/2026/glbx-mdp3-2026050[45].mbp-10.SI.n.0.parquet",
        "HG": "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/HG.n.0/2026/glbx-mdp3-2026050[45].mbp-10.HG.n.0.parquet"
    }

    df_final = df_tgt.sort("ts_event")

    for prefix, file_path in macro_configs.items():
        try:
            print(f"   Processing and Feature Engineering macro feed: {prefix}")
            df_macro_raw = load_equity_data(file_path)
            df_macro_bars = process_raw_data(df_macro_raw)
            df_macro_feat = extract_features(df_macro_bars)
            
            ts_macro_col = f"ts_{prefix}"
            df_macro_feat = df_macro_feat.select(
                [pl.col("ts_event"), pl.col("ts_event").alias(ts_macro_col)] + 
                [pl.col(c).alias(f"{prefix}_{c}") for c in df_macro_feat.columns if c != "ts_event"]
            ).sort("ts_event")
            
            df_final = df_final.join_asof(df_macro_feat, on="ts_event", strategy="backward")
            
            age_col = f"{prefix}_age_ms"
            stale_col = f"{prefix}_stale_flag"
            
            df_final = df_final.with_columns([
                (pl.col("ts_event") - pl.col(ts_macro_col)).dt.total_milliseconds().alias(age_col).fill_null(999999),
            ]).with_columns([
                (pl.col(age_col) > 5000).cast(pl.Float32).alias(stale_col)
            ])
            
        except Exception as e:
            print(f"   Warning: Macro ticker {prefix} skipped or error locating source file: {e}")

    # -------------------------------------------------------------------------
    # SECURE AUTOMATED FEATURE ARRAY SELECTION
    # -------------------------------------------------------------------------
    model_feature_cols = [
        col for col in df_final.columns 
        if col.endswith("_z180") or col.endswith("_z600") or col in [
            "microprice_dev_bps", "obi_L1", "obi_L5", "depth_entropy_bid", "depth_entropy_ask",
            "distance_weighted_imbalance", "cancel_burst_bid", "cancel_burst_ask", "large_trade_flag",
            "trade_count_buy", "trade_count_sell", "aggressor_streak_buy", "aggressor_streak_sell",
            "spread_duration", "quote_stability", "price_level_flip_count", "depth_slope_bid",
            "depth_slope_ask", "depth_curvature_bid", "depth_curvature_ask", "spread_bps"
        ] 
        or col.startswith("rank_")
        or (
            any(col.startswith(f"{p}_") for p in macro_configs.keys()) 
            and not col.endswith("_symbol") 
            and not col.endswith("_action")
        )
    ]

    target_col = args.target
    if target_col not in df_final.columns:
        raise ValueError(f"Selected target column '{target_col}' doesn't match engineered dataset headers.")

    # Drop non-targeted horizons and scrub target boundaries
    all_targets = [c for c in df_final.columns if c.startswith("ret_") and c != "ret_bps"]
    df_final = df_final.drop([t for t in all_targets if t != target_col])
    df_final = df_final.drop_nulls(subset=model_feature_cols + [target_col])
    df_final = df_final.filter(pl.col(target_col) != 0.0)

    print("5. Chronological Train/Test Data Split with Purge Buffering...")
    min_time, max_time = df_final["ts_event"].min(), df_final["ts_event"].max()
    train_end = min_time + (max_time - min_time) * 0.8
    test_start = train_end + pl.duration(seconds=900)
    
    df_train = df_final.filter(pl.col("ts_event") <= train_end)
    df_test = df_final.filter(pl.col("ts_event") >= test_start)
    
    assets = df_train["symbol"].unique().sort().to_list()
    print(f"   Train samples: {len(df_train)} | Test samples: {len(df_test)} | Nodes count: {len(assets)}")

    print("6. Instantiating Spatio-Temporal Graph Model Architecture...")
    F_in = len(model_feature_cols)
    seq_len = 60
    
    model = RGATModel(
        in_channels=F_in, 
        hidden_channels=args.hidden_channels, 
        num_relations=1, 
        out_channels=1,
        num_layers=2,
        edge_dim=10
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"7. Beginning Multi-Epoch Training Loop ({args.epochs} Epochs)...")
    total_batches = df_train["ts_event"].n_unique() - seq_len + 1
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        train_gen = build_temporal_batches(df_train, model_feature_cols, target_col, assets, seq_len=seq_len)
        
        for ts, x_seq, y, r_5m, r_30m, group_t in tqdm(train_gen, total=total_batches, desc=f"Epoch {epoch}/{args.epochs}"):
            edge_index, edge_attr = compute_rich_dynamic_edges(
                returns_5m=r_5m, 
                returns_30m=r_30m, 
                node_features_t=group_t, 
                threshold=0.05
            )
            edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)
            
            optimizer.zero_grad()
            out = model(x_seq, edge_index, edge_type, edge_attr=edge_attr)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
        print(f"   Epoch {epoch} Complete. Average Train MSE Loss: {epoch_loss / max(1, batch_count):.6f}")

    print("8. Generating Out-of-Sample Test Evaluation Pass...")
    model.eval()
    total_test_batches = df_test["ts_event"].n_unique() - seq_len + 1
    test_gen = build_temporal_batches(df_test, model_feature_cols, target_col, assets, seq_len=seq_len)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for ts, x_seq, y, r_5m, r_30m, group_t in tqdm(test_gen, total=total_test_batches, desc="Evaluating Test Set"):
            edge_index, edge_attr = compute_rich_dynamic_edges(
                returns_5m=r_5m, 
                returns_30m=r_30m, 
                node_features_t=group_t, 
                threshold=0.05
            )
            edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)
            
            out = model(x_seq, edge_index, edge_type, edge_attr=edge_attr)
            all_preds.append(out.numpy())
            all_targets.append(y.numpy())

    if not all_preds:
        print("No valid out-of-sample sequences available after lookback window allocations.")
        return
        
    preds_np = np.vstack(all_preds).flatten()
    targets_np = np.vstack(all_targets).flatten()
    test_metrics = calculate_metrics(targets_np, preds_np)

    output_path = "output.txt"
    print(f"Logging structural summary run parameters straight into {output_path}")
    with open(output_path, "w") as f:
        f.write("========================================================\n")
        f.write("                         RGAT REPORT                     \n")
        f.write("========================================================\n\n")
        f.write(f"Isolated Target Horizon Logged : {target_col}\n")
        f.write(f"Total Backpropagation Epochs   : {args.epochs}\n")
        f.write(f"Hidden Dim Channel Size Layer  : {args.hidden_channels}\n")
        f.write(f"TCN Lookback Frame Sequence    : {seq_len} seconds\n\n")
        f.write("--- OUT-OF-SAMPLE RESULTS ---\n")
        for k, v in test_metrics.items():
            if isinstance(v, tuple):
                f.write(f"  {k}: {v[0]:.5f} (p-val: {v[1]:.4e})\n")
            else:
                f.write(f"  {k}: {v:.5f}\n")
        f.write("========================================================\n")
    print("Execution accomplished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Horizon Spatio-Temporal Graph Engine Executive Driver")
    parser.add_argument("--epochs", type=int, default=1, help="Training iterations")
    parser.add_argument("--hidden_channels", type=int, default=32, help="Hidden dimensions width sizing")
    parser.add_argument("--target", type=str, default="ret_10_30s", help="Target horizon string column mapping name")
    
    args = parser.parse_args()
    train_and_evaluate(args)