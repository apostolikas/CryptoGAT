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

from data_processor import load_equity_data, process_raw_data, engineer_targets
from features import extract_features
from kalman_pricing import apply_kalman_filter
from graph_builder import build_graph, compute_correlation_edges
from model import RGATModel

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Step 8: Calculate Backtest & Statistical Performance Metrics
    Expects 1D arrays for a specific target horizon.
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

def train_and_evaluate(args):
    """
    Institutional Pipeline Orchestration
    """
    pattern = "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/XNAS.ITCH/*/2026/xnas-itch-2026050[45].mbp-10.*.parquet"

    print("1. Loading raw data...")
    try:
        lf_raw = load_equity_data(pattern)
        df_bars = process_raw_data(lf_raw)
    except FileNotFoundError:
        print(f"No files matched pattern {pattern}. Exiting.")
        return

    print("2. Computing stationary features...")
    df_feat = extract_features(df_bars)

    print("3. Engineering targets...")
    df_tgt = engineer_targets(df_feat)

    print("4. Loading Macro drivers and applying Kalman Filtering...")
    try:
        gold_pattern = "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/GC.n.0/2026/glbx-mdp3-2026050[45].mbp-10.GC.n.0.parquet"
        df_gold = process_raw_data(load_equity_data(gold_pattern))
        
        nq_pattern = "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/NG.n.0/2026/glbx-mdp3-2026050[45].mbp-10.NG.n.0.parquet"
        df_nq = process_raw_data(load_equity_data(nq_pattern))
        
        bz_pattern = "/home/apostolikas/orderbook_data/Databento/mbp-10/raw/GLBX.MDP3/BZ.n.0/2026/glbx-mdp3-2026050[45].mbp-10.BZ.n.0.parquet"
        df_bz = process_raw_data(load_equity_data(bz_pattern))
        
        macro_gold = df_gold.select(["ts_event", pl.col("price").alias("price_gold")])
        macro_nq   = df_nq.select(["ts_event", pl.col("price").alias("price_nq")])
        macro_bz   = df_bz.select(["ts_event", pl.col("price").alias("price_bz")])

        df_macros = (
            macro_gold
            .join(macro_nq, on="ts_event", how="outer_coalesce")
            .join(macro_bz, on="ts_event", how="outer_coalesce")
            .sort("ts_event")
        ).with_columns([
            pl.col("price_gold").forward_fill(),
            pl.col("price_nq").forward_fill(),
            pl.col("price_bz").forward_fill()
        ])
        
        df_pipeline = df_tgt.join(df_macros, on="ts_event", how="left")

        macro_drivers = {"gold": ["price_gold"], "nq": ["price_nq"], "bz": ["price_bz"]}
        df_filtered_pairs = df_pipeline
        
        for label, col_list in macro_drivers.items():
            df_tmp = apply_kalman_filter(df_filtered_pairs, asset_col="price", macro_cols=col_list, window_size=180)
            df_filtered_pairs = df_tmp.rename({
                "kalman_epsilon": f"kalman_epsilon_{label}",
                "kalman_z_score": f"kalman_z_score_{label}"
            })
            
        df_final = df_filtered_pairs
    except Exception as e:
        print(f"Macro data missing or error: {e}. Falling back to standard targets.")
        df_final = df_tgt

    # -------------------------------------------------------------------------
    # FEATURE SELECTION FOR GNN NODES
    # -------------------------------------------------------------------------
    model_feature_cols = [
        col for col in df_final.columns 
        if col.endswith("_z180") or col.endswith("_z600") or col in [
            "microprice_dev_bps", "obi_L1", "obi_L5", 
            "queue_convexity_bid", "queue_convexity_ask",
            "depth_ratio_bid_L2", "depth_ratio_ask_L2", 
            "depth_ratio_bid_L5", "depth_ratio_ask_L5",
            "bid_remove_ratio", "bid_add_ratio",
            "vol_expand_60_300", "vol_expand_180_600",
            "vol_spread_ratio_30s", "vol_spread_ratio_180s"
        ] 
        or col.startswith("trade_side_")
        or col.startswith("kalman_z_score_")
    ]

    # Validate specified target column exists
    target_col = args.target
    if target_col not in df_final.columns:
        raise ValueError(f"Target column '{target_col}' not found in engineered targets. Available: {[c for c in df_final.columns if c.startswith('ret_')]}")

    # Drop all other prediction target horizons to save space and ensure single-target training integrity
    all_targets = [c for c in df_final.columns if c.startswith("ret_") and c != "ret_bps"]
    drop_targets = [t for t in all_targets if t != target_col]
    df_final = df_final.drop(drop_targets)

    # Clean the dataset: Drop nulls, and explicitly filter out where the chosen target column is exactly 0
    df_final = df_final.drop_nulls(subset=model_feature_cols + [target_col])
    df_final = df_final.filter(pl.col(target_col) != 0.0)

    print("5. Chronological Train/Test Split with 900-second Purge Gap...")
    min_time = df_final["ts_event"].min()
    max_time = df_final["ts_event"].max()
    duration = max_time - min_time
    
    train_end = min_time + duration * 0.8
    test_start = train_end + pl.duration(seconds=900)
    
    df_train = df_final.filter(pl.col("ts_event") <= train_end)
    df_test = df_final.filter(pl.col("ts_event") >= test_start)
    
    print(f"   Train samples: {len(df_train)} | Test samples: {len(df_test)}")
    if len(df_train) == 0 or len(df_test) == 0:
        print("Insufficient data after filtering. Exiting.")
        return

    print("6. Building Graph Edges (Strictly on Train Set)...")
    assets = df_train["symbol"].unique().sort().to_list()
    
    # Check if we have an early proxy return available to build continuous correlation links
    pivot_col = "ret_0_5s" if "ret_0_5s" in df_train.columns else target_col
    train_returns_pd = df_train.select(["ts_event", "symbol", pivot_col]).pivot(
        index="ts_event", columns="symbol", values=pivot_col
    ).to_pandas().set_index("ts_event")
    
    edge_index, edge_attr = compute_correlation_edges(train_returns_pd, threshold=0.01)
    edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)

    print("7. Initializing Relational Graph Architecture...")
    F_in = len(model_feature_cols)
    out_dim = 1  # Changed to 1 since we dropped the rest of the target horizons
    
    model = RGATModel(
        in_channels=F_in, 
        hidden_channels=args.hidden_channels, 
        num_relations=1, 
        out_channels=out_dim,
        edge_dim=1
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Pre-calculate batch iterators to save optimization execution overhead
    train_grouped = df_train.group_by("ts_event", maintain_order=True)
    total_batches = df_train["ts_event"].n_unique()

    print(f"8. Training Model for {args.epochs} Epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for ts, group in tqdm(train_grouped, total=total_batches, desc=f"Epoch {epoch}/{args.epochs}"):
            group = group.sort("symbol")
            x = torch.tensor(group.select(model_feature_cols).to_numpy(), dtype=torch.float32)
            y = torch.tensor(group.select([target_col]).to_numpy(), dtype=torch.float32)
            
            if x.shape[0] != len(assets):
                continue

            optimizer.zero_grad()
            out = model(x, edge_index, edge_type, edge_attr)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
        print(f"   Epoch {epoch} complete. Avg Training MSE Loss: {epoch_loss / max(1, batch_count):.6f}")

    # Generate Training predictions for metrics report
    print("Evaluating Model performance on Train Set...")
    model.eval()
    all_train_preds = []
    all_train_targets = []
    with torch.no_grad():
        for ts, group in train_grouped:
            group = group.sort("symbol")
            x = torch.tensor(group.select(model_feature_cols).to_numpy(), dtype=torch.float32)
            y = torch.tensor(group.select([target_col]).to_numpy(), dtype=torch.float32)
            if x.shape[0] == len(assets):
                out = model(x, edge_index, edge_type, edge_attr)
                all_train_preds.append(out.numpy())
                all_train_targets.append(y.numpy())
                
    train_preds_np = np.vstack(all_train_preds).flatten()
    train_targets_np = np.vstack(all_train_targets).flatten()
    train_metrics = calculate_metrics(train_targets_np, train_preds_np)

    print("9. Evaluating Model performance on Out-of-Sample Test Set...")
    test_grouped = df_test.group_by("ts_event", maintain_order=True)
    total_eval_batches = df_test["ts_event"].n_unique()
    
    all_test_preds = []
    all_test_targets = []
    
    with torch.no_grad():
        for ts, group in tqdm(test_grouped, total=total_eval_batches, desc="Evaluating Test Set"):
            group = group.sort("symbol")
            x = torch.tensor(group.select(model_feature_cols).to_numpy(), dtype=torch.float32)
            y = torch.tensor(group.select([target_col]).to_numpy(), dtype=torch.float32)
            
            if x.shape[0] != len(assets):
                continue
                
            out = model(x, edge_index, edge_type, edge_attr)
            all_test_preds.append(out.numpy())
            all_test_targets.append(y.numpy())
            
    if not all_test_preds:
        print("No valid test batches found.")
        return
        
    test_preds_np = np.vstack(all_test_preds).flatten()
    test_targets_np = np.vstack(all_test_targets).flatten()
    test_metrics = calculate_metrics(test_targets_np, test_preds_np)

    # -------------------------------------------------------------------------
    # WRITE EXPORT METRICS TO OUTPUT.TXT
    # -------------------------------------------------------------------------
    output_path = "/home/apostolikas/rwa/gnn/output.txt"
    print(f"Writing experiment summary report securely to {output_path}...")
    with open(output_path, "w") as f:
        f.write("========================================================\n")
        f.write("                RGAT PIPELINE RUN LOG                   \n")
        f.write("========================================================\n\n")
        
        f.write("--- INPUT ARGUMENTS ---\n")
        f.write(f"Target Horizon Logged: {args.target}\n")
        f.write(f"Epochs Trained       : {args.epochs}\n")
        f.write(f"Hidden Dim Channels  : {args.hidden_channels}\n\n")
        
        f.write("--- INPUT MODEL FEATURES ---\n")
        for i, col in enumerate(model_feature_cols, 1):
            f.write(f"  [{i:02d}] {col}\n")
        f.write("\n")
        
        f.write("--- TRAINING PERFORMANCE METRICS ---\n")
        for k, v in train_metrics.items():
            if isinstance(v, tuple):
                f.write(f"  {k}: {v[0]:.5f} (p-val: {v[1]:.4e})\n")
            else:
                f.write(f"  {k}: {v:.5f}\n")
        f.write("\n")
        
        f.write("--- OUT-OF-SAMPLE TEST PERFORMANCE METRICS ---\n")
        for k, v in test_metrics.items():
            if isinstance(v, tuple):
                f.write(f"  {k}: {v[0]:.5f} (p-val: {v[1]:.4e})\n")
            else:
                f.write(f"  {k}: {v:.5f}\n")
        f.write("========================================================\n")

    print("\nPipeline run accomplished successfully. Output logged.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Institutional Multi-Horizon Graph Neural Network Execution Driver")
    parser.add_argument("--epochs", type=int, default=3, help="Number of full dataset backprop epochs to train")
    parser.add_argument("--hidden_channels", type=int, default=32, help="Internal hidden channel feature dimensionality size for the RGATConv layers")
    parser.add_argument("--target", type=str, default="ret_0_5s", help="The target column to track and predict (e.g. ret_0_5s, ret_10_30s, ret_300_600s)")
    
    args = parser.parse_args()
    train_and_evaluate(args)