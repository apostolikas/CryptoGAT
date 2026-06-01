import polars as pl
import numpy as np

def extract_features(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """
    Step 2: Extract Microstructure Features using Polars
    Incorporating core ideas from seattle_features.py (Groups A-F).
    """
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    
    # -------------------------------------------------------------------------
    # Group A: Book Microstructure
    # -------------------------------------------------------------------------
    lf = lf.with_columns([
        ((pl.col("bid_px_00") + pl.col("ask_px_00")) / 2.0).alias("mid_price"),
    ])
    
    lf = lf.with_columns([
        ((pl.col("ask_px_00") - pl.col("bid_px_00")) / pl.col("mid_price") * 10000).alias("spread_bps"),
        
        ((pl.col("ask_px_00") * pl.col("bid_sz_00") + pl.col("bid_px_00") * pl.col("ask_sz_00")) / 
         (pl.col("bid_sz_00") + pl.col("ask_sz_00")).fill_nan(None)).alias("microprice"),
         
        ((pl.col("bid_sz_00") - pl.col("ask_sz_00")) / 
         (pl.col("bid_sz_00") + pl.col("ask_sz_00")).fill_nan(None)).alias("obi_level_L1"),
         
        (pl.col("bid_sz_01") / pl.col("bid_sz_00").fill_nan(None)).alias("depth_ratio_bid_L2"),
        (pl.col("ask_sz_01") / pl.col("ask_sz_00").fill_nan(None)).alias("depth_ratio_ask_L2"),
        (pl.col("bid_sz_04") / pl.col("bid_sz_00").fill_nan(None)).alias("depth_ratio_bid_L5"),
        (pl.col("ask_sz_04") / pl.col("ask_sz_00").fill_nan(None)).alias("depth_ratio_ask_L5"),
    ])
    
    lf = lf.with_columns([
        ((pl.col("microprice") - pl.col("mid_price")) / pl.col("mid_price") * 10000).alias("microprice_dev_bps"),
    ])
    
    # -------------------------------------------------------------------------
    # Group B: Book State Raw (top-K size, OFI)
    # -------------------------------------------------------------------------
    top5_cols = []
    for i in range(5):
        top5_cols.extend([pl.col(f"bid_sz_{i:02d}"), pl.col(f"ask_sz_{i:02d}")])
    
    top10_cols = []
    for i in range(10):
        top10_cols.extend([pl.col(f"bid_sz_{i:02d}"), pl.col(f"ask_sz_{i:02d}")])
        
    lf = lf.with_columns([
        pl.sum_horizontal(top5_cols).alias("top5_book_size"),
        pl.sum_horizontal(top10_cols).alias("top10_book_size"),
    ])
    
    # Multilevel OFI
    ofi_exprs = []
    for lv in range(5):
        bid_px = pl.col(f"bid_px_{lv:02d}")
        bid_sz = pl.col(f"bid_sz_{lv:02d}")
        ask_px = pl.col(f"ask_px_{lv:02d}")
        ask_sz = pl.col(f"ask_sz_{lv:02d}")
        
        bid_px_prev = bid_px.shift(1)
        bid_sz_prev = bid_sz.shift(1)
        ask_px_prev = ask_px.shift(1)
        ask_sz_prev = ask_sz.shift(1)
        
        e_bid = pl.when(bid_px > bid_px_prev).then(bid_sz) \
                 .when(bid_px < bid_px_prev).then(-bid_sz_prev) \
                 .otherwise(bid_sz - bid_sz_prev)
                 
        e_ask = pl.when(ask_px < ask_px_prev).then(ask_sz) \
                 .when(ask_px > ask_px_prev).then(-ask_sz_prev) \
                 .otherwise(ask_sz - ask_sz_prev)
                 
        w = 1.0 / (1.0 + lv)
        ofi_exprs.append((w * (e_bid - e_ask)).fill_null(0))
        
    lf = lf.with_columns([
        pl.sum_horizontal(ofi_exprs).alias("ofi_multilevel")
    ])
    
    # -------------------------------------------------------------------------
    # Group C: Trade Tape (CVD)
    # -------------------------------------------------------------------------
    # CVD = rolling sum of (trade_buy - trade_sell)
    lf = lf.with_columns([
         (pl.col("bid_sz_00") + pl.col("ask_sz_00"))).alias("obi_level_L1"),
         
        # Depth Ratios (Bounded 0 to 1)
        (pl.col("bid_sz_00") / (pl.col("bid_sz_00") + pl.col("bid_sz_01"))).alias("depth_ratio_bid_L2"),
        (pl.col("ask_sz_00") / (pl.col("ask_sz_00") + pl.col("ask_sz_01"))).alias("depth_ratio_ask_L2"),
    ])

    lf = lf.with_columns([
        # Microprice deviation
        (((pl.col("microprice") - ((pl.col("bid_px_00") + pl.col("ask_px_00")) * 0.5)) / 
          ((pl.col("bid_px_00") + pl.col("ask_px_00")) * 0.5)) * 1e4).alias("microprice_dev_bps")
    ])

    # -------------------------------------------------------------------------
    # Group B & C: Book Size, OFI, and Tape
    # -------------------------------------------------------------------------
    lf = lf.with_columns([
        # Top 5 and Top 10 aggregate book sizes
        (pl.sum_horizontal([f"bid_sz_0{i}" for i in range(5)]) + 
         pl.sum_horizontal([f"ask_sz_0{i}" for i in range(5)])).alias("top5_book_size"),
        (pl.sum_horizontal([f"bid_sz_0{i}" for i in range(10)]) + 
         pl.sum_horizontal([f"ask_sz_0{i}" for i in range(10)])).alias("top10_book_size"),
    ])

    lf = lf.with_columns([
        # Multilevel OFI proxy (Net Additions/Cancellations)
        (pl.col("size_added_bid") - pl.col("size_canceled_bid") - 
         (pl.col("size_added_ask") - pl.col("size_canceled_ask"))).alias("ofi_multilevel_proxy"),
         
        # Trade Tape / Cumulative Volume Deltas
        (pl.col("trade_size_buy") - pl.col("trade_size_sell")).alias("signed_trade_size"),
        
        # Net book pressure
        (pl.col("size_added_bid") + pl.col("size_canceled_ask") - 
         (pl.col("size_added_ask") + pl.col("size_canceled_bid"))).alias("net_book_pressure"),
    ])

    # -------------------------------------------------------------------------
    # Rolling & Stationarity Adjustments (Group F & EWMA Z-Scores)
    # -------------------------------------------------------------------------
    # Raw returns
    lf = lf.with_columns([
        ((pl.col("microprice") / pl.col("microprice").shift(1).over("symbol") - 1) * 1e4).alias("ret_bps")
    ])

    halflife = 180  # 3 minute halflife for EWMA z-scores

    lf = lf.with_columns([
        # Rolling CVDs
        pl.col("signed_trade_size").rolling_sum(window_size=30).over("symbol").alias("cvd_30s"),
        pl.col("signed_trade_size").rolling_sum(window_size=180).over("symbol").alias("cvd_180s"),
        
        # Realized volatility
        pl.col("ret_bps").rolling_std(window_size=60).over("symbol").alias("rvol_60s"),
    ])

    # -------------------------------------------------------------------------
    # EWMA Z-Score normalization per-symbol to ensure stationarity
    # -------------------------------------------------------------------------
    pos_features = ["top5_book_size", "top10_book_size", "rvol_60s", "spread_bps"]
    signed_features = ["ofi_multilevel_proxy", "signed_trade_size", "net_book_pressure", "cvd_30s", "cvd_180s"]

    exprs = []
    # For strictly positive unbounded features, log1p to compress right-tails before z-scoring
    for f in pos_features:
        expr = pl.col(f).fill_null(0)
        expr = (expr + 1).log()
        z_expr = compute_ewma_zscore(expr, halflife).over("symbol").alias(f"{f}_z")
        exprs.append(z_expr)
        
    # For signed unbounded features, z-score directly
    for f in signed_features:
        expr = pl.col(f).fill_null(0)
        z_expr = compute_ewma_zscore(expr, halflife).over("symbol").alias(f"{f}_z")
        exprs.append(z_expr)

    lf = lf.with_columns(exprs)

    # -------------------------------------------------------------------------
    # Clean up NaNs / Nulls that result from EWMA warmup
    # -------------------------------------------------------------------------
    engineered_cols = [
        "spread_bps_z", "microprice", "microprice_dev_bps", "obi_level_L1", "depth_ratio_bid_L2", "depth_ratio_ask_L2",
        "top5_book_size_z", "top10_book_size_z", "ofi_multilevel_proxy_z", "signed_trade_size_z",
        "net_book_pressure_z", "ret_bps", "cvd_30s_z", "cvd_180s_z", "rvol_60s_z"
    ]
    
    lf = lf.with_columns([
        pl.col(c).fill_nan(0).fill_null(0) for c in engineered_cols
    ])

    return lf.collect()

if __name__ == "__main__":
    pass
