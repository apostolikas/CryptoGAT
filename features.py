import polars as pl

def compute_ewma_zscore(expr: pl.Expr, halflife: int) -> pl.Expr:
    """Computes streaming EWMA Z-score for a given expression."""
    mean = expr.ewm_mean(half_life=halflife)
    std = expr.ewm_std(half_life=halflife)
    return ((expr - mean) / (std + 1e-12))

def extract_features(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """
    Step 2: Extract High-Frequency Microstructure Features.
    Optimized for short horizons (1s to 10m) with Dual EWMA Regime Normalization.
    """
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    
    # -------------------------------------------------------------------------
    # Base Price and Returns
    # -------------------------------------------------------------------------
    lf = lf.with_columns([
        ((pl.col("bid_px_00") + pl.col("ask_px_00")) / 2.0).alias("mid_price"),
    ])
    
    lf = lf.with_columns([
        (((pl.col("mid_price") / pl.col("mid_price").shift(1).over("symbol")) - 1.0) * 1e4).alias("ret_bps_raw"),
    ])
    
    # -------------------------------------------------------------------------
    # Group A: Book Microstructure & Shape (Bounded / Raw)
    # -------------------------------------------------------------------------
    lf = lf.with_columns([
        pl.col("ret_bps_raw").alias("ret_bps").fill_null(0.0),
        ((pl.col("ask_px_00") - pl.col("bid_px_00")) / pl.col("mid_price") * 10000).alias("spread_bps"),
        
        ((pl.col("ask_px_00") * pl.col("bid_sz_00") + pl.col("bid_px_00") * pl.col("ask_sz_00")) / 
         (pl.col("bid_sz_00") + pl.col("ask_sz_00") + 1e-9)).alias("microprice"),
         
        ((pl.col("bid_sz_00") - pl.col("ask_sz_00")) / 
         (pl.col("bid_sz_00") + pl.col("ask_sz_00") + 1e-9)).alias("obi_L1"),
         
        # Limit Order Book Queue Shape / Convexity at the touch
        ((pl.col("bid_sz_00") - pl.col("bid_sz_01")) / (pl.col("bid_sz_00") + pl.col("bid_sz_01") + 1e-9)).alias("queue_convexity_bid"),
        ((pl.col("ask_sz_00") - pl.col("ask_sz_01")) / (pl.col("ask_sz_00") + pl.col("ask_sz_01") + 1e-9)).alias("queue_convexity_ask"),
         
        # Depth Ratios anchored strictly to L1
        (pl.col("bid_sz_01") / (pl.col("bid_sz_00") + 1e-9)).alias("depth_ratio_bid_L2"),
        (pl.col("ask_sz_01") / (pl.col("ask_sz_00") + 1e-9)).alias("depth_ratio_ask_L2"),
        (pl.col("bid_sz_04") / (pl.col("bid_sz_00") + 1e-9)).alias("depth_ratio_bid_L5"),
        (pl.col("ask_sz_04") / (pl.col("ask_sz_00") + 1e-9)).alias("depth_ratio_ask_L5"),
    ])
    
    lf = lf.with_columns([
        ((pl.col("microprice") - pl.col("mid_price")) / pl.col("mid_price") * 10000).alias("microprice_dev_bps"),
        
        # OBI L5
        ((pl.sum_horizontal([f"bid_sz_0{i}" for i in range(5)]) - pl.sum_horizontal([f"ask_sz_0{i}" for i in range(5)])) / 
         (pl.sum_horizontal([f"bid_sz_0{i}" for i in range(5)]) + pl.sum_horizontal([f"ask_sz_0{i}" for i in range(5)]) + 1e-9)).alias("obi_L5"),
         
        # Order Imbalance Momentum (OIM) - Acceleration of imbalance over 2s
        (pl.col("obi_L1") - pl.col("obi_L1").shift(2).over("symbol")).alias("obi_mom_2s")
    ])
    
    # -------------------------------------------------------------------------
    # Group B: Book State Raw (Top-K, Msg Count, OFI)
    # -------------------------------------------------------------------------
    lf = lf.with_columns([
        (pl.sum_horizontal([f"bid_sz_{i:02d}" for i in range(5)]) + 
         pl.sum_horizontal([f"ask_sz_{i:02d}" for i in range(5)])).alias("top5_book_size"),
    ])
    
    # Strict Cont-Kukanov Multilevel OFI
    ofi_exprs = []
    for lv in range(5):
        w = 1.0 / (1.0 + lv)
        bp, bs = pl.col(f"bid_px_0{lv}"), pl.col(f"bid_sz_0{lv}")
        ap, as_ = pl.col(f"ask_px_0{lv}"), pl.col(f"ask_sz_0{lv}")
        
        e_bid = pl.when(bp > bp.shift(1)).then(bs).when(bp < bp.shift(1)).then(-bs.shift(1)).otherwise(bs - bs.shift(1)).fill_null(0.0)
        e_ask = pl.when(ap < ap.shift(1)).then(as_).when(ap > ap.shift(1)).then(-as_.shift(1)).otherwise(as_ - as_.shift(1)).fill_null(0.0)
                 
        ofi_exprs.append(w * (e_bid - e_ask))
        
    lf = lf.with_columns([pl.sum_horizontal(ofi_exprs).alias("ofi_raw")])

    # -------------------------------------------------------------------------
    # Group C & D: Tape, Rates, and Flow Pressure
    # -------------------------------------------------------------------------
    lf = lf.with_columns([
        (pl.col("trade_size_buy") - pl.col("trade_size_sell")).alias("signed_trade_size"),
        (pl.col("size_added_bid") + pl.col("size_canceled_ask") - 
         (pl.col("size_added_ask") + pl.col("size_canceled_bid"))).alias("net_book_pressure"),
         
        # Bounded Ratios
        (pl.col("size_canceled_bid") / (pl.col("size_canceled_bid") + pl.col("size_canceled_ask") + 1e-9)).alias("bid_remove_ratio"),
        (pl.col("size_added_bid") / (pl.col("size_added_bid") + pl.col("size_added_ask") + 1e-9)).alias("bid_add_ratio"),
        
        # Level 1 Cancellation Intensity (Hidden Liquidity Pulling)
        ((pl.col("size_canceled_bid") + pl.col("size_canceled_ask")) / 
         (pl.col("bid_sz_00") + pl.col("ask_sz_00") + 1e-9)).alias("l1_cancel_intensity"),
         
        # Trade Aggression Ratio (Trades vs Limit Order Replenishment)
        ((pl.col("trade_size_buy") + pl.col("trade_size_sell")) / 
         (pl.col("size_added_bid") + pl.col("size_added_ask") + 1e-9)).alias("trade_to_order_ratio")
    ])

    # -------------------------------------------------------------------------
    # Group E & F: Multi-Window Aggregations (Tape, Volatility, Absorption)
    # -------------------------------------------------------------------------
    windows = [30, 60, 180, 300, 600]
    agg_exprs = []

    lf = lf.with_columns([
        pl.when(pl.col("ret_bps") > 0).then(pl.col("ret_bps")).otherwise(0.0).alias("move_up"),
        pl.when(pl.col("ret_bps") < 0).then(-pl.col("ret_bps")).otherwise(0.0).alias("move_dn"),
        
        # High-Frequency Microprice Momentum (Velocity over 2s, 5s, 10s)
        ((pl.col("microprice") / pl.col("microprice").shift(2).over("symbol")) - 1.0).alias("micro_mom_2s"),
        ((pl.col("microprice") / pl.col("microprice").shift(5).over("symbol")) - 1.0).alias("micro_mom_5s"),
        ((pl.col("microprice") / pl.col("microprice").shift(10).over("symbol")) - 1.0).alias("micro_mom_10s"),
    ])

    for w in windows:
        agg_exprs.append(pl.col("signed_trade_size").rolling_sum(window_size=w).over("symbol").alias(f"cvd_{w}s"))
        agg_exprs.append(pl.col("ret_bps").rolling_std(window_size=w).over("symbol").alias(f"rvol_{w}s"))
        
        buy_vol_w = pl.col("trade_size_buy").rolling_sum(window_size=w).over("symbol")
        sell_vol_w = pl.col("trade_size_sell").rolling_sum(window_size=w).over("symbol")
        move_up_w = pl.col("move_up").rolling_sum(window_size=w).over("symbol") + 1e-9
        move_dn_w = pl.col("move_dn").rolling_sum(window_size=w).over("symbol") + 1e-9
        
        agg_exprs.append((buy_vol_w / move_up_w).alias(f"buy_abs_{w}s"))
        agg_exprs.append((sell_vol_w / move_dn_w).alias(f"sell_abs_{w}s"))
        agg_exprs.append((buy_vol_w / (buy_vol_w + sell_vol_w + 1e-9)).alias(f"trade_side_{w}s"))

    lf = lf.with_columns(agg_exprs)

    # Volatility Spread Ratio & Variance Expansion
    lf = lf.with_columns([
        (pl.col("rvol_60s") / (pl.col("rvol_300s") + 1e-9)).alias("vol_expand_60_300"),
        (pl.col("rvol_180s") / (pl.col("rvol_600s") + 1e-9)).alias("vol_expand_180_600"),
        (pl.col("rvol_30s") / (pl.col("spread_bps") + 1e-9)).alias("vol_spread_ratio_30s"),
        (pl.col("rvol_180s") / (pl.col("spread_bps") + 1e-9)).alias("vol_spread_ratio_180s")
    ])

    # -------------------------------------------------------------------------
    # EWMA Z-Score Normalization (Dual Regime Normalization)
    # -------------------------------------------------------------------------
    pos_features = [
        "top5_book_size", "msg_count", "size_added_bid", "size_added_ask", 
        "size_canceled_bid", "size_canceled_ask", "trade_size_buy", "trade_size_sell",
        "l1_cancel_intensity", "trade_to_order_ratio"
    ]
    pos_features += [f"rvol_{w}s" for w in windows] + [f"buy_abs_{w}s" for w in windows] + [f"sell_abs_{w}s" for w in windows]
    
    signed_features = [
        "ofi_raw", "net_book_pressure", "obi_mom_2s", 
        "micro_mom_2s", "micro_mom_5s", "micro_mom_10s"
    ] + [f"cvd_{w}s" for w in windows]

    z_exprs = []
    z_cols = []  # Explicitly store names to avoid alias method collision
    halflives = [180, 600] 

    for hl in halflives:
        for f in pos_features:
            col_name = f"{f}_z{hl}"
            expr = (pl.col(f).fill_null(0.0) + 1.0).log()
            z_exprs.append(compute_ewma_zscore(expr, hl).over("symbol").alias(col_name))
            z_cols.append(col_name)
            
        for f in signed_features:
            col_name = f"{f}_z{hl}"
            expr = pl.col(f).fill_null(0.0)
            z_exprs.append(compute_ewma_zscore(expr, hl).over("symbol").alias(col_name))
            z_cols.append(col_name)

    lf = lf.with_columns(z_exprs)

    # -------------------------------------------------------------------------
    # Final Cleanup
    # -------------------------------------------------------------------------
    raw_bounded = [
        "spread_bps", "microprice", "microprice_dev_bps", "obi_L1", "obi_L5",
        "queue_convexity_bid", "queue_convexity_ask",
        "depth_ratio_bid_L2", "depth_ratio_ask_L2", "depth_ratio_bid_L5", "depth_ratio_ask_L5",
        "bid_remove_ratio", "bid_add_ratio",
        "vol_expand_60_300", "vol_expand_180_600",
        "vol_spread_ratio_30s", "vol_spread_ratio_180s"
    ] + [f"trade_side_{w}s" for w in windows]
    
    lf = lf.with_columns([
        pl.col(c).fill_nan(0.0).fill_null(0.0) for c in raw_bounded + z_cols
    ])

    lf = lf.drop(["ret_bps_raw", "move_up", "move_dn"])
    return lf.collect()