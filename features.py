import math
import polars as pl

PI_HALF = math.pi / 2.0
EPS = 1e-9


def compute_ewma_zscore(expr: pl.Expr, halflife: int) -> pl.Expr:
    mean = expr.ewm_mean(half_life=halflife)
    std = expr.ewm_std(half_life=halflife)
    return ((expr - mean) / (std + 1e-12))


def extract_features(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    # 1. Base Prices & Returns
    lf = lf.with_columns([
        ((pl.col("bid_px_00") + pl.col("ask_px_00")) / 2.0).alias("mid_price"),
    ])
    lf = lf.with_columns([
        (((pl.col("mid_price") / pl.col("mid_price").shift(1).over("symbol")) - 1.0) * 1e4).alias("ret_bps_raw"),
        ((pl.col("ask_px_00") - pl.col("bid_px_00")) / pl.col("mid_price") * 10000).alias("spread_bps"),
    ])

    # 2. Advanced LOB Shape & Physics
    sum_bid_sz = pl.sum_horizontal([f"bid_sz_0{i}" for i in range(5)]) + 1e-9
    sum_ask_sz = pl.sum_horizontal([f"ask_sz_0{i}" for i in range(5)]) + 1e-9

    entropy_bid_exprs, entropy_ask_exprs, dwi_bid_exprs, dwi_ask_exprs = [], [], [], []
    for i in range(5):
        p_bid = pl.col(f"bid_sz_0{i}") / sum_bid_sz
        p_ask = pl.col(f"ask_sz_0{i}") / sum_ask_sz
        entropy_bid_exprs.append(pl.when(p_bid > 0).then(-p_bid * p_bid.log()).otherwise(0.0))
        entropy_ask_exprs.append(pl.when(p_ask > 0).then(-p_ask * p_ask.log()).otherwise(0.0))
        dwi_bid_exprs.append(pl.col(f"bid_sz_0{i}") / (pl.col("mid_price") - pl.col(f"bid_px_0{i}") + 1e-5))
        dwi_ask_exprs.append(pl.col(f"ask_sz_0{i}") / (pl.col(f"ask_px_0{i}") - pl.col("mid_price") + 1e-5))

    lf = lf.with_columns([
        (sum_bid_sz + sum_ask_sz).alias("top5_book_size"),
        pl.sum_horizontal(entropy_bid_exprs).alias("depth_entropy_bid"),
        pl.sum_horizontal(entropy_ask_exprs).alias("depth_entropy_ask"),
        ((pl.sum_horizontal(dwi_bid_exprs) - pl.sum_horizontal(dwi_ask_exprs)) /
         (pl.sum_horizontal(dwi_bid_exprs) + pl.sum_horizontal(dwi_ask_exprs) + 1e-9)).alias("distance_weighted_imbalance"),
        ((pl.col("bid_sz_01") - pl.col("bid_sz_00")) / (pl.col("bid_px_00") - pl.col("bid_px_01") + 1e-5)).alias("depth_slope_bid"),
        ((pl.col("ask_sz_01") - pl.col("ask_sz_00")) / (pl.col("ask_px_01") - pl.col("ask_px_00") + 1e-5)).alias("depth_slope_ask"),
        (pl.col("bid_sz_02") - 2 * pl.col("bid_sz_01") + pl.col("bid_sz_00")).alias("depth_curvature_bid"),
        (pl.col("ask_sz_02") - 2 * pl.col("ask_sz_01") + pl.col("ask_sz_00")).alias("depth_curvature_ask"),
    ])

    lf = lf.with_columns([
        pl.col("ret_bps_raw").alias("ret_bps").fill_null(0.0),
        ((pl.col("ask_px_00") * pl.col("bid_sz_00") + pl.col("bid_px_00") * pl.col("ask_sz_00")) /
         (pl.col("bid_sz_00") + pl.col("ask_sz_00") + 1e-9)).alias("microprice"),
        ((pl.col("bid_sz_00") - pl.col("ask_sz_00")) / (pl.col("bid_sz_00") + pl.col("ask_sz_00") + 1e-9)).alias("obi_L1"),
        (pl.col("spread_bps") - pl.col("spread_bps").shift(1).over("symbol")).alias("spread_change"),
    ])

    # obi_L5: full-book imbalance (was referenced downstream but never defined)
    lf = lf.with_columns([
        ((sum_bid_sz - sum_ask_sz) / (sum_bid_sz + sum_ask_sz)).alias("obi_L5"),
        ((pl.col("microprice") - pl.col("mid_price")) / pl.col("mid_price") * 10000).alias("microprice_dev_bps"),
        (pl.col("spread_change") == 0).cast(pl.Float32).rolling_sum(30).over("symbol").alias("spread_duration"),
        pl.col("mid_price").rolling_std(30).over("symbol").alias("quote_stability"),
        (pl.col("mid_price") != pl.col("mid_price").shift(1).over("symbol")).cast(pl.Float32).rolling_sum(30).over("symbol").alias("price_level_flip_count"),
    ])

    # ---- Kalman / residual feature (FIX #5) ---------------------------------
    # A local-level (random-walk + noise) Kalman filter has a steady-state gain
    # that makes its one-step state estimate an EWMA of the observation. The
    # standardized innovation (observation - prior estimate)/innovation_std is
    # therefore an EWMA-innovation z-score. This is the `kalman_z_score_*` the
    # graph builder/base-feature filter already expect (previously never built).
    lf = lf.with_columns([
        (pl.col("mid_price") - pl.col("mid_price").ewm_mean(half_life=30).over("symbol")).alias("_kalman_innov"),
    ])
    lf = lf.with_columns([
        (pl.col("_kalman_innov") / (pl.col("_kalman_innov").ewm_std(half_life=30).over("symbol") + 1e-9)
         ).alias("kalman_z_score_mid"),
    ])

    lf = lf.with_columns([
        (pl.col("bid_sz_00") - pl.col("bid_sz_00").shift(1).over("symbol")).alias("queue_change_bid"),
        (pl.col("ask_sz_00") - pl.col("ask_sz_00").shift(1).over("symbol")).alias("queue_change_ask"),
        (pl.col("msg_count") - pl.col("msg_count").shift(1).over("symbol")).alias("message_count_acceleration"),
    ])

    lf = lf.with_columns([
        pl.when(pl.col("queue_change_bid") < 0).then(pl.col("queue_change_bid").abs()).otherwise(0).alias("queue_depletion_bid"),
        pl.when(pl.col("queue_change_bid") > 0).then(pl.col("queue_change_bid")).otherwise(0).alias("queue_replenishment_bid"),
        pl.when(pl.col("queue_change_ask") < 0).then(pl.col("queue_change_ask").abs()).otherwise(0).alias("queue_depletion_ask"),
        pl.when(pl.col("queue_change_ask") > 0).then(pl.col("queue_change_ask")).otherwise(0).alias("queue_replenishment_ask"),
        (pl.col("size_canceled_bid") > 2 * pl.col("size_canceled_bid").rolling_mean(30).over("symbol")).cast(pl.Float32).alias("cancel_burst_bid"),
        (pl.col("size_canceled_ask") > 2 * pl.col("size_canceled_ask").rolling_mean(30).over("symbol")).cast(pl.Float32).alias("cancel_burst_ask"),
    ])

    # 3. Order Flow Imbalance (level-weighted), isolated per symbol
    ofi_exprs = []
    for lv in range(5):
        w = 1.0 / (1.0 + lv)
        bp, bs = pl.col(f"bid_px_0{lv}"), pl.col(f"bid_sz_0{lv}")
        ap, as_ = pl.col(f"ask_px_0{lv}"), pl.col(f"ask_sz_0{lv}")

        bp_prev = bp.shift(1).over("symbol")
        bs_prev = bs.shift(1).over("symbol")
        ap_prev = ap.shift(1).over("symbol")
        as_prev = as_.shift(1).over("symbol")

        e_bid = pl.when(bp > bp_prev).then(bs).when(bp < bp_prev).then(-bs_prev).otherwise(bs - bs_prev).fill_null(0.0)
        e_ask = pl.when(ap < ap_prev).then(as_).when(ap > ap_prev).then(-as_prev).otherwise(as_ - as_prev).fill_null(0.0)
        ofi_exprs.append(w * (e_bid - e_ask))

    lf = lf.with_columns([pl.sum_horizontal(ofi_exprs).alias("ofi_raw")])

    lf = lf.with_columns([
        (pl.col("trade_size_buy") - pl.col("trade_size_sell")).alias("signed_trade_size"),
        (pl.col("trade_size_buy") + pl.col("trade_size_sell")).alias("trade_volume"),
        (pl.col("trade_size_buy") > 0).cast(pl.Float32).rolling_sum(60).over("symbol").alias("trade_count_buy"),
        (pl.col("trade_size_sell") > 0).cast(pl.Float32).rolling_sum(60).over("symbol").alias("trade_count_sell"),
        (pl.col("trade_size_buy") > pl.col("trade_size_sell")).cast(pl.Float32).rolling_sum(15).over("symbol").alias("aggressor_streak_buy"),
        (pl.col("trade_size_sell") > pl.col("trade_size_buy")).cast(pl.Float32).rolling_sum(15).over("symbol").alias("aggressor_streak_sell"),
        (pl.col("trade_size_buy") + pl.col("trade_size_sell") >
         (pl.col("trade_size_buy") + pl.col("trade_size_sell")).rolling_quantile(0.95, window_size=180).over("symbol")
         ).cast(pl.Float32).alias("large_trade_flag"),
    ])

    # ---- VPIN / order-flow toxicity ----------------------------------------
    # Volume-synchronised probability of informed trading, approximated on the
    # 1s grid: rolling |buy-sell| / rolling total volume over a 60s bucket.
    lf = lf.with_columns([
        (pl.col("signed_trade_size").abs().rolling_sum(60).over("symbol") /
         (pl.col("trade_volume").rolling_sum(60).over("symbol") + EPS)).alias("vpin_60s"),
    ])

    # ---- Hawkes-style self-exciting message / trade intensity ---------------
    # Short-EWMA / long-EWMA of arrivals proxies the self-excitation ratio of a
    # Hawkes process: >1 means a recent burst (clustering), <1 means cooling.
    lf = lf.with_columns([
        (pl.col("msg_count").ewm_mean(half_life=5).over("symbol") /
         (pl.col("msg_count").ewm_mean(half_life=60).over("symbol") + EPS)).alias("msg_intensity_ratio"),
        (pl.col("trade_volume").ewm_mean(half_life=5).over("symbol") /
         (pl.col("trade_volume").ewm_mean(half_life=60).over("symbol") + EPS)).alias("trade_intensity_ratio"),
    ])

    # ---- Realized variance, bipower variation & jump ratio ------------------
    # RV captures total quadratic variation; BV is jump-robust. RV-BV isolates
    # the jump component; jump_ratio in [0,1] flags discontinuous moves.
    lf = lf.with_columns([
        (pl.col("ret_bps") ** 2).rolling_sum(60).over("symbol").alias("rv_60s"),
        (PI_HALF * (pl.col("ret_bps").abs() * pl.col("ret_bps").abs().shift(1).over("symbol"))
         ).rolling_sum(60).over("symbol").alias("bv_60s"),
    ])
    lf = lf.with_columns([
        ((pl.col("rv_60s") - pl.col("bv_60s")).clip(lower_bound=0.0) /
         (pl.col("rv_60s") + EPS)).alias("jump_ratio"),
    ])

    # ---- Amihud illiquidity & Kyle's lambda (price impact) ------------------
    # Amihud: |return| per unit traded volume (higher = more illiquid / impactful).
    # Kyle lambda: rolling regression slope of return on signed order flow,
    # cov(ret, signed)/var(signed) over 60s = marginal price impact per unit flow.
    lf = lf.with_columns([
        (pl.col("ret_bps").abs() / (pl.col("trade_volume") + 1.0)).rolling_mean(60).over("symbol").alias("amihud_60s"),
    ])
    lf = lf.with_columns([
        (pl.col("ret_bps") * pl.col("signed_trade_size")).rolling_mean(60).over("symbol").alias("_xy"),
        pl.col("ret_bps").rolling_mean(60).over("symbol").alias("_mx"),
        pl.col("signed_trade_size").rolling_mean(60).over("symbol").alias("_my"),
        (pl.col("signed_trade_size") ** 2).rolling_mean(60).over("symbol").alias("_yy"),
    ])
    lf = lf.with_columns([
        ((pl.col("_xy") - pl.col("_mx") * pl.col("_my")) /
         (pl.col("_yy") - pl.col("_my") ** 2 + EPS)).alias("kyle_lambda_60s"),
    ])

    windows = [30, 60, 180, 300, 600]
    agg_exprs = []
    for w in windows:
        agg_exprs.append(pl.col("signed_trade_size").rolling_sum(window_size=w).over("symbol").alias(f"cvd_{w}s"))
        agg_exprs.append(pl.col("ret_bps").rolling_std(window_size=w).over("symbol").alias(f"rvol_{w}s"))
    lf = lf.with_columns(agg_exprs)

    pos_features = [
        "queue_depletion_bid", "queue_depletion_ask", "queue_replenishment_bid", "queue_replenishment_ask",
        "trade_size_buy", "trade_size_sell",
        "vpin_60s", "rv_60s", "bv_60s", "amihud_60s",
        "msg_intensity_ratio", "trade_intensity_ratio",
    ] + [f"rvol_{w}s" for w in windows]
    signed_features = [
        "ofi_raw", "message_count_acceleration", "spread_change",
        "depth_slope_bid", "depth_slope_ask", "depth_curvature_bid", "depth_curvature_ask",
        "kyle_lambda_60s", "jump_ratio",
    ] + [f"cvd_{w}s" for w in windows]

    z_exprs, z_cols = [], []
    for hl in [180, 600]:
        for f in pos_features:
            col_name = f"{f}_z{hl}"
            z_exprs.append(compute_ewma_zscore((pl.col(f).fill_null(0.0).clip(lower_bound=0.0) + 1.0).log(), hl).over("symbol").alias(col_name))
            z_cols.append(col_name)
        for f in signed_features:
            col_name = f"{f}_z{hl}"
            z_exprs.append(compute_ewma_zscore(pl.col(f).fill_null(0.0), hl).over("symbol").alias(col_name))
            z_cols.append(col_name)

    lf = lf.with_columns(z_exprs)

    cs_features = [
        pl.col("ret_bps").rank(descending=True).over("ts_event").alias("rank_ret_1s"),
        pl.col("ofi_raw_z180").rank(descending=True).over("ts_event").alias("rank_ofi_z180"),
        pl.col("obi_L1").rank(descending=True).over("ts_event").alias("rank_obi_L1"),
        pl.col("spread_bps").rank(descending=False).over("ts_event").alias("rank_spread_bps"),
        pl.col("cvd_60s_z180").rank(descending=True).over("ts_event").alias("rank_cvd_60s"),
    ]
    lf = lf.with_columns(cs_features)

    raw_bounded = [
        "spread_bps", "microprice", "microprice_dev_bps", "obi_L1", "obi_L5",
        "depth_entropy_bid", "depth_entropy_ask",
        "distance_weighted_imbalance", "cancel_burst_bid", "cancel_burst_ask", "large_trade_flag",
        "trade_count_buy", "trade_count_sell", "aggressor_streak_buy", "aggressor_streak_sell",
        "spread_duration", "quote_stability", "price_level_flip_count", "top5_book_size",
        "kalman_z_score_mid", "vpin_60s", "jump_ratio", "msg_intensity_ratio", "trade_intensity_ratio",
        "rank_ret_1s", "rank_ofi_z180", "rank_obi_L1", "rank_spread_bps", "rank_cvd_60s",
    ]

    lf = lf.with_columns([pl.col(c).fill_nan(0.0).fill_null(0.0) for c in raw_bounded + z_cols])
    lf = lf.drop([
        "ret_bps_raw", "queue_change_bid", "queue_change_ask",
        "_kalman_innov", "_xy", "_mx", "_my", "_yy",
    ])
    return lf.collect()