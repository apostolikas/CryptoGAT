import polars as pl


def engineer_targets(
    df: pl.DataFrame | pl.LazyFrame,
    time_col: str = "ts_event",
    ticker_col: str = "symbol",
    price_col: str = "microprice",
    prediction_time_col: str = "predictionTimestamp",
) -> pl.DataFrame:
    """Create forward return targets based on the 1-second price."""
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    
    # NEW CONTIGUOUS TARGETS
    horizon_chunks_seconds = {
        "0_5s": (0, 5),
        "5_10s": (5, 10),
        "10_30s": (10, 30),
        "30_60s": (30, 60),
        "60_300s": (60, 300),
        "300_600s": (300, 600),
        "600_900s": (600, 900),

    }
    
    price = pl.col(price_col).cast(pl.Float64)

    lf = lf.sort([ticker_col, time_col]).with_columns([
        price.alias(price_col),
        (pl.col(time_col) + pl.duration(seconds=1)).alias(prediction_time_col),
    ])

    target_exprs = []
    # By anchoring to `t+1` and calculating strictly between window bounds,
    # overlap and lookahead leakage are mathematically impossible.
    for label, (start_s, end_s) in horizon_chunks_seconds.items():
        p_start = price.shift(-(1 + start_s)).over(ticker_col)
        p_end = price.shift(-(1 + end_s)).over(ticker_col)
        target_exprs.append(((p_end / p_start) - 1.0).alias(f"ret_{label}"))

    lf = lf.with_columns(target_exprs)
    return lf.collect().sort([prediction_time_col, ticker_col])