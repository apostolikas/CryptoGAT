"""
data_processor.py
=================
Step 1: Ingest raw Databento MBP-10 Parquet files and produce strict
1-second time bars per symbol.
"""

import glob
import polars as pl

# Databento uses this as a sentinel for "no level present"
MAX_INT_SENTINEL = 9223372036854775807

def load_equity_data(pattern: str) -> pl.LazyFrame:
    """
    Load raw MBP-10 Parquet files matching the given glob pattern.
    Deduplicates on (ts_event, symbol, sequence), keeping the last update,
    drops nulls on ts_event, and sorts by [symbol, ts_event].
    """
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    print(f"Planning query execution graph for {len(files)} equity files")

    lf = (
        pl.scan_parquet(files)
        .drop_nulls(subset=["ts_event"])
        .unique(subset=["ts_event", "symbol", "sequence"], keep="last")
        .sort(["symbol", "ts_event"])
    )
    return lf

def process_raw_data(lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Aggregate raw ticks into 1-second bars per symbol.
    Preserves and casts key metrics: depth, size, and price.
    ts_event is treated as a nanosecond epoch datetime.
    """
    schema_names = lf.collect_schema().names()

    # ── 1. Timestamp conversion ──────────────────────────────────────────
    lf = lf.with_columns(
        pl.from_epoch(pl.col("ts_event"), time_unit="ns").alias("ts_event"),
    )

    # ── 2. Price and Numeric scaling ─────────────────────────────────────
    # Adjust prices by dividing by 1e-9 (standard Databento format)
    price_cols = [c for c in schema_names if "px" in c or c == "price"]
    if price_cols:
        lf = lf.with_columns([
            (
                pl.when(pl.col(c).cast(pl.Float64) == MAX_INT_SENTINEL)
                .then(None)
                .otherwise(pl.col(c).cast(pl.Float64) * 1e-9)
                .alias(c)
            )
            for c in price_cols
        ])

    # Cast other requested variables to Float64 so they handle downstream math properly
    numeric_cols = [c for c in ["size", "depth"] if c in schema_names]
    if numeric_cols:
        lf = lf.with_columns([
            pl.col(c).cast(pl.Float64).alias(c) for c in numeric_cols
        ])

    # ── 3. Dynamic 1-second grouping by symbol ───────────────────────────
    # Dynamically extract depth, size, and price columns plus any level snapshots (px, sz, ct)
    additional_preserves = ["depth", "size", "price"]
    snapshot_cols = [
        c for c in schema_names 
        if ("px" in c or "sz" in c or "ct" in c or c in additional_preserves) and c != "ts_event"
    ]
    snapshot_exprs = [pl.col(c).last().alias(c) for c in snapshot_cols]

    lf_grouped = lf.group_by_dynamic(
        "ts_event", every="1s", by="symbol"
    ).agg([
        *snapshot_exprs,
        pl.len().alias("msg_count"),
        pl.col("size").filter((pl.col("action") == "A") & (pl.col("side") == "B")).sum().alias("size_added_bid"),
        pl.col("size").filter((pl.col("action") == "A") & (pl.col("side") == "A")).sum().alias("size_added_ask"),
        pl.col("size").filter((pl.col("action") == "C") & (pl.col("side") == "B")).sum().alias("size_canceled_bid"),
        pl.col("size").filter((pl.col("action") == "C") & (pl.col("side") == "A")).sum().alias("size_canceled_ask"),
        pl.col("size").filter((pl.col("action") == "T") & (pl.col("side") == "A")).sum().alias("trade_size_buy"),
        pl.col("size").filter((pl.col("action") == "T") & (pl.col("side") == "B")).sum().alias("trade_size_sell"),
    ])

    df_agg = lf_grouped.collect()

    if len(df_agg) == 0:
        return df_agg

    # ── 4. Fill gaps per symbol on a contiguous 1-second grid ────────────
    flow_cols = [
        "msg_count", "size_added_bid", "size_added_ask",
        "size_canceled_bid", "size_canceled_ask",
        "trade_size_buy", "trade_size_sell",
    ]

    symbols = df_agg["symbol"].unique().sort().to_list()
    parts = []

    for sym in symbols:
        sym_df = df_agg.filter(pl.col("symbol") == sym)
        min_ts = sym_df["ts_event"].min()
        max_ts = sym_df["ts_event"].max()

        # time_unit="ns" ensures grid types align perfectly to prevent type mismatch crashes on join
        full_grid = pl.DataFrame({
            "ts_event": pl.datetime_range(min_ts, max_ts, "1s", time_unit="ns", eager=True),
        }).with_columns(pl.lit(sym).alias("symbol"))

        sym_df = full_grid.join(sym_df, on=["ts_event", "symbol"], how="left")

        # forward_fill handles snapshot_cols, which includes depth, size, and price
        sym_df = sym_df.with_columns([
            pl.col(c).forward_fill() for c in snapshot_cols
        ]).with_columns([
            pl.col(c).fill_null(0) for c in flow_cols
        ])

        parts.append(sym_df)

    df_out = pl.concat(parts).sort(["symbol", "ts_event"])
    return df_out


def engineer_targets(
    df: pl.DataFrame | pl.LazyFrame,
    time_col: str = "ts_event",
    ticker_col: str = "symbol",
    price_col: str = "price",
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

if __name__ == "__main__":
    pass
