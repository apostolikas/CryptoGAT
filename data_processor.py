"""
data_processor.py
=================
Step 1: Ingest raw Databento MBP-10 Parquet files and produce strict
1-second time bars per symbol.
"""

import glob
import os
import re
import polars as pl

# Databento uses this as a sentinel for "no level present"
MAX_INT_SENTINEL = 9223372036854775807


def list_days(pattern: str) -> dict:
    """Map each trading day -> list of files, parsed from the YYYYMMDD in each
    filename. Lets the pipeline stream the month one day-block at a time instead
    of loading every file at once (the source of the OOM). Returns an ordered
    dict {date_str: [files]} sorted by date."""
    files = glob.glob(pattern)
    days = {}
    for f in files:
        m = re.search(r"(20\d{6})", os.path.basename(f))
        if m:
            days.setdefault(m.group(1), []).append(f)
    return dict(sorted(days.items()))


def load_equity_data(pattern_or_files) -> pl.LazyFrame:
    """
    Load raw MBP-10 Parquet files from a glob pattern OR an explicit file list.
    Deduplicates on (ts_event, symbol, sequence), keeping the last update,
    drops nulls on ts_event, and sorts by [symbol, ts_event].
    """
    if isinstance(pattern_or_files, str):
        files = glob.glob(pattern_or_files)
    else:
        files = list(pattern_or_files)
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern_or_files}")

    print(f"Planning query execution graph for {len(files)} files")

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

    # 1. Timestamp conversion
    lf = lf.with_columns(
        pl.from_epoch(pl.col("ts_event"), time_unit="ns").alias("ts_event"),
    )

    # 2. Price and Numeric scaling
    # Databento fixed-point prices are integers scaled by 1e9, so the real
    # price is (raw * 1e-9). The sentinel marks an empty book level -> null.
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

    numeric_cols = [c for c in ["size", "depth"] if c in schema_names]
    if numeric_cols:
        lf = lf.with_columns([
            pl.col(c).cast(pl.Float64).alias(c) for c in numeric_cols
        ])

    # 3. Dynamic 1-second grouping by symbol
    additional_preserves = ["depth", "size", "price"]
    snapshot_cols = [
        c for c in schema_names
        if ("px" in c or "sz" in c or "ct" in c or c in additional_preserves) and c != "ts_event"
    ]
    snapshot_exprs = [pl.col(c).last().alias(c) for c in snapshot_cols]

    # NOTE on Databento trade-side convention (FIX #6):
    # For Trade ('T') messages, `side` encodes the AGGRESSOR side: 'B' = a buy
    # aggressor lifting the ask, 'A' = a sell aggressor hitting the bid. The
    # previous code mapped side=='A' -> buy, which inverted CVD / aggressor
    # streaks / signed flow. Mapping is now B->buy, A->sell. If your specific
    # Databento dataset documents the opposite convention, swap these two lines.
    lf_grouped = lf.group_by_dynamic(
        "ts_event", every="1s", by="symbol"
    ).agg([
        *snapshot_exprs,
        pl.len().alias("msg_count"),
        pl.col("size").filter((pl.col("action") == "A") & (pl.col("side") == "B")).sum().alias("size_added_bid"),
        pl.col("size").filter((pl.col("action") == "A") & (pl.col("side") == "A")).sum().alias("size_added_ask"),
        pl.col("size").filter((pl.col("action") == "C") & (pl.col("side") == "B")).sum().alias("size_canceled_bid"),
        pl.col("size").filter((pl.col("action") == "C") & (pl.col("side") == "A")).sum().alias("size_canceled_ask"),
        pl.col("size").filter((pl.col("action") == "T") & (pl.col("side") == "B")).sum().alias("trade_size_buy"),
        pl.col("size").filter((pl.col("action") == "T") & (pl.col("side") == "A")).sum().alias("trade_size_sell"),
    ])

    df_agg = lf_grouped.collect()

    if len(df_agg) == 0:
        return df_agg

    # 4. Fill gaps on a contiguous 1-second grid, PER (symbol, day).
    # Critical for multi-day data: building the grid from global min->max ts
    # would forward-fill stale books across every overnight/weekend gap,
    # fabricating hours of constant "bars". Gridding per calendar day keeps the
    # fill within each session and leaves the cross-day gap empty (as it should).
    flow_cols = [
        "msg_count", "size_added_bid", "size_added_ask",
        "size_canceled_bid", "size_canceled_ask",
        "trade_size_buy", "trade_size_sell",
    ]

    df_agg = df_agg.with_columns(pl.col("ts_event").dt.date().alias("date"))
    parts = []
    for (sym, day), sym_df in df_agg.group_by(["symbol", "date"], maintain_order=True):
        min_ts = sym_df["ts_event"].min()
        max_ts = sym_df["ts_event"].max()
        full_grid = pl.DataFrame({
            "ts_event": pl.datetime_range(min_ts, max_ts, "1s", time_unit="ns", eager=True),
        }).with_columns([pl.lit(sym).alias("symbol"), pl.lit(day).alias("date")])

        sym_df = full_grid.join(sym_df, on=["ts_event", "symbol", "date"], how="left")
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
    price_col: str = "mid_price",
    prediction_time_col: str = "predictionTimestamp",
) -> pl.DataFrame:
    """
    Create forward return targets.

    FIX #1: targets are now built from `mid_price` (the same series all the
    features are derived from) rather than `price` (the last raw MBP-10 message
    price in the second, which could be a deep-level update far from the touch).
    This removes the feature/target inconsistency that was injecting noise.
    """
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

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

    # Day-local partition: forward returns must stay inside one session. Without
    # the date in the partition key, shift(-k) at the tail of each day would pull
    # the NEXT day's opening price -> an overnight-gap return contaminating the
    # target (and a subtle leak). With it, the last ~900s of each day get null
    # targets and are dropped downstream, which is correct: you genuinely cannot
    # observe a 900s-forward return at the close.
    if "date" not in lf.collect_schema().names():
        lf = lf.with_columns(pl.col(time_col).dt.date().alias("date"))
    grp = [ticker_col, "date"]

    lf = lf.sort([ticker_col, time_col]).with_columns([
        (pl.col(time_col) + pl.duration(seconds=1)).alias(prediction_time_col),
    ])

    target_exprs = []
    # Anchored at t+1; returns measured strictly between window bounds so the
    # chunks are non-overlapping and contain no lookahead leakage.
    for label, (start_s, end_s) in horizon_chunks_seconds.items():
        p_start = price.shift(-(1 + start_s)).over(grp)
        p_end = price.shift(-(1 + end_s)).over(grp)
        target_exprs.append(((p_end / p_start) - 1.0).alias(f"ret_{label}"))

    lf = lf.with_columns(target_exprs)
    return lf.collect().sort([prediction_time_col, ticker_col])


if __name__ == "__main__":
    pass