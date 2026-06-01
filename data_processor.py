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
    Deduplicates on (ts_event, symbol, sequence) keeping the last update,
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
    """
    schema_names = lf.collect_schema().names()

    # ── 1. Timestamp conversion ──────────────────────────────────────────
    lf = lf.with_columns(
        pl.from_epoch(pl.col("ts_event"), time_unit="ns").alias("ts_event"),
    )

    # ── 2. Price scaling ─────────────────────────────────────────────────
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

    # Cast size columns to float for downstream arithmetic
    size_cols = [c for c in schema_names if "sz" in c or c == "size"]
    if size_cols:
        lf = lf.with_columns([
            pl.col(c).cast(pl.Float64).alias(c) for c in size_cols
        ])

    # ── 3. Dynamic 1-second grouping by symbol ───────────────────────────
    snapshot_cols = [c for c in schema_names if ("px" in c or "sz" in c or "ct" in c)]
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

        full_grid = pl.DataFrame({
            "ts_event": pl.datetime_range(min_ts, max_ts, "1s", eager=True),
        }).with_columns(pl.lit(sym).alias("symbol"))

        sym_df = full_grid.join(sym_df, on=["ts_event", "symbol"], how="left")

        sym_df = sym_df.with_columns([
            pl.col(c).forward_fill() for c in snapshot_cols
        ]).with_columns([
            pl.col(c).fill_null(0) for c in flow_cols
        ])

        parts.append(sym_df)

    df_out = pl.concat(parts).sort(["symbol", "ts_event"])
    return df_out

if __name__ == "__main__":
    pass
