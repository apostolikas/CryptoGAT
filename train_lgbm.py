# """
# train_lgbm.py
# =============
# Per-horizon LightGBM baseline, deliberately apples-to-apples with the GNN.

# It reuses the EXACT same machinery as train_evaluate.py:
#   - build_dataset(): identical data, features, macro join, warm-up drop
#   - make_folds():    identical expanding-window walk-forward + purge
#   - cross_sectional_ic() / calculate_metrics(): identical metrics
#   - cross-sectionally demeaned target (matches the GNN's cross-sectional loss)

# The only difference is the model: a gradient-boosted tree per horizon that sees
# each (timestamp, name) row's own equity features from features.py. It does NOT
# get the macro nodes or the cross-asset graph. So:

#     if LGBM_IC >= GNN_IC, the graph + TCN are not earning their complexity.

# This is the cheapest, most decisive experiment you can run before investing
# more in the deep model.
# """

# import argparse
# import warnings
# import numpy as np
# import polars as pl
# import lightgbm as lgb

# warnings.filterwarnings("ignore", message="X does not have valid feature names")

# from main import (
#     build_dataset, make_folds, cross_sectional_ic, calculate_metrics,
#     precompute_daily_cache, stream_fold_frames, TARGET_COLS, HORIZON_END_S,
# )
# from backtest import run_backtests


# def _group_indices(df_sorted: pl.DataFrame):
#     """Index arrays of rows sharing each ts_event (df must be ts-sorted)."""
#     codes = df_sorted["ts_event"].to_physical().to_numpy()
#     cut = np.flatnonzero(np.diff(codes)) + 1
#     return np.split(np.arange(len(codes)), cut)


# def _dense_grids(df_te, preds_by_h, n_targets):
#     """Scatter per-row LGBM predictions back onto a dense (time x name) grid so
#     the SAME portfolio backtester used for the GNN can run on the tree's signal.
#     Returns (times, scores, valid, ret1s, spread_bps, day_code, n_eq)."""
#     ts_phys = df_te["ts_event"].to_physical().to_numpy()
#     ts_u, ti = np.unique(ts_phys, return_inverse=True)
#     syms = df_te["symbol"].to_numpy()
#     assets = np.array(sorted(set(syms.tolist())))
#     si_map = {s: i for i, s in enumerate(assets)}
#     si = np.array([si_map[s] for s in syms])
#     T, n_eq = len(ts_u), len(assets)

#     scores = np.full((T, n_eq, n_targets), np.nan)
#     for hi in range(n_targets):
#         scores[ti, si, hi] = preds_by_h[hi]
#     valid = np.zeros((T, n_eq), dtype=bool)
#     valid[ti, si] = True
#     ret1s = np.zeros((T, n_eq))
#     ret1s[ti, si] = np.nan_to_num(df_te["ret_bps"].to_numpy()) / 1e4
#     spread = np.full((T, n_eq), 1e6)                      # missing cells -> never traded
#     sp_src = df_te["spread_bps"].to_numpy() if "spread_bps" in df_te.columns else np.full(len(df_te), 2.0)
#     spread[ti, si] = np.nan_to_num(sp_src, nan=1e6)
#     # day id from the date of each unique timestamp (sorted), via run-length id
#     day_code = (df_te.select("ts_event").unique().sort("ts_event")
#                 .select(pl.col("ts_event").dt.date()).to_series().rle_id().to_numpy())
#     return np.arange(T), scores, valid, ret1s, spread, day_code, n_eq


# def _features(df: pl.DataFrame, feat_cols):
#     """Feature matrix (built ONCE per fold and reused across all horizons)."""
#     return np.nan_to_num(df.select(feat_cols).to_numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# def _target(df: pl.DataFrame, tcol):
#     """Cross-sectionally demeaned target (demean per ts uses only the
#     contemporaneous cross-section -> no lookahead)."""
#     y = (df.select((pl.col(tcol) - pl.col(tcol).mean().over("ts_event")).alias("_ydm"))["_ydm"]
#          .to_numpy().astype(np.float32))
#     return np.nan_to_num(y, nan=0.0)


# def run_lgbm_fold(df_tr, df_te, feat_cols, target_cols, horizon_end_s, micro_col, mom_col, args, fold_id):
#     if len(df_tr) < 1000 or len(df_te) < 100:
#         print(f"   [Fold {fold_id}] too little data (train={len(df_tr)}, test={len(df_te)}); skipping.")
#         return None

#     df_te = df_te.sort("ts_event")
#     groups = _group_indices(df_te)
#     Xte_full = np.nan_to_num(df_te.select(feat_cols).to_numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
#     micro = df_te[micro_col].to_numpy() if micro_col in df_te.columns else np.zeros(len(df_te))
#     mom = df_te[mom_col].to_numpy() if mom_col in df_te.columns else np.zeros(len(df_te))
#     pw_micro = [micro[g] for g in groups]
#     pw_mom = [mom[g] for g in groups]

#     params = dict(
#         objective="huber", alpha=1.0, n_estimators=args.n_estimators,
#         learning_rate=args.learning_rate, num_leaves=args.num_leaves,
#         min_child_samples=args.min_child_samples, subsample=0.8, subsample_freq=1,
#         colsample_bytree=0.5, reg_lambda=5.0, n_jobs=-1, verbosity=-1, random_state=fold_id,
#     )

#     res = {}
#     preds_by_h = []
#     Xtr = _features(df_tr, feat_cols)          # built ONCE per fold (was rebuilt 7x)
#     for hi, col in enumerate(target_cols):
#         ytr = _target(df_tr, col)
#         model = lgb.LGBMRegressor(**params)
#         model.fit(Xtr, ytr)
#         preds = model.predict(Xte_full)
#         preds_by_h.append(preds)
#         # cross-sectionally demeaned true target for the pooled view (ranking is
#         # invariant to the per-ts demean, so the cross-sectional IC is unaffected)
#         ytrue = _target(df_te, col)
#         pw_pred = [preds[g] for g in groups]
#         pw_true = [ytrue[g] for g in groups]
#         res[col] = {
#             "model": cross_sectional_ic(pw_pred, pw_true, overlap_lag=horizon_end_s[hi]),
#             "microprice_dev": cross_sectional_ic(pw_micro, pw_true, overlap_lag=horizon_end_s[hi]),
#             "mom_60s": cross_sectional_ic(pw_mom, pw_true, overlap_lag=horizon_end_s[hi]),
#             "pooled": calculate_metrics(ytrue, preds, overlap_lag=horizon_end_s[hi]),
#         }
#     del Xtr

#     # same net-of-cost portfolio backtest as the GNN, on the tree's signal
#     times, scores, valid, ret1s, spread, day_code, n_eq = _dense_grids(df_te, preds_by_h, len(target_cols))
#     res["backtest"] = run_backtests(
#         times, scores, valid, ret1s, spread, day_code, n_eq, horizon_end_s,
#         k=args.bt_k, buffer_m=(2 * args.bt_k if args.bt_buffer_m <= 0 else args.bt_buffer_m),
#         band=args.bt_band, edge_gate=args.bt_edge_gate,
#         score_scale=np.ones(len(target_cols)),       # LGBM preds are already fractional returns
#         extra_cost_bps=args.cost_bps,
#     )
#     print(f"   [Fold {fold_id}] trained {len(target_cols)} LGBM models on {len(df_tr)} rows.")
#     return res


# def train_lgbm(args):
#     if args.stream:
#         cached_days, base_feature_cols, macro_keys, _assets = precompute_daily_cache(args)
#         target_cols, horizon_end_s = TARGET_COLS, HORIZON_END_S
#         # column membership check needs a sample schema
#         sample_cols = pl.scan_parquet(
#             __import__("os").path.join(args.cache_dir, f"proc_{cached_days[0]}.parquet")
#         ).collect_schema().names()
#         feat_cols = [c for c in base_feature_cols if c in sample_cols]
#     else:
#         df_final, base_feature_cols, macro_keys, target_cols, horizon_end_s = build_dataset(args)
#         if df_final is None:
#             print("Could not load equity data.")
#             return
#         feat_cols = [c for c in base_feature_cols if c in df_final.columns]
#         sample_cols = df_final.columns

#     micro_col = "microprice_dev_bps"
#     mom_col = "mom_60s_z180" if "mom_60s_z180" in sample_cols else "rank_mom_60s"
#     print(f"   LGBM features: {len(feat_cols)} equity features from features.py (no macro/graph)")

#     if args.stream:
#         print(f"4. Streaming walk-forward: rolling {args.train_days or 'all'}d train, "
#               f"{args.test_days}d test, over {len(cached_days)} cached days")
#         folds_iter = stream_fold_frames(cached_days, args)
#     else:
#         folds, Ttot = make_folds(df_final, args.n_splits, args.purge_seconds, train_days=args.train_days)
#         print(f"4. Walk-forward: {args.n_splits} folds "
#               f"({'rolling ' + str(args.train_days) + 'd' if args.train_days > 0 else 'expanding'}), "
#               f"purge={args.purge_seconds}s")
#         folds_iter = ((k, dtr, dte) for k, dtr, dte in folds)

#     fold_results = []
#     for k, df_tr, df_te in folds_iter:
#         print(f"   Fold {k}: train rows={len(df_tr)}  test rows={len(df_te)}")
#         r = run_lgbm_fold(df_tr, df_te, feat_cols, target_cols, horizon_end_s, micro_col, mom_col, args, k)
#         if r is not None:
#             fold_results.append(r)
#         del df_tr, df_te

#     if not fold_results:
#         print("   No valid folds completed.")
#         return

#     out = "output_lgbm.txt"
#     with open(out, "w") as f:
#         f.write("========================================================\n")
#         f.write("   WALK-FORWARD LIGHTGBM BASELINE RUN REPORT\n")
#         f.write("========================================================\n\n")
#         f.write(f"Trees/fold/horizon : {args.n_estimators} (lr={args.learning_rate}, leaves={args.num_leaves})\n")
#         f.write(f"Warm-up dropped    : {args.warmup_seconds}s/symbol   Purge: {args.purge_seconds}s\n")
#         f.write(f"Walk-forward       : {len(fold_results)} folds completed ({'streaming roll ' + str(args.train_days) + 'd' if args.stream else ('rolling ' + str(args.train_days) + 'd' if args.train_days>0 else 'expanding')})\n")
#         f.write(f"Features           : {len(feat_cols)}\n\n")
#         f.write("=== CROSS-SECTIONAL RANK IC, mean +/- std ACROSS FOLDS (headline) ===\n")
#         f.write("    Compare model_IC directly against the GNN's output.txt table. If\n")
#         f.write("    LGBM >= GNN, the graph/TCN complexity is not adding alpha.\n")
#         f.write(f"    {'horizon':<12}{'model_IC':>16}{'IC_tstat':>10}{'micro_IC':>10}{'mom_IC':>10}\n")
#         for col in target_cols:
#             mic = np.array([r[col]["model"][0] for r in fold_results])
#             mir = np.array([r[col]["model"][1] for r in fold_results])
#             mcc = np.array([r[col]["microprice_dev"][0] for r in fold_results])
#             mmm = np.array([r[col]["mom_60s"][0] for r in fold_results])
#             f.write(f"    {col:<12}{mic.mean():>8.4f}+/-{mic.std():<5.4f}{mir.mean():>10.2f}"
#                     f"{mcc.mean():>10.4f}{mmm.mean():>10.4f}\n")
#         f.write("\n=== PER-FOLD MODEL CROSS-SECTIONAL IC ===\n")
#         for col in target_cols:
#             vals = "  ".join(f"{r[col]['model'][0]:+.4f}" for r in fold_results)
#             f.write(f"    {col:<12}: {vals}\n")

#         f.write(f"\n=== PORTFOLIO BACKTEST (market-neutral top/bottom-{args.bt_k}, "
#                 f"hold=horizon, costs ON) ===\n")
#         bm = 2 * args.bt_k if args.bt_buffer_m <= 0 else args.bt_buffer_m
#         f.write(f"    turnover controls: hysteresis exit rank m={bm}, no-trade band={args.bt_band}, "
#                 f"cost gate={args.bt_edge_gate}x round-trip spread, extra cost={args.cost_bps}bp/side\n")
#         f.write("    Same simulator as the GNN. Watch net_Sharpe; held->0 means the gate\n")
#         f.write("    refused ~all trades (nothing cleared the spread at that horizon).\n")
#         f.write(f"    {'horizon':<12}{'net_Shrp':>9}{'+/-':>7}{'gross_Shrp':>11}"
#                 f"{'netRet/yr':>10}{'turn/yr':>9}{'costDrag':>9}{'maxDD':>8}{'hit':>6}{'held':>6}\n")
#         for hi, col in enumerate(target_cols):
#             ns = np.array([r["backtest"][hi]["net_sharpe"] for r in fold_results])
#             gs = np.array([r["backtest"][hi]["gross_sharpe"] for r in fold_results])
#             nr = np.array([r["backtest"][hi]["net_ann_return"] for r in fold_results])
#             tu = np.array([r["backtest"][hi]["turnover_ann"] for r in fold_results])
#             cd = np.array([r["backtest"][hi]["cost_drag_ann"] for r in fold_results])
#             dd = np.array([r["backtest"][hi]["max_drawdown"] for r in fold_results])
#             ht = np.array([r["backtest"][hi]["hit_rate"] for r in fold_results])
#             hp = np.array([r["backtest"][hi]["avg_n_pos"] for r in fold_results])
#             f.write(f"    {col:<12}{ns.mean():>9.2f}{ns.std():>7.2f}{gs.mean():>11.2f}"
#                     f"{nr.mean():>10.3f}{tu.mean():>9.0f}{cd.mean():>9.3f}{dd.mean():>8.3f}"
#                     f"{ht.mean():>6.2f}{hp.mean():>6.1f}\n")
#         f.write("\n========================================================\n")
#     print(f"   Report written to {out}")


# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     # data / split args -- keep identical to the GNN run for a fair comparison
#     p.add_argument("--warmup_seconds", type=int, default=900)
#     p.add_argument("--session_start", type=str, default="13:30")
#     p.add_argument("--session_end", type=str, default="20:00")
#     p.add_argument("--include_extended", action="store_true")
#     p.add_argument("--n_splits", type=int, default=5)
#     p.add_argument("--train_days", type=int, default=0,
#                    help="Rolling-window train size in trading days (0 = expanding). Use e.g. 5 on a month.")
#     p.add_argument("--stream", action="store_true", help="Stream the month day-by-day (cache then walk forward).")
#     p.add_argument("--cache_dir", type=str, default="/tmp/obp_cache")
#     p.add_argument("--rebuild_cache", action="store_true")
#     p.add_argument("--test_days", type=int, default=1)
#     p.add_argument("--equity_glob", type=str, default=None)
#     p.add_argument("--purge_seconds", type=int, default=900)
#     # lightgbm args
#     p.add_argument("--n_estimators", type=int, default=100)
#     p.add_argument("--learning_rate", type=float, default=0.03)
#     p.add_argument("--num_leaves", type=int, default=31)
#     p.add_argument("--min_child_samples", type=int, default=200)
#     # backtest args -- keep identical to the GNN run for a fair comparison
#     p.add_argument("--bt_k", type=int, default=5)
#     p.add_argument("--bt_buffer_m", type=int, default=0, help="Hysteresis exit rank (>k); 0 => 2*bt_k.")
#     p.add_argument("--bt_band", type=float, default=0.0, help="No-trade band on L1 weight change.")
#     p.add_argument("--bt_edge_gate", type=float, default=0.0, help="Cost gate: |edge| >= gate * round-trip spread.")
#     p.add_argument("--cost_bps", type=float, default=0.0)
#     args = p.parse_args()
#     train_lgbm(args)
    
# # nohup python train_lgbm.py --stream --train_days 5 --test_days 2 --cache_dir /home/apostolikas/rwa/gnn/obp_cache --bt_k 5 --bt_buffer_m 10 --bt_band 0.1 --bt_edge_gate 1.0 > output_lgbm_nohup.log 2>&1 &

# # python train_lgbm.py --stream --train_days 5 --test_days 2 --cache_dir /home/apostolikas/rwa/gnn/obp_cache --bt_k 5 --bt_buffer_m 10 --bt_band 0.1 --bt_edge_gate 1.0


"""
train_lgbm.py
=============
Per-horizon LightGBM baseline, deliberately apples-to-apples with the GNN.

It reuses the EXACT same machinery as train_evaluate.py:
  - build_dataset(): identical data, features, macro join, warm-up drop
  - make_folds():    identical expanding-window walk-forward + purge
  - cross_sectional_ic() / calculate_metrics(): identical metrics
  - cross-sectionally demeaned target (matches the GNN's cross-sectional loss)

The only difference is the model: a gradient-boosted tree per horizon that sees
each (timestamp, name) row's own equity features from features.py. It does NOT
get the macro nodes or the cross-asset graph. So:

    if LGBM_IC >= GNN_IC, the graph + TCN are not earning their complexity.

This is the cheapest, most decisive experiment you can run before investing
more in the deep model.
"""

import argparse
import warnings
import numpy as np
import polars as pl
import lightgbm as lgb

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from main import (
    build_dataset, make_folds, cross_sectional_ic, calculate_metrics,
    precompute_daily_cache, stream_fold_frames, TARGET_COLS, HORIZON_END_S,
)
from backtest import run_backtests
from market_making import run_market_making


def _group_indices(df_sorted: pl.DataFrame):
    """Index arrays of rows sharing each ts_event (df must be ts-sorted)."""
    codes = df_sorted["ts_event"].to_physical().to_numpy()
    cut = np.flatnonzero(np.diff(codes)) + 1
    return np.split(np.arange(len(codes)), cut)


def _dense_grids(df_te, preds_by_h, n_targets):
    """Scatter per-row LGBM predictions back onto a dense (time x name) grid so
    the SAME backtester used for the GNN can run on the tree's signal. Also
    returns the queue/flow grids the market-making sim needs.
    Returns (times, scores, valid, ret1s, spread_bps, day_code, n_eq, mm) where
    mm = dict(bid_sz, ask_sz, buy_vol, sell_vol)."""
    ts_phys = df_te["ts_event"].to_physical().to_numpy()
    ts_u, ti = np.unique(ts_phys, return_inverse=True)
    syms = df_te["symbol"].to_numpy()
    assets = np.array(sorted(set(syms.tolist())))
    si_map = {s: i for i, s in enumerate(assets)}
    si = np.array([si_map[s] for s in syms])
    T, n_eq = len(ts_u), len(assets)

    scores = np.full((T, n_eq, n_targets), np.nan)
    for hi in range(n_targets):
        scores[ti, si, hi] = preds_by_h[hi]
    valid = np.zeros((T, n_eq), dtype=bool)
    valid[ti, si] = True
    ret1s = np.zeros((T, n_eq))
    ret1s[ti, si] = np.nan_to_num(df_te["ret_bps"].to_numpy()) / 1e4
    spread = np.full((T, n_eq), 1e6)                      # missing cells -> never traded
    sp_src = df_te["spread_bps"].to_numpy() if "spread_bps" in df_te.columns else np.full(len(df_te), 2.0)
    spread[ti, si] = np.nan_to_num(sp_src, nan=1e6)
    day_code = (df_te.select("ts_event").unique().sort("ts_event")
                .select(pl.col("ts_event").dt.date()).to_series().rle_id().to_numpy())

    def _grid(col, default):
        g = np.full((T, n_eq), default, dtype=np.float64)
        if col in df_te.columns:
            g[ti, si] = np.nan_to_num(df_te[col].to_numpy(), nan=default)
        return g
    mm = {
        "bid_sz": _grid("bid_sz_00", 0.0),
        "ask_sz": _grid("ask_sz_00", 0.0),
        "buy_vol": _grid("trade_size_buy", 0.0),
        "sell_vol": _grid("trade_size_sell", 0.0),
    }
    return np.arange(T), scores, valid, ret1s, spread, day_code, n_eq, mm


def _features(df: pl.DataFrame, feat_cols):
    """Feature matrix (built ONCE per fold and reused across all horizons)."""
    return np.nan_to_num(df.select(feat_cols).to_numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _target(df: pl.DataFrame, tcol):
    """Cross-sectionally demeaned target (demean per ts uses only the
    contemporaneous cross-section -> no lookahead)."""
    y = (df.select((pl.col(tcol) - pl.col(tcol).mean().over("ts_event")).alias("_ydm"))["_ydm"]
         .to_numpy().astype(np.float32))
    return np.nan_to_num(y, nan=0.0)


def run_lgbm_fold(df_tr, df_te, feat_cols, target_cols, horizon_end_s, micro_col, mom_col, args, fold_id):
    if len(df_tr) < 1000 or len(df_te) < 100:
        print(f"   [Fold {fold_id}] too little data (train={len(df_tr)}, test={len(df_te)}); skipping.")
        return None

    df_te = df_te.sort("ts_event")
    groups = _group_indices(df_te)
    Xte_full = np.nan_to_num(df_te.select(feat_cols).to_numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    micro = df_te[micro_col].to_numpy() if micro_col in df_te.columns else np.zeros(len(df_te))
    mom = df_te[mom_col].to_numpy() if mom_col in df_te.columns else np.zeros(len(df_te))
    pw_micro = [micro[g] for g in groups]
    pw_mom = [mom[g] for g in groups]

    params = dict(
        objective="huber", alpha=1.0, n_estimators=args.n_estimators,
        learning_rate=args.learning_rate, num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.5, reg_lambda=5.0, n_jobs=-1, verbosity=-1, random_state=fold_id,
    )

    res = {}
    preds_by_h = []
    Xtr = _features(df_tr, feat_cols)          # built ONCE per fold (was rebuilt 7x)
    for hi, col in enumerate(target_cols):
        ytr = _target(df_tr, col)
        model = lgb.LGBMRegressor(**params)
        model.fit(Xtr, ytr)
        preds = model.predict(Xte_full)
        preds_by_h.append(preds)
        # cross-sectionally demeaned true target for the pooled view (ranking is
        # invariant to the per-ts demean, so the cross-sectional IC is unaffected)
        ytrue = _target(df_te, col)
        pw_pred = [preds[g] for g in groups]
        pw_true = [ytrue[g] for g in groups]
        res[col] = {
            "model": cross_sectional_ic(pw_pred, pw_true, overlap_lag=horizon_end_s[hi]),
            "microprice_dev": cross_sectional_ic(pw_micro, pw_true, overlap_lag=horizon_end_s[hi]),
            "mom_60s": cross_sectional_ic(pw_mom, pw_true, overlap_lag=horizon_end_s[hi]),
            "pooled": calculate_metrics(ytrue, preds, overlap_lag=horizon_end_s[hi]),
        }
    del Xtr

    # same net-of-cost portfolio backtest as the GNN, on the tree's signal
    times, scores, valid, ret1s, spread, day_code, n_eq, mm = _dense_grids(df_te, preds_by_h, len(target_cols))
    res["backtest"] = run_backtests(
        times, scores, valid, ret1s, spread, day_code, n_eq, horizon_end_s,
        k=args.bt_k, buffer_m=(2 * args.bt_k if args.bt_buffer_m <= 0 else args.bt_buffer_m),
        band=args.bt_band, edge_gate=args.bt_edge_gate,
        score_scale=np.ones(len(target_cols)),       # LGBM preds are already fractional returns
        extra_cost_bps=args.cost_bps,
    )
    if args.mm:
        res["mm"] = run_market_making(
            times, scores, valid, ret1s, spread, mm["bid_sz"], mm["ask_sz"],
            mm["buy_vol"], mm["sell_vol"], day_code, n_eq,
            signal_hi=args.mm_signal_hi, score_scale=1.0,     # LGBM preds already fractional returns
            kappa=args.mm_kappa, inv_max=args.mm_inv_max, lam=args.mm_lambda, fee_bps=args.cost_bps,
        )
    print(f"   [Fold {fold_id}] trained {len(target_cols)} LGBM models on {len(df_tr)} rows.")
    return res


def train_lgbm(args):
    if args.stream:
        cached_days, base_feature_cols, macro_keys, _assets = precompute_daily_cache(args)
        target_cols, horizon_end_s = TARGET_COLS, HORIZON_END_S
        # column membership check needs a sample schema
        sample_cols = pl.scan_parquet(
            __import__("os").path.join(args.cache_dir, f"proc_{cached_days[0]}.parquet")
        ).collect_schema().names()
        feat_cols = [c for c in base_feature_cols if c in sample_cols]
    else:
        df_final, base_feature_cols, macro_keys, target_cols, horizon_end_s = build_dataset(args)
        if df_final is None:
            print("Could not load equity data.")
            return
        feat_cols = [c for c in base_feature_cols if c in df_final.columns]
        sample_cols = df_final.columns

    micro_col = "microprice_dev_bps"
    mom_col = "mom_60s_z180" if "mom_60s_z180" in sample_cols else "rank_mom_60s"
    print(f"   LGBM features: {len(feat_cols)} equity features from features.py (no macro/graph)")

    if args.stream:
        print(f"4. Streaming walk-forward: rolling {args.train_days or 'all'}d train, "
              f"{args.test_days}d test, over {len(cached_days)} cached days")
        folds_iter = stream_fold_frames(cached_days, args)
    else:
        folds, Ttot = make_folds(df_final, args.n_splits, args.purge_seconds, train_days=args.train_days)
        print(f"4. Walk-forward: {args.n_splits} folds "
              f"({'rolling ' + str(args.train_days) + 'd' if args.train_days > 0 else 'expanding'}), "
              f"purge={args.purge_seconds}s")
        folds_iter = ((k, dtr, dte) for k, dtr, dte in folds)

    fold_results = []
    for k, df_tr, df_te in folds_iter:
        print(f"   Fold {k}: train rows={len(df_tr)}  test rows={len(df_te)}")
        r = run_lgbm_fold(df_tr, df_te, feat_cols, target_cols, horizon_end_s, micro_col, mom_col, args, k)
        if r is not None:
            fold_results.append(r)
        del df_tr, df_te

    if not fold_results:
        print("   No valid folds completed.")
        return

    out = "output_lgbm.txt"
    with open(out, "w") as f:
        f.write("========================================================\n")
        f.write("   WALK-FORWARD LIGHTGBM BASELINE RUN REPORT\n")
        f.write("========================================================\n\n")
        f.write(f"Trees/fold/horizon : {args.n_estimators} (lr={args.learning_rate}, leaves={args.num_leaves})\n")
        f.write(f"Warm-up dropped    : {args.warmup_seconds}s/symbol   Purge: {args.purge_seconds}s\n")
        f.write(f"Walk-forward       : {len(fold_results)} folds completed ({'streaming roll ' + str(args.train_days) + 'd' if args.stream else ('rolling ' + str(args.train_days) + 'd' if args.train_days>0 else 'expanding')})\n")
        f.write(f"Features           : {len(feat_cols)}\n\n")
        f.write("=== CROSS-SECTIONAL RANK IC, mean +/- std ACROSS FOLDS (headline) ===\n")
        f.write("    Compare model_IC directly against the GNN's output.txt table. If\n")
        f.write("    LGBM >= GNN, the graph/TCN complexity is not adding alpha.\n")
        f.write(f"    {'horizon':<12}{'model_IC':>16}{'IC_tstat':>10}{'micro_IC':>10}{'mom_IC':>10}\n")
        for col in target_cols:
            mic = np.array([r[col]["model"][0] for r in fold_results])
            mir = np.array([r[col]["model"][1] for r in fold_results])
            mcc = np.array([r[col]["microprice_dev"][0] for r in fold_results])
            mmm = np.array([r[col]["mom_60s"][0] for r in fold_results])
            f.write(f"    {col:<12}{mic.mean():>8.4f}+/-{mic.std():<5.4f}{mir.mean():>10.2f}"
                    f"{mcc.mean():>10.4f}{mmm.mean():>10.4f}\n")
        f.write("\n=== PER-FOLD MODEL CROSS-SECTIONAL IC ===\n")
        for col in target_cols:
            vals = "  ".join(f"{r[col]['model'][0]:+.4f}" for r in fold_results)
            f.write(f"    {col:<12}: {vals}\n")

        f.write(f"\n=== PORTFOLIO BACKTEST (market-neutral top/bottom-{args.bt_k}, "
                f"hold=horizon, costs ON) ===\n")
        bm = 2 * args.bt_k if args.bt_buffer_m <= 0 else args.bt_buffer_m
        f.write(f"    turnover controls: hysteresis exit rank m={bm}, no-trade band={args.bt_band}, "
                f"cost gate={args.bt_edge_gate}x round-trip spread, extra cost={args.cost_bps}bp/side\n")
        f.write("    Same simulator as the GNN. Watch net_Sharpe; held->0 means the gate\n")
        f.write("    refused ~all trades (nothing cleared the spread at that horizon).\n")
        f.write(f"    {'horizon':<12}{'net_Shrp':>9}{'+/-':>7}{'gross_Shrp':>11}"
                f"{'netRet/yr':>10}{'turn/yr':>9}{'costDrag':>9}{'maxDD':>8}{'hit':>6}{'held':>6}\n")
        for hi, col in enumerate(target_cols):
            ns = np.array([r["backtest"][hi]["net_sharpe"] for r in fold_results])
            gs = np.array([r["backtest"][hi]["gross_sharpe"] for r in fold_results])
            nr = np.array([r["backtest"][hi]["net_ann_return"] for r in fold_results])
            tu = np.array([r["backtest"][hi]["turnover_ann"] for r in fold_results])
            cd = np.array([r["backtest"][hi]["cost_drag_ann"] for r in fold_results])
            dd = np.array([r["backtest"][hi]["max_drawdown"] for r in fold_results])
            ht = np.array([r["backtest"][hi]["hit_rate"] for r in fold_results])
            hp = np.array([r["backtest"][hi]["avg_n_pos"] for r in fold_results])
            f.write(f"    {col:<12}{ns.mean():>9.2f}{ns.std():>7.2f}{gs.mean():>11.2f}"
                    f"{nr.mean():>10.3f}{tu.mean():>9.0f}{cd.mean():>9.3f}{dd.mean():>8.3f}"
                    f"{ht.mean():>6.2f}{hp.mean():>6.1f}\n")

        if args.mm and any("mm" in r for r in fold_results):
            sig_hi_name = target_cols[args.mm_signal_hi]
            f.write(f"\n=== MARKET-MAKING A/B (passive quoting; skew signal = {sig_hi_name}) ===\n")
            f.write(f"    fill kappa={args.mm_kappa} (<1 = back-of-queue, conservative), "
                    f"inv_max={args.mm_inv_max}, inv_aversion lambda={args.mm_lambda}, fee={args.cost_bps}bp\n")
            f.write("    BASELINE = inventory-managed maker with NO signal (symmetric quoting).\n")
            f.write("    SIGNAL   = same maker, but the forecast skews quotes to dodge toxic fills.\n")
            f.write("    The SIGNAL's value to a maker is the LIFT of signal over baseline. Absolute\n")
            f.write("    levels depend on the fill assumption (kappa) -- read the lift, not the level.\n")
            f.write(f"    {'arm':<10}{'Sharpe':>9}{'+/-':>7}{'pnl_bps/nm/day':>16}{'spreadCap':>11}{'adverse':>10}{'avgInv':>9}{'fills':>8}\n")
            for arm in ["baseline", "signal"]:
                sh = np.array([r["mm"][arm]["sharpe"] for r in fold_results if "mm" in r])
                pb = np.array([r["mm"][arm]["pnl_bps_per_name_day"] for r in fold_results if "mm" in r])
                scp = np.array([r["mm"][arm]["spread_capture"] for r in fold_results if "mm" in r])
                adv = np.array([r["mm"][arm]["adverse_sel"] for r in fold_results if "mm" in r])
                inv = np.array([r["mm"][arm]["avg_abs_inv"] for r in fold_results if "mm" in r])
                fl = np.array([r["mm"][arm]["fill_rate"] for r in fold_results if "mm" in r])
                f.write(f"    {arm:<10}{sh.mean():>9.2f}{sh.std():>7.2f}{pb.mean():>16.3f}"
                        f"{scp.mean():>11.5f}{adv.mean():>10.5f}{inv.mean():>9.2f}{fl.mean():>8.2f}\n")
            sig_pb = np.array([r["mm"]["signal"]["pnl_bps_per_name_day"] for r in fold_results if "mm" in r])
            base_pb = np.array([r["mm"]["baseline"]["pnl_bps_per_name_day"] for r in fold_results if "mm" in r])
            lift = sig_pb - base_pb
            f.write(f"    SIGNAL LIFT (per-name bps/day): {lift.mean():+.3f} +/- {lift.std():.3f} "
                    f"across {len(lift)} folds; positive & stable => the forecast is worth money to a maker.\n")
            f.write("    NOTE: a making backtest depends on the fill model; re-run with --mm_kappa 0.1 and\n"
                    "    0.6 to bracket the result. The LIFT is far more robust than the absolute P&L.\n")

        f.write("\n========================================================\n")
    print(f"   Report written to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # data / split args -- keep identical to the GNN run for a fair comparison
    p.add_argument("--warmup_seconds", type=int, default=900)
    p.add_argument("--session_start", type=str, default="13:30")
    p.add_argument("--session_end", type=str, default="20:00")
    p.add_argument("--include_extended", action="store_true")
    p.add_argument("--n_splits", type=int, default=4)
    p.add_argument("--train_days", type=int, default=0,
                   help="Rolling-window train size in trading days (0 = expanding). Use e.g. 5 on a month.")
    p.add_argument("--stream", action="store_true", help="Stream the month day-by-day (cache then walk forward).")
    p.add_argument("--cache_dir", type=str, default="/tmp/obp_cache")
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--test_days", type=int, default=2)
    p.add_argument("--equity_glob", type=str, default=None)
    p.add_argument("--purge_seconds", type=int, default=900)
    # lightgbm args
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--learning_rate", type=float, default=0.03)
    p.add_argument("--num_leaves", type=int, default=31)
    p.add_argument("--min_child_samples", type=int, default=200)
    # backtest args -- keep identical to the GNN run for a fair comparison
    p.add_argument("--bt_k", type=int, default=5)
    p.add_argument("--bt_buffer_m", type=int, default=0, help="Hysteresis exit rank (>k); 0 => 2*bt_k.")
    p.add_argument("--bt_band", type=float, default=0.0, help="No-trade band on L1 weight change.")
    p.add_argument("--bt_edge_gate", type=float, default=0.0, help="Cost gate: |edge| >= gate * round-trip spread.")
    p.add_argument("--cost_bps", type=float, default=0.0)
    # market-making A/B
    p.add_argument("--mm", action="store_true", help="Run the passive market-making A/B (signal-skew vs symmetric).")
    p.add_argument("--mm_signal_hi", type=int, default=0, help="Horizon index used as the quote-skew signal (0 = 0-5s).")
    p.add_argument("--mm_kappa", type=float, default=0.25, help="Fill aggressiveness (<1 = back-of-queue, conservative).")
    p.add_argument("--mm_inv_max", type=float, default=5.0, help="Per-name inventory limit (quote-size units).")
    p.add_argument("--mm_lambda", type=float, default=0.5, help="Inventory aversion (skew strength to mean-revert q).")
    args = p.parse_args()
    train_lgbm(args)
