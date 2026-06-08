"""
backtest.py
===========
Institutional-grade event-driven backtester for the cross-sectional signal.

Models what an IC number cannot:
  - ONE book carried through time (inventory); trade only the DELTA each step.
  - Realistic cost: cross HALF the spread on every share traded (entry, every
    rebalance, end-of-day flatten) + optional commission/slippage in bps.
  - Market-neutral: equal-weight top-k long / bottom-k short, gross 1.0, net ~0.
  - Intraday discipline: flat at each session close, never holds across days.

Turnover controls (the levers that decide whether a real-but-fast signal can be
harvested net of cost):
  1. HYSTERESIS BUFFER (`buffer_m`): enter a name on rank<=k, but only EXIT it
     once it falls out of rank<=m (m>k). Stops churn from names oscillating
     around the k boundary -- "only trade structural rank breakdowns."
  2. NO-TRADE BAND (`band`): skip any rebalance whose total weight change is
     below `band` (L1 weight units) -- kills micro-adjustments.
  3. COST-AWARE ENTRY GATE (`edge_gate`): a name is eligible only if its
     PREDICTED move exceeds `edge_gate` x round-trip spread. Needs the score in
     return units, so a per-horizon `score_scale` maps raw score -> expected
     fractional return. Refuses trades that cannot pay for themselves.

Inputs are sanitized (returns clipped to a sane intraday band, spreads guarded)
so a single bad pre-market tick cannot detonate the P&L.

No leakage: weights at t use only the score and spread known at t; P&L is the
realized mid return from t+1..t+H.
"""

import numpy as np

TRADING_DAYS = 252
RET_CLIP = 0.02          # |1s mid return| > 2% is bad data (crossed/zero book), not a move


def _select_side(order, rankpos, prev_held, k, m):
    """Choose up to k names for one side with hysteresis.
    order     : eligible indices sorted best-first for this side
    rankpos   : dict idx -> rank (0 = best)
    prev_held : set of indices currently held on this side
    Keep held names still inside top-m (best k of them), then fill remaining
    slots from top-k, then from the rest by rank."""
    n = len(order)
    kk, mm = min(k, n), min(m, n)
    if kk == 0:
        return []
    top_k = set(order[:kk])
    chosen = [i for i in order if i in prev_held and rankpos[i] < mm][:kk]   # retain incumbents in top-m
    if len(chosen) < kk:
        for i in order:                       # add fresh entrants from top-k
            if len(chosen) >= kk:
                break
            if i in top_k and i not in chosen:
                chosen.append(i)
    if len(chosen) < kk:
        for i in order:                       # backfill if still short
            if len(chosen) >= kk:
                break
            if i not in chosen:
                chosen.append(i)
    return chosen[:kk]


def simulate_long_short(times, scores_h, valid, ret1s, spread_bps, day_code,
                        n_eq, rebal, k=5, buffer_m=None, band=0.0,
                        edge_gate=0.0, score_scale=1.0, extra_cost_bps=0.0):
    """One-horizon long/short simulation. See module docstring for the levers."""
    T = ret1s.shape[0]
    m = buffer_m if (buffer_m is not None and buffer_m > k) else k

    # ---- sanitize inputs: one bad pre-market tick must not blow up P&L -------
    ret1s = np.clip(np.nan_to_num(ret1s, nan=0.0, posinf=0.0, neginf=0.0), -RET_CLIP, RET_CLIP)
    spread_bps = np.nan_to_num(spread_bps, nan=1e6, posinf=1e6, neginf=1e6)
    half_spread = np.clip(spread_bps, 0.0, None) / 2.0 / 1e4 + extra_cost_bps / 1e4

    score_dense = np.full((T, n_eq), np.nan)
    valid_dense = np.zeros((T, n_eq), dtype=bool)
    score_dense[times] = scores_h
    valid_dense[times] = valid

    net_rets, gross_rets, turnovers, gross_exp, net_exp, n_pos = [], [], [], [], [], []
    intervals_per_day = []

    for d in np.unique(day_code):
        di = np.where(day_code == d)[0]
        if di.size < rebal + 2:
            continue
        a, b = di[0], di[-1]
        w = np.zeros(n_eq)
        n_int_day = 0
        for t in range(a, b - rebal + 1, rebal):
            pos = np.searchsorted(times, t, side="right") - 1
            if pos >= 0 and day_code[times[pos]] == d:
                s = score_dense[times[pos]]
                trad = valid_dense[times[pos]] & np.isfinite(s)
                if edge_gate > 0:                       # cost-aware entry gate
                    exp_ret = np.abs(s * score_scale)
                    roundtrip = 2.0 * half_spread[t]    # cross spread on entry AND exit
                    trad = trad & (exp_ret >= edge_gate * roundtrip)
                elig = np.where(trad)[0]
                if elig.size >= 2:
                    long_order = elig[np.argsort(-s[elig])]
                    short_order = elig[np.argsort(s[elig])]
                    lpos = {idx: r for r, idx in enumerate(long_order)}
                    spos = {idx: r for r, idx in enumerate(short_order)}
                    prev_long = set(np.where(w > 0)[0])
                    prev_short = set(np.where(w < 0)[0])
                    longs = _select_side(long_order, lpos, prev_long, k, m)
                    shorts = _select_side(short_order, spos, prev_short, k, m)
                    shorts = [i for i in shorts if i not in set(longs)]
                    target = np.zeros(n_eq)
                    if longs and shorts:
                        target[longs] = 0.5 / len(longs)
                        target[shorts] = -0.5 / len(shorts)
                else:
                    target = np.zeros(n_eq)
            else:
                target = w                              # no fresh signal -> hold

            if band > 0 and np.sum(np.abs(target - w)) < band:
                target = w                              # no-trade band: skip micro-moves

            dw = target - w
            cost = float(np.sum(np.abs(dw) * half_spread[t]))
            seg = ret1s[t + 1:t + rebal + 1]
            cumret = np.prod(1.0 + seg, axis=0) - 1.0
            gross = float(target @ cumret)
            net_rets.append(gross - cost)
            gross_rets.append(gross)
            turnovers.append(float(np.sum(np.abs(dw))))
            gross_exp.append(float(np.sum(np.abs(target))))
            net_exp.append(float(np.sum(target)))
            n_pos.append(int(np.sum(np.abs(target) > 1e-9)))
            w = target
            n_int_day += 1

        flat_cost = float(np.sum(np.abs(w) * half_spread[b]))   # pay to flatten at close
        if net_rets:
            net_rets[-1] -= flat_cost
            turnovers[-1] += float(np.sum(np.abs(w)))
        if n_int_day:
            intervals_per_day.append(n_int_day)

    n = len(net_rets)
    if n < 5:
        return {"n_intervals": n, "net_sharpe": 0.0, "net_ann_return": 0.0,
                "gross_sharpe": 0.0, "turnover_ann": 0.0, "cost_drag_ann": 0.0,
                "max_drawdown": 0.0, "hit_rate": 0.0, "avg_gross_exp": 0.0,
                "avg_net_exp": 0.0, "avg_n_pos": 0.0, "rebal_s": rebal}

    net = np.asarray(net_rets)
    gross = np.asarray(gross_rets)
    ipd = float(np.mean(intervals_per_day)) if intervals_per_day else 1.0
    per_year = ipd * TRADING_DAYS
    net_mu, net_sd = net.mean(), net.std()
    gross_mu, gross_sd = gross.mean(), gross.std()
    equity = np.cumsum(net)
    peak = np.maximum.accumulate(equity)

    return {
        "n_intervals": n,
        "rebal_s": rebal,
        "net_sharpe": float(net_mu / net_sd * np.sqrt(per_year)) if net_sd > 0 else 0.0,
        "net_ann_return": float(net_mu * per_year),
        "gross_sharpe": float(gross_mu / gross_sd * np.sqrt(per_year)) if gross_sd > 0 else 0.0,
        "turnover_ann": float(np.mean(turnovers) * per_year),
        "cost_drag_ann": float((gross_mu - net_mu) * per_year),
        "max_drawdown": float(np.max(peak - equity)),
        "hit_rate": float(np.mean(net > 0)),
        "avg_gross_exp": float(np.mean(gross_exp)),
        "avg_net_exp": float(np.mean(net_exp)),
        "avg_n_pos": float(np.mean(n_pos)),          # avg names held (0 => gate killed all trades)
        "net_series": net,
    }


def run_backtests(times, scores, valid, ret1s, spread_bps, day_code, n_eq,
                  horizon_end_s, k=5, buffer_m=None, band=0.0, edge_gate=0.0,
                  score_scale=None, extra_cost_bps=0.0):
    """Run the long/short sim once per horizon (hold = that horizon's length).
    `score_scale` is a per-horizon array mapping raw score -> fractional return
    (target_std for the GNN; ones for the LGBM whose preds are already returns)."""
    out = {}
    for hi, H in enumerate(horizon_end_s):
        sc = 1.0 if score_scale is None else float(score_scale[hi])
        out[hi] = simulate_long_short(
            times, scores[:, :, hi], valid, ret1s, spread_bps, day_code, n_eq,
            rebal=max(1, H), k=k, buffer_m=buffer_m, band=band,
            edge_gate=edge_gate, score_scale=sc, extra_cost_bps=extra_cost_bps,
        )
    return out