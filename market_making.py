"""
market_making.py
================
Passive market-making simulator for the short-horizon mid-direction signal.

The taker backtest (backtest.py) proved the 0-5s signal can't pay the spread by
CROSSING it. A maker instead POSTS at the bid/ask and earns the spread, and the
signal is used to SKEW quoting to dodge adverse selection (toxic fills right
before a move) and to lean into the predicted direction.

P&L decomposition, per name, per second (everything in fractional/return units
so it sums across names and annualizes cleanly):

    hold inventory q into [t, t+1]; passive fills f_b (buy at bid) / f_s (sell at
    ask) occur at ~mid_t; mark everything to mid_{t+1}:

        pnl_t = half_spread * (f_b + f_s)          # spread captured
              + dmid_{t+1} * (q + f_b - f_s)        # position P&L = ADVERSE SELECTION
              - fee * (f_b + f_s)                   # optional commission/rebate

    The position term is where toxicity bites: if you were just filled long
    (f_b>0) right before a down move (dmid<0), you lose. The signal's job is to
    make f_b happen before UP moves and f_s before DOWN moves.

Fill model (explicit + conservative, one main knob kappa):
    a passive BID is hit by SELL-aggressor volume; we fill a fraction
        f_b = bid_active * clip(kappa * sell_vol / (bid_size + 1), 0, 1)
    i.e. we sit behind the displayed queue and fill in proportion to how much
    aggressor volume arrives relative to the size ahead. kappa<1 = back of queue
    (conservative); kappa=1 = aggressive/front. Symmetric for the ask.

HEADLINE METRIC: run WITH signal-skew vs WITHOUT (symmetric quoting). The lift
of the former over the latter is the signal's marginal value to a maker; the
absolute level depends on the fill assumption and should be read with kappa
sensitivity, not as a promise.

No leakage: quotes at t use the signal & book known at t; fills/P&L use t+1.
"""

import numpy as np

TRADING_DAYS = 252
RET_CLIP = 0.02


def _latest_score_index(times, T, day_code):
    """For every second t, the index into `times` of the most recent score at or
    before t within the same session (-1 if none)."""
    lat = np.searchsorted(times, np.arange(T), side="right") - 1
    ok = lat >= 0
    same = np.zeros(T, dtype=bool)
    same[ok] = day_code[times[lat[ok]]] == day_code[np.arange(T)[ok]]
    lat[~same] = -1
    return lat


def simulate_market_making(times, score_h, valid, ret1s, spread_bps, bid_sz, ask_sz,
                           buy_vol, sell_vol, day_code, n_eq, score_scale=1.0,
                           kappa=0.25, inv_max=5.0, quote_size=1.0, lam=0.5,
                           gate=0.0, use_signal=True, fee_bps=0.0):
    """One pass of the maker. use_signal=False => symmetric quoting (no-alpha
    baseline). Returns metrics + the per-day P&L series. P&L is in units of
    'fraction of one quote_size notional', summed across names."""
    T = ret1s.shape[0]
    ret1s = np.clip(np.nan_to_num(ret1s, nan=0.0, posinf=0.0, neginf=0.0), -RET_CLIP, RET_CLIP)
    hs_all = np.clip(np.nan_to_num(spread_bps, nan=0.0), 0.0, None) / 2.0 / 1e4
    bid_sz = np.nan_to_num(bid_sz, nan=0.0)
    ask_sz = np.nan_to_num(ask_sz, nan=0.0)
    buy_vol = np.nan_to_num(buy_vol, nan=0.0)
    sell_vol = np.nan_to_num(sell_vol, nan=0.0)
    fee = fee_bps / 1e4

    score_dense = np.full((T, n_eq), np.nan)
    valid_dense = np.zeros((T, n_eq), dtype=bool)
    score_dense[times] = score_h
    valid_dense[times] = valid
    lat = _latest_score_index(times, T, day_code)

    day_pnl, day_spread, day_adverse, inv_abs, fills, n_days = [], [], [], [], [], 0
    spread_tot = adverse_tot = fee_tot = 0.0

    for d in np.unique(day_code):
        di = np.where(day_code == d)[0]
        if di.size < 3:
            continue
        a, b = di[0], di[-1]
        q = np.zeros(n_eq)
        p_day = sp_day = adv_day = 0.0
        for t in range(a, b):                     # hold set at t, realize over t->t+1
            li = lat[t]
            if li >= 0:
                s = np.nan_to_num(score_dense[li]) * score_scale
                tradeable = valid_dense[li]
            else:
                s = np.zeros(n_eq); tradeable = np.zeros(n_eq, dtype=bool)

            hs = hs_all[t]
            # Both arms manage inventory (skew to mean-revert q); only the
            # predictive signal differs. This makes the baseline an inventory-
            # managed, SIGNAL-FREE maker, so the signal-arm's lift is the
            # marginal value of the forecast -- not just inventory control
            # (otherwise even a noise 'signal' would 'win' by reducing inventory).
            sig = s if use_signal else np.zeros(n_eq)
            eff = sig - lam * q
            up = (eff > gate)
            down = (eff < -gate)
            ask_active = np.where(up, 0.0, 1.0)     # lean long: pull ask
            bid_active = np.where(down, 0.0, 1.0)   # lean short: pull bid
            # hard inventory limit
            bid_active = np.where(q >= inv_max, 0.0, bid_active)
            ask_active = np.where(q <= -inv_max, 0.0, ask_active)

            fb = bid_active * quote_size * np.clip(kappa * sell_vol[t] / (bid_sz[t] + 1.0), 0.0, 1.0)
            fs = ask_active * quote_size * np.clip(kappa * buy_vol[t] / (ask_sz[t] + 1.0), 0.0, 1.0)
            fb = np.minimum(fb, np.maximum(0.0, inv_max - q))     # don't breach limit
            fs = np.minimum(fs, np.maximum(0.0, inv_max + q))
            qn = q + fb - fs
            dmid = ret1s[t + 1]

            sp = hs * (fb + fs)                   # spread captured (>=0)
            adv = dmid * qn                        # position P&L (adverse selection lives here)
            fees = fee * (fb + fs)
            p_day += float(np.sum(sp + adv - fees))
            sp_day += float(np.sum(sp)); adv_day += float(np.sum(adv))
            fills.append(float(np.sum(fb + fs))); inv_abs.append(float(np.sum(np.abs(qn))))
            q = qn

        flat = float(np.sum(np.abs(q) * hs_all[b]))   # flatten across the spread at close
        p_day -= flat
        day_pnl.append(p_day); day_spread.append(sp_day); day_adverse.append(adv_day)
        spread_tot += sp_day; adverse_tot += adv_day; fee_tot += 0.0
        n_days += 1

    if n_days < 2:
        return {"n_days": n_days, "sharpe": 0.0, "pnl_per_day": 0.0, "pnl_bps_per_name_day": 0.0,
                "spread_capture": 0.0, "adverse_sel": 0.0, "avg_abs_inv": 0.0, "fill_rate": 0.0}

    dp = np.asarray(day_pnl)
    sharpe = float(dp.mean() / dp.std() * np.sqrt(TRADING_DAYS)) if dp.std() > 0 else 0.0
    return {
        "n_days": n_days,
        "sharpe": sharpe,                                  # annualized, on daily P&L
        "pnl_per_day": float(dp.mean()),                   # total, all names, per day (frac units)
        "pnl_bps_per_name_day": float(dp.mean() / n_eq * 1e4),   # avg per-name daily P&L in bps
        "spread_capture": float(np.mean(day_spread)),      # per day, all names
        "adverse_sel": float(np.mean(day_adverse)),        # per day (negative = picked off)
        "avg_abs_inv": float(np.mean(inv_abs)),            # avg gross inventory (units)
        "fill_rate": float(np.mean(fills)),                # avg filled size/sec, all names
    }


def run_market_making(times, scores, valid, ret1s, spread_bps, bid_sz, ask_sz,
                      buy_vol, sell_vol, day_code, n_eq, signal_hi=0, score_scale=1.0,
                      kappa=0.25, inv_max=5.0, lam=0.5, gate=0.0, fee_bps=0.0):
    """Run the maker WITH the signal skew and WITHOUT (symmetric baseline), using
    the `signal_hi` horizon's score (default 0-5s) to skew. The signal's value is
    the lift of 'signal' over 'baseline'."""
    sc = score_scale if np.isscalar(score_scale) else float(score_scale[signal_hi])
    common = dict(times=times, score_h=scores[:, :, signal_hi], valid=valid, ret1s=ret1s,
                  spread_bps=spread_bps, bid_sz=bid_sz, ask_sz=ask_sz, buy_vol=buy_vol,
                  sell_vol=sell_vol, day_code=day_code, n_eq=n_eq, score_scale=sc,
                  kappa=kappa, inv_max=inv_max, lam=lam, gate=gate, fee_bps=fee_bps)
    return {
        "signal": simulate_market_making(use_signal=True, **common),
        "baseline": simulate_market_making(use_signal=False, **common),
    }