"""Quick Alpha Scanner — short-term (3-7 day) high-probability long setups.

Algorithm: pullback-to-trend + triple momentum confluence.

A candidate must pass HARD GATES. Survivors are scored 0-100 and ranked.
Every pick targets a 1-week swing, not a multi-week breakout.

Hard gates (all must pass):
    1. Uptrend intact  — EMA20 > EMA50 > EMA200, all three rising
    2. Pullback reset  — price touched / crossed EMA20 within last 5 bars
    3. Bounce confirmed — today's close back above EMA20 OR < 1% below it
    4. MACD turning up — bullish cross within 3 bars OR line>signal w/ hist↑
    5. RSI in launch zone — 45-68, rising (RSI[-1] > RSI[-3])
    6. Stoch RSI confluence — %K crossed above %D in last 3 bars, %K < 80
    7. Volume confirm — today's vol >= 1.1x 20d avg
    8. No distribution — no bar in last 5 that dropped >4% on above-avg vol
    9. Liquidity — price >= INR 20, avg turnover >= 5 Cr

Soft score (0-100):
    25  MACD freshness + trajectory
    20  RSI trajectory through 50
    20  Stoch RSI cross quality
    15  Pullback quality (wick / clean bounce)
    10  Volume confirmation strength
    10  Trend strength (EMA stack + slope)

Entry, stop, target are sized for a 3-7 trading day hold.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ── Tunables ──────────────────────────────────────────────────────────────────
MIN_PRICE            = 20.0
MIN_AVG_TURNOVER     = 50_000_000      # INR 5 Cr avg daily turnover
VOL_CONFIRM_MIN      = 1.1
VOL_CONFIRM_STRONG   = 2.0
RSI_MIN              = 45.0
RSI_MAX              = 68.0
STOCH_K_MAX          = 80.0
PULLBACK_LOOKBACK    = 5               # bars to look for pullback touch
MACD_CROSS_LOOKBACK  = 3               # how fresh the MACD cross must be
STOCH_CROSS_LOOKBACK = 3
DISTRIBUTION_DROP_PCT = -4.0
DISTRIBUTION_LOOKBACK = 5
MIN_SCORE_FOR_BUY    = 70.0
# Max score = 100 + 5 weekly bonus = 105
GRADE_CUTOFFS        = [(95, "A+"), (85, "A"), (75, "B+"), (0, "B")]

# Forward-return backtest window (trading days)
BACKTEST_HOLD_DAYS   = 5
BACKTEST_WALK_DAYS   = 120             # most recent N bars used as test period


# ── Indicators ────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, w: int) -> pd.Series:
    return s.ewm(span=w, adjust=False).mean()


def _rsi(close: pd.Series, w: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    ag = gain.ewm(alpha=1 / w, min_periods=w, adjust=False).mean()
    al = loss.ewm(alpha=1 / w, min_periods=w, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    line = _ema(close, 12) - _ema(close, 26)
    signal = _ema(line, 9)
    return line, signal, line - signal


def _stoch_rsi(close: pd.Series, w: int = 14, smooth_k: int = 3, smooth_d: int = 3
               ) -> Tuple[pd.Series, pd.Series]:
    rsi = _rsi(close, w)
    low = rsi.rolling(w).min()
    high = rsi.rolling(w).max()
    stoch = (rsi - low) / (high - low).replace(0, np.nan)
    k = stoch.rolling(smooth_k).mean() * 100.0
    d = k.rolling(smooth_d).mean()
    return k, d


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, w: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / w, min_periods=w, adjust=False).mean()


def _attach_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Add `_weekly_bullish` column: True when last-completed weekly bar has
    close > 20W SMA AND weekly MACD line > signal. Uses daily date index,
    broadcasts weekly verdict forward to each daily bar (no look-ahead).
    """
    if "_weekly_bullish" in df.columns:
        return df
    try:
        wk = df[["Open", "High", "Low", "Close", "Volume"]].resample("W-FRI").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna()
        if len(wk) < 22:
            df["_weekly_bullish"] = False
            return df
        wk_close = wk["Close"].astype(float)
        wk_sma20 = wk_close.rolling(20).mean()
        wk_line, wk_sig, _ = _macd(wk_close)
        wk_rsi = _rsi(wk_close, 14)
        wk_bull = ((wk_close > wk_sma20) & (wk_line > wk_sig) & (wk_rsi > 50)).shift(1).fillna(False)
        df["_weekly_bullish"] = wk_bull.reindex(df.index, method="ffill").fillna(False).astype(bool)
    except Exception:
        df["_weekly_bullish"] = False
    return df


# ── Bar-at-index evaluation (used by both scanner and backtester) ─────────────

def _signal_at(df: pd.DataFrame, i: int) -> Optional[Dict]:
    """Evaluate gates+score at bar index `i` (0-based).

    `i` must point to a bar where all indicators are defined (~>= 260).
    Returns dict with score/levels/flags, or None if gates fail.
    """
    if i < 260 or i >= len(df):
        return None

    close  = df["Close"].astype(float)
    high   = df["High"].astype(float)
    low    = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    c = float(close.iloc[i])
    if c < MIN_PRICE:
        return None

    avg_vol20 = float(volume.iloc[i - 20:i].mean())
    if avg_vol20 * c < MIN_AVG_TURNOVER:
        return None

    ema20  = _ema(close, 20)
    ema50  = _ema(close, 50)
    ema200 = _ema(close, 200)
    rsi14  = _rsi(close, 14)
    macd_line, macd_sig, macd_hist = _macd(close)
    k, d   = _stoch_rsi(close, 14, 3, 3)
    atr14  = _atr(high, low, close, 14)

    e20_i  = float(ema20.iloc[i])
    e50_i  = float(ema50.iloc[i])
    e200_i = float(ema200.iloc[i])

    # Gate 1 — trend intact
    if not (e20_i > e50_i > e200_i):
        return None
    e20_slope  = e20_i - float(ema20.iloc[i - 5])
    e50_slope  = e50_i - float(ema50.iloc[i - 10])
    e200_slope = e200_i - float(ema200.iloc[i - 20])
    if e20_slope <= 0 or e50_slope <= 0 or e200_slope <= 0:
        return None

    # Gate 2 — pullback touched EMA20 in last N bars
    window_low = low.iloc[i - PULLBACK_LOOKBACK:i + 1]
    window_e20 = ema20.iloc[i - PULLBACK_LOOKBACK:i + 1]
    touched = (window_low <= window_e20 * 1.005).any()
    if not touched:
        return None

    # Gate 3 — bounce / back above EMA20
    bounced = (c >= e20_i * 0.99)
    if not bounced:
        return None

    # Gate 4 — MACD turning up
    line_i = float(macd_line.iloc[i])
    sig_i  = float(macd_sig.iloc[i])
    hist_i = float(macd_hist.iloc[i])

    macd_fresh_cross = False
    for look in range(1, MACD_CROSS_LOOKBACK + 1):
        prev_line = float(macd_line.iloc[i - look])
        prev_sig  = float(macd_sig.iloc[i - look])
        if prev_line <= prev_sig and line_i > sig_i:
            macd_fresh_cross = True
            break
    hist_prev = float(macd_hist.iloc[i - 1])
    hist_prev2 = float(macd_hist.iloc[i - 2])
    hist_bottomed = hist_prev <= hist_prev2 and hist_i > hist_prev and hist_i > 0
    if not (macd_fresh_cross or (line_i > sig_i and hist_bottomed)):
        return None

    # Gate 5 — RSI launch zone
    rsi_now  = float(rsi14.iloc[i])
    rsi_prev = float(rsi14.iloc[i - 3])
    if not (RSI_MIN <= rsi_now <= RSI_MAX):
        return None
    if rsi_now <= rsi_prev:
        return None

    # Gate 6 — Stoch RSI cross
    k_i = float(k.iloc[i]) if not pd.isna(k.iloc[i]) else np.nan
    d_i = float(d.iloc[i]) if not pd.isna(d.iloc[i]) else np.nan
    if pd.isna(k_i) or pd.isna(d_i):
        return None
    if k_i >= STOCH_K_MAX:
        return None
    stoch_fresh_cross = False
    for look in range(1, STOCH_CROSS_LOOKBACK + 1):
        k_prev = float(k.iloc[i - look]) if not pd.isna(k.iloc[i - look]) else np.nan
        d_prev = float(d.iloc[i - look]) if not pd.isna(d.iloc[i - look]) else np.nan
        if not pd.isna(k_prev) and not pd.isna(d_prev):
            if k_prev <= d_prev and k_i > d_i:
                stoch_fresh_cross = True
                break
    if not stoch_fresh_cross:
        return None

    # Gate 7 — volume confirm today
    vol_today = float(volume.iloc[i])
    vol_ratio = vol_today / (avg_vol20 + 1e-9)
    if vol_ratio < VOL_CONFIRM_MIN:
        return None

    # Gate 8 — no distribution bar in last 5
    for look in range(1, DISTRIBUTION_LOOKBACK + 1):
        c_l = float(close.iloc[i - look])
        c_lp = float(close.iloc[i - look - 1])
        v_l = float(volume.iloc[i - look])
        bar_ret = (c_l - c_lp) / (c_lp + 1e-9) * 100.0
        if bar_ret <= DISTRIBUTION_DROP_PCT and v_l > avg_vol20 * 1.2:
            return None

    atr_now = float(atr14.iloc[i]) if not pd.isna(atr14.iloc[i]) else c * 0.02

    # ── Additional indicators for new scoring matrix ─────────────────────────
    sma20_full  = close.rolling(20).mean()
    sma50_full  = close.rolling(50).mean()
    sma200_full = close.rolling(200).mean()
    s20  = float(sma20_full.iloc[i])
    s50  = float(sma50_full.iloc[i])
    s200 = float(sma200_full.iloc[i])
    s20_prev  = float(sma20_full.iloc[i - 5])
    s50_prev  = float(sma50_full.iloc[i - 10])
    s200_prev = float(sma200_full.iloc[i - 20])

    # Bollinger Band (20, 2) width as % of close
    bb_mid   = sma20_full
    bb_std   = close.rolling(20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width_pct = ((bb_upper - bb_lower) / close) * 100.0
    bb_now = float(bb_width_pct.iloc[i])
    bb_window = bb_width_pct.iloc[max(0, i - 59):i + 1].dropna()
    if len(bb_window) >= 20:
        bb_pct_rank = (bb_window < bb_now).mean()  # 0 = tightest, 1 = widest
    else:
        bb_pct_rank = 0.5

    # 52W high proximity
    wk52_high = float(high.rolling(252, min_periods=60).max().iloc[i])
    pct_below_52w = (wk52_high - c) / wk52_high * 100.0 if wk52_high > 0 else 100.0

    # Weekly confirmation (precomputed series stored on df)
    weekly_bullish = bool(df["_weekly_bullish"].iloc[i]) if "_weekly_bullish" in df.columns else False

    # ── NEW SCORING MATRIX (100 + 5 bonus) ───────────────────────────────────
    score = 0.0
    reasons: List[str] = []

    # Trend — 30  (price vs 20/50/200 SMA, alignment + slope)
    trend_pts = 0.0
    if c > s20:        trend_pts += 7.5
    if s20 > s50:      trend_pts += 7.5
    if s50 > s200:     trend_pts += 7.5
    slopes_ok = (s20 > s20_prev) and (s50 > s50_prev) and (s200 > s200_prev)
    if slopes_ok:      trend_pts += 7.5
    score += trend_pts
    reasons.append(f"Trend {trend_pts:.0f}/30")

    # Momentum — 25  (RSI zone 15 + MACD direction 10)
    rsi_pts = 0.0
    rsi_delta = rsi_now - rsi_prev
    if 50 <= rsi_now <= 65 and rsi_delta >= 2:
        rsi_pts = 15.0
    elif 45 <= rsi_now <= 70 and rsi_delta > 0:
        rsi_pts = 11.0
    elif rsi_delta > 0:
        rsi_pts = 6.0
    else:
        rsi_pts = 2.0
    macd_pts = 0.0
    if macd_fresh_cross:
        macd_pts = 10.0
    elif line_i > sig_i and hist_i > hist_prev and hist_i > 0:
        macd_pts = 8.0
    elif line_i > sig_i:
        macd_pts = 5.0
    else:
        macd_pts = 2.0
    score += rsi_pts + macd_pts
    reasons.append(f"RSI{rsi_now:.0f}/MACD{'×' if macd_fresh_cross else '+'}")

    # Volume — 20  (today vs 20d avg)
    if vol_ratio >= 2.5:
        vol_pts = 20.0
    elif vol_ratio >= VOL_CONFIRM_STRONG:
        vol_pts = 17.0
    elif vol_ratio >= 1.5:
        vol_pts = 13.0
    elif vol_ratio >= 1.2:
        vol_pts = 9.0
    else:
        vol_pts = 5.0
    score += vol_pts
    reasons.append(f"Vol×{vol_ratio:.1f}")

    # Setup — 25  (52W proximity 13 + BB squeeze 12)
    if pct_below_52w <= 2:
        prox_pts = 13.0
    elif pct_below_52w <= 8:
        prox_pts = 10.0
    elif pct_below_52w <= 15:
        prox_pts = 6.0
    else:
        prox_pts = 3.0
    if bb_pct_rank <= 0.2:
        squeeze_pts = 12.0
    elif bb_pct_rank <= 0.4:
        squeeze_pts = 8.0
    elif bb_pct_rank <= 0.6:
        squeeze_pts = 5.0
    else:
        squeeze_pts = 2.0
    score += prox_pts + squeeze_pts
    reasons.append(f"52W −{pct_below_52w:.1f}% BBrk{bb_pct_rank*100:.0f}%")

    # Weekly bonus — +5  (weekly trend confirmation)
    weekly_pts = 5.0 if weekly_bullish else 0.0
    score += weekly_pts
    if weekly_bullish:
        reasons.append("Wk↑")

    if k_i < 40:
        reasons.append(f"StochK {k_i:.0f}")

    score = round(min(105.0, score), 1)
    # Expose dist_200ema for downstream display
    dist_200 = (c - e200_i) / e200_i * 100.0

    # ── Levels for 1-week swing ──────────────────────────────────────────────
    entry = c
    swing_low_5 = float(low.iloc[i - 4:i + 1].min())
    stop_atr    = c - 1.5 * atr_now
    stop = max(swing_low_5, stop_atr) * 0.995
    if stop >= entry:
        stop = entry - 1.25 * atr_now
    risk = entry - stop
    target1 = entry + 2.0 * risk
    target2 = entry + 3.0 * risk
    # T3 = recent swing high (20d) if above T2, else T2 * 1.05
    swing_high_20 = float(high.iloc[max(0, i - 19):i + 1].max())
    target3 = max(swing_high_20 if swing_high_20 > target2 else target2 * 1.05, target2 * 1.02)

    return {
        "score":          score,
        "reasons":        reasons,
        "macd_fresh":     macd_fresh_cross,
        "rsi":            rsi_now,
        "stoch_k":        k_i,
        "stoch_d":        d_i,
        "vol_ratio":      vol_ratio,
        "atr":            atr_now,
        "entry":          entry,
        "stop":           stop,
        "target1":        target1,
        "target2":        target2,
        "target3":        target3,
        "risk":           risk,
        "dist_200ema":    dist_200,
    }


# ── Per-stock scan (latest bar only) ──────────────────────────────────────────

def _evaluate(ticker: str, df: pd.DataFrame) -> Optional[Dict]:
    if df is None or len(df) < 260:
        return None
    df = _attach_weekly(df)
    sig = _signal_at(df, len(df) - 1)
    if sig is None or sig["score"] < MIN_SCORE_FOR_BUY:
        return None

    c = sig["entry"]
    risk = sig["risk"]
    grade = next(g for cut, g in GRADE_CUTOFFS if sig["score"] >= cut)
    prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else c
    change_pct = (c - prev_close) / (prev_close + 1e-9) * 100.0
    rr = (sig["target1"] - c) / risk if risk > 0 else 0.0

    return {
        "ticker":       ticker,
        "cmp":          round(c, 2),
        "change_pct":   round(change_pct, 2),
        "score":        sig["score"],
        "grade":        grade,
        "macd_fresh":   sig["macd_fresh"],
        "rsi":          round(sig["rsi"], 1),
        "stoch_k":      round(sig["stoch_k"], 1),
        "vol_surge":    round(sig["vol_ratio"], 2),
        "dist_200ema":  round(sig["dist_200ema"], 2),
        "entry":        round(c, 2),
        "stop":         round(sig["stop"], 2),
        "target1":      round(sig["target1"], 2),
        "target2":      round(sig["target2"], 2),
        "target3":      round(sig["target3"], 2),
        "risk_pct":     round(risk / c * 100.0, 2) if c > 0 else 0.0,
        "rr_ratio":     round(rr, 2),
        "reasons":      " · ".join(sig["reasons"]),
    }


# ── Universe scan ─────────────────────────────────────────────────────────────

def scan_universe(
    stocks_data: Dict[str, pd.DataFrame],
    bench_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Scan universe for Quick Alpha long setups. Returns ranked DataFrame."""
    from tickers import get_company_name, get_sector

    rows: List[Dict] = []
    for ticker, df in stocks_data.items():
        try:
            res = _evaluate(ticker, df)
        except Exception as exc:
            logger.warning("Scanner failed for %s: %s", ticker, exc)
            continue
        if res is None:
            continue
        res["name"]   = get_company_name(ticker)
        res["sector"] = get_sector(ticker)
        rows.append(res)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.sort_values("score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.insert(0, "rank", df.index + 1)

    cols_order = [
        "rank", "ticker", "name", "sector", "grade", "score",
        "cmp", "change_pct", "vol_surge", "rsi", "stoch_k",
        "dist_200ema", "entry", "stop", "target1", "target2", "target3",
        "risk_pct", "rr_ratio", "reasons",
    ]
    return df[[c for c in cols_order if c in df.columns]]


# ── Walk-forward backtest ─────────────────────────────────────────────────────

def backtest_signals(
    stocks_data: Dict[str, pd.DataFrame],
    hold_days: int = BACKTEST_HOLD_DAYS,
    walk_days: int = BACKTEST_WALK_DAYS,
    min_score: float = MIN_SCORE_FOR_BUY,
) -> Dict[str, float]:
    """Walk forward over last `walk_days` bars, fire signals on each bar,
    record `hold_days` forward return. Return aggregate stats.
    """
    returns: List[float] = []
    hits = 0
    stop_hits = 0
    total_signals = 0

    for ticker, df in stocks_data.items():
        if df is None or len(df) < 260 + hold_days + 5:
            continue
        df = _attach_weekly(df)
        close = df["Close"].astype(float)
        low   = df["Low"].astype(float)
        high  = df["High"].astype(float)
        n = len(df)
        start_i = max(260, n - walk_days - hold_days)
        end_i   = n - hold_days - 1

        i = start_i
        while i <= end_i:
            try:
                sig = _signal_at(df, i)
            except Exception:
                sig = None
            if sig is None or sig["score"] < min_score:
                i += 1
                continue

            entry  = sig["entry"]
            stop   = sig["stop"]
            target = sig["target1"]

            # Walk the next `hold_days` bars
            exit_price = float(close.iloc[i + hold_days])
            hit_stop = False
            hit_tgt  = False
            for fwd in range(1, hold_days + 1):
                bar_high = float(high.iloc[i + fwd])
                bar_low  = float(low.iloc[i + fwd])
                if bar_low <= stop:
                    exit_price = stop
                    hit_stop = True
                    break
                if bar_high >= target:
                    exit_price = target
                    hit_tgt = True
                    break

            ret = (exit_price - entry) / entry * 100.0
            returns.append(ret)
            if ret > 0:
                hits += 1
            if hit_stop:
                stop_hits += 1
            total_signals += 1

            # cooldown to avoid overlapping same-ticker signals
            i += hold_days
            i += 1

    if not returns:
        return {
            "signals": 0, "hit_rate": 0.0, "avg_return": 0.0,
            "median_return": 0.0, "stop_rate": 0.0, "best": 0.0, "worst": 0.0,
        }

    arr = np.array(returns)
    return {
        "signals":       total_signals,
        "hit_rate":      round(hits / total_signals * 100.0, 1),
        "avg_return":    round(float(arr.mean()), 2),
        "median_return": round(float(np.median(arr)), 2),
        "stop_rate":     round(stop_hits / total_signals * 100.0, 1),
        "best":          round(float(arr.max()), 2),
        "worst":         round(float(arr.min()), 2),
    }
