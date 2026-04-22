"""Buy-only confluence breakout scanner.

Returns a ranked long-only watchlist. A stock must clear ALL hard gates to
qualify (volume, trend, momentum, relative strength, consolidation,
breakout proximity, liquidity). Remaining candidates are scored 0-100 and
sorted. Entry / stop / target levels are computed per candidate.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ── Tunable thresholds ────────────────────────────────────────────────────────
MIN_PRICE         = 20.0       # INR
MIN_AVG_DOLLAR_VOL = 50_000_000   # 5 Cr INR turnover
VOL_SURGE_MIN     = 1.5        # latest vol / 20d avg vol
VOL_SURGE_STRONG  = 2.5
RSI_MIN           = 50.0
RSI_MAX           = 78.0
CONSOLIDATION_MAX_PCT = 15.0   # 20d range / low
RS_LOOKBACK_DAYS  = 63         # ~3 months
PROXIMITY_MAX_PCT = 8.0        # close within this % below 52W high
BREAKOUT_ABOVE_MAX_PCT = 5.0   # or up to 5% past recent 20d high
MIN_SCORE_FOR_BUY = 65.0
GRADE_CUTOFFS = [(85, "A+"), (75, "A"), (65, "B+"), (0, "B")]


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    line = ema12 - ema26
    signal = _ema(line, 9)
    hist = line - signal
    return line, signal, hist


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


# ── Per-stock evaluation ──────────────────────────────────────────────────────

def _evaluate(
    ticker: str,
    df: pd.DataFrame,
    bench_returns: Optional[Dict[int, float]] = None,
) -> Optional[Dict]:
    """Evaluate one stock. Return candidate dict or None if rejected."""
    if df is None or len(df) < 200:
        return None

    close  = df["Close"].astype(float)
    high   = df["High"].astype(float)
    low    = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    c = float(close.iloc[-1])
    if c < MIN_PRICE:
        return None

    avg_vol20 = float(volume.iloc[-20:].mean())
    avg_dollar_vol = avg_vol20 * c
    if avg_dollar_vol < MIN_AVG_DOLLAR_VOL:
        return None

    # Indicators
    ema20  = _ema(close, 20)
    ema50  = _ema(close, 50)
    ema100 = _ema(close, 100)
    ema200 = _ema(close, 200)
    rsi14  = _rsi(close, 14)
    macd_line, macd_sig, macd_hist = _macd(close)
    atr14  = _atr(high, low, close, 14)

    e20  = float(ema20.iloc[-1])
    e50  = float(ema50.iloc[-1])
    e100 = float(ema100.iloc[-1])
    e200 = float(ema200.iloc[-1])
    rsi_now   = float(rsi14.iloc[-1])
    hist_now  = float(macd_hist.iloc[-1])
    hist_prev = float(macd_hist.iloc[-2]) if len(macd_hist) >= 2 else hist_now
    macd_now  = float(macd_line.iloc[-1])
    sig_now   = float(macd_sig.iloc[-1])
    atr_now   = float(atr14.iloc[-1]) if not pd.isna(atr14.iloc[-1]) else c * 0.02

    # Gate 1 — Trend alignment (close > EMA20 > EMA50 > EMA200, EMA20 rising)
    ema20_slope = (e20 - float(ema20.iloc[-5])) / (abs(float(ema20.iloc[-5])) + 1e-9)
    trend_ok = (c > e20 > e50 > e200) and (ema20_slope > 0)
    if not trend_ok:
        return None

    # Gate 2 — Volume confirmation
    vol_today = float(volume.iloc[-1])
    vol_ratio = vol_today / (avg_vol20 + 1e-9)
    if vol_ratio < VOL_SURGE_MIN:
        return None

    # Gate 3 — Momentum (RSI zone + MACD bullish)
    if not (RSI_MIN <= rsi_now <= RSI_MAX):
        return None
    macd_bull = (macd_now > sig_now) and (hist_now > 0) and (hist_now >= hist_prev * 0.9)
    if not macd_bull:
        return None

    # Gate 4 — Relative Strength vs benchmark (3-month)
    rs_3m_pct = None
    if bench_returns is not None and len(close) >= RS_LOOKBACK_DAYS:
        stock_ret = (c / float(close.iloc[-RS_LOOKBACK_DAYS]) - 1.0) * 100.0
        bench_ret = bench_returns.get(RS_LOOKBACK_DAYS, 0.0)
        rs_3m_pct = stock_ret - bench_ret
        if rs_3m_pct <= 0:
            return None
    else:
        rs_3m_pct = 0.0  # tolerate if bench missing

    # Gate 5 — Breakout proximity
    wk52_high = float(high.rolling(252, min_periods=60).max().iloc[-1])
    base20_high = float(high.iloc[-20:-1].max())
    pct_below_52w = (wk52_high - c) / wk52_high * 100.0 if wk52_high > 0 else 100.0
    pct_above_base = (c - base20_high) / base20_high * 100.0 if base20_high > 0 else 0.0

    near_52w = pct_below_52w <= PROXIMITY_MAX_PCT
    fresh_breakout = 0.0 <= pct_above_base <= BREAKOUT_ABOVE_MAX_PCT
    if not (near_52w or fresh_breakout):
        return None

    # Gate 6 — Consolidation tightness over prior 20 bars (excluding today)
    base_slice = df.iloc[-21:-1]
    base_high = float(base_slice["High"].max())
    base_low  = float(base_slice["Low"].min())
    consolidation_pct = (base_high - base_low) / (base_low + 1e-9) * 100.0
    if consolidation_pct > CONSOLIDATION_MAX_PCT * 1.5:
        # too wide a base → not a clean breakout
        return None

    # ── Scoring (0-100) ────────────────────────────────────────────────────────
    score = 0.0
    reasons: List[str] = []

    # Volume (0-20)
    if vol_ratio >= VOL_SURGE_STRONG:
        score += 20.0
        reasons.append(f"vol×{vol_ratio:.1f}")
    else:
        score += 10.0 + (vol_ratio - VOL_SURGE_MIN) / (VOL_SURGE_STRONG - VOL_SURGE_MIN) * 10.0
        reasons.append(f"vol×{vol_ratio:.1f}")

    # Trend (0-20) — deeper above 200 EMA = stronger trend, but not overextended
    dist_200 = (c - e200) / e200 * 100.0
    if 5 <= dist_200 <= 25:
        score += 20.0
        reasons.append("trend↑")
    elif 0 < dist_200 < 5:
        score += 12.0
        reasons.append("trend↑ early")
    elif 25 < dist_200 <= 40:
        score += 14.0
        reasons.append("trend↑ extended")
    else:
        score += 8.0
        reasons.append("trend↑ stretched")

    # Momentum (0-15) — RSI sweet spot + MACD acceleration
    if 55 <= rsi_now <= 68:
        score += 10.0
    else:
        score += 6.0
    if hist_now > hist_prev and hist_prev > 0:
        score += 5.0
        reasons.append("MACD↑")
    else:
        score += 2.0
        reasons.append("MACD+")

    # Relative Strength (0-15)
    if rs_3m_pct >= 20:
        score += 15.0
        reasons.append(f"RS+{rs_3m_pct:.0f}%")
    elif rs_3m_pct >= 10:
        score += 11.0
        reasons.append(f"RS+{rs_3m_pct:.0f}%")
    elif rs_3m_pct > 0:
        score += 6.0
        reasons.append(f"RS+{rs_3m_pct:.0f}%")

    # Consolidation tightness (0-15)
    if consolidation_pct <= 7:
        score += 15.0
        reasons.append(f"base {consolidation_pct:.1f}%")
    elif consolidation_pct <= 12:
        score += 10.0
        reasons.append(f"base {consolidation_pct:.1f}%")
    else:
        score += 5.0
        reasons.append(f"base {consolidation_pct:.1f}%")

    # Breakout proximity (0-15)
    if fresh_breakout:
        score += 15.0
        reasons.append(f"B/O +{pct_above_base:.1f}%")
    elif pct_below_52w <= 3:
        score += 12.0
        reasons.append(f"52Wh −{pct_below_52w:.1f}%")
    else:
        score += 7.0
        reasons.append(f"near 52Wh")

    score = round(min(100.0, score), 1)

    if score < MIN_SCORE_FOR_BUY:
        return None

    # Grade
    grade = next(g for cut, g in GRADE_CUTOFFS if score >= cut)

    # Entry / Stop / Target
    entry = c
    swing_low = float(low.iloc[-10:].min())
    stop_atr  = c - 2.0 * atr_now
    stop = max(swing_low, stop_atr) * 0.995
    if stop >= entry:
        stop = entry - 1.5 * atr_now
    risk = entry - stop
    target1 = entry + 2.0 * risk
    target2 = entry + 3.0 * risk
    # Target3 = next major resistance above
    candidates = [wk52_high]
    for mult in [10, 50, 100, 250, 500, 1000, 2000, 5000]:
        r = (entry // mult + 1) * mult
        if r > target2:
            candidates.append(r)
            break
    target3 = max(c * 1.25, min(x for x in candidates if x > target2)) if any(x > target2 for x in candidates) else target2 * 1.1
    rr = (target1 - entry) / risk if risk > 0 else 0.0

    prev_close = float(close.iloc[-2]) if len(close) >= 2 else c
    change_pct = (c - prev_close) / (prev_close + 1e-9) * 100.0

    return {
        "ticker":           ticker,
        "cmp":              round(c, 2),
        "change_pct":       round(change_pct, 2),
        "score":            score,
        "grade":            grade,
        "vol_surge":        round(vol_ratio, 2),
        "rsi":              round(rsi_now, 1),
        "rs_3m_pct":        round(rs_3m_pct, 2),
        "consolidation":    round(consolidation_pct, 2),
        "pct_below_52w":    round(pct_below_52w, 2),
        "pct_above_base":   round(pct_above_base, 2),
        "dist_200ema":      round(dist_200, 2),
        "entry":            round(entry, 2),
        "stop":             round(stop, 2),
        "target1":          round(target1, 2),
        "target2":          round(target2, 2),
        "target3":          round(target3, 2),
        "risk_pct":         round(risk / entry * 100.0, 2) if entry > 0 else 0.0,
        "rr_ratio":         round(rr, 2),
        "reasons":          " · ".join(reasons),
    }


# ── Universe-wide scan ────────────────────────────────────────────────────────

def _bench_returns(bench_df: Optional[pd.DataFrame]) -> Dict[int, float]:
    """Compute benchmark %-return over a set of lookback windows."""
    if bench_df is None or len(bench_df) < RS_LOOKBACK_DAYS + 5:
        return {}
    bc = bench_df["Close"].astype(float)
    out: Dict[int, float] = {}
    for lb in (21, 63, 126, 252):
        if len(bc) >= lb:
            out[lb] = (float(bc.iloc[-1]) / float(bc.iloc[-lb]) - 1.0) * 100.0
    return out


def scan_universe(
    stocks_data: Dict[str, pd.DataFrame],
    bench_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run the buy-only scan across all stocks.

    Parameters
    ----------
    stocks_data : dict of {ticker: daily OHLCV DataFrame}
    bench_df    : benchmark (Nifty 50) daily data for relative strength

    Returns
    -------
    DataFrame ranked by score desc, one row per qualified long candidate.
    Empty DataFrame if nothing passes the gates.
    """
    from tickers import get_company_name, get_sector

    bench_rets = _bench_returns(bench_df)

    rows: List[Dict] = []
    skipped = 0
    errored = 0

    for ticker, df in stocks_data.items():
        try:
            res = _evaluate(ticker, df, bench_rets)
        except Exception as exc:
            errored += 1
            logger.warning("Scanner failed for %s: %s", ticker, exc)
            continue
        if res is None:
            skipped += 1
            continue
        res["name"]   = get_company_name(ticker)
        res["sector"] = get_sector(ticker)
        rows.append(res)

    if not rows:
        logger.info("Scan: 0 BUY setups from %d tickers (%d skipped, %d errored)",
                    len(stocks_data), skipped, errored)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.sort_values("score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.insert(0, "rank", df.index + 1)

    cols_order = [
        "rank", "ticker", "name", "sector", "grade", "score",
        "cmp", "change_pct", "vol_surge", "rsi", "rs_3m_pct",
        "consolidation", "pct_below_52w", "pct_above_base", "dist_200ema",
        "entry", "stop", "target1", "target2", "target3",
        "risk_pct", "rr_ratio", "reasons",
    ]
    return df[[c for c in cols_order if c in df.columns]]
