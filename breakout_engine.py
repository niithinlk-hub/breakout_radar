"""Breakout Probability Score (BPS) engine — composite 0-100 scoring."""

from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import ta

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class BreakoutEngine:
    """Compute BPS and all sub-factor scores for a single stock."""

    WEIGHTS = {
        "volume":     0.20,
        "consolidation": 0.20,
        "ma_alignment":  0.15,
        "resistance":    0.15,
        "momentum":      0.10,
        "rel_strength":  0.10,
        "fundamentals":  0.10,
    }

    def __init__(self, df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None):
        """
        Parameters
        ----------
        df           : Daily OHLCV DataFrame (auto-adjusted, sorted ascending)
        benchmark_df : Nifty 50 daily data for relative-strength calculation
        """
        self.df = df.copy().sort_index()
        self.bench = benchmark_df
        self._add_indicators()

    # ──────────────────────────────────────────────────────────────────────────
    # Indicator computation
    # ──────────────────────────────────────────────────────────────────────────

    def _add_indicators(self) -> None:
        df = self.df
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # EMAs
        for w in [20, 50, 100, 200]:
            df[f"ema{w}"] = ta.trend.EMAIndicator(close=close, window=w).ema_indicator()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()
        df["bb_pct"] = bb.bollinger_pband()

        # RSI
        df["rsi"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

        # MACD
        macd_ind = ta.trend.MACD(close=close)
        df["macd"] = macd_ind.macd()
        df["macd_signal"] = macd_ind.macd_signal()
        df["macd_hist"] = macd_ind.macd_diff()

        # ROC
        df["roc10"] = ta.momentum.ROCIndicator(close=close, window=10).roc()

        # Stochastic RSI
        stoch = ta.momentum.StochRSIIndicator(close=close)
        df["stoch_k"] = stoch.stochrsi_k()
        df["stoch_d"] = stoch.stochrsi_d()

        # OBV
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()

        # ATR
        df["atr"] = ta.volatility.AverageTrueRange(high=high, low=low, close=close).average_true_range()

        # Volume avg
        df["vol_avg20"] = volume.rolling(20).mean()
        df["vol_ratio"] = volume / df["vol_avg20"].replace(0, np.nan)

        self.df = df

    # ──────────────────────────────────────────────────────────────────────────
    # Individual factor scorers (each returns 0-10)
    # ──────────────────────────────────────────────────────────────────────────

    def score_volume(self) -> float:
        df = self.df
        if len(df) < 21:
            return 0.0

        latest = df.iloc[-1]
        score = 0.0

        # Volume surge ratio
        vol_ratio = latest.get("vol_ratio", 1.0)
        if pd.isna(vol_ratio):
            vol_ratio = 1.0
        surge_score = min(10.0, (vol_ratio / 3.0) * 10.0)
        score += surge_score * 0.5

        # Progressive volume increase (last 5 sessions)
        vols = df["Volume"].iloc[-5:].values
        if len(vols) >= 5:
            increases = sum(vols[i] > vols[i-1] for i in range(1, len(vols)))
            score += (increases / 4.0) * 10.0 * 0.25

        # OBV trend: rising OBV vs flat/declining price
        if len(df) >= 10:
            obv_slope = df["obv"].iloc[-10:].values
            price_slope = df["Close"].iloc[-10:].values
            obv_up = (obv_slope[-1] - obv_slope[0]) / (abs(obv_slope[0]) + 1e-9) > 0.01
            price_flat = abs((price_slope[-1] - price_slope[0]) / (price_slope[0] + 1e-9)) < 0.03
            if obv_up and price_flat:
                score += 2.5  # stealth accumulation bonus

        return min(10.0, score)

    def score_consolidation(self) -> float:
        df = self.df
        if len(df) < 20:
            return 0.0

        last20 = df.iloc[-20:]
        high = last20["High"].max()
        low = last20["Low"].min()
        price_range_pct = (high - low) / (low + 1e-9) * 100

        # Range tightness score
        if price_range_pct <= 5:
            range_score = 10.0
        elif price_range_pct >= 25:
            range_score = 0.0
        else:
            range_score = 10.0 - (price_range_pct - 5) / 20.0 * 10.0

        # BB width (narrower = higher score)
        bb_width = df["bb_width"].iloc[-1]
        if pd.isna(bb_width):
            bb_score = 5.0
        else:
            # Normalize against 20-day average BB width
            avg_bb = df["bb_width"].iloc[-60:].mean() if len(df) >= 60 else df["bb_width"].mean()
            relative_width = bb_width / (avg_bb + 1e-9)
            bb_score = max(0.0, min(10.0, (1 - relative_width) * 15.0 + 5.0))

        # Days within 5% band
        mid = last20["Close"].median()
        in_band = ((last20["Close"] - mid).abs() / (mid + 1e-9) < 0.05).sum()
        band_score = min(10.0, in_band / 20.0 * 10.0)

        return (range_score * 0.4 + bb_score * 0.35 + band_score * 0.25)

    def score_ma_alignment(self) -> float:
        df = self.df
        if len(df) < 5:
            return 0.0

        latest = df.iloc[-1]
        close = latest["Close"]
        emas = {w: latest.get(f"ema{w}", np.nan) for w in [20, 50, 100, 200]}

        # Perfect alignment: price > ema20 > ema50 > ema100 > ema200
        score = 10.0
        deductions = 0

        checks = [
            (close, emas[20]),
            (emas[20], emas[50]),
            (emas[50], emas[100]),
            (emas[100], emas[200]),
        ]
        for a, b in checks:
            if pd.isna(a) or pd.isna(b):
                deductions += 1
            elif a <= b:
                deductions += 2

        score -= deductions

        # Bonus: 20 EMA curling up
        if len(df) >= 5:
            ema20_slope = df["ema20"].iloc[-5:].diff().mean()
            if not pd.isna(ema20_slope) and ema20_slope > 0:
                score += 1.5

        # Penalty: overextension (price > 30% above 200 EMA)
        if not pd.isna(emas[200]) and emas[200] > 0:
            dist_from_200 = (close - emas[200]) / emas[200]
            if dist_from_200 > 0.30:
                score -= 2.0

        return max(0.0, min(10.0, score))

    def score_resistance_proximity(self) -> float:
        df = self.df
        if len(df) < 20:
            return 0.0

        close = df["Close"].iloc[-1]
        resistance_levels = self._get_resistance_levels()

        if not resistance_levels:
            return 5.0

        best_score = 0.0
        for level in resistance_levels:
            if level <= 0 or pd.isna(level):
                continue
            dist_pct = (level - close) / close * 100

            if 0 <= dist_pct <= 2:
                # Very close to breakout point
                s = 9.0 + (2 - dist_pct) / 2 * 1.0
            elif 2 < dist_pct <= 5:
                # Ideal setup zone
                s = 10.0
            elif 5 < dist_pct <= 15:
                s = 10.0 - (dist_pct - 5) / 10.0 * 6.0
            elif dist_pct < 0:
                # Already broken out — check if clean break
                s = 7.0 if abs(dist_pct) < 3 else 4.0
            else:
                s = 0.0

            best_score = max(best_score, s)

        return min(10.0, best_score)

    def _get_resistance_levels(self) -> list:
        df = self.df
        levels = []

        # 52-week high
        wk52_high = df["High"].rolling(252, min_periods=50).max().iloc[-1]
        if not pd.isna(wk52_high):
            levels.append(wk52_high)

        # Pivot point (last week's high)
        if len(df) >= 5:
            levels.append(df["High"].iloc[-5:].max())

        # All-time high from available data
        levels.append(df["High"].max())

        # Round numbers near current price
        close = df["Close"].iloc[-1]
        for mult in [10, 50, 100, 250, 500, 1000, 2000, 5000]:
            candidate = (close // mult + 1) * mult
            levels.append(candidate)

        # Fibonacci retracement from swing low to swing high
        high = df["High"].rolling(60, min_periods=20).max().iloc[-1]
        low = df["Low"].rolling(60, min_periods=20).min().iloc[-1]
        if not pd.isna(high) and not pd.isna(low):
            levels.append(low + (high - low) * 0.618)
            levels.append(high)

        return [l for l in levels if not pd.isna(l) and l > 0]

    def score_momentum(self) -> float:
        df = self.df
        if len(df) < 15:
            return 0.0

        score = 0.0
        latest = df.iloc[-1]

        # RSI score
        rsi = latest.get("rsi", 50)
        if pd.isna(rsi):
            rsi_score = 5.0
        elif 55 <= rsi <= 70:
            rsi_score = 10.0
        elif 45 <= rsi < 55:
            rsi_score = 6.0 + (rsi - 45) / 10.0 * 4.0
        elif rsi < 45:
            rsi_score = max(0.0, rsi / 45.0 * 6.0)
        else:
            rsi_score = max(0.0, 10.0 - (rsi - 70) / 30.0 * 10.0)
        score += rsi_score * 0.35

        # MACD histogram positive and increasing
        hist = latest.get("macd_hist", 0)
        if not pd.isna(hist) and hist > 0:
            prev_hist = df["macd_hist"].iloc[-2] if len(df) >= 2 else 0
            score += 10.0 * 0.30 if (not pd.isna(prev_hist) and hist > prev_hist) else 6.0 * 0.30
        elif not pd.isna(hist) and hist < 0:
            score += 2.0 * 0.30

        # ROC positive and accelerating
        roc = latest.get("roc10", 0)
        if not pd.isna(roc) and roc > 0:
            prev_roc = df["roc10"].iloc[-2] if len(df) >= 2 else 0
            accelerating = not pd.isna(prev_roc) and roc > prev_roc
            score += (10.0 if accelerating else 7.0) * 0.20

        # Stochastic RSI crossing up from oversold
        k = latest.get("stoch_k", 0.5)
        d = latest.get("stoch_d", 0.5)
        if not pd.isna(k) and not pd.isna(d):
            if k > d and k < 0.8:
                score += 10.0 * 0.15
            elif k > d:
                score += 5.0 * 0.15

        return min(10.0, score)

    def score_relative_strength(self, bench_df: Optional[pd.DataFrame] = None) -> float:
        df = self.df
        bench = bench_df if bench_df is not None else self.bench
        if bench is None or len(df) < 20:
            return 5.0

        score = 0.0
        try:
            bench_close = bench["Close"]
            if hasattr(bench_close.index, "tz") and bench_close.index.tz is not None:
                bench_close = bench_close.tz_localize(None)
            stock_idx = df.index.tz_localize(None) if hasattr(df.index, "tz") and df.index.tz is not None else df.index
            bench_aligned = bench_close.reindex(stock_idx, method="ffill")
        except Exception:
            return 5.0

        windows = [(21, 0.33), (63, 0.33), (126, 0.34)]
        for lookback, w in windows:
            if len(df) < lookback:
                score += 5.0 * w
                continue
            stock_ret = (df["Close"].iloc[-1] / df["Close"].iloc[-lookback] - 1) * 100
            b_last = bench_aligned.dropna().iloc[-1] if not bench_aligned.dropna().empty else np.nan
            b_start = bench_aligned.dropna().iloc[max(0, len(bench_aligned.dropna()) - lookback)] if not bench_aligned.dropna().empty else np.nan
            if pd.isna(b_last) or pd.isna(b_start) or b_start == 0:
                score += 5.0 * w
                continue
            bench_ret = (b_last / b_start - 1) * 100

            outperf = stock_ret - bench_ret
            if outperf >= 10:
                s = 10.0
            elif outperf >= 0:
                s = 5.0 + outperf / 10.0 * 5.0
            elif outperf >= -10:
                s = 5.0 + outperf / 10.0 * 5.0
            else:
                s = 0.0
            score += s * w

        return min(10.0, max(0.0, score))

    def score_fundamentals(self, market_cap: float = 0) -> float:
        """Proxy fundamental score using market cap category and delivery volume signals."""
        score = 5.0  # default neutral

        # Mid-cap bonus (more breakout potential)
        if 5000 < market_cap <= 50000:
            score += 2.0
        elif market_cap > 50000:
            score += 1.0

        # Recent price action near earnings (proxy: increased volatility)
        if len(self.df) >= 10:
            recent_atr = self.df["atr"].iloc[-5:].mean()
            avg_atr = self.df["atr"].iloc[-60:].mean() if len(self.df) >= 60 else self.df["atr"].mean()
            if not pd.isna(recent_atr) and not pd.isna(avg_atr) and avg_atr > 0:
                if recent_atr / avg_atr > 1.3:
                    score += 1.5  # elevated volatility → possible catalyst

        return min(10.0, max(0.0, score))

    # ──────────────────────────────────────────────────────────────────────────
    # Composite BPS
    # ──────────────────────────────────────────────────────────────────────────

    def compute_bps(
        self,
        bench_df: Optional[pd.DataFrame] = None,
        market_cap: float = 0,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Returns (bps_score_0_100, factor_scores_dict).
        Each factor score is 0-10 before weighting.
        """
        scores = {
            "volume":        self.score_volume(),
            "consolidation": self.score_consolidation(),
            "ma_alignment":  self.score_ma_alignment(),
            "resistance":    self.score_resistance_proximity(),
            "momentum":      self.score_momentum(),
            "rel_strength":  self.score_relative_strength(bench_df),
            "fundamentals":  self.score_fundamentals(market_cap),
        }
        bps = sum(scores[k] * self.WEIGHTS[k] * 10 for k in scores)
        return round(bps, 1), scores

    def get_key_levels(self) -> Dict[str, float]:
        """Return entry, stop, and target levels."""
        df = self.df
        close = df["Close"].iloc[-1]
        atr = df["atr"].iloc[-1] if not pd.isna(df["atr"].iloc[-1]) else close * 0.02

        resistances = sorted(self._get_resistance_levels())
        # Nearest resistance above current price
        targets = [r for r in resistances if r > close]
        target1 = targets[0] if targets else close * 1.10
        target2 = targets[1] if len(targets) > 1 else target1 * 1.05
        target3 = target2 * 1.05

        # Stop: below recent low or 1.5x ATR below close
        recent_low = df["Low"].iloc[-10:].min()
        stop = max(recent_low - atr * 0.5, close - atr * 1.5)
        risk = close - stop
        reward = target1 - close

        return {
            "entry":   round(close, 2),
            "stop":    round(stop, 2),
            "target1": round(target1, 2),
            "target2": round(target2, 2),
            "target3": round(target3, 2),
            "rr_ratio": round(reward / risk, 2) if risk > 0 else 0.0,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Universe-wide computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics_for_universe(
    stocks_data: Dict[str, pd.DataFrame],
    bench_df: Optional[pd.DataFrame] = None,
    market_cap_map: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Run BPS scoring for every stock in stocks_data.

    Returns a DataFrame with one row per ticker, all relevant metrics.
    """
    from tickers import get_company_name, get_sector

    from tickers import NIFTY_50, NIFTY_NEXT_50, NIFTY_MIDCAP_150

    def _infer_mc_category(t: str) -> str:
        base = t.replace(".NS", "")
        if base in NIFTY_50 or base in NIFTY_NEXT_50:
            return "Large Cap"
        elif base in NIFTY_MIDCAP_150:
            return "Mid Cap"
        return "Small Cap"

    rows = []
    first_error: str = ""
    for ticker, df in stocks_data.items():
        if df is None or len(df) < 20:
            continue
        try:
            engine = BreakoutEngine(df, bench_df)
            mc = (market_cap_map or {}).get(ticker, 0)
            bps, factor_scores = engine.compute_bps(bench_df, mc)
            levels = engine.get_key_levels()
            close = df["Close"].iloc[-1]
            prev_close = df["Close"].iloc[-2] if len(df) >= 2 else close
            change_pct = (close - prev_close) / (prev_close + 1e-9) * 100

            vol_ratio = df["Volume"].iloc[-1] / (df["Volume"].iloc[-20:].mean() + 1e-9)
            rsi = df["rsi"].iloc[-1] if "rsi" in df.columns else float("nan")

            rows.append({
                "ticker":       ticker,
                "name":         get_company_name(ticker),
                "sector":       get_sector(ticker),
                "cmp":          round(float(close), 2),
                "change_pct":   round(float(change_pct), 2),
                "bps":          bps,
                "vol_surge":    round(float(vol_ratio), 2),
                "rsi":          round(float(rsi), 1) if not pd.isna(rsi) else None,
                "mc_category":  _infer_mc_category(ticker),
                **{f"score_{k}": round(v, 2) for k, v in factor_scores.items()},
                **levels,
            })
        except Exception as exc:
            if not first_error:
                first_error = f"{ticker}: {type(exc).__name__}: {exc}"
            logger.warning("Engine failed for %s: %s", ticker, exc)
            continue

    if first_error and not rows:
        logger.error("compute_metrics returned 0 rows. First error: %s", first_error)

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out.sort_values("bps", ascending=False, inplace=True)
        df_out.reset_index(drop=True, inplace=True)
    return df_out
