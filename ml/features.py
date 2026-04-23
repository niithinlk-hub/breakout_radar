"""Feature engineering — base technicals + U6 expansions.

Sector-aware RS is deferred to Phase 2 (requires ticker_sector_map.csv).
Patterns as features land in Phase 2 (UPGRADE 7+8).

All features are computed per ticker from OHLCV + optional benchmark.
`build_features(df, bench_df)` returns a DataFrame of the same index
as df with NaN-padded rows where warm-up isn't satisfied.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator


FEATURE_COLS: List[str] = [
    # returns
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    # trend / EMA relative
    "px_over_ema8", "px_over_ema21", "px_over_ema50", "px_over_ema200",
    "ema8_over_ema21", "ema21_over_ema50", "ema50_over_ema200",
    "ema50_slope_20",
    # momentum
    "rsi_14", "rsi_14_slope_5",
    "stoch_k", "stoch_d", "stoch_k_minus_d",
    "macd_line", "macd_hist", "macd_hist_slope_3",
    "adx_14", "adx_rising",
    # volatility / bands
    "bb_width", "bb_pct_b", "atr_pct",
    "dist_ema50_in_atr",
    # volume
    "vol_ratio_20", "vol_ratio_5", "mfi_14", "obv_z_20",
    # highs
    "px_over_20d_high", "px_over_52w_high", "dist_20d_high_pct",
    # microstructure / gap (U6)
    "gap_pct", "gap_up_count_5d", "intraday_range_over_atr",
    "close_loc_in_range", "accum_3d",
    # higher-timeframe (U6)
    "wk_rsi_14", "wk_close_over_ema21", "monthly_ret",
    # volume profile (U6)
    "vwma_20_dist_pct",
    # relative strength vs benchmark
    "rs_20d_vs_bench", "rs_60d_vs_bench",
]


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace(0, np.nan)) - 1.0


def _weekly_resample(df: pd.DataFrame) -> pd.DataFrame:
    wk = df[["Open", "High", "Low", "Close", "Volume"]].resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()
    return wk


def build_features(df: pd.DataFrame,
                   bench_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Build feature DataFrame aligned to df.index."""
    if df is None or len(df) < 60:
        return pd.DataFrame(index=df.index if df is not None else None,
                            columns=FEATURE_COLS, dtype=float)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]
    openp = df["Open"]

    f = pd.DataFrame(index=df.index)

    # Returns
    f["ret_1d"] = close.pct_change(1)
    f["ret_5d"] = close.pct_change(5)
    f["ret_10d"] = close.pct_change(10)
    f["ret_20d"] = close.pct_change(20)
    f["ret_60d"] = close.pct_change(60)

    # EMAs
    ema8 = EMAIndicator(close, window=8).ema_indicator()
    ema21 = EMAIndicator(close, window=21).ema_indicator()
    ema50 = EMAIndicator(close, window=50).ema_indicator()
    ema200 = EMAIndicator(close, window=200).ema_indicator()
    f["px_over_ema8"] = _safe_div(close, ema8)
    f["px_over_ema21"] = _safe_div(close, ema21)
    f["px_over_ema50"] = _safe_div(close, ema50)
    f["px_over_ema200"] = _safe_div(close, ema200)
    f["ema8_over_ema21"] = _safe_div(ema8, ema21)
    f["ema21_over_ema50"] = _safe_div(ema21, ema50)
    f["ema50_over_ema200"] = _safe_div(ema50, ema200)
    f["ema50_slope_20"] = (ema50 - ema50.shift(20)) / ema50.shift(20)

    # Momentum
    rsi = RSIIndicator(close, window=14).rsi()
    f["rsi_14"] = rsi
    f["rsi_14_slope_5"] = rsi - rsi.shift(5)

    stoch = StochasticOscillator(high=high, low=low, close=close,
                                 window=14, smooth_window=3)
    f["stoch_k"] = stoch.stoch()
    f["stoch_d"] = stoch.stoch_signal()
    f["stoch_k_minus_d"] = f["stoch_k"] - f["stoch_d"]

    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    f["macd_line"] = macd.macd()
    f["macd_hist"] = macd.macd_diff()
    f["macd_hist_slope_3"] = f["macd_hist"] - f["macd_hist"].shift(3)

    adx = ADXIndicator(high=high, low=low, close=close, window=14).adx()
    f["adx_14"] = adx
    f["adx_rising"] = (adx - adx.shift(5)).gt(0).astype(float)

    # Volatility / bands
    bb = BollingerBands(close, window=20, window_dev=2)
    bb_high = bb.bollinger_hband()
    bb_low = bb.bollinger_lband()
    bb_mid = bb.bollinger_mavg()
    f["bb_width"] = (bb_high - bb_low) / bb_mid.replace(0, np.nan)
    f["bb_pct_b"] = (close - bb_low) / (bb_high - bb_low).replace(0, np.nan)

    atr = AverageTrueRange(high=high, low=low, close=close,
                           window=14).average_true_range()
    f["atr_pct"] = atr / close.replace(0, np.nan)
    f["dist_ema50_in_atr"] = (close - ema50) / atr.replace(0, np.nan)

    # Volume
    vol_ma20 = vol.rolling(20).mean()
    vol_ma5 = vol.rolling(5).mean()
    f["vol_ratio_20"] = vol / vol_ma20.replace(0, np.nan)
    f["vol_ratio_5"] = vol_ma5 / vol_ma20.replace(0, np.nan)
    f["mfi_14"] = MFIIndicator(high=high, low=low, close=close, volume=vol,
                               window=14).money_flow_index()
    obv = OnBalanceVolumeIndicator(close=close, volume=vol).on_balance_volume()
    obv_mean = obv.rolling(20).mean()
    obv_std = obv.rolling(20).std()
    f["obv_z_20"] = (obv - obv_mean) / obv_std.replace(0, np.nan)

    # Highs
    roll20_high = high.rolling(20).max()
    roll252_high = high.rolling(252).max()
    f["px_over_20d_high"] = close / roll20_high.replace(0, np.nan)
    f["px_over_52w_high"] = close / roll252_high.replace(0, np.nan)
    f["dist_20d_high_pct"] = (close - roll20_high) / roll20_high.replace(0, np.nan)

    # U6 microstructure / gap
    prev_close = close.shift(1)
    f["gap_pct"] = (openp - prev_close) / prev_close.replace(0, np.nan)
    f["gap_up_count_5d"] = (f["gap_pct"] > 0.005).rolling(5).sum()
    day_range = (high - low).replace(0, np.nan)
    f["intraday_range_over_atr"] = day_range / atr.replace(0, np.nan)
    f["close_loc_in_range"] = (close - low) / day_range
    accum_bar = ((close > (high + low) / 2) & (vol > vol_ma20)).astype(float)
    f["accum_3d"] = accum_bar.rolling(3).sum()

    # U6 higher-timeframe (shift to avoid look-ahead)
    wk = _weekly_resample(df)
    if len(wk) >= 30:
        wk_rsi = RSIIndicator(wk["Close"], window=14).rsi()
        wk_ema21 = EMAIndicator(wk["Close"], window=21).ema_indicator()
        wk_flag = (wk["Close"] > wk_ema21).astype(float)
        wk_rsi_sh = wk_rsi.shift(1)
        wk_flag_sh = wk_flag.shift(1)
        f["wk_rsi_14"] = wk_rsi_sh.reindex(df.index, method="ffill")
        f["wk_close_over_ema21"] = wk_flag_sh.reindex(df.index, method="ffill")
    else:
        f["wk_rsi_14"] = np.nan
        f["wk_close_over_ema21"] = np.nan
    f["monthly_ret"] = close.pct_change(21)

    # U6 volume profile
    typical = (high + low + close) / 3
    vwma_num = (typical * vol).rolling(20).sum()
    vwma_den = vol.rolling(20).sum().replace(0, np.nan)
    vwma20 = vwma_num / vwma_den
    f["vwma_20_dist_pct"] = (close - vwma20) / vwma20.replace(0, np.nan)

    # Relative strength vs benchmark
    if bench_df is not None and "Close" in bench_df.columns and len(bench_df) > 60:
        bench_close = bench_df["Close"].reindex(df.index, method="ffill")
        stock_20 = close.pct_change(20)
        stock_60 = close.pct_change(60)
        bench_20 = bench_close.pct_change(20)
        bench_60 = bench_close.pct_change(60)
        f["rs_20d_vs_bench"] = stock_20 - bench_20
        f["rs_60d_vs_bench"] = stock_60 - bench_60
    else:
        f["rs_20d_vs_bench"] = np.nan
        f["rs_60d_vs_bench"] = np.nan

    return f[FEATURE_COLS]
