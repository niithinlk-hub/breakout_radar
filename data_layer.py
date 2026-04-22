"""Data fetching layer — single bulk yf.download, no cooldowns, no retries.

Mirrors the approach used in the stock-pattern-screener reference repo
which reliably pulls 500+ tickers from Yahoo without rate-limit trips.
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

MIN_ROWS = 60


def _clean_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Drop NaN rows, strip timezone, enforce column subset."""
    if df is None or df.empty:
        return None
    df = df.dropna(how="all")
    if df.empty or len(df) < MIN_ROWS:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df.columns = [str(c) for c in df.columns]
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(df.columns)):
        return None
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


@st.cache_data(ttl=14400, show_spinner=False)
def fetch_universe(
    tickers: Tuple[str, ...],
    period: str = "1y",
    interval: str = "1d",
) -> Tuple[Dict[str, pd.DataFrame], List[str], str]:
    """Bulk-download OHLCV for all tickers in a single yf.download call.

    Returns (data_dict, failed_list, timestamp_str).
    tickers must be a tuple for st.cache_data hashability.
    """
    tickers_list = list(tickers)
    total = len(tickers_list)
    if total == 0:
        return {}, [], datetime.now().strftime("%Y-%m-%d %H:%M IST")

    progress_bar = st.progress(0.0, text=f"Downloading {total} tickers in one bulk request…")

    try:
        raw = yf.download(
            tickers=tickers_list,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
    except Exception as exc:
        progress_bar.empty()
        logger.error("yfinance bulk download error: %s", exc)
        st.error(f"Download failed: {exc}")
        return {}, tickers_list, datetime.now().strftime("%Y-%m-%d %H:%M IST")

    progress_bar.progress(0.5, text="Parsing OHLCV frames…")

    result: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []

    if total == 1:
        cleaned = _clean_df(raw)
        if cleaned is not None:
            result[tickers_list[0]] = cleaned
        else:
            failed.append(tickers_list[0])
    else:
        for t in tickers_list:
            try:
                if t not in raw.columns.get_level_values(0):
                    failed.append(t)
                    continue
                sub = raw[t]
                cleaned = _clean_df(sub.copy())
                if cleaned is None:
                    failed.append(t)
                else:
                    result[t] = cleaned
            except (KeyError, TypeError, AttributeError):
                failed.append(t)

    progress_bar.progress(1.0, text=f"Loaded {len(result)}/{total} tickers")
    progress_bar.empty()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M IST")
    return result, failed, timestamp


@st.cache_data(ttl=14400, show_spinner=False)
def fetch_hourly(
    tickers: Tuple[str, ...],
    period: str = "3mo",
) -> Dict[str, pd.DataFrame]:
    """Fetch hourly data for a subset (deep-dive view)."""
    result, _, _ = fetch_universe(tickers, period=period, interval="1h")
    return result


def get_nifty50_data() -> Optional[pd.DataFrame]:
    """Fetch ^NSEI (Nifty 50 index) for market regime / relative-strength."""
    try:
        raw = yf.download(
            "^NSEI",
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        return _clean_df(raw)
    except Exception as exc:
        logger.warning("Nifty 50 fetch failed: %s", exc)
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_nifty50_cached() -> Optional[pd.DataFrame]:
    return get_nifty50_data()
