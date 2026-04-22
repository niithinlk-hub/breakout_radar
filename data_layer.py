"""Data fetching layer: batch download, retry, caching."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf

logger = logging.getLogger(__name__)

BATCH_SIZE = 50
MAX_RETRIES = 3
BACKOFF_BASE = 2.0   # seconds


def _download_batch(
    tickers: List[str],
    period: str = "1y",
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """Download one batch; returns {ticker: ohlcv_df}."""
    result: Dict[str, pd.DataFrame] = {}
    try:
        raw = yf.download(
            tickers,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
    except Exception as exc:
        logger.warning("Batch download error: %s", exc)
        return result

    if len(tickers) == 1:
        t = tickers[0]
        df = raw.copy()
        df.dropna(how="all", inplace=True)
        if not df.empty:
            result[t] = df
        return result

    for t in tickers:
        try:
            df = raw[t].copy()
            df.dropna(how="all", inplace=True)
            if not df.empty and len(df) >= 20:
                result[t] = df
        except (KeyError, TypeError):
            pass
    return result


def _retry_singles(
    failed: List[str],
    period: str,
    interval: str,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Retry failed tickers one-by-one with exponential backoff."""
    success: Dict[str, pd.DataFrame] = {}
    still_failed: List[str] = []

    for t in failed:
        fetched = False
        for attempt in range(MAX_RETRIES):
            try:
                df = yf.download(t, period=period, interval=interval,
                                 auto_adjust=True, progress=False)
                df.dropna(how="all", inplace=True)
                if not df.empty and len(df) >= 20:
                    success[t] = df
                    fetched = True
                    break
            except Exception:
                pass
            time.sleep(BACKOFF_BASE ** attempt)
        if not fetched:
            still_failed.append(t)

    return success, still_failed


@st.cache_data(ttl=900, show_spinner=False)
def fetch_universe(
    tickers: Tuple[str, ...],
    period: str = "1y",
    interval: str = "1d",
) -> Tuple[Dict[str, pd.DataFrame], List[str], str]:
    """
    Fetch OHLCV data for all tickers in batches of BATCH_SIZE.

    Parameters
    ----------
    tickers : tuple of ticker strings (tuple for st.cache_data hashability)
    period  : yfinance period string
    interval: yfinance interval string

    Returns
    -------
    (data_dict, failed_list, timestamp_str)
    """
    tickers_list = list(tickers)
    total = len(tickers_list)
    batches = [tickers_list[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    n_batches = len(batches)

    all_data: Dict[str, pd.DataFrame] = {}
    batch_failed: List[str] = []

    progress_bar = st.progress(0.0, text="Initialising download…")
    status_text = st.empty()

    for i, batch in enumerate(batches):
        frac = i / n_batches
        loaded = len(all_data)
        progress_bar.progress(frac, text=f"Fetching batch {i+1}/{n_batches} — {loaded}/{total} stocks loaded")
        status_text.caption(f"Downloading: {', '.join(b.replace('.NS','') for b in batch[:5])}…")

        batch_result = _download_batch(batch, period, interval)
        all_data.update(batch_result)

        missed = [t for t in batch if t not in batch_result]
        batch_failed.extend(missed)

        time.sleep(0.3)

    # Retry failed
    if batch_failed:
        status_text.caption(f"Retrying {len(batch_failed)} failed tickers…")
        recovered, still_failed = _retry_singles(batch_failed, period, interval)
        all_data.update(recovered)
        final_failed = still_failed
    else:
        final_failed = []

    progress_bar.progress(1.0, text=f"Done — {len(all_data)}/{total} stocks loaded")
    status_text.empty()
    progress_bar.empty()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M IST")
    return all_data, final_failed, timestamp


@st.cache_data(ttl=900, show_spinner=False)
def fetch_hourly(
    tickers: Tuple[str, ...],
    period: str = "3mo",
) -> Dict[str, pd.DataFrame]:
    """Fetch hourly data for a smaller subset (used in deep-dive)."""
    result, _, _ = fetch_universe(tickers, period=period, interval="1h")
    return result


def get_nifty50_data() -> Optional[pd.DataFrame]:
    """Fetch ^NSEI (Nifty 50 index) for market regime detection."""
    try:
        df = yf.download("^NSEI", period="1y", interval="1d",
                         auto_adjust=True, progress=False)
        df.dropna(how="all", inplace=True)
        return df if not df.empty else None
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def get_nifty50_cached() -> Optional[pd.DataFrame]:
    return get_nifty50_data()
