"""Data fetching layer: batch download, retry, caching."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf

logger = logging.getLogger(__name__)

BATCH_SIZE = 20
MAX_RETRIES = 2
BACKOFF_BASE = 1.5
MAX_RETRY_TICKERS = 40   # cap to prevent multi-minute hangs
INTER_BATCH_SLEEP = 0.5


def _extract_ticker_data(raw: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """Extract OHLCV for one ticker from a yfinance download result.

    Handles:
    - Flat DataFrame (single-ticker downloads in most yfinance versions)
    - Old MultiIndex: (ticker, field) — ticker at level 0
    - New MultiIndex (yfinance ≥ 0.2.38): (field, ticker) — field at level 0
    """
    if raw is None or raw.empty:
        return None
    try:
        if not isinstance(raw.columns, pd.MultiIndex):
            df = raw.copy()
        elif ticker in raw.columns.get_level_values(1):
            # New yfinance format: Price field at level-0, ticker at level-1
            df = raw.xs(ticker, axis=1, level=1).copy()
        elif ticker in raw.columns.get_level_values(0):
            # Old yfinance format: ticker at level-0
            sub = raw[ticker]
            if isinstance(sub, pd.Series):
                return None
            df = sub.copy()
        else:
            return None

        df.dropna(how="all", inplace=True)
        if df.empty or len(df) < 20:
            return None
        # Ensure standard OHLCV columns exist
        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(set(df.columns)):
            return None
        return df
    except Exception:
        return None


def _download_batch(
    tickers: List[str],
    period: str = "1y",
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """Download one batch; returns {ticker: ohlcv_df}."""
    result: Dict[str, pd.DataFrame] = {}
    try:
        kwargs: dict = dict(
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if len(tickers) > 1:
            kwargs["group_by"] = "ticker"
        raw = yf.download(tickers, **kwargs)
    except Exception as exc:
        logger.warning("Batch download error: %s", exc)
        return result

    if raw is None or (hasattr(raw, "empty") and raw.empty):
        return result

    for t in tickers:
        df = _extract_ticker_data(raw, t)
        if df is not None:
            result[t] = df
    return result


def _retry_singles(
    failed: List[str],
    period: str,
    interval: str,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Retry up to MAX_RETRY_TICKERS failed tickers one-by-one.

    Capped to avoid multi-minute hangs when Yahoo blocks the full universe.
    """
    to_retry = failed[:MAX_RETRY_TICKERS]
    success: Dict[str, pd.DataFrame] = {}
    still_failed: List[str] = list(failed[MAX_RETRY_TICKERS:])

    for t in to_retry:
        fetched = False
        for attempt in range(MAX_RETRIES):
            try:
                raw = yf.download(t, period=period, interval=interval,
                                  auto_adjust=True, progress=False)
                df = _extract_ticker_data(raw, t)
                if df is not None:
                    success[t] = df
                    fetched = True
                    break
            except Exception:
                pass
            time.sleep(BACKOFF_BASE ** attempt)
        if not fetched:
            still_failed.append(t)
        time.sleep(0.2)

    return success, still_failed


@st.cache_data(ttl=900, show_spinner=False)
def fetch_universe(
    tickers: Tuple[str, ...],
    period: str = "1y",
    interval: str = "1d",
) -> Tuple[Dict[str, pd.DataFrame], List[str], str]:
    """Fetch OHLCV for all tickers in batches.

    Returns (data_dict, failed_list, timestamp_str).
    tickers must be a tuple for st.cache_data hashability.
    """
    tickers_list = list(tickers)
    total = len(tickers_list)
    batches = [tickers_list[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    n_batches = len(batches)

    all_data: Dict[str, pd.DataFrame] = {}
    batch_failed: List[str] = []

    progress_bar = st.progress(0.0, text="Initialising download…")
    status_text = st.empty()

    try:
        for i, batch in enumerate(batches):
            frac = i / n_batches
            progress_bar.progress(
                frac,
                text=f"Fetching batch {i + 1}/{n_batches} — {len(all_data)}/{total} stocks loaded",
            )
            status_text.caption(
                f"Downloading: {', '.join(b.replace('.NS', '') for b in batch[:5])}…"
            )

            batch_result = _download_batch(batch, period, interval)
            all_data.update(batch_result)
            missed = [t for t in batch if t not in batch_result]
            batch_failed.extend(missed)

            time.sleep(INTER_BATCH_SLEEP)

        if batch_failed:
            status_text.caption(
                f"Retrying {min(len(batch_failed), MAX_RETRY_TICKERS)} "
                f"of {len(batch_failed)} failed tickers…"
            )
            recovered, still_failed = _retry_singles(batch_failed, period, interval)
            all_data.update(recovered)
            final_failed = still_failed
        else:
            final_failed = []

        progress_bar.progress(1.0, text=f"Done — {len(all_data)}/{total} stocks loaded")

    finally:
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
        raw = yf.download("^NSEI", period="1y", interval="1d",
                          auto_adjust=True, progress=False)
        return _extract_ticker_data(raw, "^NSEI")
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def get_nifty50_cached() -> Optional[pd.DataFrame]:
    return get_nifty50_data()
