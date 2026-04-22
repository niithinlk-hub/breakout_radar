"""Data fetching layer: batch download, retry, caching."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf
import yfinance.shared as yf_shared

logger = logging.getLogger(__name__)

BATCH_SIZE = 20
DOWNLOAD_THREADS = 1
REQUEST_TIMEOUT = 6
RETRY_BATCH_SIZE = 4
RETRY_THREADS = 1
INTER_BATCH_SLEEP_SECONDS = 1.0
RETRY_SLEEP_SECONDS = 1.0
RATE_LIMIT_COOLDOWN_SECONDS = 20


@contextmanager
def _suppress_yfinance_errors():
    """Temporarily mute noisy yfinance error logs; we surface cleaner app messages instead."""
    yf_logger = logging.getLogger("yfinance")
    previous_level = yf_logger.level
    previous_propagate = yf_logger.propagate
    yf_logger.setLevel(logging.CRITICAL)
    yf_logger.propagate = False
    try:
        yield
    finally:
        yf_logger.setLevel(previous_level)
        yf_logger.propagate = previous_propagate


def _clean_frame(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Return a usable OHLCV frame or None when the result is too sparse."""
    cleaned = df.copy()
    cleaned.dropna(how="all", inplace=True)
    if cleaned.empty or len(cleaned) < 20:
        return None
    return cleaned


def _error_kind(message: str) -> str:
    lowered = message.lower()
    if "ratelimit" in lowered or "too many requests" in lowered:
        return "rate_limited"
    if (
        "possibly delisted" in lowered
        or "no data found" in lowered
        or "pricesmissing" in lowered
        or "tickermissing" in lowered
        or "empty or insufficient data" in lowered
    ):
        return "no_data"
    return "transient"


def _download_batch(
    tickers: List[str],
    period: str = "1y",
    interval: str = "1d",
    *,
    threads: int | bool = DOWNLOAD_THREADS,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Download one batch and return ({ticker: frame}, {ticker: error_string})."""
    result: Dict[str, pd.DataFrame] = {}
    errors: Dict[str, str] = {}

    try:
        with _suppress_yfinance_errors():
            raw = yf.download(
                tickers,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=True,
                threads=threads,
                progress=False,
                timeout=REQUEST_TIMEOUT,
            )
    except Exception as exc:
        logger.warning("Batch download error for %s: %s", tickers, exc)
        return result, {ticker: repr(exc) for ticker in tickers}

    yf_errors = {
        ticker.upper(): message
        for ticker, message in getattr(yf_shared, "_ERRORS", {}).items()
    }

    if len(tickers) == 1:
        ticker = tickers[0]
        cleaned = _clean_frame(raw)
        if cleaned is not None:
            result[ticker] = cleaned
        else:
            errors[ticker] = yf_errors.get(ticker.upper(), "Empty or insufficient data")
        return result, errors

    for ticker in tickers:
        try:
            cleaned = _clean_frame(raw[ticker])
            if cleaned is not None:
                result[ticker] = cleaned
                continue
        except (KeyError, TypeError):
            pass

        errors[ticker] = yf_errors.get(ticker.upper(), "Empty or insufficient data")

    return result, errors


def _retry_failed_tickers(
    failed_errors: Dict[str, str],
    period: str,
    interval: str,
    progress_callback: Optional[Callable[[str, float, int], None]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Retry only plausible transient failures and back off once on Yahoo rate limiting."""
    recovered: Dict[str, pd.DataFrame] = {}
    final_errors = dict(failed_errors)

    no_data_failures = {
        ticker: message
        for ticker, message in failed_errors.items()
        if _error_kind(message) == "no_data"
    }
    retry_queue = [
        ticker
        for ticker, message in failed_errors.items()
        if _error_kind(message) != "no_data"
    ]

    if not retry_queue:
        return recovered, no_data_failures

    if any(_error_kind(failed_errors[ticker]) == "rate_limited" for ticker in retry_queue):
        if progress_callback is not None:
            progress_callback(
                f"Yahoo rate limited the scan. Cooling down for {RATE_LIMIT_COOLDOWN_SECONDS}s before one final retry...",
                0.0,
                0,
            )
        time.sleep(RATE_LIMIT_COOLDOWN_SECONDS)

    retry_batches = [
        retry_queue[i:i + RETRY_BATCH_SIZE]
        for i in range(0, len(retry_queue), RETRY_BATCH_SIZE)
    ]
    total_steps = len(retry_batches)

    remaining_errors = {
        ticker: failed_errors[ticker]
        for ticker in retry_queue
    }

    for index, batch in enumerate(retry_batches, start=1):
        if progress_callback is not None:
            progress_callback(
                f"Retry batch {index}/{len(retry_batches)} - {len(remaining_errors)} tickers still missing",
                (index - 1) / max(total_steps, 1),
                len(recovered),
            )

        batch_result, batch_errors = _download_batch(
            batch,
            period,
            interval,
            threads=RETRY_THREADS,
        )
        recovered.update(batch_result)

        for ticker in batch_result:
            remaining_errors.pop(ticker, None)
            final_errors.pop(ticker, None)

        for ticker, message in batch_errors.items():
            remaining_errors[ticker] = message
            final_errors[ticker] = message

        time.sleep(RETRY_SLEEP_SECONDS)

    # If Yahoo is still rate-limiting after the cooldown pass, stop immediately.
    persistent_rate_limits = {
        ticker: message
        for ticker, message in remaining_errors.items()
        if _error_kind(message) == "rate_limited"
    }
    non_rate_remaining = {
        ticker: message
        for ticker, message in remaining_errors.items()
        if _error_kind(message) != "rate_limited"
    }

    # Do one last single-symbol pass only for non-rate transient failures.
    single_pass_total = len(non_rate_remaining)
    for index, ticker in enumerate(list(non_rate_remaining.keys()), start=1):
        if progress_callback is not None:
            progress_callback(
                f"Final single retry {index}/{single_pass_total} - {ticker.replace('.NS', '')}",
                (len(retry_batches) + index - 1) / max(len(retry_batches) + single_pass_total, 1),
                len(recovered),
            )

        result, errors = _download_batch([ticker], period, interval, threads=False)
        if ticker in result:
            recovered[ticker] = result[ticker]
            final_errors.pop(ticker, None)
        else:
            final_errors[ticker] = errors.get(ticker, non_rate_remaining[ticker])

        if index < single_pass_total:
            time.sleep(RETRY_SLEEP_SECONDS)

    final_errors.update(persistent_rate_limits)
    final_errors.update(no_data_failures)
    return recovered, final_errors


@st.cache_data(ttl=3600, show_spinner=False)
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
    batch_failed_errors: Dict[str, str] = {}

    progress_bar = st.progress(0.0, text="Initializing download...")
    status_text = st.empty()

    for index, batch in enumerate(batches, start=1):
        frac = (index - 1) / max(n_batches, 1)
        loaded = len(all_data)
        progress_bar.progress(frac, text=f"Fetching batch {index}/{n_batches} - {loaded}/{total} stocks loaded")
        status_text.caption(f"Downloading: {', '.join(ticker.replace('.NS', '') for ticker in batch[:5])}...")

        batch_result, batch_errors = _download_batch(batch, period, interval)
        all_data.update(batch_result)
        batch_failed_errors.update(batch_errors)

        time.sleep(INTER_BATCH_SLEEP_SECONDS)

    if batch_failed_errors:
        base_progress = len(all_data) / max(total, 1)

        def update_retry_progress(message: str, retry_fraction: float, recovered_count: int) -> None:
            current_progress = base_progress + (1.0 - base_progress) * retry_fraction
            progress_bar.progress(
                min(current_progress, 0.99),
                text=f"Retrying failures - {len(all_data) + recovered_count}/{total} stocks loaded",
            )
            status_text.caption(message)

        status_text.caption(f"Retrying {len(batch_failed_errors)} failed tickers with rate-limit protection...")
        recovered, final_errors = _retry_failed_tickers(
            batch_failed_errors,
            period,
            interval,
            progress_callback=update_retry_progress,
        )
        all_data.update(recovered)
        final_failed = list(final_errors.keys())
    else:
        final_failed = []

    progress_bar.progress(1.0, text=f"Done - {len(all_data)}/{total} stocks loaded")
    status_text.empty()
    progress_bar.empty()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M IST")
    return all_data, final_failed, timestamp


@st.cache_data(ttl=3600, show_spinner=False)
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
        with _suppress_yfinance_errors():
            df = yf.download(
                "^NSEI",
                period="1y",
                interval="1d",
                auto_adjust=True,
                progress=False,
                timeout=REQUEST_TIMEOUT,
                threads=False,
            )
        cleaned = _clean_frame(df)
        return cleaned if cleaned is not None else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_nifty50_cached() -> Optional[pd.DataFrame]:
    return get_nifty50_data()
