"""Bulk OHLCV loader with local Parquet cache.

Single `yf.download` per call — no per-ticker loops, no cooldowns.
Cache is local-only (Streamlit Cloud FS ephemeral).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MIN_HISTORY_DAYS = 252
MIN_TURNOVER_INR = 5e7  # ₹5 Cr daily avg
BENCH_TICKER = "^NSEI"
VIX_TICKER = "^INDIAVIX"


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("^", "_").replace("/", "_")
    return CACHE_DIR / f"{safe}.parquet"


def _is_fresh(path: Path, max_age_days: int = 1) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime) < timedelta(days=max_age_days)


def _clean_single(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output for a single ticker."""
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    needed = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in needed if c in df.columns]].dropna(how="all")
    return df


def fetch_universe(
    tickers: List[str],
    period: str = "3y",
    use_cache: bool = True,
    max_age_days: int = 1,
) -> Dict[str, pd.DataFrame]:
    """Bulk-download OHLCV for full universe. Returns {ticker: df}.

    Serves from Parquet cache when fresh. On cache miss, does ONE
    `yf.download` for the uncached set.
    """
    out: Dict[str, pd.DataFrame] = {}
    to_fetch: List[str] = []

    if use_cache:
        for t in tickers:
            p = _cache_path(t)
            if _is_fresh(p, max_age_days):
                try:
                    out[t] = pd.read_parquet(p)
                except Exception:
                    to_fetch.append(t)
            else:
                to_fetch.append(t)
    else:
        to_fetch = list(tickers)

    if to_fetch:
        raw = yf.download(
            tickers=to_fetch,
            period=period,
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False,
            auto_adjust=True,
        )
        if len(to_fetch) == 1:
            df = _clean_single(raw)
            if not df.empty:
                out[to_fetch[0]] = df
                if use_cache:
                    df.to_parquet(_cache_path(to_fetch[0]))
        else:
            for t in to_fetch:
                try:
                    sub = raw[t] if t in raw.columns.get_level_values(0) else None
                except Exception:
                    sub = None
                if sub is None:
                    continue
                df = _clean_single(sub)
                if df.empty:
                    continue
                out[t] = df
                if use_cache:
                    try:
                        df.to_parquet(_cache_path(t))
                    except Exception:
                        pass
    return out


def apply_liquidity_filter(
    data: Dict[str, pd.DataFrame],
    min_turnover_inr: float = MIN_TURNOVER_INR,
    min_history: int = MIN_HISTORY_DAYS,
) -> Dict[str, pd.DataFrame]:
    """Keep only tickers with enough history and avg daily turnover."""
    kept: Dict[str, pd.DataFrame] = {}
    for t, df in data.items():
        if df is None or len(df) < min_history:
            continue
        recent = df.iloc[-20:]
        turnover = (recent["Close"] * recent["Volume"]).mean()
        if turnover < min_turnover_inr:
            continue
        kept[t] = df
    return kept


def get_benchmark(period: str = "3y", use_cache: bool = True,
                  max_age_days: int = 1) -> pd.DataFrame:
    p = _cache_path(BENCH_TICKER)
    if use_cache and _is_fresh(p, max_age_days):
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    raw = yf.download(BENCH_TICKER, period=period, interval="1d",
                      progress=False, auto_adjust=True)
    df = _clean_single(raw)
    if use_cache and not df.empty:
        try:
            df.to_parquet(p)
        except Exception:
            pass
    return df


def get_vix(period: str = "3y", use_cache: bool = True,
            max_age_days: int = 1) -> Optional[pd.DataFrame]:
    """INDIAVIX — graceful skip if unavailable."""
    try:
        p = _cache_path(VIX_TICKER)
        if use_cache and _is_fresh(p, max_age_days):
            return pd.read_parquet(p)
        raw = yf.download(VIX_TICKER, period=period, interval="1d",
                          progress=False, auto_adjust=True)
        df = _clean_single(raw)
        if df.empty:
            return None
        if use_cache:
            try:
                df.to_parquet(p)
            except Exception:
                pass
        return df
    except Exception:
        return None


def universe_health(data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Summary stats for Universe Health tab."""
    if not data:
        return {"tickers": 0, "avg_history_days": 0, "avg_turnover_cr": 0.0,
                "latest_bar": "n/a"}
    hist = [len(d) for d in data.values()]
    turnovers = [(d["Close"].iloc[-20:] * d["Volume"].iloc[-20:]).mean()
                 for d in data.values() if len(d) >= 20]
    last = max((d.index[-1] for d in data.values()), default=None)
    return {
        "tickers": len(data),
        "avg_history_days": float(sum(hist) / max(1, len(hist))),
        "avg_turnover_cr": float(sum(turnovers) / max(1, len(turnovers)) / 1e7),
        "latest_bar": last.strftime("%Y-%m-%d") if last is not None else "n/a",
    }
