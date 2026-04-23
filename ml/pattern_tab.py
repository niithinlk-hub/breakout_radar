"""Standalone Pattern Scanner tab — independent of ML model.

Two modes:
  - Browse by pattern: pick one of 20 patterns, see all firing tickers.
  - Universe overview: every ticker ranked by (pattern_count, avg_conf).
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import pandas as pd
import streamlit as st

from . import data_loader
from .patterns import (PATTERN_NAMES, PatternScanner, scan_ticker_list,
                       scan_universe)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 4)
def _load_universe(tickers: tuple, period: str = "3y") -> Dict[str, pd.DataFrame]:
    data = data_loader.fetch_universe(list(tickers), period=period, use_cache=True)
    data = data_loader.apply_liquidity_filter(data)
    return data


def _resolve_tickers() -> List[str]:
    try:
        from tickers import get_tickers_ns
        return list(get_tickers_ns())
    except Exception:
        try:
            from tickers import NIFTY_50_TICKERS_NS
            return list(NIFTY_50_TICKERS_NS)
        except Exception:
            return []


def _render_overview(stocks: Dict[str, pd.DataFrame]) -> None:
    st.markdown("**Universe overview — every ticker with ≥1 pattern firing**")
    df = scan_universe(stocks)
    if df.empty:
        st.info("No patterns firing across the universe right now.")
        return

    c1, c2 = st.columns(2)
    min_patterns = c1.slider("Min # patterns", 1, 5, 1)
    min_conf = c2.slider("Min avg confidence", 0.0, 1.0, 0.5, step=0.05)

    view = df[(df["pattern_count"] >= min_patterns) & (df["avg_confidence"] >= min_conf)].copy()
    if view.empty:
        st.warning("No tickers pass filters.")
        return

    show = view[["ticker", "pattern_count", "avg_confidence",
                 "close", "patterns", "last_bar"]].copy()
    st.dataframe(
        show, hide_index=True, use_container_width=True, height=500,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker"),
            "pattern_count": st.column_config.NumberColumn("# Patterns"),
            "avg_confidence": st.column_config.NumberColumn("Avg Conf", format="%.2f"),
            "close": st.column_config.NumberColumn("Close", format="%.2f"),
            "patterns": st.column_config.TextColumn("Patterns Detected", width="large"),
            "last_bar": st.column_config.TextColumn("Bar"),
        },
    )
    csv = show.to_csv(index=False)
    st.download_button(
        "Download overview CSV", data=csv,
        file_name=f"pattern_overview_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    # Drill-down: pick a ticker to see details
    picked = st.selectbox("Inspect ticker details", ["—"] + view["ticker"].tolist())
    if picked != "—":
        row = view[view["ticker"] == picked].iloc[0]
        rows = []
        for name, d in row["details"].items():
            extras = {k: v for k, v in d.items() if k not in ("confidence", "category")}
            rows.append({
                "Pattern": name,
                "Confidence": f"{d.get('confidence', 0):.2f}",
                "Category": d.get("category", "—"),
                "Details": ", ".join(f"{k}={v}" for k, v in extras.items()),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)


def _render_by_pattern(stocks: Dict[str, pd.DataFrame]) -> None:
    st.markdown("**Browse by single pattern**")
    pattern = st.selectbox("Pattern", PATTERN_NAMES)
    if not pattern:
        return
    df = scan_ticker_list(pattern, stocks)
    if df.empty:
        st.info(f"No tickers currently firing **{pattern}**.")
        return
    st.caption(f"{len(df)} tickers firing — sorted by confidence")
    show = df[["ticker", "confidence", "category", "close", "last_bar"]].copy()
    st.dataframe(
        show, hide_index=True, use_container_width=True, height=500,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker"),
            "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
            "category": st.column_config.TextColumn("Type"),
            "close": st.column_config.NumberColumn("Close", format="%.2f"),
            "last_bar": st.column_config.TextColumn("Bar"),
        },
    )

    # Drill-down details for top picks
    picked = st.selectbox("Inspect ticker details", ["—"] + df["ticker"].tolist()[:50])
    if picked != "—":
        row = df[df["ticker"] == picked].iloc[0]
        st.json(row["details"])


def render_pattern_scanner() -> None:
    """Main entry — call from app.py."""
    st.markdown("### 🔍 Pattern Scanner — 20 bullish patterns")
    st.caption(
        "Pure technical scanner. Independent of the ML model — works even before "
        "the model is trained. Each pattern returns a confidence (0-1) derived "
        "from its own structural fit, not from the classifier."
    )

    tickers = _resolve_tickers()
    if not tickers:
        st.error("No tickers resolved from `tickers.py`.")
        return
    with st.spinner(f"Loading cached data for {len(tickers)} tickers…"):
        stocks = _load_universe(tuple(tickers))

    if not stocks:
        st.warning("No data available. Run `python -m ml.train` once to populate the cache, "
                   "or wait for the bulk downloader to finish.")
        return

    tab_over, tab_by = st.tabs(["Universe Overview", "Browse by Pattern"])
    with tab_over:
        _render_overview(stocks)
    with tab_by:
        _render_by_pattern(stocks)
