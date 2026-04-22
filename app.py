"""Indian Market Breakout Radar — Bloomberg-grade NSE/BSE breakout scanner."""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config must be first ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Market Breakout Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Local modules
from styles import inject_css
from tickers import (
    NIFTY_50, NIFTY_NEXT_50, NIFTY_MIDCAP_150, NIFTY_SMALLCAP_250,
    ADDITIONAL_LIQUID, SECTOR_LIST, get_universe, get_tickers_ns,
    get_company_name, get_sector,
)
from data_layer import fetch_universe, get_nifty50_cached
from breakout_engine import BreakoutEngine, compute_metrics_for_universe
from pattern_detection import PatternDetector, get_pattern_labels
import charts as ch

inject_css()

# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "metrics_df": None,
        "stocks_data": None,
        "failed_tickers": [],
        "data_timestamp": None,
        "bench_df": None,
        "watchlist": [],
        "selected_ticker": None,
        "universe_category": "full",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> Dict:
    """Render sidebar controls and return filter settings."""
    with st.sidebar:
        st.markdown(
            '<div class="terminal-banner">📡 BREAKOUT RADAR<br>'
            '<span style="font-size:0.7rem;color:#8B949E">NSE/BSE Technical Scanner</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-header">Universe</div>', unsafe_allow_html=True)
        universe_options = {
            "Full 475":             "full",
            "Nifty 50":             "nifty50",
            "Nifty Next 50":        "next50",
            "Nifty Midcap 150":     "midcap150",
            "Nifty Smallcap 250":   "smallcap250",
        }
        universe_label = st.selectbox(
            "Stock Universe", list(universe_options.keys()), index=0, label_visibility="collapsed"
        )
        universe_cat = universe_options[universe_label]

        # Custom ticker input
        custom_raw = st.text_area(
            "Add custom tickers (NSE symbols, comma-separated)",
            placeholder="TATATECH, MANKIND, JSWENERGY",
            height=68,
        )
        custom_tickers = []
        if custom_raw.strip():
            custom_tickers = [t.strip().upper() + ".NS" for t in custom_raw.split(",") if t.strip()]

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.metrics_df = None
            st.session_state.stocks_data = None
            st.rerun()

        if st.session_state.data_timestamp:
            st.caption(f"Last refresh: {st.session_state.data_timestamp}")

        st.divider()
        st.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)

        min_bps = st.slider("Min BPS Score", 0, 100, 60, step=5, help="Breakout Probability Score threshold")

        sectors = ["All Sectors"] + sorted(SECTOR_LIST)
        selected_sectors = st.multiselect("Sector Filter", sectors[1:], default=[])

        mc_filter = st.multiselect("Market Cap", ["Large Cap", "Mid Cap", "Small Cap"], default=[])

        pattern_options = [
            "Cup and Handle", "Ascending Triangle", "Bull Flag/Pennant",
            "Flat Base", "VCP (Minervini)", "Pocket Pivot",
            "Inside Day Breakout Setup", "Darvas Box",
        ]
        selected_patterns = st.multiselect("Pattern Filter", pattern_options, default=[])

        st.divider()
        st.markdown('<div class="section-header">Chart</div>', unsafe_allow_html=True)
        timeframe = st.radio("Timeframe", ["Daily", "Weekly"], horizontal=True)

        st.divider()
        # Failed tickers expander
        if st.session_state.failed_tickers:
            with st.expander(f"⚠ Failed Tickers ({len(st.session_state.failed_tickers)})"):
                for t in st.session_state.failed_tickers:
                    st.caption(t.replace(".NS", ""))

    return {
        "universe_cat":    universe_cat,
        "custom_tickers":  custom_tickers,
        "min_bps":         min_bps,
        "selected_sectors":selected_sectors,
        "mc_filter":       mc_filter,
        "selected_patterns": selected_patterns,
        "timeframe":       timeframe,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(universe_cat: str, custom_tickers: List[str]) -> None:
    """Fetch data for universe if not already loaded."""
    base_tickers = get_tickers_ns(universe_cat)
    all_tickers = list(dict.fromkeys(base_tickers + custom_tickers))

    # Check if we need to re-fetch
    if (st.session_state.stocks_data is not None and
            st.session_state.universe_category == universe_cat and
            st.session_state.metrics_df is not None):
        return

    st.session_state.universe_category = universe_cat

    with st.spinner("Loading market data…"):
        stocks_data, failed, timestamp = fetch_universe(
            tuple(all_tickers), period="1y", interval="1d"
        )
        bench_df = get_nifty50_cached()

    st.session_state.stocks_data = stocks_data
    st.session_state.bench_df = bench_df
    st.session_state.failed_tickers = failed
    st.session_state.data_timestamp = timestamp

    with st.spinner("Computing breakout scores…"):
        metrics_df = compute_metrics_for_universe(stocks_data, bench_df)

    # Detect patterns and add to metrics
    pattern_col = []
    for _, row in metrics_df.iterrows():
        t = row["ticker"]
        df_stock = stocks_data.get(t)
        if df_stock is not None and len(df_stock) >= 20:
            try:
                detector = PatternDetector(df_stock)
                best_name, _ = detector.get_best_pattern()
                pattern_col.append(best_name)
            except Exception:
                pattern_col.append("—")
        else:
            pattern_col.append("—")

    metrics_df["pattern"] = pattern_col
    st.session_state.metrics_df = metrics_df


# ─────────────────────────────────────────────────────────────────────────────
# Market regime detection
# ─────────────────────────────────────────────────────────────────────────────

def get_market_regime(bench_df: Optional[pd.DataFrame]) -> Tuple[str, str]:
    """Return (label, color) for market regime."""
    if bench_df is None or len(bench_df) < 200:
        return "Unknown", "#8B949E"
    close = bench_df["Close"].iloc[-1]
    ema200 = bench_df["Close"].ewm(span=200).mean().iloc[-1]
    if close > ema200 * 1.02:
        return "BULL", "#00FF88"
    elif close < ema200 * 0.98:
        return "BEAR", "#FF4444"
    return "NEUTRAL", "#FFB800"


# ─────────────────────────────────────────────────────────────────────────────
# Apply filters
# ─────────────────────────────────────────────────────────────────────────────

def apply_filters(df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    filtered = df.copy()
    filtered = filtered[filtered["bps"] >= settings["min_bps"]]
    if settings["selected_sectors"]:
        filtered = filtered[filtered["sector"].isin(settings["selected_sectors"])]
    if settings["mc_filter"]:
        filtered = filtered[filtered["mc_category"].isin(settings["mc_filter"])]
    if settings["selected_patterns"]:
        pat_mask = filtered["pattern"].apply(
            lambda p: any(pat in str(p) for pat in settings["selected_patterns"])
        )
        filtered = filtered[pat_mask]
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Breakout Radar
# ─────────────────────────────────────────────────────────────────────────────

def tab_breakout_radar(filtered_df: pd.DataFrame, full_df: pd.DataFrame, settings: Dict) -> None:
    bench_df = st.session_state.bench_df
    regime_label, regime_color = get_market_regime(bench_df)

    # ── Summary metrics ──
    total_scanned = len(full_df)
    breakout_candidates = len(full_df[full_df["bps"] >= 70])
    avg_rsi = full_df["rsi"].mean() if "rsi" in full_df.columns else 50
    breadth = f"{avg_rsi:.1f}"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Stocks Scanned", f"{total_scanned:,}")
    c2.metric("Breakout Candidates (BPS≥70)", f"{breakout_candidates:,}")
    c3.metric("Showing (filtered)", f"{len(filtered_df):,}")
    c4.metric("Avg RSI (universe)", breadth)
    c5.metric(
        "Market Regime",
        regime_label,
        delta=None,
        help="Based on Nifty 50 vs 200 EMA",
    )

    st.divider()

    if filtered_df.empty:
        st.info("No stocks match the current filters. Try lowering the BPS threshold or adjusting filters.")
        return

    # ── Display columns selection ──
    display_cols = [
        "ticker", "name", "sector", "cmp", "change_pct",
        "bps", "vol_surge", "rsi", "pattern",
        "target1", "rr_ratio",
    ]
    col_renames = {
        "ticker": "Ticker", "name": "Company", "sector": "Sector",
        "cmp": "CMP (₹)", "change_pct": "Chg %",
        "bps": "BPS", "vol_surge": "Vol Surge",
        "rsi": "RSI", "pattern": "Pattern",
        "target1": "T1 (₹)", "rr_ratio": "R:R",
    }

    display_df = filtered_df[display_cols].copy()
    display_df.rename(columns=col_renames, inplace=True)

    # Color-code BPS
    def bps_color(val: float) -> str:
        if val >= 70:
            return "color: #00FF88; font-weight:700"
        elif val >= 50:
            return "color: #FFB800; font-weight:700"
        return "color: #FF4444"

    def chg_color(val: float) -> str:
        return "color: #00FF88" if val >= 0 else "color: #FF4444"

    styled = display_df.style \
        .map(bps_color, subset=["BPS"]) \
        .map(chg_color, subset=["Chg %"]) \
        .format({
            "CMP (₹)": "₹{:.2f}",
            "Chg %": "{:+.2f}%",
            "BPS": "{:.1f}",
            "Vol Surge": "{:.2f}x",
            "RSI": "{:.0f}",
            "T1 (₹)": "₹{:.2f}",
            "R:R": "{:.2f}",
        }, na_rep="—") \
        .background_gradient(subset=["BPS"], cmap="RdYlGn", vmin=0, vmax=100)

    st.dataframe(
        styled,
        use_container_width=True,
        height=min(600, 36 + len(display_df) * 35),
        on_select="ignore",
    )

    # ── Quick select for deep dive ──
    st.caption("Select a stock for deep-dive analysis →")
    top_tickers = filtered_df["ticker"].str.replace(".NS", "").tolist()
    selected = st.selectbox(
        "Open in Analysis tab",
        top_tickers,
        index=0,
        key="radar_select",
        label_visibility="collapsed",
    )
    if st.button("Analyse →", use_container_width=False):
        st.session_state.selected_ticker = selected + ".NS"
        st.info("Switch to the 'Analysis' tab to see the deep-dive.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Analysis
# ─────────────────────────────────────────────────────────────────────────────

def tab_analysis(full_df: pd.DataFrame, settings: Dict) -> None:
    stocks_data = st.session_state.stocks_data
    bench_df = st.session_state.bench_df

    if stocks_data is None:
        st.warning("Load data first.")
        return

    available_tickers = sorted([t.replace(".NS", "") for t in stocks_data.keys()])

    default_idx = 0
    if st.session_state.selected_ticker:
        plain = st.session_state.selected_ticker.replace(".NS", "")
        if plain in available_tickers:
            default_idx = available_tickers.index(plain)

    col_sel, col_empty = st.columns([3, 1])
    with col_sel:
        chosen = st.selectbox(
            "Select stock for analysis",
            available_tickers,
            index=default_idx,
            label_visibility="collapsed",
        )

    ticker_ns = chosen + ".NS"
    df_stock = stocks_data.get(ticker_ns)

    if df_stock is None or df_stock.empty:
        st.error(f"No data for {chosen}.")
        return

    # Recompute engine for this stock
    engine = BreakoutEngine(df_stock, bench_df)
    bps, factor_scores = engine.compute_bps(bench_df)
    levels = engine.get_key_levels()

    # Detect patterns
    detector = PatternDetector(df_stock)
    all_patterns = detector.detect_all()

    # ── Header row ──
    st.markdown(
        f"<h2 style='color:#4488FF;margin-bottom:4px'>{get_company_name(ticker_ns)} "
        f"<span style='color:#8B949E;font-size:1rem'>({chosen})</span></h2>"
        f"<p style='color:#8B949E;margin:0'>{get_sector(ticker_ns)} · "
        f"CMP: <b style='color:#E6EDF3'>₹{levels['entry']:.2f}</b></p>",
        unsafe_allow_html=True,
    )

    # ── Chart + Scorecard ──
    col_chart, col_score = st.columns([3, 1])

    with col_chart:
        fig = ch.build_candlestick(df_stock, chosen, all_patterns, levels)
        st.plotly_chart(fig, use_container_width=True)

    with col_score:
        gauge_fig = ch.build_bps_gauge(bps)
        st.plotly_chart(gauge_fig, use_container_width=True)

        # Key levels
        st.markdown('<div class="section-header">Key Levels</div>', unsafe_allow_html=True)
        lvl_data = {
            "Entry":      f"₹{levels['entry']:.2f}",
            "Stop Loss":  f"₹{levels['stop']:.2f}",
            "Target 1":   f"₹{levels['target1']:.2f}",
            "Target 2":   f"₹{levels['target2']:.2f}",
            "Target 3":   f"₹{levels['target3']:.2f}",
            "Risk:Reward": f"{levels['rr_ratio']:.2f}",
        }
        for label, val in lvl_data.items():
            color = "#00FF88" if "Target" in label else "#FF4444" if "Stop" in label else "#E6EDF3"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #30363D'>"
                f"<span style='color:#8B949E;font-size:0.8rem'>{label}</span>"
                f"<span style='color:{color};font-weight:700;font-size:0.85rem'>{val}</span></div>",
                unsafe_allow_html=True,
            )

    # ── Score breakdown ──
    st.markdown('<div class="section-header">Factor Score Breakdown</div>', unsafe_allow_html=True)
    breakdown_fig = ch.build_score_breakdown(factor_scores)
    st.plotly_chart(breakdown_fig, use_container_width=True)

    # ── Detected patterns ──
    st.markdown('<div class="section-header">Pattern Analysis</div>', unsafe_allow_html=True)
    detected = {k: v for k, v in all_patterns.items() if v.get("detected")}
    if not detected:
        st.caption("No classic patterns detected in current timeframe.")
    else:
        pat_cols = st.columns(min(4, len(detected)))
        for i, (key, pat) in enumerate(detected.items()):
            with pat_cols[i % len(pat_cols)]:
                q = pat["quality"]
                stars = pat.get("quality_stars", "—")
                st.markdown(
                    f"<div style='background:#1C2230;border:1px solid #30363D;border-radius:8px;"
                    f"padding:10px;margin-bottom:8px'>"
                    f"<div style='color:#4488FF;font-weight:700;font-size:0.8rem'>{pat['description']}</div>"
                    f"<div style='color:#FFB800;font-size:1rem'>{stars}</div>"
                    f"<div style='color:#8B949E;font-size:0.72rem'>Quality: {q:.1f}/10</div>"
                    f"<div style='color:#00FF88;font-size:0.8rem'>Target: ₹{pat.get('target', 0):.0f}</div>"
                    f"<div style='color:#FF4444;font-size:0.8rem'>Stop: ₹{pat.get('stop', 0):.0f}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Quick backtest for this stock ──
    with st.expander("Historical signal accuracy (this stock)"):
        _render_stock_backtest(df_stock, engine)


def _render_stock_backtest(df: pd.DataFrame, engine: BreakoutEngine) -> None:
    """Simple rolling BPS history + forward returns."""
    if len(df) < 60:
        st.caption("Not enough data for backtest.")
        return

    results = []
    lookback = 30
    for i in range(lookback, len(df) - 20, 10):
        sub = df.iloc[:i].copy()
        if len(sub) < 20:
            continue
        try:
            e = BreakoutEngine(sub)
            score, _ = e.compute_bps()
            fwd_ret = (df["Close"].iloc[i + 20] - df["Close"].iloc[i]) / df["Close"].iloc[i] * 100
            results.append({"date": df.index[i], "bps": score, "fwd_20d_ret": fwd_ret})
        except Exception:
            continue

    if not results:
        st.caption("No backtest results.")
        return

    bt_df = pd.DataFrame(results)
    high_bps = bt_df[bt_df["bps"] >= 70]
    st.metric("Avg 20-day return when BPS≥70", f"{high_bps['fwd_20d_ret'].mean():.1f}%")
    st.metric("Hit rate (positive return when BPS≥70)",
              f"{(high_bps['fwd_20d_ret'] > 0).mean() * 100:.0f}%")
    st.dataframe(bt_df.tail(10).reset_index(drop=True), use_container_width=True, height=220)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Market Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def tab_heatmap(full_df: pd.DataFrame) -> None:
    if full_df.empty:
        st.info("No data loaded.")
        return

    st.markdown("### Treemap — All Stocks by Sector & BPS Score")
    st.caption("Size = equal weight · Color = BPS (green=high, red=low) · Click to zoom into sector")
    treemap_fig = ch.build_treemap(full_df)
    st.plotly_chart(treemap_fig, use_container_width=True)

    st.divider()
    st.markdown("### Sector Average BPS")
    sector_fig = ch.build_sector_heatmap(full_df)
    st.plotly_chart(sector_fig, use_container_width=True)

    col_pie, col_stats = st.columns([1, 1])
    with col_pie:
        st.markdown("#### Sector Coverage")
        pie_fig = ch.build_sector_pie(full_df)
        st.plotly_chart(pie_fig, use_container_width=True)

    with col_stats:
        st.markdown("#### Universe Stats")
        total = len(full_df)
        high_bps = len(full_df[full_df["bps"] >= 70])
        mid_bps = len(full_df[(full_df["bps"] >= 50) & (full_df["bps"] < 70)])
        low_bps = len(full_df[full_df["bps"] < 50])
        failed = len(st.session_state.failed_tickers)

        stats = {
            "Total tickers loaded": total,
            "Failed tickers": failed,
            "BPS ≥ 70 (Breakout zone)": high_bps,
            "BPS 50–70 (Watch zone)": mid_bps,
            "BPS < 50 (Avoid zone)": low_bps,
            "Data freshness": st.session_state.data_timestamp or "—",
            "Sectors covered": full_df["sector"].nunique(),
        }
        for k, v in stats.items():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:5px 0;"
                f"border-bottom:1px solid #30363D'>"
                f"<span style='color:#8B949E;font-size:0.82rem'>{k}</span>"
                f"<span style='color:#E6EDF3;font-weight:700;font-size:0.82rem'>{v}</span></div>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4: Backtest
# ─────────────────────────────────────────────────────────────────────────────

def tab_backtest(full_df: pd.DataFrame) -> None:
    st.markdown("### Portfolio Backtest")
    st.caption(
        "Simulate buying all BPS≥70 stocks equally weighted on the score date, "
        "hold 20 trading days, then rebalance. Compare vs Nifty 50."
    )

    stocks_data = st.session_state.stocks_data
    bench_df = st.session_state.bench_df

    if stocks_data is None or bench_df is None:
        st.info("Load data first.")
        return

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        lookback_days = st.slider("Lookback (trading days)", 60, 252, 126, step=21)
    with col_b:
        bps_threshold = st.slider("BPS threshold for entry", 50, 90, 70, step=5)
    with col_c:
        hold_days = st.slider("Hold period (days)", 5, 40, 20, step=5)

    if st.button("Run Backtest", use_container_width=True):
        with st.spinner("Running backtest…"):
            portfolio_ret, nifty_ret, trades_df = _run_backtest(
                stocks_data, bench_df, lookback_days, bps_threshold, hold_days
            )

        if portfolio_ret is not None:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            final_pf = portfolio_ret.iloc[-1] if not portfolio_ret.empty else 100
            final_ni = nifty_ret.iloc[-1] if not nifty_ret.empty else 100
            col_m1.metric("Portfolio Return", f"{final_pf - 100:.1f}%")
            col_m2.metric("Nifty Return", f"{final_ni - 100:.1f}%")
            col_m3.metric("Alpha", f"{final_pf - final_ni:.1f}%")
            if not trades_df.empty:
                hit_rate = (trades_df["return_pct"] > 0).mean() * 100
                col_m4.metric("Win Rate", f"{hit_rate:.0f}%")

            eq_fig = ch.build_equity_curve(portfolio_ret, nifty_ret)
            st.plotly_chart(eq_fig, use_container_width=True)

            if not trades_df.empty:
                st.markdown("#### Trade Log")
                st.dataframe(trades_df.tail(30), use_container_width=True, height=280)
        else:
            st.warning("Not enough data to run backtest with selected parameters.")


def _run_backtest(
    stocks_data: Dict[str, pd.DataFrame],
    bench_df: pd.DataFrame,
    lookback_days: int,
    bps_threshold: float,
    hold_days: int,
) -> Tuple[Optional[pd.Series], Optional[pd.Series], pd.DataFrame]:
    """Run simplified backtest on historical data."""
    # Use a subset of liquid stocks for speed
    liquid = list(stocks_data.items())[:100]
    trades = []

    for step in range(0, lookback_days, hold_days):
        cutoff = len(bench_df) - lookback_days + step
        if cutoff < 20 or cutoff + hold_days >= len(bench_df):
            continue

        entry_date_idx = cutoff
        exit_date_idx = min(cutoff + hold_days, len(bench_df) - 1)
        period_rets = []

        for ticker, df in liquid:
            if len(df) < entry_date_idx + hold_days:
                continue
            try:
                sub = df.iloc[:entry_date_idx].copy()
                if len(sub) < 20:
                    continue
                eng = BreakoutEngine(sub)
                score, _ = eng.compute_bps()
                if score < bps_threshold:
                    continue
                entry_price = df["Close"].iloc[entry_date_idx]
                exit_price = df["Close"].iloc[exit_date_idx]
                ret = (exit_price - entry_price) / entry_price * 100
                period_rets.append(ret)
                trades.append({
                    "ticker": ticker,
                    "bps": round(score, 1),
                    "entry": round(entry_price, 2),
                    "exit": round(exit_price, 2),
                    "return_pct": round(ret, 2),
                    "period": step,
                })
            except Exception:
                continue

        # No-op if no trades

    if not trades:
        return None, None, pd.DataFrame()

    trades_df = pd.DataFrame(trades)

    # Build equity curves
    pf_curve = pd.Series([100.0])
    ni_curve = pd.Series([100.0])

    for step in range(0, lookback_days, hold_days):
        period_trades = trades_df[trades_df["period"] == step]
        avg_ret = period_trades["return_pct"].mean() if not period_trades.empty else 0
        pf_curve = pd.concat([pf_curve, pd.Series([pf_curve.iloc[-1] * (1 + avg_ret / 100)])])

        # Nifty return for same period
        cutoff = len(bench_df) - lookback_days + step
        exit_idx = min(cutoff + hold_days, len(bench_df) - 1)
        if cutoff < len(bench_df) and exit_idx < len(bench_df):
            ni_ret = (bench_df["Close"].iloc[exit_idx] - bench_df["Close"].iloc[cutoff]) / \
                     bench_df["Close"].iloc[cutoff] * 100
        else:
            ni_ret = 0
        ni_curve = pd.concat([ni_curve, pd.Series([ni_curve.iloc[-1] * (1 + ni_ret / 100)])])

    pf_curve.index = range(len(pf_curve))
    ni_curve.index = range(len(ni_curve))

    return pf_curve, ni_curve, trades_df


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5: Watchlist
# ─────────────────────────────────────────────────────────────────────────────

def tab_watchlist(full_df: pd.DataFrame) -> None:
    stocks_data = st.session_state.stocks_data

    # ── Add to watchlist ──
    st.markdown("### Watchlist")
    col_add, col_btn = st.columns([3, 1])
    with col_add:
        available = sorted(full_df["ticker"].str.replace(".NS", "").tolist()) if not full_df.empty else []
        to_add = st.selectbox("Add stock to watchlist", ["— select —"] + available, label_visibility="collapsed")
    with col_btn:
        if st.button("+ Add", use_container_width=True) and to_add != "— select —":
            t = to_add + ".NS"
            if t not in st.session_state.watchlist:
                st.session_state.watchlist.append(t)
                st.success(f"Added {to_add}")

    if not st.session_state.watchlist:
        st.info("Watchlist is empty. Add stocks using the selector above.")
        return

    # ── Display watchlist ──
    st.divider()
    ready_to_break = []

    for ticker in st.session_state.watchlist:
        plain = ticker.replace(".NS", "")
        row = full_df[full_df["ticker"] == ticker]
        bps = row["bps"].iloc[0] if not row.empty else 0
        cmp = row["cmp"].iloc[0] if not row.empty else 0
        pattern = row["pattern"].iloc[0] if not row.empty else "—"
        vol_surge = row["vol_surge"].iloc[0] if not row.empty else 1.0

        if bps >= 80 and vol_surge >= 2.0:
            ready_to_break.append(plain)

        bps_class = "bps-green" if bps >= 70 else "bps-yellow" if bps >= 50 else "bps-red"

        col_name, col_cmp, col_bps, col_pat, col_vol, col_rm = st.columns([2, 1, 1, 2, 1, 1])
        with col_name:
            st.markdown(f"**{plain}**<br><span style='color:#8B949E;font-size:0.75rem'>{get_company_name(ticker)}</span>", unsafe_allow_html=True)
        with col_cmp:
            st.markdown(f"₹{cmp:.2f}" if cmp else "—")
        with col_bps:
            st.markdown(f'<span class="{bps_class}">{bps:.0f}</span>', unsafe_allow_html=True)
        with col_pat:
            st.caption(pattern)
        with col_vol:
            st.caption(f"{vol_surge:.1f}x")
        with col_rm:
            if st.button("✕", key=f"rm_{ticker}"):
                st.session_state.watchlist.remove(ticker)
                st.rerun()

    # ── Ready to Break alert ──
    if ready_to_break:
        st.divider()
        st.markdown(
            f"<div style='background:rgba(0,255,136,0.08);border:1px solid #00FF88;border-radius:8px;"
            f"padding:12px;margin-top:8px'>"
            f"<span style='color:#00FF88;font-weight:700'>🚨 READY TO BREAK:</span> "
            f"<span style='color:#E6EDF3'>{', '.join(ready_to_break)}</span><br>"
            f"<span style='color:#8B949E;font-size:0.75rem'>BPS≥80 + Volume Surge ≥2x</span></div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    settings = render_sidebar()

    # ── App header ──
    st.markdown(
        '<div class="terminal-banner">'
        '📡 INDIAN MARKET BREAKOUT RADAR &nbsp;|&nbsp; '
        '<span style="color:#8B949E">NSE/BSE Technical Scanner &nbsp;·&nbsp; '
        'Powered by Breakout Probability Score™</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Load data ──
    load_data(settings["universe_cat"], settings["custom_tickers"])

    full_df = st.session_state.metrics_df
    if full_df is None or full_df.empty:
        st.warning("Data loading in progress or no data available. Click 'Refresh Data' to retry.")
        return

    filtered_df = apply_filters(full_df, settings)

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📡 Breakout Radar",
        "🔬 Analysis",
        "🗺 Market Heatmap",
        "📊 Backtest",
        "⭐ Watchlist",
    ])

    with tab1:
        tab_breakout_radar(filtered_df, full_df, settings)

    with tab2:
        tab_analysis(full_df, settings)

    with tab3:
        tab_heatmap(full_df)

    with tab4:
        tab_backtest(full_df)

    with tab5:
        tab_watchlist(full_df)


if __name__ == "__main__":
    main()
