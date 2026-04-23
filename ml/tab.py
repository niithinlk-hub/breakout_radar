"""Streamlit ML Screener tab — 4 sub-tabs.

Today's Picks / Backtest / Model Diagnostics / Universe Health.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from styles import plotly_layout

from . import backtest as bt
from . import data_loader, screener
from .model import MLBundle, feature_importance, BUNDLE_PATH
from .risk import compute_levels, size_position, TradeLevels


# ───────────────────────────── helpers ──────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=60 * 60 * 4)
def _load_universe(tickers: tuple, period: str = "3y") -> Dict[str, pd.DataFrame]:
    data = data_loader.fetch_universe(list(tickers), period=period, use_cache=True)
    data = data_loader.apply_liquidity_filter(data)
    return data


@st.cache_data(show_spinner=False, ttl=60 * 60 * 4)
def _load_bench(period: str = "3y") -> pd.DataFrame:
    return data_loader.get_benchmark(period=period, use_cache=True)


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


def _missing_model_banner() -> None:
    st.warning(
        "**ML bundle not found.** Train the model offline before using this tab.\n\n"
        "From repo root:\n\n"
        "```bash\npython -m breakout_radar.ml.train --tickers all --period 3y\n```\n\n"
        f"Expected output path: `{BUNDLE_PATH}`\n\n"
        "Commit and push the resulting `models/ml_bundle.joblib` — the Streamlit app loads it at runtime."
    )


def _honesty_disclaimer() -> None:
    st.caption(
        "⚠️ **Honesty note:** Phase 1 baseline uses TimeSeriesSplit CV (not Purged K-Fold). "
        "Reported numbers may slightly overstate true generalization until Phase 3 lands. "
        "Triple-barrier labels: +5% / −3% / 5-day horizon. Calibrated probabilities (isotonic). "
        "No regime gating yet — trade smaller in choppy markets."
    )


# ─────────────────────────── Today's Picks ──────────────────────────────────

def _today_picks(bundle: MLBundle, stocks: Dict[str, pd.DataFrame],
                 bench: pd.DataFrame) -> None:
    st.subheader("Today's Picks")

    scored = screener.score_universe(bundle, stocks, bench)
    if scored.empty:
        st.info("No scorable tickers for latest bar.")
        return

    c1, c2, c3, c4 = st.columns(4)
    min_prob = c1.slider("Min probability", 0.30, 0.90, 0.60, step=0.01)
    min_rr = c2.slider("Min R:R", 0.5, 5.0, 1.2, step=0.1)
    top_n = c3.slider("Top N", 5, 50, 20, step=1)
    mode = c4.selectbox("Position sizer", ["kelly", "fixed"], index=0)

    capital = st.number_input("Capital (₹)", min_value=10000, value=500000,
                              step=10000, format="%d")

    picks = scored[(scored["prob"] >= min_prob) & (scored["rr"] >= min_rr)].head(top_n).copy()
    if picks.empty:
        st.warning("No picks pass the filters. Lower the probability or R:R threshold.")
        return

    upper = bundle.label_params.get("upper_pct", 0.05)
    lower = bundle.label_params.get("lower_pct", -0.03)

    sized = []
    for _, row in picks.iterrows():
        df = stocks.get(row["ticker"])
        if df is None:
            continue
        lvls = compute_levels(df, upper_pct=upper, lower_pct=lower)
        sz = size_position(capital, row["prob"], lvls, mode=mode)
        sized.append({**row.to_dict(), **{
            "kelly_frac": sz.get("kelly_frac", 0.0),
            "capital_pct": sz.get("capital_pct", 0.0),
            "capital_alloc": sz.get("capital_alloc", 0.0),
            "qty": sz.get("qty", 0),
        }})

    view = pd.DataFrame(sized)
    display_cols = ["ticker", "prob_pct", "entry", "stop", "target1", "target2",
                    "rr", "risk_pct", "reward_pct", "capital_pct", "qty",
                    "last_bar"]
    display_cols = [c for c in display_cols if c in view.columns]

    st.dataframe(
        view[display_cols],
        use_container_width=True, hide_index=True,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker"),
            "prob_pct": st.column_config.NumberColumn("Prob %", format="%.1f"),
            "entry": st.column_config.NumberColumn("Entry", format="%.2f"),
            "stop": st.column_config.NumberColumn("Stop", format="%.2f"),
            "target1": st.column_config.NumberColumn("T1 (+5%)", format="%.2f"),
            "target2": st.column_config.NumberColumn("T2", format="%.2f"),
            "rr": st.column_config.NumberColumn("R:R", format="%.2f"),
            "risk_pct": st.column_config.NumberColumn("Risk %", format="%.2%"),
            "reward_pct": st.column_config.NumberColumn("Reward %", format="%.2%"),
            "capital_pct": st.column_config.NumberColumn("Alloc %", format="%.2%"),
            "qty": st.column_config.NumberColumn("Qty"),
            "last_bar": st.column_config.TextColumn("Bar"),
        },
    )

    csv = view[display_cols].to_csv(index=False)
    st.download_button("Download CSV", data=csv,
                       file_name=f"ml_picks_{datetime.now().strftime('%Y%m%d')}.csv",
                       mime="text/csv")


# ──────────────────────────────── Backtest ──────────────────────────────────

def _backtest_tab(bundle: MLBundle, stocks: Dict[str, pd.DataFrame],
                  bench: pd.DataFrame) -> None:
    st.subheader("Walk-Forward Backtest")
    st.caption("Retrains monthly on trailing 18-month window. Each test set "
               "is one future month, never seen by the model.")

    upper = bundle.label_params.get("upper_pct", 0.05)
    lower = bundle.label_params.get("lower_pct", -0.03)
    horizon = bundle.label_params.get("horizon", 5)

    c1, c2, c3 = st.columns(3)
    train_months = c1.number_input("Train window (months)", 6, 36, 18)
    refit_months = c2.number_input("Refit / test step (months)", 1, 6, 1)
    run = c3.button("Run backtest", type="primary")

    key = "ml_bt_cache"
    if run:
        with st.spinner("Running walk-forward backtest — this can take minutes…"):
            result = bt.walk_forward(
                stocks=stocks, bench_df=bench,
                upper=upper, lower=lower, horizon=horizon,
                train_months=int(train_months),
                refit_months=int(refit_months),
            )
            st.session_state[key] = result

    result = st.session_state.get(key)
    if result is None:
        st.info("Click **Run backtest** to generate fresh results.")
        return

    summ = result.summary or {}
    if "error" in summ:
        st.error(f"Backtest error: {summ['error']}")
        return

    a, b, c, d = st.columns(4)
    a.metric("Out-of-fold AUC", f"{summ.get('auc_oof', float('nan')):.3f}")
    b.metric("Out-of-fold Brier", f"{summ.get('brier_oof', float('nan')):.3f}")
    c.metric("Base rate (+5% hit)", f"{summ.get('base_rate', 0):.1%}")
    d.metric("# folds", summ.get("folds", 0))

    st.markdown("**Precision / coverage at probability thresholds**")
    thr_rows = [{
        "threshold": thr,
        "precision": summ.get(f"precision@{thr}"),
        "coverage": summ.get(f"coverage@{thr}"),
        "n_signals": summ.get(f"n@{thr}"),
    } for thr in (0.5, 0.6, 0.65, 0.7, 0.75)]
    st.dataframe(pd.DataFrame(thr_rows), hide_index=True, use_container_width=True)

    if result.decile_table is not None and not result.decile_table.empty:
        st.markdown("**Hit rate by probability decile**")
        dec = result.decile_table
        fig = go.Figure(go.Bar(
            x=[f"D{int(d)+1}" for d in dec["decile"]],
            y=dec["hit_rate"],
            text=[f"{v:.1%}" for v in dec["hit_rate"]],
            textposition="outside",
            marker_color="#4488FF",
        ))
        fig.add_hline(y=summ.get("base_rate", 0), line_dash="dash",
                      line_color="#8B949E", annotation_text="base rate")
        fig.update_layout(**plotly_layout(
            height=340,
            xaxis=dict(title="Probability decile (D10 = highest)"),
            yaxis=dict(title="Realized +5% hit rate", tickformat=".0%"),
            margin=dict(l=40, r=20, t=20, b=40),
        ))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Per-fold metrics**")
    st.dataframe(pd.DataFrame(result.folds), hide_index=True,
                 use_container_width=True)


# ───────────────────────── Model Diagnostics ────────────────────────────────

def _diagnostics_tab(bundle: MLBundle) -> None:
    st.subheader("Model Diagnostics")

    meta_cols = st.columns(4)
    meta_cols[0].metric("Trained at (UTC)", bundle.trained_at or "n/a")
    meta_cols[1].metric("# training rows", f"{bundle.n_train:,}")
    meta_cols[2].metric("# tickers", bundle.n_tickers)
    meta_cols[3].metric("# features", len(bundle.feature_names))

    st.markdown("**Held-out tail metrics** (final 20% by time)")
    m = bundle.metrics or {}
    grid = st.columns(4)
    grid[0].metric("AUC", f"{m.get('auc_tail', float('nan')):.3f}")
    grid[1].metric("Brier", f"{m.get('brier_tail', float('nan')):.3f}")
    grid[2].metric("LogLoss", f"{m.get('logloss_tail', float('nan')):.3f}")
    grid[3].metric("Base rate", f"{m.get('base_rate_tail', 0):.1%}")

    thr_rows = []
    for thr in (0.5, 0.6, 0.65, 0.7):
        thr_rows.append({
            "threshold": thr,
            "precision": m.get(f"precision@{thr}"),
            "coverage": m.get(f"coverage@{thr}"),
        })
    st.dataframe(pd.DataFrame(thr_rows), hide_index=True,
                 use_container_width=True)

    st.markdown("**Top 20 features by importance**")
    fi = feature_importance(bundle, top_n=20)
    if fi.empty:
        st.info("Feature importance unavailable.")
    else:
        fig = go.Figure(go.Bar(
            x=fi["importance"][::-1], y=fi["feature"][::-1],
            orientation="h", marker_color="#00FF88",
        ))
        fig.update_layout(**plotly_layout(
            height=520,
            xaxis=dict(title="avg gain importance"),
            margin=dict(l=200, r=40, t=10, b=40),
        ))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Hyperparameters + label config"):
        st.json({"params": bundle.params,
                 "label_params": bundle.label_params})


# ───────────────────────── Universe Health ──────────────────────────────────

def _universe_health_tab(stocks: Dict[str, pd.DataFrame]) -> None:
    st.subheader("Universe Health")
    stats = data_loader.universe_health(stocks)
    a, b, c, d = st.columns(4)
    a.metric("Tickers loaded", stats["tickers"])
    b.metric("Avg history (days)", f"{stats['avg_history_days']:.0f}")
    c.metric("Avg turnover (₹ Cr / day)", f"{stats['avg_turnover_cr']:.1f}")
    d.metric("Latest bar", stats["latest_bar"])

    if not stocks:
        st.info("No data loaded.")
        return

    # Per-ticker table (history + turnover)
    rows = []
    for t, df in stocks.items():
        if df is None or df.empty:
            continue
        turn = float((df["Close"].iloc[-20:] * df["Volume"].iloc[-20:]).mean())
        rows.append({
            "ticker": t,
            "bars": len(df),
            "first": df.index[0].strftime("%Y-%m-%d"),
            "last": df.index[-1].strftime("%Y-%m-%d"),
            "avg_turnover_cr": round(turn / 1e7, 2),
        })
    tbl = pd.DataFrame(rows).sort_values("avg_turnover_cr", ascending=False)
    st.dataframe(tbl, hide_index=True, use_container_width=True, height=420)


# ───────────────────────────── entrypoint ───────────────────────────────────

def render_ml_screener(settings: Dict) -> None:
    """Main entry — call from app.py."""
    st.markdown("### 🤖 ML Screener — Phase 1 (baseline XGB)")

    bundle = MLBundle.load()
    if bundle is None:
        _missing_model_banner()
        _honesty_disclaimer()
        return

    tickers = _resolve_tickers()
    if not tickers:
        st.error("No tickers resolved from `tickers.py`.")
        return

    with st.spinner(f"Loading cached data for {len(tickers)} tickers…"):
        stocks = _load_universe(tuple(tickers))
        bench = _load_bench()

    _honesty_disclaimer()

    sub_picks, sub_bt, sub_diag, sub_uni = st.tabs(
        ["Today's Picks", "Backtest", "Model Diagnostics", "Universe Health"]
    )
    with sub_picks:
        _today_picks(bundle, stocks, bench)
    with sub_bt:
        _backtest_tab(bundle, stocks, bench)
    with sub_diag:
        _diagnostics_tab(bundle)
    with sub_uni:
        _universe_health_tab(stocks)
