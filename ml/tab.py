"""Streamlit ML Screener tab — Phase 3+4+5 meta pipeline.

Sub-tabs: Today's Picks / Backtest / Model Diagnostics / Universe Health.
Adds:
  - Regime banner (red if risk-off — blocks picks)
  - Composite ranking column + probability + pattern count + SHAP drivers
  - Naive vs Purged CV honesty badge
  - Per-regime + per-pattern-count backtest tables
  - What-if slider: filter picks to signals with >= K patterns
  - Equity curve + meta-model calibration reliability plot
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
from .meta import MetaBundle, META_PATH_DEFAULT
from .optuna_tune import load_best_params
from .primary import PrimaryConfig
from .regime import (HMM_PATH_DEFAULT, REGIME_LABELS, RegimeBundle,
                     current_regime, regime_series)
from .risk import compute_levels, size_position


REGIME_COLORS = {0: "#00C96B", 1: "#AAAAAA", 2: "#FF4B4B", -1: "#888888"}


# ───────────────────────────── loaders ──────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=60 * 60 * 4)
def _load_universe(tickers: tuple, period: str = "3y") -> Dict[str, pd.DataFrame]:
    data = data_loader.fetch_universe(list(tickers), period=period, use_cache=True)
    return data_loader.apply_liquidity_filter(data)


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


def _missing_bundle_banner() -> None:
    st.warning(
        "**Meta bundle not found.** Train the full Phase 3+4+5 pipeline offline.\n\n"
        "```bash\npython -m breakout_radar.ml.train --tickers all --period 3y\n```\n\n"
        f"Expected output: `{META_PATH_DEFAULT}`\n\n"
        "Commit `models/meta.pkl` + `models/regime_hmm.pkl` + `models/primary.pkl`."
    )


def _honesty_disclaimer() -> None:
    st.caption(
        "**Honesty:** purged K-fold CV numbers are the honest ones (Lopez de Prado AFML Ch.7). "
        "Naive TimeSeriesSplit lets label-end-time overlap cross the boundary — shown only to "
        "quantify leakage. Patterns firing <100 times historically are experimental (UI-only, "
        "not fed to the meta-model). The composite score cannot inflate a mediocre base prob."
    )


# ─────────────────────────── regime banner ──────────────────────────────────

def _regime_banner(regime_df: pd.DataFrame) -> Dict:
    snap = current_regime(regime_df)
    state = snap["state"]
    color = REGIME_COLORS.get(state, "#888888")
    label = snap["label"]
    pbull, pchop, pro = snap["p_bull"], snap["p_choppy"], snap["p_riskoff"]
    block = snap["block_new_picks"]
    banner_css = (
        f"background:{color}22;border-left:4px solid {color};"
        "padding:10px 14px;border-radius:6px;margin-bottom:10px;"
    )
    lock = " 🔒 **NEW PICKS BLOCKED**" if block else ""
    st.markdown(
        f"<div style='{banner_css}'>"
        f"<strong>Market regime:</strong> {label}{lock}<br>"
        f"<span style='font-size:0.88em;color:#999'>"
        f"P(bull)={pbull:.2f} · P(choppy)={pchop:.2f} · "
        f"P(risk-off)={pro:.2f}"
        "</span></div>",
        unsafe_allow_html=True,
    )
    return snap


# ─────────────────────────── Today's Picks ──────────────────────────────────

def _today_picks(meta_bundle: MetaBundle, stocks: Dict[str, pd.DataFrame],
                 bench: pd.DataFrame, regime_df: pd.DataFrame,
                 primary_cfg: PrimaryConfig) -> None:
    st.subheader("Today's Picks")

    snap = current_regime(regime_df)
    override = False
    if snap["block_new_picks"]:
        st.error(
            f"🔒 Risk-off regime active (P={snap['p_riskoff']:.2f} > 0.60). "
            "Honest risk gate blocks new picks by default."
        )
        override = st.checkbox(
            "Override risk-off gate (show picks anyway — use at your own risk)",
            value=False,
        )
        if not override:
            return
        st.warning(
            "⚠ Gate overridden. Picks shown are probabilities under the model's "
            "training distribution, not conditioned on the current risk-off regime. "
            "Size smaller than usual."
        )

    with st.spinner("Scoring universe…"):
        scored = screener.score_universe(
            meta_bundle, stocks, bench,
            regime_df=regime_df, primary_cfg=primary_cfg,
            include_shap=True, shap_top_n=10,
            override_regime_gate=override,
        )
    if scored.empty:
        st.info("No tickers passed the primary filter on the latest bar.")
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
        st.warning("No picks pass the filters.")
        return

    upper = meta_bundle.label_params.get("upper_pct", 0.05)
    lower = meta_bundle.label_params.get("lower_pct", -0.03)

    rows = []
    for _, row in picks.iterrows():
        df = stocks.get(row["ticker"])
        lvls = compute_levels(df, upper_pct=upper, lower_pct=lower) if df is not None else None
        sz = size_position(capital, row["prob"], lvls, mode=mode) if lvls is not None else {}
        rows.append({**row.to_dict(), **{
            "qty": sz.get("qty", 0),
            "capital_alloc": sz.get("capital_alloc", 0.0),
            "capital_pct": sz.get("capital_pct", row.get("capital_pct", 0.0)),
        }})
    view = pd.DataFrame(rows)

    display_cols = ["ticker", "composite", "prob_pct", "pattern_count",
                    "pattern_avg_conf", "primary_confidence",
                    "entry", "stop", "target1", "target2", "rr",
                    "risk_pct", "reward_pct",
                    "kelly_frac", "capital_pct", "qty",
                    "pattern_list", "last_bar"]
    display_cols = [c for c in display_cols if c in view.columns]

    st.dataframe(
        view[display_cols],
        use_container_width=True, hide_index=True,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker"),
            "composite": st.column_config.ProgressColumn(
                "Composite 0-1", min_value=0.0, max_value=1.0, format="%.3f"),
            "prob_pct": st.column_config.NumberColumn("Prob %", format="%.1f"),
            "pattern_count": st.column_config.NumberColumn("# Pat", format="%d"),
            "pattern_avg_conf": st.column_config.NumberColumn("Pat Conf", format="%.2f"),
            "primary_confidence": st.column_config.NumberColumn("Primary", format="%.2f"),
            "entry": st.column_config.NumberColumn("Entry", format="%.2f"),
            "stop": st.column_config.NumberColumn("Stop", format="%.2f"),
            "target1": st.column_config.NumberColumn("T1 (+5%)", format="%.2f"),
            "target2": st.column_config.NumberColumn("T2", format="%.2f"),
            "rr": st.column_config.NumberColumn("R:R", format="%.2f"),
            "risk_pct": st.column_config.NumberColumn("Risk %", format="%.2%"),
            "reward_pct": st.column_config.NumberColumn("Reward %", format="%.2%"),
            "kelly_frac": st.column_config.NumberColumn("Kelly f*", format="%.2%"),
            "capital_pct": st.column_config.NumberColumn("Alloc %", format="%.2%"),
            "qty": st.column_config.NumberColumn("Qty"),
            "pattern_list": st.column_config.TextColumn("Patterns", width="large"),
            "last_bar": st.column_config.TextColumn("Bar"),
        },
    )

    csv = view[display_cols].to_csv(index=False)
    st.download_button(
        "Download CSV", data=csv,
        file_name=f"ml_picks_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    with st.expander("🔍 Top-10 SHAP drivers per pick"):
        shown = 0
        for _, row in view.iterrows():
            drivers = row.get("shap_drivers") or []
            if not drivers:
                continue
            shown += 1
            st.markdown(f"**{row['ticker']}** — meta prob {row['prob_pct']:.1f}%  ·  "
                        f"composite {row['composite']:.3f}")
            dfd = pd.DataFrame([{
                "feature": d["feature"],
                "value": d.get("value"),
                "shap": d["shap"],
                "direction": d["direction"],
            } for d in drivers])
            fig = go.Figure(go.Bar(
                x=dfd["shap"][::-1], y=dfd["feature"][::-1],
                orientation="h",
                marker_color=[("#00C96B" if s > 0 else "#FF4B4B")
                              for s in dfd["shap"][::-1]],
                text=[f"{v:+.3f}" for v in dfd["shap"][::-1]],
                textposition="auto",
            ))
            fig.update_layout(**plotly_layout(
                height=260, showlegend=False,
                xaxis=dict(title=dict(text="SHAP value (-> higher prob)")),
                margin=dict(l=160, r=40, t=10, b=30),
            ))
            st.plotly_chart(fig, use_container_width=True)
        if shown == 0:
            st.caption("SHAP unavailable (library missing or non-tree base).")


# ──────────────────────────────── Backtest ──────────────────────────────────

def _backtest_tab(meta_bundle: MetaBundle, stocks: Dict[str, pd.DataFrame],
                  bench: pd.DataFrame, primary_cfg: PrimaryConfig) -> None:
    st.subheader("Walk-Forward Backtest (primary-gated + ensemble)")
    st.caption("Retrains the calibrated ensemble monthly on trailing 18-month "
               "window of **primary-fired** bars. Each fold is out-of-sample.")

    upper = meta_bundle.label_params.get("upper_pct", 0.05)
    lower = meta_bundle.label_params.get("lower_pct", -0.03)
    horizon = meta_bundle.label_params.get("horizon", 5)

    c1, c2, c3 = st.columns(3)
    train_months = c1.number_input("Train window (months)", 6, 36, 18)
    refit_months = c2.number_input("Refit step (months)", 1, 6, 1)
    run = c3.button("Run backtest", type="primary")

    key = "ml_bt_cache_meta"
    if run:
        with st.spinner("Training + scoring walk-forward folds — this takes minutes…"):
            bp = load_best_params()
            result = bt.walk_forward(
                stocks=stocks, bench_df=bench,
                upper=upper, lower=lower, horizon=horizon,
                train_months=int(train_months),
                refit_months=int(refit_months),
                best_params=bp,
                primary_cfg=primary_cfg,
            )
            st.session_state[key] = result

    result = st.session_state.get(key)
    if result is None:
        st.info("Click **Run backtest** to produce fresh OOF numbers.")
        return

    summ = result.summary or {}
    if "error" in summ:
        st.error(f"Backtest error: {summ['error']}")
        return

    a, b, c, d = st.columns(4)
    a.metric("OOF AUC", f"{summ.get('auc_oof', float('nan')):.3f}")
    b.metric("OOF Brier", f"{summ.get('brier_oof', float('nan')):.3f}")
    c.metric("Base rate", f"{summ.get('base_rate', 0):.1%}")
    d.metric("Folds", summ.get("folds", 0))

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
            textposition="outside", marker_color="#4488FF",
        ))
        fig.add_hline(y=summ.get("base_rate", 0), line_dash="dash",
                      line_color="#8B949E", annotation_text="base rate")
        fig.update_layout(**plotly_layout(
            height=340,
            xaxis=dict(title=dict(text="Probability decile (D10 = highest)")),
            yaxis=dict(title=dict(text="Realized +5% hit rate"), tickformat=".0%"),
            margin=dict(l=40, r=20, t=20, b=40),
        ))
        st.plotly_chart(fig, use_container_width=True)

    if result.regime_table is not None:
        st.markdown("**Hit rate by regime (prob ≥ 0.60)**")
        st.dataframe(result.regime_table, hide_index=True, use_container_width=True)

    if result.pattern_count_table is not None:
        st.markdown("**Hit rate by # patterns firing (what-if slider)**")
        min_pat = st.slider("Only count signals with ≥ N patterns", 0, 6, 2)
        wi = bt.filter_picks_min_patterns(result, min_patterns=min_pat,
                                          prob_threshold=0.60)
        w1, w2, w3 = st.columns(3)
        w1.metric("Base hit rate (prob≥0.60)",
                  f"{wi.get('base_hit_rate', 0):.1%}" if not np.isnan(wi.get('base_hit_rate', np.nan)) else "n/a",
                  help=f"n={wi.get('n_base', 0)}")
        w2.metric(f"With ≥{min_pat} patterns",
                  f"{wi.get('hit_rate', 0):.1%}" if not np.isnan(wi.get('hit_rate', np.nan)) else "n/a",
                  help=f"n={wi.get('n', 0)}")
        lift = wi.get("lift", float("nan"))
        w3.metric("Lift",
                  f"{lift:+.1%}" if not np.isnan(lift) else "n/a")
        st.dataframe(result.pattern_count_table, hide_index=True,
                     use_container_width=True)

    if result.equity_curve is not None and not result.equity_curve.empty:
        eq = result.equity_curve
        fig = go.Figure(go.Scatter(
            x=eq["date"], y=eq["equity"], mode="lines",
            line=dict(color="#00FF88", width=2),
            name="Cumulative R"
        ))
        fig.update_layout(**plotly_layout(
            height=300, xaxis=dict(title=dict(text="")), yaxis=dict(title=dict(text="Cumulative R")),
            margin=dict(l=40, r=20, t=10, b=40),
        ))
        st.markdown("**Equity curve at prob ≥ 0.60** (+upper on hit, −|lower| on miss)")
        st.plotly_chart(fig, use_container_width=True)
        if result.max_drawdown_by_regime:
            st.caption("Max drawdown by regime (in R units): "
                       + " · ".join(f"{k}={v:+.2f}"
                                    for k, v in result.max_drawdown_by_regime.items()))

    st.markdown("**Per-fold metrics**")
    st.dataframe(pd.DataFrame(result.folds), hide_index=True,
                 use_container_width=True)


# ───────────────────────── Model Diagnostics ────────────────────────────────

def _diagnostics_tab(meta_bundle: MetaBundle) -> None:
    st.subheader("Model Diagnostics")

    m = meta_bundle.metrics or {}
    meta_cols = st.columns(4)
    meta_cols[0].metric("Trained at (UTC)", meta_bundle.trained_at or "n/a")
    meta_cols[1].metric("# training rows", f"{meta_bundle.n_train:,}")
    meta_cols[2].metric("# tickers", meta_bundle.n_tickers)
    meta_cols[3].metric("# features", len(meta_bundle.feature_names))

    # ── Naive-vs-Purged honesty badge ──
    cv = meta_bundle.cv_compare or {}
    if cv:
        st.markdown("**Naive vs Purged CV (honesty)**")
        gap = cv.get("gap", float("nan"))
        leak = cv.get("leakage_warning", False)
        ncol1, ncol2, ncol3 = st.columns(3)
        ncol1.metric("Naive AUC", f"{cv.get('naive_auc_mean', float('nan'))}")
        ncol2.metric("Purged AUC", f"{cv.get('purged_auc_mean', float('nan'))}")
        ncol3.metric("Gap (naive − purged)",
                     f"{gap:+.3f}" if isinstance(gap, (int, float)) and not np.isnan(gap) else "n/a",
                     delta="LEAKAGE" if leak else "ok",
                     delta_color="inverse")
        if leak:
            st.error("⚠️ Naive CV overstated AUC by > 0.05 vs purged. Older pipelines "
                     "that reported TSS scores materially overstated performance.")

    st.markdown("**Held-out tail metrics** (final 20% by time)")
    grid = st.columns(4)
    grid[0].metric("AUC", f"{m.get('auc', float('nan')):.3f}")
    grid[1].metric("Brier", f"{m.get('brier', float('nan')):.3f}")
    grid[2].metric("LogLoss", f"{m.get('logloss', float('nan')):.3f}")
    grid[3].metric("Base rate", f"{m.get('base_rate_tail', 0):.1%}")

    thr_rows = []
    for thr in (0.5, 0.6, 0.65, 0.7, 0.75):
        thr_rows.append({
            "threshold": thr,
            "precision": m.get(f"precision@{thr}"),
            "coverage": m.get(f"coverage@{thr}"),
            "n_signals": m.get(f"n@{thr}"),
        })
    st.dataframe(pd.DataFrame(thr_rows), hide_index=True, use_container_width=True)

    # ── Pattern firing counts + experimental gating ──
    pfc = meta_bundle.pattern_firing_counts or {}
    exp = set(meta_bundle.experimental_patterns or [])
    if pfc:
        st.markdown("**Pattern detector firings (training window)**")
        rows = [{
            "pattern": name,
            "firings": n,
            "experimental (<100)": "YES" if name in exp else "no",
        } for name, n in sorted(pfc.items(), key=lambda kv: -kv[1])]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     height=380)
        if exp:
            st.caption(f"⚠️ {len(exp)} experimental patterns excluded from the meta features.")

    # ── Feature importance (gain fallback) ──
    try:
        imp_rows = []
        for cal in getattr(meta_bundle.ensemble, "calibrated_bases_", []) or []:
            for ccb in getattr(cal, "calibrated_classifiers_", []):
                est = getattr(ccb, "estimator", None) or getattr(ccb, "base_estimator", None)
                if est is None or not hasattr(est, "feature_importances_"):
                    continue
                vals = np.asarray(est.feature_importances_, dtype=float)
                if len(vals) == len(meta_bundle.feature_names):
                    imp_rows.append(vals)
        if imp_rows:
            avg = np.mean(np.vstack(imp_rows), axis=0)
            fi = (pd.DataFrame({"feature": meta_bundle.feature_names, "importance": avg})
                  .sort_values("importance", ascending=False).head(20))
            st.markdown("**Top 20 features by gain importance**")
            fig = go.Figure(go.Bar(
                x=fi["importance"][::-1], y=fi["feature"][::-1],
                orientation="h", marker_color="#00FF88",
            ))
            fig.update_layout(**plotly_layout(
                height=520, xaxis=dict(title=dict(text="avg gain importance")),
                margin=dict(l=220, r=40, t=10, b=40),
            ))
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    with st.expander("Label config + primary rules"):
        st.json({"label_params": meta_bundle.label_params,
                 "primary_cfg": meta_bundle.primary_cfg})


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
    st.markdown("### 🤖 ML Screener — meta-labeling + ensemble + regime gate")

    meta_bundle = MetaBundle.load()
    if meta_bundle is None or meta_bundle.ensemble is None:
        _missing_bundle_banner()
        _honesty_disclaimer()
        return

    tickers = _resolve_tickers()
    if not tickers:
        st.error("No tickers resolved from `tickers.py`.")
        return

    with st.spinner(f"Loading cached data for {len(tickers)} tickers…"):
        stocks = _load_universe(tuple(tickers))
        bench = _load_bench()

    # Regime series (live)
    regime_bundle = RegimeBundle.load()
    regime_df = regime_series(regime_bundle, bench) if regime_bundle is not None else pd.DataFrame()
    _regime_banner(regime_df)

    primary_cfg = PrimaryConfig.load() or PrimaryConfig(**(meta_bundle.primary_cfg or {}))

    _honesty_disclaimer()

    sub_picks, sub_bt, sub_diag, sub_uni = st.tabs(
        ["Today's Picks", "Backtest", "Model Diagnostics", "Universe Health"]
    )
    with sub_picks:
        _today_picks(meta_bundle, stocks, bench, regime_df, primary_cfg)
    with sub_bt:
        _backtest_tab(meta_bundle, stocks, bench, primary_cfg)
    with sub_diag:
        _diagnostics_tab(meta_bundle)
    with sub_uni:
        _universe_health_tab(stocks)
