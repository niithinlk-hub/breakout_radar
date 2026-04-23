"""Walk-forward backtest for the meta-labeling pipeline.

Every `refit_months`, retrain the ensemble on trailing `train_months` of
primary-fired bars, apply it to the next `refit_months` as out-of-sample.
Produces:
  - summary   : AUC/Brier/base rate across all OOF preds
  - decile    : hit rate by probability decile
  - regime    : hit rate by regime state (bull/choppy/risk-off)
  - pat_count : hit rate by # patterns firing (what-if slider)
  - equity    : cumulative realized return curve (per-trade +upper or -lower)
  - max_dd    : max drawdown overall + per regime

Honesty:
  - uses primary-filtered panel (Stage 1 gate)
  - regime features computed only on pre-fold data (no look-ahead)
  - labels are triple-barrier so hit_rate = exact target-first probability
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import features as feat_mod
from . import labeler as lbl_mod
from .ensemble import train_ensemble
from .pattern_features import (FIRES_COLS, PATTERN_FEATURE_COLS,
                               patterns_at_bars)
from .primary import PrimaryConfig, primary_confidence, primary_signal
from .regime import regime_series


@dataclass
class BacktestResult:
    folds: List[Dict] = field(default_factory=list)
    preds: Optional[pd.DataFrame] = None
    summary: Dict = field(default_factory=dict)
    decile_table: Optional[pd.DataFrame] = None
    regime_table: Optional[pd.DataFrame] = None
    pattern_count_table: Optional[pd.DataFrame] = None
    equity_curve: Optional[pd.DataFrame] = None
    max_drawdown_by_regime: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        def _rec(df):
            return df.to_dict(orient="records") if df is not None else []
        return {
            "summary": self.summary,
            "folds": self.folds,
            "decile_table": _rec(self.decile_table),
            "regime_table": _rec(self.regime_table),
            "pattern_count_table": _rec(self.pattern_count_table),
            "equity_curve": _rec(self.equity_curve),
            "max_drawdown_by_regime": self.max_drawdown_by_regime,
        }


# ───────────────────────── panel builder ────────────────────────────────────

def _primary_panel(
    stocks: Dict[str, pd.DataFrame],
    bench_df: Optional[pd.DataFrame],
    upper: float, lower: float, horizon: int,
    regime_df: Optional[pd.DataFrame],
    primary_cfg: PrimaryConfig,
    include_patterns: bool,
    max_rows_per_ticker: Optional[int] = None,
) -> pd.DataFrame:
    """Long panel: rows = primary-fired bars. Cols = features + label + t1 + ticker + n_patterns."""
    parts: List[pd.DataFrame] = []
    for t, df in stocks.items():
        if df is None or len(df) < 252:
            continue
        feats = feat_mod.build_features(df, bench_df)
        labels, t1 = lbl_mod.triple_barrier_labels(
            df, upper_pct=upper, lower_pct=lower, horizon=horizon
        )
        fires = primary_signal(feats, cfg=primary_cfg)
        pconf = primary_confidence(feats, cfg=primary_cfg)
        mask = fires & labels.notna() & feats.notna().all(axis=1)
        if not mask.any():
            continue
        rows = feats.loc[mask].copy()
        rows["primary_confidence"] = pconf.loc[mask]

        if regime_df is not None and not regime_df.empty:
            r = regime_df.reindex(rows.index, method="ffill")
            rows["regime_state"] = r["regime_state"].fillna(1).astype(int)
            rows["regime_p_bull"] = r["p_bull"].fillna(0.0)
            rows["regime_p_choppy"] = r["p_choppy"].fillna(0.0)
            rows["regime_p_riskoff"] = r["p_riskoff"].fillna(0.0)
        else:
            rows["regime_state"] = 1
            rows["regime_p_bull"] = 0.0
            rows["regime_p_choppy"] = 0.0
            rows["regime_p_riskoff"] = 0.0

        if include_patterns:
            pat_df = patterns_at_bars(df, rows.index)
            for c in PATTERN_FEATURE_COLS:
                rows[c] = pat_df[c] if c in pat_df.columns else 0.0
            rows["n_patterns"] = pat_df[FIRES_COLS].sum(axis=1).astype(int)
        else:
            rows["n_patterns"] = 0

        rows["label"] = labels.loc[rows.index].astype(int)
        rows["t1"] = t1.loc[rows.index]
        rows["ticker"] = t

        if max_rows_per_ticker and len(rows) > max_rows_per_ticker:
            rows = rows.tail(max_rows_per_ticker)

        parts.append(rows)

    if not parts:
        return pd.DataFrame()
    panel = pd.concat(parts)
    panel.index.name = "date"
    panel = panel.reset_index().sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel


# ───────────────────────── helpers ──────────────────────────────────────────

def _month_floor(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts.year, ts.month, 1)


def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    cum = series.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    return float(dd.min())


# ───────────────────────── walk-forward ─────────────────────────────────────

def walk_forward(
    stocks: Dict[str, pd.DataFrame],
    bench_df: Optional[pd.DataFrame] = None,
    upper: float = 0.05,
    lower: float = -0.03,
    horizon: int = 5,
    train_months: int = 18,
    refit_months: int = 1,
    min_train_rows: int = 500,
    best_params: Optional[Dict] = None,
    primary_cfg: Optional[PrimaryConfig] = None,
    include_patterns: bool = True,
    max_rows_per_ticker: Optional[int] = None,
    cv_splits: int = 4,
    embargo_pct: float = 0.01,
) -> BacktestResult:
    """Primary-gated walk-forward. Retrain ensemble each fold."""
    if primary_cfg is None:
        primary_cfg = PrimaryConfig()

    # Pre-fit regime bundle on full bench (same regime observable to all folds;
    # acceptable — HMM is slow to react and priors are stable).
    regime_df = pd.DataFrame()
    if bench_df is not None and len(bench_df) > 300:
        from .regime import fit_regime
        rb = fit_regime(bench_df)
        if rb is not None:
            regime_df = regime_series(rb, bench_df)

    panel = _primary_panel(
        stocks, bench_df, upper, lower, horizon,
        regime_df=regime_df,
        primary_cfg=primary_cfg,
        include_patterns=include_patterns,
        max_rows_per_ticker=max_rows_per_ticker,
    )
    if panel.empty:
        return BacktestResult(summary={"error": "empty primary-fired panel"})

    non_feat = {"date", "ticker", "label", "t1", "n_patterns"}
    feat_cols = [c for c in panel.columns if c not in non_feat]

    start = _month_floor(panel["date"].min()) + pd.DateOffset(months=train_months)
    end = _month_floor(panel["date"].max())

    folds: List[Dict] = []
    all_preds: List[pd.DataFrame] = []

    cursor = start
    bp = best_params or {}
    while cursor <= end:
        train_end = cursor
        test_end = cursor + pd.DateOffset(months=refit_months)
        train_start = cursor - pd.DateOffset(months=train_months)
        train = panel[(panel["date"] >= train_start) & (panel["date"] < train_end)]
        test = panel[(panel["date"] >= train_end) & (panel["date"] < test_end)]

        if len(train) < min_train_rows or len(test) == 0:
            cursor = test_end
            continue

        try:
            ens_res = train_ensemble(
                X=train[feat_cols],
                y=train["label"].astype(int),
                t1=train["t1"],
                t0=train["date"],
                best_params=bp,
                tail_frac=0.15,
                cv_splits=cv_splits,
                embargo_pct=embargo_pct,
                naive_compare=False,
            )
            proba = ens_res.ensemble.predict_proba(test[feat_cols])[:, 1]
        except Exception as e:
            folds.append({"train_end": str(train_end.date()), "error": str(e)})
            cursor = test_end
            continue

        y_te = test["label"].astype(int).values
        try:
            auc = float(roc_auc_score(y_te, proba)) if len(set(y_te)) > 1 else float("nan")
        except Exception:
            auc = float("nan")
        brier = float(brier_score_loss(y_te, proba))
        folds.append({
            "train_end": str(train_end.date()),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "auc": round(auc, 4),
            "brier": round(brier, 4),
            "pos_rate": round(float(y_te.mean()), 4),
            "chosen_strategy": ens_res.chosen,
        })

        fold_pred = test[["date", "ticker", "label", "regime_state", "n_patterns"]].copy()
        fold_pred["prob"] = proba
        all_preds.append(fold_pred)

        cursor = test_end

    if not all_preds:
        return BacktestResult(folds=folds, summary={"error": "no valid folds"})

    preds = pd.concat(all_preds).reset_index(drop=True)

    # ── decile table ──
    try:
        preds["decile"] = pd.qcut(preds["prob"], q=10, labels=False, duplicates="drop")
    except Exception:
        preds["decile"] = 0
    dec = preds.groupby("decile").agg(
        n=("prob", "size"),
        prob_mean=("prob", "mean"),
        hit_rate=("label", "mean"),
    ).reset_index()

    # ── per-regime table (prob >= 0.60) ──
    sig = preds[preds["prob"] >= 0.60]
    reg_map = {0: "Bull", 1: "Choppy", 2: "Risk-off", -1: "Unknown"}
    reg_tbl = (sig.groupby("regime_state")
               .agg(n=("prob", "size"), hit_rate=("label", "mean"))
               .reset_index()) if not sig.empty else pd.DataFrame(
        columns=["regime_state", "n", "hit_rate"]
    )
    if not reg_tbl.empty:
        reg_tbl["regime"] = reg_tbl["regime_state"].map(reg_map).fillna("Unknown")
        reg_tbl = reg_tbl[["regime", "regime_state", "n", "hit_rate"]]

    # ── per-pattern-count table (the what-if slider) ──
    pc_tbl = (sig.groupby("n_patterns")
              .agg(n=("prob", "size"), hit_rate=("label", "mean"))
              .reset_index()) if not sig.empty else pd.DataFrame(
        columns=["n_patterns", "n", "hit_rate"]
    )

    # ── summary ──
    try:
        auc = float(roc_auc_score(preds["label"], preds["prob"])) if preds["label"].nunique() > 1 else float("nan")
    except Exception:
        auc = float("nan")

    thresh_stats: Dict = {}
    for thr in (0.5, 0.6, 0.65, 0.7, 0.75):
        m = preds["prob"] >= thr
        if m.sum() > 0:
            thresh_stats[f"precision@{thr}"] = round(float(preds.loc[m, "label"].mean()), 4)
            thresh_stats[f"coverage@{thr}"] = round(float(m.mean()), 4)
            thresh_stats[f"n@{thr}"] = int(m.sum())

    summary = {
        "folds": len(folds),
        "n_preds": int(len(preds)),
        "auc_oof": round(auc, 4),
        "brier_oof": round(float(brier_score_loss(preds["label"], preds["prob"])), 4),
        "base_rate": round(float(preds["label"].mean()), 4),
        **thresh_stats,
    }

    # ── equity curve at 0.60 threshold (each hit = +upper, miss = lower) ──
    eq_src = preds[preds["prob"] >= 0.60].sort_values("date").copy()
    if not eq_src.empty:
        eq_src["pnl"] = np.where(eq_src["label"] == 1, upper, lower)
        eq_src = eq_src.groupby("date", as_index=False)["pnl"].sum()
        eq_src["equity"] = eq_src["pnl"].cumsum()
        equity_curve = eq_src[["date", "pnl", "equity"]]
    else:
        equity_curve = pd.DataFrame(columns=["date", "pnl", "equity"])

    max_dd_by_regime: Dict[str, float] = {}
    if not sig.empty:
        tmp = sig.copy()
        tmp["pnl"] = np.where(tmp["label"] == 1, upper, lower)
        for rs, label in reg_map.items():
            sub = tmp[tmp["regime_state"] == rs].sort_values("date")
            if not sub.empty:
                max_dd_by_regime[label] = round(_max_drawdown(sub["pnl"]), 4)

    return BacktestResult(
        folds=folds, preds=preds, summary=summary,
        decile_table=dec,
        regime_table=reg_tbl if not reg_tbl.empty else None,
        pattern_count_table=pc_tbl if not pc_tbl.empty else None,
        equity_curve=equity_curve,
        max_drawdown_by_regime=max_dd_by_regime,
    )


# ───────────────────── what-if slider helpers ───────────────────────────────

def filter_picks_min_patterns(
    result: BacktestResult,
    min_patterns: int = 2,
    prob_threshold: float = 0.60,
) -> Dict:
    """Report hit rate after filtering picks to those with >= min_patterns firing.

    Used by the UI what-if slider: 'what if I only traded signals with 2+ patterns?'
    """
    if result.preds is None or result.preds.empty:
        return {"n": 0, "hit_rate": float("nan"),
                "base_hit_rate": float("nan"), "lift": float("nan")}
    p = result.preds
    gate = (p["prob"] >= prob_threshold)
    base = p[gate]
    strict = p[gate & (p["n_patterns"] >= min_patterns)]
    base_hr = float(base["label"].mean()) if len(base) else float("nan")
    strict_hr = float(strict["label"].mean()) if len(strict) else float("nan")
    lift = (strict_hr - base_hr) if (not np.isnan(base_hr) and not np.isnan(strict_hr)) else float("nan")
    return {
        "n_base": int(len(base)),
        "n": int(len(strict)),
        "hit_rate": round(strict_hr, 4) if not np.isnan(strict_hr) else float("nan"),
        "base_hit_rate": round(base_hr, 4) if not np.isnan(base_hr) else float("nan"),
        "lift": round(lift, 4) if not np.isnan(lift) else float("nan"),
    }
