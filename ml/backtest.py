"""Walk-forward backtest for the ML bundle.

Retrains every `refit_months` over trailing `train_months` window,
applies to the next `refit_months` window as test. Reports per-decile
hit rate + realized outcome using triple-barrier labels as ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import features, labeler, model


@dataclass
class BacktestResult:
    folds: List[Dict] = field(default_factory=list)
    preds: pd.DataFrame = None           # type: ignore
    summary: Dict = field(default_factory=dict)
    decile_table: pd.DataFrame = None    # type: ignore

    def to_dict(self) -> Dict:
        return {
            "summary": self.summary,
            "folds": self.folds,
            "decile_table": (self.decile_table.to_dict(orient="records")
                             if self.decile_table is not None else []),
        }


def _build_panel(
    stocks: Dict[str, pd.DataFrame],
    bench_df: Optional[pd.DataFrame],
    upper: float, lower: float, horizon: int,
) -> pd.DataFrame:
    """Long panel: rows = (ticker, date); cols = features + label + t1."""
    parts = []
    for t, df in stocks.items():
        feats = features.build_features(df, bench_df)
        labels, t1 = labeler.triple_barrier_labels(
            df, upper_pct=upper, lower_pct=lower, horizon=horizon
        )
        combo = feats.copy()
        combo["label"] = labels
        combo["t1"] = t1
        combo["ticker"] = t
        combo = combo.dropna(subset=["label"]).dropna(subset=feats.columns.tolist())
        if not combo.empty:
            parts.append(combo)
    if not parts:
        return pd.DataFrame()
    panel = pd.concat(parts)
    panel.index.name = "date"
    panel = panel.reset_index().sort_values(["date", "ticker"])
    return panel


def _month_floor(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts.year, ts.month, 1)


def walk_forward(
    stocks: Dict[str, pd.DataFrame],
    bench_df: Optional[pd.DataFrame] = None,
    upper: float = 0.05,
    lower: float = -0.03,
    horizon: int = 5,
    train_months: int = 18,
    refit_months: int = 1,
    min_train_rows: int = 2000,
) -> BacktestResult:
    panel = _build_panel(stocks, bench_df, upper, lower, horizon)
    if panel.empty:
        return BacktestResult(summary={"error": "empty panel"})

    feat_cols = [c for c in panel.columns
                 if c not in ("date", "ticker", "label", "t1")]

    start = _month_floor(panel["date"].min()) + pd.DateOffset(months=train_months)
    end = _month_floor(panel["date"].max())

    folds: List[Dict] = []
    all_preds: List[pd.DataFrame] = []

    cursor = start
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
            bundle = model.train_bundle(
                X=train[feat_cols], y=train["label"].astype(int),
                calibration_splits=3, tail_frac=0.15,
                label_params={"upper_pct": upper, "lower_pct": lower,
                              "horizon": horizon},
            )
            proba = bundle.predict_proba(test[feat_cols])
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
        })

        fold_pred = test[["date", "ticker", "label"]].copy()
        fold_pred["prob"] = proba
        all_preds.append(fold_pred)

        cursor = test_end

    if not all_preds:
        return BacktestResult(folds=folds, summary={"error": "no valid folds"})

    preds = pd.concat(all_preds).reset_index(drop=True)

    # Decile table (probability → realized hit rate)
    preds["decile"] = pd.qcut(preds["prob"], q=10, labels=False, duplicates="drop")
    dec = preds.groupby("decile").agg(
        n=("prob", "size"),
        prob_mean=("prob", "mean"),
        hit_rate=("label", "mean"),
    ).reset_index()

    # Overall summary
    try:
        auc = float(roc_auc_score(preds["label"], preds["prob"])) if preds["label"].nunique() > 1 else float("nan")
    except Exception:
        auc = float("nan")

    # Precision / coverage @ thresholds
    thresh_stats = {}
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

    return BacktestResult(folds=folds, preds=preds,
                          summary=summary, decile_table=dec)
