"""Model wrapper — Phase 1 baseline.

XGBoost + isotonic CalibratedClassifierCV + TimeSeriesSplit for
calibration folds. Hard-coded hyperparams now; Optuna replaces in Phase 3.

Train/load via `MLBundle.save(path)` / `MLBundle.load(path)`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (brier_score_loss, log_loss, precision_score,
                             roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BUNDLE_PATH = MODELS_DIR / "ml_bundle.joblib"

DEFAULT_PARAMS: Dict = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}


@dataclass
class MLBundle:
    model: object = None
    feature_names: List[str] = field(default_factory=list)
    label_params: Dict = field(default_factory=dict)
    metrics: Dict = field(default_factory=dict)
    params: Dict = field(default_factory=dict)
    trained_at: str = ""
    n_train: int = 0
    n_tickers: int = 0

    def save(self, path: Path = BUNDLE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path = BUNDLE_PATH) -> Optional["MLBundle"]:
        if not Path(path).exists():
            return None
        try:
            return joblib.load(path)
        except Exception:
            return None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_names]
        return self.model.predict_proba(X)[:, 1]


def _holdout_split(X: pd.DataFrame, y: pd.Series,
                   tail_frac: float = 0.2) -> Tuple:
    n = len(X)
    k = max(1, int(n * (1.0 - tail_frac)))
    return X.iloc[:k], y.iloc[:k], X.iloc[k:], y.iloc[k:]


def train_bundle(
    X: pd.DataFrame,
    y: pd.Series,
    params: Optional[Dict] = None,
    calibration_splits: int = 4,
    tail_frac: float = 0.2,
    label_params: Optional[Dict] = None,
    n_tickers: int = 0,
) -> MLBundle:
    """Train XGB + calibrate + eval on tail 20%."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    feature_names = list(X.columns)

    X_tr, y_tr, X_te, y_te = _holdout_split(X, y, tail_frac=tail_frac)

    base = XGBClassifier(**params)

    cv = TimeSeriesSplit(n_splits=calibration_splits)
    clf = CalibratedClassifierCV(estimator=base, method="isotonic", cv=cv)
    clf.fit(X_tr, y_tr)

    # Metrics on held-out tail
    proba_te = clf.predict_proba(X_te)[:, 1]
    metrics = {
        "auc_tail": float(roc_auc_score(y_te, proba_te)) if y_te.nunique() > 1 else float("nan"),
        "brier_tail": float(brier_score_loss(y_te, proba_te)),
        "logloss_tail": float(log_loss(y_te, np.clip(proba_te, 1e-6, 1 - 1e-6))),
        "base_rate_tail": float(y_te.mean()),
        "n_tail": int(len(y_te)),
    }
    for thr in (0.5, 0.6, 0.65, 0.7):
        pred = (proba_te >= thr).astype(int)
        if pred.sum() > 0:
            metrics[f"precision@{thr}"] = float(precision_score(y_te, pred, zero_division=0))
            metrics[f"coverage@{thr}"] = float(pred.mean())
        else:
            metrics[f"precision@{thr}"] = float("nan")
            metrics[f"coverage@{thr}"] = 0.0

    bundle = MLBundle(
        model=clf,
        feature_names=feature_names,
        label_params=label_params or {},
        metrics=metrics,
        params=params,
        trained_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        n_train=int(len(X_tr)),
        n_tickers=int(n_tickers),
    )
    return bundle


def feature_importance(bundle: MLBundle, top_n: int = 20) -> pd.DataFrame:
    """Average gain importance across calibrated estimators."""
    clf = bundle.model
    importances: Dict[str, float] = {f: 0.0 for f in bundle.feature_names}
    n = 0
    cc_list = getattr(clf, "calibrated_classifiers_", [])
    for cc in cc_list:
        est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if est is None or not hasattr(est, "feature_importances_"):
            continue
        vals = est.feature_importances_
        for name, v in zip(bundle.feature_names, vals):
            importances[name] += float(v)
        n += 1
    if n > 0:
        importances = {k: v / n for k, v in importances.items()}
    df = pd.DataFrame({"feature": list(importances.keys()),
                       "importance": list(importances.values())})
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
