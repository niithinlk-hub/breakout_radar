"""Gradient-boosting ensemble: XGB + LightGBM + CatBoost.

Two blend strategies:
  - "blend": simple mean of isotonically-calibrated probabilities
  - "stack": logistic-regression meta-learner over the three base probs,
             fit via PurgedKFold out-of-fold probs

Pick whichever has lower Brier on a held-out tail 20% — report both.
Missing learners (no lightgbm / no catboost) are gracefully dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .purged_cv import PurgedKFold


# ────────────────────── base learner factories ──────────────────────────────

def make_xgb(params: Optional[Dict] = None):
    from xgboost import XGBClassifier
    p = dict(params or {})
    p.setdefault("objective", "binary:logistic")
    p.setdefault("eval_metric", "logloss")
    p.setdefault("tree_method", "hist")
    p.setdefault("random_state", 42)
    p.setdefault("n_jobs", -1)
    return XGBClassifier(**p)


def make_lgbm(params: Optional[Dict] = None):
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return None
    p = dict(params or {})
    p.setdefault("objective", "binary")
    p.setdefault("verbose", -1)
    p.setdefault("random_state", 42)
    p.setdefault("n_jobs", -1)
    return LGBMClassifier(**p)


def make_cat(params: Optional[Dict] = None):
    try:
        from catboost import CatBoostClassifier
    except Exception:
        return None
    p = dict(params or {})
    p.setdefault("loss_function", "Logloss")
    p.setdefault("verbose", False)
    p.setdefault("random_seed", 42)
    p.setdefault("allow_writing_files", False)
    return CatBoostClassifier(**p)


# ──────────────────── calibrated wrapper ────────────────────────────────────

class CalibratedEnsemble(BaseEstimator, ClassifierMixin):
    """Holds calibrated base learners + blend/stack head."""

    def __init__(self, base_names: List[str], strategy: str = "blend"):
        self.base_names = base_names
        self.strategy = strategy
        self.calibrated_bases_: List = []
        self.stacker_: Optional[LogisticRegression] = None
        self.chosen_strategy_: str = strategy
        self.feature_names_: List[str] = []

    def fit(self, X, y, t1=None,
            calibrators=None, stacker=None,
            chosen_strategy: Optional[str] = None,
            feature_names: Optional[List[str]] = None):
        self.calibrated_bases_ = list(calibrators or [])
        self.stacker_ = stacker
        self.chosen_strategy_ = chosen_strategy or self.strategy
        self.feature_names_ = list(feature_names or X.columns.tolist())
        self.classes_ = np.array([0, 1])
        return self

    def _base_probs(self, X) -> np.ndarray:
        probs = []
        for cal in self.calibrated_bases_:
            try:
                p = cal.predict_proba(X[self.feature_names_])[:, 1]
            except Exception:
                p = np.full(len(X), 0.5)
            probs.append(p)
        return np.vstack(probs).T   # (n, k)

    def predict_proba(self, X) -> np.ndarray:
        bp = self._base_probs(X)
        if self.chosen_strategy_ == "stack" and self.stacker_ is not None:
            p1 = self.stacker_.predict_proba(bp)[:, 1]
        else:
            p1 = bp.mean(axis=1)
        return np.vstack([1 - p1, p1]).T

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ──────────────────── train an ensemble ─────────────────────────────────────

@dataclass
class EnsembleTrainResult:
    ensemble: Optional[CalibratedEnsemble] = None
    base_names: List[str] = field(default_factory=list)
    per_base_metrics: Dict[str, Dict] = field(default_factory=dict)
    blend_metrics: Dict = field(default_factory=dict)
    stack_metrics: Dict = field(default_factory=dict)
    chosen: str = "blend"
    cv_compare: Dict = field(default_factory=dict)


def _calibrate_on_tail(est, X_tr, y_tr, n_splits: int = 3) -> CalibratedClassifierCV:
    """Isotonic calibration via sklearn TSS (lightweight)."""
    from sklearn.model_selection import TimeSeriesSplit
    cv = TimeSeriesSplit(n_splits=n_splits)
    cal = CalibratedClassifierCV(estimator=est, method="isotonic", cv=cv)
    cal.fit(X_tr, y_tr)
    return cal


def _metrics(y_true, proba) -> Dict:
    out = {}
    try:
        out["auc"] = float(roc_auc_score(y_true, proba)) if len(set(y_true)) > 1 else float("nan")
    except Exception:
        out["auc"] = float("nan")
    out["brier"] = float(brier_score_loss(y_true, proba))
    out["logloss"] = float(log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6)))
    return out


def train_ensemble(
    X: pd.DataFrame, y: pd.Series, t1: pd.Series,
    best_params: Dict[str, Optional[Dict]],
    t0: Optional[pd.Series] = None,
    tail_frac: float = 0.2,
    cv_splits: int = 4,
    embargo_pct: float = 0.01,
    naive_compare: bool = True,
) -> EnsembleTrainResult:
    """Train three base learners, calibrate, compare blend vs stack on tail."""
    n = len(X)
    k = max(1, int(n * (1.0 - tail_frac)))
    X_tr, y_tr = X.iloc[:k], y.iloc[:k]
    X_te, y_te = X.iloc[k:], y.iloc[k:]

    base_builders = {
        "xgb":  make_xgb(best_params.get("xgb")),
        "lgbm": make_lgbm(best_params.get("lgbm")),
        "cat":  make_cat(best_params.get("cat")),
    }
    built = {k2: v for k2, v in base_builders.items() if v is not None}
    if not built:
        raise RuntimeError("No base learners available.")

    # Out-of-fold probs for stacker (purged CV on training half)
    oof_matrix = np.zeros((len(X_tr), len(built)))
    per_base_metrics: Dict[str, Dict] = {}
    calibrators: List = []
    base_name_order: List[str] = []

    cv = PurgedKFold(n_splits=cv_splits, embargo_pct=embargo_pct)
    t1_tr = t1.iloc[:k]
    t0_tr = t0.iloc[:k] if t0 is not None else None

    for j, (name, est) in enumerate(built.items()):
        # OOF probs
        oof = np.full(len(X_tr), 0.5)
        for tr_idx, te_idx in cv.split(X_tr, y_tr, t1=t1_tr, t0=t0_tr):
            if len(tr_idx) < 100 or len(te_idx) < 20:
                continue
            e = clone(est)
            try:
                e.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
                oof[te_idx] = e.predict_proba(X_tr.iloc[te_idx])[:, 1]
            except Exception:
                pass
        oof_matrix[:, j] = oof

        # Full-fit calibrated model for test-set evaluation + live use
        try:
            cal = _calibrate_on_tail(est, X_tr, y_tr, n_splits=3)
        except Exception:
            continue
        proba_te = cal.predict_proba(X_te)[:, 1]
        per_base_metrics[name] = _metrics(y_te.values, proba_te)
        calibrators.append(cal)
        base_name_order.append(name)

    if not calibrators:
        raise RuntimeError("All base calibrations failed.")

    # Blend: simple mean of calibrated probs on tail
    tail_base_probs = np.vstack(
        [cal.predict_proba(X_te)[:, 1] for cal in calibrators]
    ).T
    blend_proba = tail_base_probs.mean(axis=1)
    blend_metrics = _metrics(y_te.values, blend_proba)

    # Stack: fit logistic on OOF probs for learners we kept
    keep_cols = [list(built.keys()).index(n) for n in base_name_order]
    oof_used = oof_matrix[:, keep_cols]
    stacker = LogisticRegression(max_iter=200)
    try:
        stacker.fit(oof_used, y_tr)
        stack_tail_probs = np.vstack(
            [cal.predict_proba(X_te)[:, 1] for cal in calibrators]
        ).T
        stack_proba = stacker.predict_proba(stack_tail_probs)[:, 1]
        stack_metrics = _metrics(y_te.values, stack_proba)
    except Exception:
        stacker = None
        stack_metrics = {"auc": float("nan"), "brier": 1.0, "logloss": float("inf")}

    chosen = "stack" if stack_metrics.get("brier", 1.0) < blend_metrics.get("brier", 1.0) else "blend"

    ensemble = CalibratedEnsemble(base_names=base_name_order, strategy=chosen)
    ensemble.fit(X_tr, y_tr,
                 calibrators=calibrators,
                 stacker=stacker,
                 chosen_strategy=chosen,
                 feature_names=list(X.columns))

    # Naive vs purged CV comparison (AUC) on XGB, for honesty banner (U2)
    cv_compare: Dict = {}
    if naive_compare and len(X) > 500:
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score as ras
        base = clone(built[list(built.keys())[0]])
        naive_cv = TimeSeriesSplit(n_splits=cv_splits)
        naive_scores = []
        for tr, te in naive_cv.split(X):
            e = clone(base)
            try:
                e.fit(X.iloc[tr], y.iloc[tr])
                p = e.predict_proba(X.iloc[te])[:, 1]
                naive_scores.append(float(ras(y.iloc[te], p)) if y.iloc[te].nunique() > 1 else float("nan"))
            except Exception:
                naive_scores.append(float("nan"))

        purged_cv = PurgedKFold(n_splits=cv_splits, embargo_pct=embargo_pct)
        purged_scores = []
        for tr, te in purged_cv.split(X, y, t1=t1, t0=t0):
            if len(tr) < 100 or len(te) < 20:
                continue
            e = clone(base)
            try:
                e.fit(X.iloc[tr], y.iloc[tr])
                p = e.predict_proba(X.iloc[te])[:, 1]
                purged_scores.append(float(ras(y.iloc[te], p)) if y.iloc[te].nunique() > 1 else float("nan"))
            except Exception:
                purged_scores.append(float("nan"))

        naive_mean = float(np.nanmean(naive_scores)) if naive_scores else float("nan")
        purged_mean = float(np.nanmean(purged_scores)) if purged_scores else float("nan")
        gap = (naive_mean - purged_mean) if not (np.isnan(naive_mean) or np.isnan(purged_mean)) else float("nan")
        cv_compare = {
            "naive_auc_mean": round(naive_mean, 4),
            "purged_auc_mean": round(purged_mean, 4),
            "gap": round(gap, 4) if not np.isnan(gap) else float("nan"),
            "leakage_warning": (not np.isnan(gap)) and gap > 0.05,
        }

    return EnsembleTrainResult(
        ensemble=ensemble,
        base_names=base_name_order,
        per_base_metrics=per_base_metrics,
        blend_metrics=blend_metrics,
        stack_metrics=stack_metrics,
        chosen=chosen,
        cv_compare=cv_compare,
    )
