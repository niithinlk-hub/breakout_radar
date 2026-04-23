"""Optuna hyperparameter tuning for XGBoost / LightGBM / CatBoost.

Objective: Brier score on PurgedKFold — rewards calibration, not just ranking.
100 trials (default), TPE sampler + MedianPruner.

Returns best params dict per learner. Saved collectively to
`models/best_params.json`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .purged_cv import PurgedKFold

try:
    import optuna                                          # type: ignore
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False


BEST_PARAMS_PATH = Path(__file__).resolve().parent.parent / "models" / "best_params.json"


def _brier_cv_score(
    build_estimator: Callable,
    params: Dict,
    X: pd.DataFrame, y: pd.Series, t1: pd.Series,
    t0: Optional[pd.Series] = None,
    n_splits: int = 4, embargo_pct: float = 0.01,
) -> float:
    """Mean Brier across purged CV folds (lower is better)."""
    from sklearn.metrics import brier_score_loss
    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
    scores: List[float] = []
    for tr, te in cv.split(X, y, t1=t1, t0=t0):
        if len(tr) < 100 or len(te) < 50:
            continue
        est = build_estimator(params)
        try:
            est.fit(X.iloc[tr], y.iloc[tr])
            p = est.predict_proba(X.iloc[te])[:, 1]
            scores.append(float(brier_score_loss(y.iloc[te], p)))
        except Exception:
            scores.append(1.0)
    return float(np.mean(scores)) if scores else 1.0


# ────────────────────────── search spaces ───────────────────────────────────

def _xgb_space(trial):
    return {
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1000, step=50),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "gamma":            trial.suggest_float("gamma", 0, 5),
        "objective":        "binary:logistic",
        "eval_metric":      "logloss",
        "tree_method":      "hist",
        "random_state":     42,
        "n_jobs":           -1,
    }


def _lgbm_space(trial):
    return {
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1000, step=50),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 15, 127),
        "objective":        "binary",
        "verbose":          -1,
        "random_state":     42,
        "n_jobs":           -1,
    }


def _cat_space(trial):
    return {
        "depth":            trial.suggest_int("depth", 4, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "iterations":       trial.suggest_int("iterations", 200, 1000, step=50),
        "l2_leaf_reg":      trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "loss_function":    "Logloss",
        "verbose":          False,
        "random_seed":      42,
        "allow_writing_files": False,
    }


# ────────────────────────── builders ────────────────────────────────────────

def _build_xgb(params: Dict):
    from xgboost import XGBClassifier
    return XGBClassifier(**params)


def _build_lgbm(params: Dict):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**params)


def _build_cat(params: Dict):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(**params)


# ────────────────────────── tuning entry ────────────────────────────────────

def tune_learner(
    name: str,
    X: pd.DataFrame, y: pd.Series, t1: pd.Series,
    t0: Optional[pd.Series] = None,
    n_trials: int = 100,
    n_splits: int = 4,
    embargo_pct: float = 0.01,
) -> Optional[Dict]:
    """Tune one of 'xgb'/'lgbm'/'cat'. Returns best params or None if skipped."""
    if not OPTUNA_AVAILABLE:
        print(f"[optuna] unavailable — using default params for {name}")
        return None

    if name == "xgb":
        space_fn, builder = _xgb_space, _build_xgb
    elif name == "lgbm":
        try:
            import lightgbm  # noqa
        except Exception:
            print("[optuna] lightgbm unavailable — skipping")
            return None
        space_fn, builder = _lgbm_space, _build_lgbm
    elif name == "cat":
        try:
            import catboost  # noqa
        except Exception:
            print("[optuna] catboost unavailable — skipping")
            return None
        space_fn, builder = _cat_space, _build_cat
    else:
        raise ValueError(f"Unknown learner: {name}")

    def objective(trial):
        params = space_fn(trial)
        return _brier_cv_score(
            lambda p: builder(p), params, X, y, t1, t0=t0,
            n_splits=n_splits, embargo_pct=embargo_pct,
        )

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[optuna/{name}] best Brier = {study.best_value:.4f}")
    return study.best_params


def save_best_params(all_params: Dict[str, Optional[Dict]],
                     path: Path = BEST_PARAMS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(all_params, f, indent=2, default=str)


def load_best_params(path: Path = BEST_PARAMS_PATH) -> Dict[str, Optional[Dict]]:
    if not Path(path).exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}
