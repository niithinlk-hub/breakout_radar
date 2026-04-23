"""SHAP explanations per pick — top-10 drivers push the probability up/down.

Uses `shap.TreeExplainer` on the first XGB base learner inside the
calibrated ensemble (TreeExplainer only works on tree models; the
isotonic calibration on top doesn't expose gradients). The SHAP values
are computed against the uncalibrated base; the rank-order of drivers
is the same up to the monotonic calibration transform.

Graceful degradation: if `shap` is not installed, return an empty
explanation dict and the UI just skips the drivers chart.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import shap                                       # type: ignore
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def explain_pick(
    ensemble,
    X_row: pd.DataFrame,
    feature_names: List[str],
    top_n: int = 10,
) -> Dict:
    """Return top_n drivers (name, value, shap) for a single pick.

    Shape: {'drivers': [...], 'base_value': float, 'available': bool}
    """
    if not SHAP_AVAILABLE or ensemble is None:
        return {"available": False, "drivers": [], "base_value": 0.0}

    # Walk calibrated bases to find a tree model (XGB/LGBM/CatBoost)
    tree_model = None
    for cal in getattr(ensemble, "calibrated_bases_", []):
        # CalibratedClassifierCV.calibrated_classifiers_[*].estimator
        for ccb in getattr(cal, "calibrated_classifiers_", []):
            est = getattr(ccb, "estimator", None) or getattr(ccb, "base_estimator", None)
            if est is None:
                continue
            # TreeExplainer is happiest with sklearn/xgboost API
            if type(est).__name__ in ("XGBClassifier", "LGBMClassifier", "CatBoostClassifier"):
                tree_model = est
                break
        if tree_model is not None:
            break

    if tree_model is None:
        return {"available": False, "drivers": [], "base_value": 0.0}

    try:
        X_use = X_row[feature_names]
        explainer = shap.TreeExplainer(tree_model)
        sv = explainer.shap_values(X_use)
        # Binary models may return (n, n_features) or a [neg, pos] list
        if isinstance(sv, list) and len(sv) == 2:
            vals = sv[1][0]
        else:
            vals = np.asarray(sv)[0]
        base_value = explainer.expected_value
        if hasattr(base_value, "__iter__"):
            base_value = list(base_value)[-1]
    except Exception:
        return {"available": False, "drivers": [], "base_value": 0.0}

    df = pd.DataFrame({
        "feature": feature_names,
        "value": X_row[feature_names].iloc[0].values,
        "shap": vals,
    })
    df["abs_shap"] = df["shap"].abs()
    df = df.sort_values("abs_shap", ascending=False).head(top_n)
    drivers = [{
        "feature": r["feature"],
        "value": float(r["value"]) if pd.notna(r["value"]) else None,
        "shap": float(r["shap"]),
        "direction": "up" if r["shap"] > 0 else "down",
    } for _, r in df.iterrows()]

    return {
        "available": True,
        "drivers": drivers,
        "base_value": float(base_value) if base_value is not None else 0.0,
    }
