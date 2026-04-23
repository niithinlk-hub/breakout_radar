"""Live screener — score latest bar per ticker with trained bundle."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import features, risk
from .model import MLBundle


def score_universe(
    bundle: MLBundle,
    stocks: Dict[str, pd.DataFrame],
    bench_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return one row per ticker with calibrated prob + levels."""
    if bundle is None or not stocks:
        return pd.DataFrame()

    upper = bundle.label_params.get("upper_pct", 0.05)
    lower = bundle.label_params.get("lower_pct", -0.03)

    rows = []
    for t, df in stocks.items():
        if df is None or len(df) < 60:
            continue
        feats = features.build_features(df, bench_df)
        last = feats.iloc[[-1]]
        if last[bundle.feature_names].isna().any().any():
            continue
        try:
            prob = float(bundle.predict_proba(last)[0])
        except Exception:
            continue
        lvls = risk.compute_levels(df, upper_pct=upper, lower_pct=lower)
        if lvls is None:
            continue
        rows.append({
            "ticker": t,
            "prob": round(prob, 4),
            "prob_pct": round(prob * 100, 1),
            "entry": lvls.entry,
            "stop": lvls.stop,
            "target1": lvls.target1,
            "target2": lvls.target2,
            "risk_pct": lvls.risk_pct,
            "reward_pct": lvls.reward_pct,
            "rr": lvls.rr,
            "atr": lvls.atr,
            "last_bar": df.index[-1].strftime("%Y-%m-%d"),
        })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("prob", ascending=False).reset_index(drop=True)
    return out


def probability_buckets(scored: pd.DataFrame) -> pd.DataFrame:
    """Count tickers per probability bucket — for diagnostics."""
    if scored.empty:
        return pd.DataFrame()
    bins = [0.0, 0.3, 0.45, 0.55, 0.65, 0.75, 1.0]
    labels = ["<30", "30-45", "45-55", "55-65", "65-75", ">=75"]
    b = pd.cut(scored["prob"], bins=bins, labels=labels,
               include_lowest=True, right=False)
    return b.value_counts().reindex(labels).fillna(0).astype(int).to_frame("count")
