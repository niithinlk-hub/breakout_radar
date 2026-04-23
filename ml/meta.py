"""Meta-labeling pipeline (UPGRADE 1 + 8).

Stage 1 — primary rules signal fires on high-recall bars.
Stage 2 — meta model trained ONLY on primary-fired rows, predicting
          whether the triple-barrier upper target hit first.

Meta features = base/HTF features + primary confidence + pattern flags
                + regime features (probabilities).

Final output of a ticker at a given bar:
    primary_on ∈ {0, 1}
    meta_prob ∈ [0, 1]
    effective_prob = primary_on * meta_prob
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .ensemble import CalibratedEnsemble
from .pattern_features import (PATTERN_FEATURE_COLS, MIN_FIRINGS_FOR_FEATURE,
                               count_firings, patterns_at_bars)
from .primary import PrimaryConfig, primary_confidence, primary_signal

META_PATH_DEFAULT = Path(__file__).resolve().parent.parent / "models" / "meta.pkl"


@dataclass
class MetaBundle:
    ensemble: Optional[CalibratedEnsemble] = None
    feature_names: List[str] = field(default_factory=list)
    label_params: Dict = field(default_factory=dict)
    primary_cfg: Dict = field(default_factory=dict)
    pattern_firing_counts: Dict[str, int] = field(default_factory=dict)
    experimental_patterns: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    cv_compare: Dict = field(default_factory=dict)
    trained_at: str = ""
    n_train: int = 0
    n_tickers: int = 0

    def save(self, path: Path = META_PATH_DEFAULT) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path = META_PATH_DEFAULT) -> Optional["MetaBundle"]:
        if not Path(path).exists():
            return None
        try:
            return joblib.load(path)
        except Exception:
            return None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X2 = X.reindex(columns=self.feature_names).fillna(0.0)
        return self.ensemble.predict_proba(X2)[:, 1]


def build_primary_filtered_panel(
    stocks: Dict[str, pd.DataFrame],
    bench_df: Optional[pd.DataFrame],
    upper: float, lower: float, horizon: int,
    regime_df: Optional[pd.DataFrame] = None,
    primary_cfg: Optional[PrimaryConfig] = None,
    include_patterns: bool = True,
    max_rows_per_ticker: Optional[int] = None,
):
    """Build training panel restricted to primary-fired bars.

    Returns: X, y, t1, ticker_col, firing_counts
    """
    from . import features as feat_mod
    from . import labeler as lbl_mod

    if primary_cfg is None:
        primary_cfg = PrimaryConfig()

    X_parts: List[pd.DataFrame] = []
    y_parts: List[pd.Series] = []
    t1_parts: List[pd.Series] = []
    tkr_parts: List[pd.Series] = []

    for t, df in stocks.items():
        if df is None or len(df) < 252:
            continue
        feats = feat_mod.build_features(df, bench_df)
        labels, t1s = lbl_mod.triple_barrier_labels(
            df, upper_pct=upper, lower_pct=lower, horizon=horizon
        )
        fires = primary_signal(feats, cfg=primary_cfg)
        pconf = primary_confidence(feats, cfg=primary_cfg)
        # Valid rows = primary-fired AND have labels AND non-NaN features
        mask = fires & labels.notna() & feats.notna().all(axis=1)
        if not mask.any():
            continue
        rows = feats.loc[mask].copy()
        rows["primary_confidence"] = pconf.loc[mask]

        # Regime features
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

        # Pattern flag features at primary-fired bars only (cost cap)
        if include_patterns:
            pat_df = patterns_at_bars(df, rows.index)
            for c in PATTERN_FEATURE_COLS:
                rows[c] = pat_df[c] if c in pat_df.columns else 0.0

        if max_rows_per_ticker and len(rows) > max_rows_per_ticker:
            rows = rows.tail(max_rows_per_ticker)

        X_parts.append(rows)
        y_parts.append(labels.loc[rows.index].astype(int))
        t1_parts.append(t1s.loc[rows.index])
        tkr_parts.append(pd.Series(t, index=rows.index))

    if not X_parts:
        raise RuntimeError("Primary signal fired on zero rows — "
                           "check thresholds or universe coverage.")

    # Concatenate per-ticker blocks in parallel (rows stay aligned).
    # DO NOT reindex — dates repeat across tickers, which would be ambiguous.
    X = pd.concat(X_parts)
    y = pd.concat(y_parts)
    t1 = pd.concat(t1_parts)
    tkr = pd.concat(tkr_parts)

    # Capture bar dates (t0) BEFORE resetting to RangeIndex — PurgedKFold needs
    # explicit t0 timestamps because X.index will become integer positions.
    t0 = pd.Series(pd.to_datetime(X.index.values))

    # Unique row ids so downstream .loc[]/.reindex() on this frame never collides.
    new_idx = pd.RangeIndex(len(X))
    X.index = new_idx
    y.index = new_idx
    t1.index = new_idx
    tkr.index = new_idx
    t0.index = new_idx

    # Sort by label time (t1) globally — downstream PurgedKFold depends on
    # a monotonic t1 for clean train/test splits.
    order = t1.sort_values().index
    X = X.loc[order].reset_index(drop=True)
    y = y.loc[order].reset_index(drop=True)
    t1 = t1.loc[order].reset_index(drop=True)
    tkr = tkr.loc[order].reset_index(drop=True)
    t0 = t0.loc[order].reset_index(drop=True)

    firing_counts = count_firings(X[[c for c in PATTERN_FEATURE_COLS if c.endswith("_fires") and c in X.columns]]) \
        if include_patterns else {}

    return X, y, t1, t0, tkr, firing_counts


def gate_experimental_patterns(
    X: pd.DataFrame,
    firing_counts: Dict[str, int],
    min_firings: int = MIN_FIRINGS_FOR_FEATURE,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop pattern feature pairs whose detector fired < min_firings times.

    Returns (X_filtered, experimental_pattern_names).
    """
    from .pattern_features import PATTERN_SLUGS
    from .patterns import PATTERN_NAMES
    drop_cols: List[str] = []
    experimental: List[str] = []
    for name, slug in zip(PATTERN_NAMES, PATTERN_SLUGS):
        n = firing_counts.get(name, 0)
        if n < min_firings:
            drop_cols.append(f"pat_{slug}_fires")
            drop_cols.append(f"pat_{slug}_conf")
            experimental.append(name)
    keep = [c for c in X.columns if c not in drop_cols]
    return X[keep], experimental
