"""Stage-1 primary signal — rules-based, high-recall.

Fires "go long" on bar t when >= 2 of:
  a) Close > EMA50 AND EMA8 > EMA21
  b) RSI(14) between 45-70 AND rising (slope over 5 bars > 0)
  c) Volume > 1.3 × 20-day avg
  d) Price within 5% of 20-day high
  e) ADX(14) > 20

Inputs expected: the base feature DataFrame from `ml.features.build_features`.
We reuse those columns — no duplicate indicator calc.

Saved as a dict config to `models/primary.pkl` (no trained parameters;
it's deterministic rules).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

PRIMARY_PATH_DEFAULT = Path(__file__).resolve().parent.parent / "models" / "primary.pkl"


@dataclass
class PrimaryConfig:
    rsi_lo: float = 45.0
    rsi_hi: float = 70.0
    vol_ratio_min: float = 1.3
    dist_20d_high_max_pct: float = -0.05   # px_over_20d_high >= 0.95
    adx_min: float = 20.0
    required_true: int = 2                 # at least N conditions must hold

    def save(self, path: Path = PRIMARY_PATH_DEFAULT) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"type": "rules", "config": self.__dict__}, path)

    @staticmethod
    def load(path: Path = PRIMARY_PATH_DEFAULT) -> Optional["PrimaryConfig"]:
        if not Path(path).exists():
            return None
        try:
            d = joblib.load(path)
            return PrimaryConfig(**d["config"])
        except Exception:
            return None


def primary_signal(
    X: pd.DataFrame,
    cfg: Optional[PrimaryConfig] = None,
) -> pd.Series:
    """Return bool Series (same index as X) where primary fires."""
    if cfg is None:
        cfg = PrimaryConfig()

    # Columns expected from features.build_features
    cond_a = (X.get("px_over_ema50", pd.Series(index=X.index, dtype=float)) > 0) & \
             (X.get("ema8_over_ema21", pd.Series(index=X.index, dtype=float)) > 0)
    rsi = X.get("rsi_14", pd.Series(index=X.index, dtype=float))
    rsi_slope = X.get("rsi_14_slope_5", pd.Series(index=X.index, dtype=float))
    cond_b = (rsi.between(cfg.rsi_lo, cfg.rsi_hi)) & (rsi_slope > 0)
    vol_ratio = X.get("vol_ratio_20", pd.Series(index=X.index, dtype=float))
    cond_c = vol_ratio > cfg.vol_ratio_min
    dist20 = X.get("dist_20d_high_pct", pd.Series(index=X.index, dtype=float))
    cond_d = dist20 >= cfg.dist_20d_high_max_pct  # within -5% of 20d high
    adx = X.get("adx_14", pd.Series(index=X.index, dtype=float))
    cond_e = adx > cfg.adx_min

    stacked = pd.concat([cond_a, cond_b, cond_c, cond_d, cond_e],
                        axis=1).fillna(False).astype(int)
    hit_count = stacked.sum(axis=1)
    return (hit_count >= cfg.required_true)


def primary_confidence(X: pd.DataFrame, cfg: Optional[PrimaryConfig] = None) -> pd.Series:
    """Soft confidence = fraction of the 5 conditions met (0.0 - 1.0).

    Fed to the meta-model as a feature so the meta learns which
    primary-fires are actually worth betting on.
    """
    if cfg is None:
        cfg = PrimaryConfig()
    cond_a = (X.get("px_over_ema50", pd.Series(index=X.index, dtype=float)) > 0) & \
             (X.get("ema8_over_ema21", pd.Series(index=X.index, dtype=float)) > 0)
    rsi = X.get("rsi_14", pd.Series(index=X.index, dtype=float))
    rsi_slope = X.get("rsi_14_slope_5", pd.Series(index=X.index, dtype=float))
    cond_b = (rsi.between(cfg.rsi_lo, cfg.rsi_hi)) & (rsi_slope > 0)
    vol_ratio = X.get("vol_ratio_20", pd.Series(index=X.index, dtype=float))
    cond_c = vol_ratio > cfg.vol_ratio_min
    dist20 = X.get("dist_20d_high_pct", pd.Series(index=X.index, dtype=float))
    cond_d = dist20 >= cfg.dist_20d_high_max_pct
    adx = X.get("adx_14", pd.Series(index=X.index, dtype=float))
    cond_e = adx > cfg.adx_min
    stacked = pd.concat([cond_a, cond_b, cond_c, cond_d, cond_e],
                        axis=1).fillna(False).astype(int)
    return (stacked.sum(axis=1) / 5.0).astype(float)
