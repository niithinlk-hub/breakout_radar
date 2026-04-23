"""Triple-barrier labeling (Lopez de Prado, AFML Ch. 3).

For each bar t:
  - upper barrier: close_t * (1 + upper_pct)
  - lower barrier: close_t * (1 + lower_pct)   # lower_pct is negative
  - time barrier:  t + horizon bars
Label = 1 if upper barrier hit first, 0 otherwise.

`t1` (label end-time) is returned alongside labels for future
PurgedKFold support (Phase 3).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def triple_barrier_labels(
    df: pd.DataFrame,
    upper_pct: float = 0.05,
    lower_pct: float = -0.03,
    horizon: int = 5,
) -> Tuple[pd.Series, pd.Series]:
    """Return (labels, t1) indexed to df.

    labels: 1 if upper hit first, 0 if lower hit first or time expires.
    t1: timestamp of the bar where the barrier resolution occurred.
    Rows without horizon bars ahead become NaN (drop before training).
    """
    if df is None or len(df) <= horizon:
        empty = pd.Series(dtype=float, index=df.index if df is not None else None)
        return empty, empty.copy()

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    idx = df.index

    n = len(df)
    labels = np.full(n, np.nan, dtype=float)
    t1_idx = np.full(n, -1, dtype=np.int64)

    for i in range(n - horizon):
        up = close[i] * (1.0 + upper_pct)
        dn = close[i] * (1.0 + lower_pct)
        resolved = False
        for j in range(i + 1, i + 1 + horizon):
            hit_up = high[j] >= up
            hit_dn = low[j] <= dn
            if hit_up and hit_dn:
                # Both in same bar — treat as lower first (conservative).
                labels[i] = 0.0
                t1_idx[i] = j
                resolved = True
                break
            if hit_up:
                labels[i] = 1.0
                t1_idx[i] = j
                resolved = True
                break
            if hit_dn:
                labels[i] = 0.0
                t1_idx[i] = j
                resolved = True
                break
        if not resolved:
            labels[i] = 0.0
            t1_idx[i] = i + horizon

    t1 = pd.Series(
        [idx[k] if k >= 0 else pd.NaT for k in t1_idx],
        index=idx, dtype="datetime64[ns]",
    )
    return pd.Series(labels, index=idx, name="label"), t1.rename("t1")


def label_summary(labels: pd.Series) -> dict:
    """Class balance + coverage."""
    valid = labels.dropna()
    if valid.empty:
        return {"n": 0, "pos_rate": 0.0, "n_pos": 0, "n_neg": 0}
    return {
        "n": int(len(valid)),
        "pos_rate": float(valid.mean()),
        "n_pos": int(valid.sum()),
        "n_neg": int((valid == 0).sum()),
    }
