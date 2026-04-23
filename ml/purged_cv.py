"""Purged K-Fold cross-validation with embargo (Lopez de Prado, AFML Ch. 7).

Naive TimeSeriesSplit lets labels leak into the training set when the
label-forming window straddles the train/test boundary. PurgedKFold:
  1. Purge training rows whose label end-time (t1) falls inside the
     test interval.
  2. Embargo an extra ceil(embargo_pct * n) rows after the test fold
     so that a train row whose features were formed under conditions
     shared with the test window isn't used.

Naive-vs-purged AUC gap quantifies the leakage the old pipeline was hiding.
"""
from __future__ import annotations

import math
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd


class PurgedKFold:
    """K-fold CV with purging on label end-times + embargo.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    t1 : pd.Series
        Indexed identically to the training panel. Values are timestamps
        at which each sample's label resolves. Passed in at .split().
    embargo_pct : float
        Fraction of n_samples to embargo after each test fold.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = int(n_splits)
        self.embargo_pct = float(embargo_pct)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(
        self,
        X: pd.DataFrame,
        y=None,
        t1: pd.Series = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if t1 is None:
            raise ValueError("PurgedKFold.split requires t1 (label end-times)")
        if not isinstance(X.index, pd.Index) or len(X) != len(t1):
            raise ValueError("X and t1 must share the same length.")

        # Ensure X is sorted by time (index assumed to be the sample-start date)
        order = np.arange(len(X))
        n = len(order)
        embargo = int(math.ceil(self.embargo_pct * n))

        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        edges: List[Tuple[int, int]] = []
        cursor = 0
        for fs in fold_sizes:
            edges.append((cursor, cursor + fs))
            cursor += fs

        # Use time index for purge logic
        t0 = pd.Series(X.index, index=X.index)
        t1_ = pd.Series(pd.to_datetime(t1.values), index=X.index)

        for f_start, f_end in edges:
            test_idx = order[f_start:f_end]
            test_times = t0.iloc[test_idx]
            test_lo = test_times.min()
            test_hi = test_times.max()

            # Purge: any sample whose (t0 <= test_hi) AND (t1 >= test_lo) overlaps test
            full_t0 = t0.values
            full_t1 = t1_.values

            overlap = (full_t0 >= np.datetime64(test_lo)) & (full_t0 <= np.datetime64(test_hi))
            overlap |= (full_t1 >= np.datetime64(test_lo)) & (full_t1 <= np.datetime64(test_hi))
            overlap |= (full_t0 <= np.datetime64(test_lo)) & (full_t1 >= np.datetime64(test_hi))

            train_mask = ~overlap
            # Drop the test indices from the train mask just in case
            train_mask_arr = np.array(train_mask, dtype=bool)
            train_mask_arr[test_idx] = False

            # Embargo rows immediately after test fold
            emb_end = min(n, f_end + embargo)
            train_mask_arr[f_end:emb_end] = False

            train_idx = order[train_mask_arr]
            yield train_idx, test_idx


def cv_score(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    scorer=None,
) -> List[float]:
    """Run purged CV; return list of per-fold scores. Default scorer = roc_auc."""
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    if scorer is None:
        scorer = roc_auc_score

    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
    scores: List[float] = []
    for tr, te in cv.split(X, y, t1=t1):
        if len(tr) == 0 or len(te) == 0:
            continue
        est = clone(estimator)
        est.fit(X.iloc[tr], y.iloc[tr])
        if hasattr(est, "predict_proba"):
            p = est.predict_proba(X.iloc[te])[:, 1]
        else:
            p = est.predict(X.iloc[te])
        try:
            s = float(scorer(y.iloc[te], p))
        except Exception:
            s = float("nan")
        scores.append(s)
    return scores
