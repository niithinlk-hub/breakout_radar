"""Per-bar pattern flag time-series.

For each ticker, walk the bar timeline and run `PatternScanner` on a
growing window. Returns a DataFrame indexed by date with:
  - <name>_fires  : int (0/1)
  - <name>_conf   : float (0.0 - 1.0)
for all 20 patterns. These columns feed the meta-model (UPGRADE 8).

Expensive — we only compute on bars where the primary signal already
fired, to cap cost.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .patterns import PATTERN_NAMES, PatternScanner


def _slug(name: str) -> str:
    return (name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
            .replace(",", "")
            .replace("__", "_"))


PATTERN_SLUGS = [_slug(n) for n in PATTERN_NAMES]

FIRES_COLS = [f"pat_{s}_fires" for s in PATTERN_SLUGS]
CONF_COLS = [f"pat_{s}_conf" for s in PATTERN_SLUGS]
PATTERN_FEATURE_COLS = FIRES_COLS + CONF_COLS


def patterns_at_bars(
    df: pd.DataFrame,
    bar_dates: pd.DatetimeIndex,
    min_history: int = 60,
) -> pd.DataFrame:
    """Scan `bar_dates` (subset of df.index). Returns len(bar_dates) × 40 cols.

    Parameters
    ----------
    df : full OHLCV for the ticker (will be sliced up to each bar).
    bar_dates : dates to score (usually primary-fired dates).
    """
    if df is None or df.empty or len(bar_dates) == 0:
        return pd.DataFrame(index=bar_dates, columns=PATTERN_FEATURE_COLS,
                            dtype=float)

    rows: List[Dict] = []
    for d in bar_dates:
        if d not in df.index:
            rows.append({c: 0.0 for c in PATTERN_FEATURE_COLS})
            continue
        sub = df.loc[:d]
        if len(sub) < min_history:
            rows.append({c: 0.0 for c in PATTERN_FEATURE_COLS})
            continue
        try:
            res = PatternScanner(sub).scan()
        except Exception:
            rows.append({c: 0.0 for c in PATTERN_FEATURE_COLS})
            continue
        row: Dict[str, float] = {c: 0.0 for c in PATTERN_FEATURE_COLS}
        for name, slug in zip(PATTERN_NAMES, PATTERN_SLUGS):
            if name in res["patterns"]:
                row[f"pat_{slug}_fires"] = 1.0
                row[f"pat_{slug}_conf"] = float(res["pattern_details"][name]["confidence"])
        rows.append(row)
    out = pd.DataFrame(rows, index=bar_dates)
    return out


def count_firings(df_patterns: pd.DataFrame) -> Dict[str, int]:
    """For U9 honesty: count historical firings per pattern to flag <100 as experimental."""
    counts: Dict[str, int] = {}
    for name, slug in zip(PATTERN_NAMES, PATTERN_SLUGS):
        col = f"pat_{slug}_fires"
        if col in df_patterns.columns:
            counts[name] = int(df_patterns[col].sum())
        else:
            counts[name] = 0
    return counts


MIN_FIRINGS_FOR_FEATURE = 100
