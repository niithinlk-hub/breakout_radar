"""Composite ranking score (UPGRADE 10).

score = 0.60 × meta_prob
      + 0.20 × (pattern_count × avg_pattern_confidence) / max_possible
      + 0.20 × regime_adjusted_expected_return

regime_adjusted_expected_return
    = (prob × upper_pct - (1 - prob) × abs(lower_pct))
      × {Bull: 1.2, Choppy: 1.0, Risk-off: 0.0}

Honesty constraint: the composite must not inflate a mediocre base prob.
The meta-prob term caps at its true value — adding patterns or a bull
regime only nudges the ranking; it never turns a 0.55 into an 85.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

PATTERN_MAX_COUNT = 6           # cap for normalization (6 concurrent = saturated)
PATTERN_MAX_SCORE = PATTERN_MAX_COUNT * 1.0


REGIME_MULTIPLIER = {
    0: 1.2,   # Bull trending
    1: 1.0,   # Choppy / Range
    2: 0.0,   # Risk-off — zeroes the regime term
    -1: 0.8,  # Unknown (HMM unavailable) — conservative
}


@dataclass
class CompositeScore:
    total: float
    prob_component: float
    pattern_component: float
    regime_component: float


def compute_composite(
    prob: float,
    pattern_count: int,
    pattern_confidence_avg: float,
    regime_state: int,
    upper_pct: float = 0.05,
    lower_pct: float = -0.03,
) -> CompositeScore:
    w_prob, w_pat, w_reg = 0.60, 0.20, 0.20

    prob_comp = w_prob * float(prob)

    # Pattern term: normalized so 6 concurrent patterns at 1.0 conf → 1.0 weight
    pat_raw = float(pattern_count) * float(pattern_confidence_avg)
    pat_norm = min(1.0, pat_raw / PATTERN_MAX_SCORE)
    pat_comp = w_pat * pat_norm

    expected = prob * upper_pct - (1.0 - prob) * abs(lower_pct)
    mult = REGIME_MULTIPLIER.get(regime_state, 1.0)
    # Normalize expected to [0, 1]: cap at +5% = 0.05, clip negatives
    expected_norm = max(0.0, min(1.0, expected / upper_pct))
    reg_comp = w_reg * expected_norm * (mult / 1.2)  # 1.2 is the max multiplier

    total = prob_comp + pat_comp + reg_comp
    return CompositeScore(
        total=round(total, 4),
        prob_component=round(prob_comp, 4),
        pattern_component=round(pat_comp, 4),
        regime_component=round(reg_comp, 4),
    )
