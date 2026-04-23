"""Risk + position sizing.

Phase 1: fixed-fractional + Kelly (UPGRADE 12).
Entry / stop / T1 / T2 derived from ATR + triple-barrier config.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange


@dataclass
class TradeLevels:
    entry: float
    stop: float
    target1: float
    target2: float
    atr: float
    risk_pct: float      # (entry - stop) / entry
    reward_pct: float    # (target1 - entry) / entry
    rr: float


def compute_levels(
    df: pd.DataFrame,
    upper_pct: float = 0.05,
    lower_pct: float = -0.03,
    atr_stop_mult: float = 1.5,
) -> Optional[TradeLevels]:
    """Levels for the latest bar. ATR stop wins if tighter than triple-barrier lower."""
    if df is None or len(df) < 30:
        return None
    close = float(df["Close"].iloc[-1])
    atr_series = AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    ).average_true_range()
    atr = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else close * 0.02

    barrier_stop = close * (1.0 + lower_pct)       # e.g. -3%
    atr_stop = close - atr * atr_stop_mult
    stop = max(barrier_stop, atr_stop)             # tighter of the two
    target1 = close * (1.0 + upper_pct)            # matches label's upper barrier
    target2 = close * (1.0 + 2.0 * upper_pct)

    risk_pct = (close - stop) / close if close > 0 else 0.0
    reward_pct = (target1 - close) / close if close > 0 else 0.0
    rr = reward_pct / risk_pct if risk_pct > 0 else 0.0

    return TradeLevels(
        entry=round(close, 2),
        stop=round(stop, 2),
        target1=round(target1, 2),
        target2=round(target2, 2),
        atr=round(atr, 2),
        risk_pct=round(risk_pct, 4),
        reward_pct=round(reward_pct, 4),
        rr=round(rr, 2),
    )


def kelly_fraction(
    prob: float,
    stop_pct: float,
    target_pct: float,
    cap: float = 0.25,
) -> float:
    """Half-Kelly capped at `cap` (default 25%).

    f* = prob - (1 - prob) * (stop / target)
    """
    if target_pct <= 0 or stop_pct <= 0:
        return 0.0
    raw = prob - (1.0 - prob) * (stop_pct / target_pct)
    return float(max(0.0, min(cap, raw * 0.5)))     # half-Kelly for robustness


def fixed_fractional(
    capital: float,
    risk_per_trade_pct: float,
    stop_pct: float,
) -> Dict[str, float]:
    """Risk a fixed % of capital per trade."""
    if stop_pct <= 0 or capital <= 0:
        return {"capital_alloc": 0.0, "capital_pct": 0.0}
    risk_inr = capital * risk_per_trade_pct
    capital_alloc = risk_inr / stop_pct
    capital_alloc = min(capital_alloc, capital)     # cap at full capital
    return {
        "capital_alloc": round(capital_alloc, 2),
        "capital_pct": round(capital_alloc / capital, 4),
    }


def size_position(
    capital: float,
    prob: float,
    levels: TradeLevels,
    mode: str = "kelly",
    risk_per_trade_pct: float = 0.01,
    kelly_cap: float = 0.25,
) -> Dict[str, float]:
    """Unified entry point. mode = 'kelly' or 'fixed'."""
    if levels is None or capital <= 0:
        return {"mode": mode, "capital_alloc": 0.0, "capital_pct": 0.0,
                "qty": 0, "kelly_frac": 0.0}
    if mode == "kelly":
        frac = kelly_fraction(prob, levels.risk_pct, levels.reward_pct, cap=kelly_cap)
        capital_alloc = capital * frac
        qty = int(capital_alloc // levels.entry) if levels.entry > 0 else 0
        return {
            "mode": "kelly",
            "capital_alloc": round(capital_alloc, 2),
            "capital_pct": round(frac, 4),
            "qty": qty,
            "kelly_frac": round(frac, 4),
        }
    else:
        ff = fixed_fractional(capital, risk_per_trade_pct, levels.risk_pct)
        qty = int(ff["capital_alloc"] // levels.entry) if levels.entry > 0 else 0
        return {
            "mode": "fixed",
            "capital_alloc": ff["capital_alloc"],
            "capital_pct": ff["capital_pct"],
            "qty": qty,
            "kelly_frac": 0.0,
        }
