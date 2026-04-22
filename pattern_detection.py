"""Classic breakout pattern detection using price action analysis."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

warnings.filterwarnings("ignore")


PatternResult = Dict[str, object]  # {detected, quality, direction, target, stop, description}


def _find_extrema(series: pd.Series, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices of local minima and maxima."""
    vals = series.values
    peaks = argrelextrema(vals, np.greater_equal, order=order)[0]
    troughs = argrelextrema(vals, np.less_equal, order=order)[0]
    return peaks, troughs


def _quality_stars(score: float) -> str:
    """Convert 0-10 quality score to star string."""
    stars = round(score / 2)
    return "★" * stars + "☆" * (5 - stars)


class PatternDetector:
    """Detect classic breakout patterns for a single stock."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().sort_index()
        self.close = df["Close"].values
        self.high = df["High"].values
        self.low = df["Low"].values
        self.volume = df["Volume"].values
        self.n = len(df)

    # ─────────────────────────────────────────────────────────────────────────

    def detect_cup_and_handle(self) -> PatternResult:
        """U-shaped base (7-65 weeks) followed by small high-handle consolidation."""
        result: PatternResult = {"detected": False, "quality": 0, "direction": "up",
                                  "target": 0.0, "stop": 0.0, "description": "Cup and Handle"}
        if self.n < 50:
            return result

        close = pd.Series(self.close)
        peaks, troughs = _find_extrema(close, order=7)

        if len(peaks) < 2 or len(troughs) < 1:
            return result

        # Look for two peaks (cup rims) with a trough in between
        for i in range(len(peaks) - 1):
            left_rim_idx = peaks[i]
            right_rim_idx = peaks[i + 1]
            cup_width = right_rim_idx - left_rim_idx

            if cup_width < 15 or cup_width > 200:
                continue

            # Find lowest point between peaks
            cup_region = close.iloc[left_rim_idx:right_rim_idx + 1]
            bottom_idx = cup_region.idxmin()
            bottom_val = cup_region.min()

            left_rim = close.iloc[left_rim_idx]
            right_rim = close.iloc[right_rim_idx]
            rim_symmetry = 1 - abs(left_rim - right_rim) / (left_rim + 1e-9)
            cup_depth = (min(left_rim, right_rim) - bottom_val) / (min(left_rim, right_rim) + 1e-9)

            if not (0.08 <= cup_depth <= 0.50):
                continue

            # Handle: consolidation after right rim (should be < 15% range)
            if right_rim_idx >= self.n - 5:
                continue
            handle_close = close.iloc[right_rim_idx:]
            handle_range = (handle_close.max() - handle_close.min()) / (right_rim + 1e-9)

            if handle_range > 0.15:
                continue

            quality = (rim_symmetry * 4 + (1 - abs(cup_depth - 0.25)) * 4 + (1 - handle_range / 0.15) * 2)
            quality = min(10.0, quality * 10 / 8)

            # Measured move target
            target = right_rim * (1 + cup_depth)
            stop = handle_close.min() * 0.99

            result.update({
                "detected": True,
                "quality": round(quality, 1),
                "target": round(target, 2),
                "stop": round(stop, 2),
                "quality_stars": _quality_stars(quality),
            })
            return result

        return result

    def detect_ascending_triangle(self) -> PatternResult:
        """Flat resistance + rising support trendline."""
        result: PatternResult = {"detected": False, "quality": 0, "direction": "up",
                                  "target": 0.0, "stop": 0.0, "description": "Ascending Triangle"}
        if self.n < 30:
            return result

        window = min(60, self.n)
        highs = pd.Series(self.high[-window:])
        lows = pd.Series(self.low[-window:])
        closes = pd.Series(self.close[-window:])

        # Flat resistance: highs cluster near a level
        peak_idxs, _ = _find_extrema(highs, order=3)
        if len(peak_idxs) < 2:
            return result

        peak_vals = highs.iloc[peak_idxs].values
        resistance_level = np.mean(peak_vals)
        resistance_flat = np.std(peak_vals) / (resistance_level + 1e-9) < 0.03

        if not resistance_flat:
            return result

        # Rising lows
        trough_idxs, _ = _find_extrema(-lows, order=3)
        trough_idxs = np.where(np.diff(lows.values) < 0, np.arange(len(lows)-1), -1)
        low_vals = lows.values
        x = np.arange(len(low_vals))
        slope, _ = np.polyfit(x, low_vals, 1)

        if slope <= 0:
            return result

        quality = min(10.0, 7.0 + (slope / resistance_level) * 100)
        target = resistance_level * (1 + (resistance_level - lows.min()) / resistance_level)
        stop = lows.iloc[-5:].min() * 0.99

        result.update({
            "detected": True,
            "quality": round(quality, 1),
            "target": round(target, 2),
            "stop": round(stop, 2),
            "resistance_level": round(resistance_level, 2),
            "quality_stars": _quality_stars(quality),
        })
        return result

    def detect_bull_flag(self) -> PatternResult:
        """Sharp pole rise + tight downward consolidation (flag)."""
        result: PatternResult = {"detected": False, "quality": 0, "direction": "up",
                                  "target": 0.0, "stop": 0.0, "description": "Bull Flag/Pennant"}
        if self.n < 20:
            return result

        close = pd.Series(self.close)

        # Detect pole: strong up-move in 5-15 days
        for pole_len in range(5, 16):
            if pole_len >= self.n - 5:
                continue
            pole_start_idx = self.n - pole_len - 10
            pole_end_idx = self.n - 10

            if pole_start_idx < 0:
                continue

            pole_return = (self.close[pole_end_idx] - self.close[pole_start_idx]) / (self.close[pole_start_idx] + 1e-9)
            if pole_return < 0.10:
                continue

            # Flag: consolidation after pole (last 5-10 sessions)
            flag = close.iloc[pole_end_idx:]
            if len(flag) < 4:
                continue

            flag_range = (flag.max() - flag.min()) / (flag.iloc[0] + 1e-9)
            flag_slope_x = np.arange(len(flag))
            flag_slope, _ = np.polyfit(flag_slope_x, flag.values, 1)

            if flag_range > 0.08 or flag_slope >= 0:
                continue

            quality = min(10.0, pole_return * 80 + (1 - flag_range / 0.08) * 4)
            target = close.iloc[-1] * (1 + pole_return)
            stop = flag.min() * 0.99

            result.update({
                "detected": True,
                "quality": round(quality, 1),
                "target": round(target, 2),
                "stop": round(stop, 2),
                "quality_stars": _quality_stars(quality),
            })
            return result

        return result

    def detect_flat_base(self) -> PatternResult:
        """5+ weeks of <15% price range after an uptrend."""
        result: PatternResult = {"detected": False, "quality": 0, "direction": "up",
                                  "target": 0.0, "stop": 0.0, "description": "Flat Base"}
        if self.n < 35:
            return result

        # Check 5-10 week window
        for weeks in range(5, 11):
            days = weeks * 5
            if days > self.n - 20:
                continue

            base = pd.Series(self.close[-days:])
            base_range = (base.max() - base.min()) / (base.min() + 1e-9)

            if base_range >= 0.15:
                continue

            # Prior uptrend (before the base)
            prior = pd.Series(self.close[-days - 20:-days])
            if len(prior) < 10:
                continue
            prior_return = (prior.iloc[-1] - prior.iloc[0]) / (prior.iloc[0] + 1e-9)
            if prior_return < 0.10:
                continue

            quality = min(10.0, (1 - base_range / 0.15) * 5 + min(5.0, prior_return * 20))
            target = base.max() * (1 + (prior_return * 0.5))
            stop = base.min() * 0.98

            result.update({
                "detected": True,
                "quality": round(quality, 1),
                "target": round(target, 2),
                "stop": round(stop, 2),
                "base_weeks": weeks,
                "quality_stars": _quality_stars(quality),
            })
            return result

        return result

    def detect_vcp(self) -> PatternResult:
        """Volatility Contraction Pattern: each contraction smaller than last."""
        result: PatternResult = {"detected": False, "quality": 0, "direction": "up",
                                  "target": 0.0, "stop": 0.0, "description": "VCP (Minervini)"}
        if self.n < 40:
            return result

        close = pd.Series(self.close)
        peaks, troughs = _find_extrema(close, order=5)

        contractions = []
        for i in range(min(len(peaks), len(troughs))):
            if i >= len(peaks) or i >= len(troughs):
                break
            peak = close.iloc[peaks[i]]
            trough = close.iloc[troughs[i]]
            cont = (peak - trough) / (peak + 1e-9) * 100
            contractions.append(cont)

        if len(contractions) < 3:
            return result

        # Each contraction should be smaller
        shrinking = all(contractions[i] > contractions[i + 1] for i in range(len(contractions) - 1))
        if not shrinking:
            return result

        last_cont = contractions[-1]
        quality = min(10.0, (1 - last_cont / contractions[0]) * 8 + len(contractions))
        target = close.iloc[-1] * (1 + contractions[0] / 100)
        stop = close.iloc[-1] * (1 - last_cont / 100 * 1.5)

        result.update({
            "detected": True,
            "quality": round(quality, 1),
            "target": round(target, 2),
            "stop": round(stop, 2),
            "n_contractions": len(contractions),
            "quality_stars": _quality_stars(quality),
        })
        return result

    def detect_pocket_pivot(self) -> PatternResult:
        """Up day on volume > any down-day volume in last 10 sessions."""
        result: PatternResult = {"detected": False, "quality": 0, "direction": "up",
                                  "target": 0.0, "stop": 0.0, "description": "Pocket Pivot"}
        if self.n < 12:
            return result

        today_close = self.close[-1]
        today_open = self.df["Open"].values[-1] if "Open" in self.df.columns else today_close * 0.99
        today_vol = self.volume[-1]
        today_up = today_close > today_open

        if not today_up:
            return result

        # Max down-day volume in last 10 sessions (excluding today)
        down_vols = []
        for i in range(2, 12):
            if i >= self.n:
                break
            c = self.close[-i]
            o = self.df["Open"].values[-i] if "Open" in self.df.columns else c * 0.99
            if c < o:
                down_vols.append(self.volume[-i])

        if not down_vols:
            return result

        max_down_vol = max(down_vols)
        if today_vol > max_down_vol:
            vol_multiple = today_vol / (max_down_vol + 1e-9)
            quality = min(10.0, 5.0 + vol_multiple * 2)
            target = today_close * 1.08
            stop = self.low[-1] * 0.99
            result.update({
                "detected": True,
                "quality": round(quality, 1),
                "target": round(target, 2),
                "stop": round(stop, 2),
                "vol_multiple": round(vol_multiple, 1),
                "quality_stars": _quality_stars(quality),
            })

        return result

    def detect_inside_day(self) -> PatternResult:
        """Today's range is entirely inside yesterday's range."""
        result: PatternResult = {"detected": False, "quality": 0, "direction": "up",
                                  "target": 0.0, "stop": 0.0, "description": "Inside Day Breakout Setup"}
        if self.n < 3:
            return result

        today_h, today_l = self.high[-1], self.low[-1]
        prev_h, prev_l = self.high[-2], self.low[-2]

        if today_h <= prev_h and today_l >= prev_l:
            compression = 1 - (today_h - today_l) / (prev_h - prev_l + 1e-9)
            quality = min(10.0, compression * 10)
            target = prev_h * 1.02
            stop = prev_l * 0.99
            result.update({
                "detected": True,
                "quality": round(quality, 1),
                "target": round(target, 2),
                "stop": round(stop, 2),
                "quality_stars": _quality_stars(quality),
            })

        return result

    def detect_darvas_box(self) -> PatternResult:
        """New high consolidation box: price makes high then boxes for 3+ weeks."""
        result: PatternResult = {"detected": False, "quality": 0, "direction": "up",
                                  "target": 0.0, "stop": 0.0, "description": "Darvas Box"}
        if self.n < 20:
            return result

        close = pd.Series(self.close)
        high_series = pd.Series(self.high)

        # Recent new high
        lookback = min(52 * 5, self.n)
        period_high = high_series.iloc[-lookback:].max()
        recent_high = high_series.iloc[-20:].max()

        if recent_high < period_high * 0.97:
            return result

        # Box: consolidation after the new high
        box = close.iloc[-15:]
        box_top = box.max()
        box_bottom = box.min()
        box_range = (box_top - box_bottom) / (box_bottom + 1e-9)

        if box_range > 0.12:
            return result

        quality = min(10.0, (1 - box_range / 0.12) * 8 + 2)
        target = box_top * (1 + box_range)
        stop = box_bottom * 0.99

        result.update({
            "detected": True,
            "quality": round(quality, 1),
            "target": round(target, 2),
            "stop": round(stop, 2),
            "box_top": round(box_top, 2),
            "box_bottom": round(box_bottom, 2),
            "quality_stars": _quality_stars(quality),
        })
        return result

    # ─────────────────────────────────────────────────────────────────────────

    def detect_all(self) -> Dict[str, PatternResult]:
        """Run all pattern detectors and return results dict."""
        return {
            "cup_handle":       self.detect_cup_and_handle(),
            "ascending_triangle": self.detect_ascending_triangle(),
            "bull_flag":        self.detect_bull_flag(),
            "flat_base":        self.detect_flat_base(),
            "vcp":              self.detect_vcp(),
            "pocket_pivot":     self.detect_pocket_pivot(),
            "inside_day":       self.detect_inside_day(),
            "darvas_box":       self.detect_darvas_box(),
        }

    def get_best_pattern(self) -> Tuple[str, float]:
        """Return (pattern_name, quality) for highest-quality detected pattern."""
        patterns = self.detect_all()
        best_name, best_quality = "None", 0.0
        for name, res in patterns.items():
            if res["detected"] and res["quality"] > best_quality:
                best_name = res["description"]
                best_quality = res["quality"]
        return best_name, best_quality


def get_pattern_labels(patterns: Dict[str, PatternResult]) -> str:
    """Comma-separated list of detected pattern names."""
    detected = [v["description"] for v in patterns.values() if v.get("detected")]
    return ", ".join(detected) if detected else "—"
