"""20-pattern detector for NSE swing-trade screening.

Output schema (per ticker):
    {
        "patterns": ["Cup and Handle", ...],
        "pattern_count": 3,
        "pattern_confidence_avg": 0.78,
        "pattern_details": {
            "Cup and Handle": {"confidence": 0.82, "breakout_price": 1245.0, "target": 1394.0},
            ...
        },
    }

Detectors (20):
    Candlestick:   Bullish Engulfing, Hammer, Morning Star, Piercing Line,
                   Three White Soldiers, Bullish Harami
    Chart:         Cup and Handle, Bull Flag, Ascending Triangle,
                   Inverse H&S, Double Bottom, VCP, Inside Bar Breakout,
                   20-day High Breakout, 52-week High Proximity, Rounding Bottom
    Trend:         Pullback to 50-EMA, Breakaway Gap, Pocket Pivot, Golden Cross

Phase-2 deliverable: UI + standalone scanner. Pattern flags feed the
meta-model in Phase 3 (pending historical-firings audit; any detector
firing <100 times across the training set is flagged 'experimental' and
kept UI-only).
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

warnings.filterwarnings("ignore")


# ──────────────────────────── helpers ───────────────────────────────────────

def _ema(s: pd.Series, window: int) -> pd.Series:
    return s.ewm(span=window, adjust=False).mean()


def _find_peaks(arr: np.ndarray, order: int = 5) -> np.ndarray:
    return argrelextrema(arr, np.greater_equal, order=order)[0]


def _find_troughs(arr: np.ndarray, order: int = 5) -> np.ndarray:
    return argrelextrema(arr, np.less_equal, order=order)[0]


def _clip01(x: float) -> float:
    if np.isnan(x):
        return 0.0
    return float(max(0.0, min(1.0, x)))


# ────────────────────────── PatternScanner ──────────────────────────────────

class PatternScanner:
    """Run all 20 detectors on a single ticker's OHLCV."""

    def __init__(self, df: pd.DataFrame):
        if df is None or len(df) < 30:
            raise ValueError("Need at least 30 bars.")
        self.df = df.sort_index()
        self.open = self.df["Open"].values
        self.high = self.df["High"].values
        self.low = self.df["Low"].values
        self.close = self.df["Close"].values
        self.vol = self.df["Volume"].values
        self.n = len(self.df)
        self.close_s = pd.Series(self.close, index=self.df.index)
        self.vol_ma20 = pd.Series(self.vol).rolling(20).mean().values

    # ─────────────── candlestick detectors (6) ──────────────────────────────

    def _det_bullish_engulfing(self) -> Tuple[bool, float, Dict]:
        if self.n < 2:
            return False, 0.0, {}
        o1, c1 = self.open[-2], self.close[-2]
        o2, c2 = self.open[-1], self.close[-1]
        red_prev = c1 < o1
        green_now = c2 > o2
        engulf = c2 >= o1 and o2 <= c1
        if not (red_prev and green_now and engulf):
            return False, 0.0, {}
        size_ratio = abs(c2 - o2) / max(1e-9, abs(c1 - o1))
        conf = _clip01(0.5 + 0.25 * min(2.0, size_ratio))
        return True, conf, {"body_ratio": round(size_ratio, 2)}

    def _det_hammer(self) -> Tuple[bool, float, Dict]:
        if self.n < 20:
            return False, 0.0, {}
        o, h, l, c = self.open[-1], self.high[-1], self.low[-1], self.close[-1]
        body = abs(c - o)
        rng = h - l
        if rng <= 0:
            return False, 0.0, {}
        upper_wick = h - max(c, o)
        lower_wick = min(c, o) - l
        if body / rng > 0.35:
            return False, 0.0, {}
        if lower_wick < 2 * body:
            return False, 0.0, {}
        if upper_wick > body:
            return False, 0.0, {}
        # At support: close near 20d low
        sup = float(np.min(self.low[-20:]))
        near_sup = (c - sup) / max(1e-9, sup) < 0.03
        if not near_sup:
            return False, 0.0, {}
        conf = _clip01(0.4 + 0.1 * (lower_wick / max(1e-9, body)))
        return True, conf, {"lower_to_body": round(lower_wick / max(1e-9, body), 2)}

    def _det_morning_star(self) -> Tuple[bool, float, Dict]:
        if self.n < 3:
            return False, 0.0, {}
        o1, c1 = self.open[-3], self.close[-3]
        o2, c2 = self.open[-2], self.close[-2]
        o3, c3 = self.open[-1], self.close[-1]
        red_1 = c1 < o1 and (o1 - c1) / max(1e-9, o1) > 0.01
        small_2 = abs(c2 - o2) / max(1e-9, o2) < 0.005
        green_3 = c3 > o3
        above_mid = c3 > (o1 + c1) / 2
        if not (red_1 and small_2 and green_3 and above_mid):
            return False, 0.0, {}
        recovery = (c3 - c1) / max(1e-9, (o1 - c1))
        conf = _clip01(0.5 + 0.2 * min(1.5, recovery))
        return True, conf, {"recovery": round(float(recovery), 2)}

    def _det_piercing_line(self) -> Tuple[bool, float, Dict]:
        if self.n < 2:
            return False, 0.0, {}
        o1, c1 = self.open[-2], self.close[-2]
        o2, c2 = self.open[-1], self.close[-1]
        if not (c1 < o1):
            return False, 0.0, {}
        if not (o2 < c1 and c2 > (o1 + c1) / 2 and c2 < o1):
            return False, 0.0, {}
        pen = (c2 - c1) / max(1e-9, (o1 - c1))
        conf = _clip01(0.4 + 0.3 * pen)
        return True, conf, {"penetration": round(float(pen), 2)}

    def _det_three_white_soldiers(self) -> Tuple[bool, float, Dict]:
        if self.n < 3:
            return False, 0.0, {}
        o = self.open[-3:]
        c = self.close[-3:]
        h = self.high[-3:]
        if not all(c[i] > o[i] for i in range(3)):
            return False, 0.0, {}
        if not (c[1] > c[0] and c[2] > c[1]):
            return False, 0.0, {}
        # Each opens within prior body
        if not (o[1] >= o[0] and o[1] <= c[0] and o[2] >= o[1] and o[2] <= c[1]):
            return False, 0.0, {}
        # Closes near highs (small upper wicks)
        close_to_high = np.mean([(c[i] - o[i]) / max(1e-9, (h[i] - o[i])) for i in range(3)])
        conf = _clip01(0.5 + 0.3 * close_to_high)
        return True, conf, {"close_to_high_ratio": round(float(close_to_high), 2)}

    def _det_bullish_harami(self) -> Tuple[bool, float, Dict]:
        if self.n < 2:
            return False, 0.0, {}
        o1, c1 = self.open[-2], self.close[-2]
        o2, c2 = self.open[-1], self.close[-1]
        red_prev = c1 < o1
        green_now = c2 > o2
        inside = (o2 > c1 and c2 < o1)
        if not (red_prev and green_now and inside):
            return False, 0.0, {}
        compress = 1.0 - abs(c2 - o2) / max(1e-9, abs(c1 - o1))
        conf = _clip01(0.4 + 0.3 * compress)
        return True, conf, {"body_compression": round(float(compress), 2)}

    # ─────────────── chart pattern detectors (10) ───────────────────────────

    def _det_cup_and_handle(self) -> Tuple[bool, float, Dict]:
        if self.n < 50:
            return False, 0.0, {}
        close = self.close_s
        peaks = _find_peaks(self.close, order=7)
        if len(peaks) < 2:
            return False, 0.0, {}
        for i in range(len(peaks) - 1):
            L, R = peaks[i], peaks[i + 1]
            width = R - L
            if width < 30 or width > 150:
                continue
            cup = close.iloc[L:R + 1]
            bottom = cup.min()
            rimL = close.iloc[L]
            rimR = close.iloc[R]
            sym = 1.0 - abs(rimL - rimR) / max(1e-9, rimL)
            depth = (min(rimL, rimR) - bottom) / max(1e-9, min(rimL, rimR))
            if not (0.12 <= depth <= 0.50):
                continue
            if sym < 0.95:
                continue
            # Handle: 5-15 bars after right rim, <12% pullback
            if R >= self.n - 5:
                continue
            handle = close.iloc[R:]
            if len(handle) < 5 or len(handle) > 30:
                continue
            handle_pull = (rimR - handle.min()) / max(1e-9, rimR)
            if handle_pull > 0.12:
                continue
            # Current bar breaks right rim on above-avg volume
            breakout = self.close[-1] > rimR
            vol_conf = self.vol[-1] > self.vol_ma20[-1] if not np.isnan(self.vol_ma20[-1]) else False
            conf = 0.4 * sym + 0.2 * (1 - abs(depth - 0.25) / 0.25)
            conf += 0.2 * (1 - handle_pull / 0.12)
            conf += 0.1 * (1.0 if breakout else 0.0) + 0.1 * (1.0 if vol_conf else 0.0)
            target = float(rimR * (1 + depth))
            return True, _clip01(conf), {
                "breakout_price": round(float(rimR), 2),
                "target": round(target, 2),
                "cup_depth": round(float(depth), 3),
                "handle_pullback": round(float(handle_pull), 3),
            }
        return False, 0.0, {}

    def _det_bull_flag(self) -> Tuple[bool, float, Dict]:
        if self.n < 20:
            return False, 0.0, {}
        # Pole: 8%+ in <=10 bars, ending 3-15 bars ago
        for flag_len in range(3, 16):
            pole_end = self.n - flag_len
            if pole_end < 10:
                continue
            for pole_len in range(5, 11):
                pole_start = pole_end - pole_len
                if pole_start < 0:
                    continue
                pole_ret = (self.close[pole_end - 1] - self.close[pole_start]) / max(1e-9, self.close[pole_start])
                if pole_ret < 0.08:
                    continue
                flag_close = self.close[pole_end:]
                flag_vol = self.vol[pole_end:]
                flag_range = (flag_close.max() - flag_close.min()) / max(1e-9, flag_close[0])
                if flag_range > 0.10:
                    continue
                # Declining volume through flag
                vol_trend = np.polyfit(np.arange(len(flag_vol)), flag_vol, 1)[0]
                if vol_trend > 0:
                    continue
                # Breakout: current bar breaks flag's upper boundary
                flag_high = flag_close[:-1].max() if len(flag_close) > 1 else flag_close[0]
                if self.close[-1] < flag_high:
                    continue
                conf = 0.35 + 0.25 * min(1.0, pole_ret / 0.15) + 0.2 * (1 - flag_range / 0.10)
                conf += 0.2 * (1.0 if vol_trend < 0 else 0.0)
                target = float(self.close[-1] * (1 + pole_ret))
                return True, _clip01(conf), {
                    "pole_return": round(float(pole_ret), 3),
                    "flag_range": round(float(flag_range), 3),
                    "breakout_price": round(float(flag_high), 2),
                    "target": round(target, 2),
                }
        return False, 0.0, {}

    def _det_ascending_triangle(self) -> Tuple[bool, float, Dict]:
        if self.n < 30:
            return False, 0.0, {}
        w = min(60, self.n)
        highs = self.high[-w:]
        lows = self.low[-w:]
        peaks = _find_peaks(highs, order=3)
        if len(peaks) < 3:
            return False, 0.0, {}
        # Pick top 3 peak clusters
        peak_vals = highs[peaks]
        top_level = float(np.median(peak_vals))
        within = np.abs(peak_vals - top_level) / max(1e-9, top_level) < 0.01
        if within.sum() < 3:
            return False, 0.0, {}
        x = np.arange(len(lows))
        slope = float(np.polyfit(x, lows, 1)[0])
        if slope <= 0:
            return False, 0.0, {}
        # Pressing resistance
        last_c = float(self.close[-1])
        press = last_c / top_level
        if press < 0.97:
            return False, 0.0, {}
        conf = 0.4 + 0.2 * min(3, within.sum()) / 3
        conf += 0.2 * min(1.0, (slope / max(1e-9, top_level)) * 200)
        conf += 0.2 * (1.0 if press >= 0.99 else 0.5)
        target = float(top_level + (top_level - lows.min()))
        return True, _clip01(conf), {
            "resistance": round(top_level, 2),
            "support_slope": round(slope, 4),
            "breakout_price": round(top_level, 2),
            "target": round(target, 2),
        }

    def _det_inverse_hs(self) -> Tuple[bool, float, Dict]:
        if self.n < 40:
            return False, 0.0, {}
        troughs = _find_troughs(self.low, order=5)
        if len(troughs) < 3:
            return False, 0.0, {}
        for i in range(len(troughs) - 2):
            l, m, r = troughs[i], troughs[i + 1], troughs[i + 2]
            if r < self.n - 30 or r > self.n - 2:
                continue
            Ll, Lm, Lr = self.low[l], self.low[m], self.low[r]
            if not (Lm < Ll and Lm < Lr):
                continue
            if abs(Ll - Lr) / max(1e-9, min(Ll, Lr)) > 0.05:
                continue
            # Neckline: max high between shoulders
            neck = float(self.high[l:r + 1].max())
            if self.close[-1] < neck:
                continue
            drop = (min(Ll, Lr) - Lm) / max(1e-9, min(Ll, Lr))
            conf = 0.45 + 0.25 * min(1.0, drop / 0.10)
            conf += 0.2 * (1.0 - abs(Ll - Lr) / max(1e-9, min(Ll, Lr)) / 0.05)
            conf += 0.1
            target = float(neck + (neck - Lm))
            return True, _clip01(conf), {
                "neckline": round(neck, 2),
                "breakout_price": round(neck, 2),
                "target": round(target, 2),
            }
        return False, 0.0, {}

    def _det_double_bottom(self) -> Tuple[bool, float, Dict]:
        if self.n < 30:
            return False, 0.0, {}
        troughs = _find_troughs(self.low, order=5)
        if len(troughs) < 2:
            return False, 0.0, {}
        for i in range(len(troughs) - 1):
            a, b = troughs[i], troughs[i + 1]
            sep = b - a
            if sep < 10 or sep > 60:
                continue
            if b < self.n - 30 or b > self.n - 2:
                continue
            La, Lb = self.low[a], self.low[b]
            if abs(La - Lb) / max(1e-9, min(La, Lb)) > 0.03:
                continue
            mid_peak = float(self.high[a:b + 1].max())
            if self.close[-1] < mid_peak:
                continue
            conf = 0.45 + 0.2 * (1 - abs(La - Lb) / max(1e-9, min(La, Lb)) / 0.03)
            conf += 0.25 * min(1.0, (self.close[-1] - mid_peak) / max(1e-9, mid_peak) * 50)
            conf += 0.1
            target = float(mid_peak + (mid_peak - min(La, Lb)))
            return True, _clip01(conf), {
                "bottom_a": round(float(La), 2),
                "bottom_b": round(float(Lb), 2),
                "breakout_price": round(mid_peak, 2),
                "target": round(target, 2),
            }
        return False, 0.0, {}

    def _det_vcp(self) -> Tuple[bool, float, Dict]:
        if self.n < 40:
            return False, 0.0, {}
        close = self.close_s
        peaks = _find_peaks(self.close, order=5)
        troughs = _find_troughs(self.close, order=5)
        if len(peaks) < 3 or len(troughs) < 2:
            return False, 0.0, {}
        # Take last 2-4 alternating peak/trough cycles
        pairs = min(4, min(len(peaks), len(troughs)))
        if pairs < 2:
            return False, 0.0, {}
        contractions = []
        for i in range(pairs):
            pv = close.iloc[peaks[-pairs + i]]
            tv = close.iloc[troughs[-pairs + i]]
            contractions.append(float((pv - tv) / max(1e-9, pv)))
        # Each <= 60% of previous
        shrinking = all(contractions[i + 1] <= 0.60 * contractions[i] + 1e-9
                        for i in range(len(contractions) - 1))
        if not shrinking:
            return False, 0.0, {}
        # Volume declining through contractions
        vol_trend = float(np.polyfit(np.arange(30),
                                     self.vol[-30:], 1)[0])
        conf = 0.4 + 0.3 * min(1.0, (1 - contractions[-1] / contractions[0]))
        conf += 0.2 * (1.0 if vol_trend < 0 else 0.0)
        conf += 0.1 * min(1.0, len(contractions) / 4)
        target = float(self.close[-1] * (1 + contractions[0]))
        return True, _clip01(conf), {
            "n_contractions": len(contractions),
            "last_contraction": round(contractions[-1], 3),
            "first_contraction": round(contractions[0], 3),
            "target": round(target, 2),
        }

    def _det_inside_bar_breakout(self) -> Tuple[bool, float, Dict]:
        if self.n < 3:
            return False, 0.0, {}
        # Check prior 1-2 bars for inside-bar structure
        for lag in (1, 2):
            if self.n < lag + 2:
                continue
            mother_h = self.high[-(lag + 2)]
            mother_l = self.low[-(lag + 2)]
            inside_h = self.high[-(lag + 1)]
            inside_l = self.low[-(lag + 1)]
            if not (inside_h <= mother_h and inside_l >= mother_l):
                continue
            if self.close[-1] <= mother_h:
                continue
            compress = 1.0 - (inside_h - inside_l) / max(1e-9, mother_h - mother_l)
            conf = 0.45 + 0.35 * compress + 0.2 * min(1.0, (self.close[-1] - mother_h) / max(1e-9, mother_h) * 30)
            return True, _clip01(conf), {
                "mother_high": round(float(mother_h), 2),
                "breakout_price": round(float(mother_h), 2),
                "compression": round(float(compress), 2),
            }
        return False, 0.0, {}

    def _det_20d_high_vol_breakout(self) -> Tuple[bool, float, Dict]:
        if self.n < 22:
            return False, 0.0, {}
        prior_high = float(np.max(self.high[-21:-1]))
        vol_ma20 = float(np.mean(self.vol[-21:-1]))
        if self.close[-1] <= prior_high:
            return False, 0.0, {}
        if self.vol[-1] < 1.5 * vol_ma20:
            return False, 0.0, {}
        excess = (self.close[-1] - prior_high) / max(1e-9, prior_high)
        vol_mult = self.vol[-1] / max(1e-9, vol_ma20)
        conf = 0.5 + 0.25 * min(1.0, excess * 30) + 0.25 * min(1.0, (vol_mult - 1.5) / 2.0)
        target = float(self.close[-1] * 1.05)
        return True, _clip01(conf), {
            "prior_20d_high": round(prior_high, 2),
            "volume_multiple": round(vol_mult, 2),
            "target": round(target, 2),
        }

    def _det_52w_high_proximity(self) -> Tuple[bool, float, Dict]:
        if self.n < 60:
            return False, 0.0, {}
        lookback = min(252, self.n)
        hh = float(np.max(self.high[-lookback:]))
        if self.close[-1] < 0.98 * hh:
            return False, 0.0, {}
        # Broke past prior resistance (60d high from 62 to 2 bars ago)
        if self.n < 62:
            return False, 0.0, {}
        prior_res = float(np.max(self.high[-62:-2]))
        if self.close[-1] < prior_res:
            return False, 0.0, {}
        proximity = self.close[-1] / hh
        conf = 0.5 + 0.3 * min(1.0, (proximity - 0.98) / 0.02) + 0.2
        return True, _clip01(conf), {
            "52w_high": round(hh, 2),
            "proximity": round(float(proximity), 3),
            "prior_resistance": round(prior_res, 2),
        }

    def _det_rounding_bottom(self) -> Tuple[bool, float, Dict]:
        if self.n < 40:
            return False, 0.0, {}
        # Fit 2nd-degree poly on windows 40-80
        best = (0.0, None)
        for w in (40, 60, 80):
            if self.n < w:
                continue
            y = self.close[-w:]
            x = np.arange(w)
            try:
                coefs = np.polyfit(x, y, 2)
            except Exception:
                continue
            if coefs[0] <= 0:
                continue  # must open upward
            y_pred = np.polyval(coefs, x)
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            if ss_tot <= 0:
                continue
            r2 = 1.0 - ss_res / ss_tot
            if r2 > best[0]:
                best = (r2, w)
        r2, w = best
        if r2 < 0.85 or w is None:
            return False, 0.0, {}
        conf = _clip01(0.3 + (r2 - 0.85) * 4)
        bottom = float(np.min(self.close[-w:]))
        target = float(self.close[-1] * 1.10)
        return True, conf, {
            "r2": round(r2, 3),
            "window": w,
            "bottom": round(bottom, 2),
            "target": round(target, 2),
        }

    # ─────────────── trend / momentum detectors (4) ─────────────────────────

    def _det_pullback_to_50ema(self) -> Tuple[bool, float, Dict]:
        if self.n < 60:
            return False, 0.0, {}
        c = self.close_s
        e8 = _ema(c, 8).values
        e21 = _ema(c, 21).values
        e50 = _ema(c, 50).values
        i = self.n - 1
        if not (e8[i] > e21[i] > e50[i]):
            return False, 0.0, {}
        touched = False
        for k in range(max(0, i - 2), i + 1):
            if self.low[k] <= e50[k] * 1.01:
                touched = True
                break
        if not touched:
            return False, 0.0, {}
        green = self.close[-1] > self.open[-1]
        if not green:
            return False, 0.0, {}
        stack_strength = min(1.0, (e8[i] - e50[i]) / max(1e-9, e50[i]) * 10)
        conf = 0.5 + 0.3 * stack_strength + 0.2
        return True, _clip01(conf), {
            "ema50": round(float(e50[i]), 2),
            "ema8_over_ema50_pct": round(float((e8[i] - e50[i]) / max(1e-9, e50[i])), 4),
        }

    def _det_breakaway_gap(self) -> Tuple[bool, float, Dict]:
        if self.n < 21:
            return False, 0.0, {}
        gap_pct = (self.open[-1] - self.close[-2]) / max(1e-9, self.close[-2])
        if gap_pct < 0.02:
            return False, 0.0, {}
        rng = self.high[-1] - self.low[-1]
        if rng <= 0:
            return False, 0.0, {}
        upper_third = self.low[-1] + (2.0 / 3.0) * rng
        if self.close[-1] < upper_third:
            return False, 0.0, {}
        vol_ma20 = float(np.mean(self.vol[-21:-1]))
        vol_mult = self.vol[-1] / max(1e-9, vol_ma20)
        if vol_mult < 2.0:
            return False, 0.0, {}
        conf = 0.5 + 0.25 * min(1.0, gap_pct / 0.05) + 0.25 * min(1.0, (vol_mult - 2.0) / 3.0)
        return True, _clip01(conf), {
            "gap_pct": round(float(gap_pct), 4),
            "volume_multiple": round(float(vol_mult), 2),
        }

    def _det_pocket_pivot(self) -> Tuple[bool, float, Dict]:
        if self.n < 12:
            return False, 0.0, {}
        if self.close[-1] <= self.open[-1]:
            return False, 0.0, {}
        down_vols = []
        for k in range(2, 12):
            if self.n < k + 1:
                break
            idx = -k
            if self.close[idx] < self.open[idx]:
                down_vols.append(self.vol[idx])
        if not down_vols:
            return False, 0.0, {}
        max_dv = float(max(down_vols))
        if self.vol[-1] <= max_dv:
            return False, 0.0, {}
        # Within a base: 10d range <12%
        rng10 = (self.high[-10:].max() - self.low[-10:].min()) / max(1e-9, self.close[-11])
        if rng10 > 0.12:
            return False, 0.0, {}
        vol_mult = self.vol[-1] / max(1e-9, max_dv)
        conf = 0.5 + 0.3 * min(1.0, (vol_mult - 1.0) / 2.0) + 0.2 * (1 - rng10 / 0.12)
        return True, _clip01(conf), {
            "volume_vs_max_down": round(float(vol_mult), 2),
            "base_range_10d": round(float(rng10), 3),
        }

    def _det_golden_cross(self) -> Tuple[bool, float, Dict]:
        if self.n < 55:
            return False, 0.0, {}
        e20 = _ema(self.close_s, 20).values
        e50 = _ema(self.close_s, 50).values
        crossed = False
        for k in range(-3, 0):
            if e20[k - 1] <= e50[k - 1] and e20[k] > e50[k]:
                crossed = True
                break
        if not crossed:
            return False, 0.0, {}
        sep = (e20[-1] - e50[-1]) / max(1e-9, e50[-1])
        conf = _clip01(0.5 + 0.4 * min(1.0, sep * 50))
        return True, conf, {
            "ema20_minus_ema50_pct": round(float(sep), 4),
        }

    # ─────────────── orchestration ──────────────────────────────────────────

    REGISTRY = [
        # name, method, category
        ("Bullish Engulfing",              "_det_bullish_engulfing",      "candle"),
        ("Hammer at Support",              "_det_hammer",                  "candle"),
        ("Morning Star",                   "_det_morning_star",            "candle"),
        ("Piercing Line",                  "_det_piercing_line",           "candle"),
        ("Three White Soldiers",           "_det_three_white_soldiers",    "candle"),
        ("Bullish Harami",                 "_det_bullish_harami",          "candle"),
        ("Cup and Handle",                 "_det_cup_and_handle",          "chart"),
        ("Bull Flag",                      "_det_bull_flag",               "chart"),
        ("Ascending Triangle",             "_det_ascending_triangle",      "chart"),
        ("Inverse Head and Shoulders",     "_det_inverse_hs",              "chart"),
        ("Double Bottom",                  "_det_double_bottom",           "chart"),
        ("VCP (Minervini)",                "_det_vcp",                     "chart"),
        ("Inside Bar Breakout",            "_det_inside_bar_breakout",     "chart"),
        ("20-day High Vol Breakout",       "_det_20d_high_vol_breakout",   "chart"),
        ("52-week High Proximity",         "_det_52w_high_proximity",      "chart"),
        ("Rounding Bottom",                "_det_rounding_bottom",         "chart"),
        ("Pullback to 50-EMA",             "_det_pullback_to_50ema",       "trend"),
        ("Breakaway Gap",                  "_det_breakaway_gap",           "trend"),
        ("Pocket Pivot",                   "_det_pocket_pivot",            "trend"),
        ("Golden Cross (short-term)",      "_det_golden_cross",            "trend"),
    ]

    def scan(self) -> Dict:
        """Run all 20 detectors; return standardized result dict."""
        detected: List[str] = []
        details: Dict[str, Dict] = {}
        for name, method, category in self.REGISTRY:
            try:
                fired, conf, d = getattr(self, method)()
            except Exception:
                fired, conf, d = False, 0.0, {}
            if fired:
                detected.append(name)
                details[name] = {"confidence": round(conf, 3),
                                 "category": category, **d}
        avg_conf = float(np.mean([v["confidence"] for v in details.values()])) if details else 0.0
        return {
            "patterns": detected,
            "pattern_count": len(detected),
            "pattern_confidence_avg": round(avg_conf, 3),
            "pattern_details": details,
        }


# ─────────────────────── universe-level helper ──────────────────────────────

def scan_universe(stocks: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Scan all tickers. Returns DataFrame sorted by pattern count + avg confidence."""
    rows = []
    for t, df in stocks.items():
        if df is None or len(df) < 30:
            continue
        try:
            res = PatternScanner(df).scan()
        except Exception:
            continue
        rows.append({
            "ticker": t,
            "pattern_count": res["pattern_count"],
            "avg_confidence": res["pattern_confidence_avg"],
            "patterns": ", ".join(res["patterns"]) if res["patterns"] else "—",
            "details": res["pattern_details"],
            "last_bar": df.index[-1].strftime("%Y-%m-%d"),
            "close": float(df["Close"].iloc[-1]),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(
        ["pattern_count", "avg_confidence"], ascending=[False, False]
    ).reset_index(drop=True)
    return out


def scan_ticker_list(
    pattern_name: str,
    stocks: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return all tickers where a specific pattern is firing, sorted by confidence."""
    rows = []
    for t, df in stocks.items():
        if df is None or len(df) < 30:
            continue
        try:
            res = PatternScanner(df).scan()
        except Exception:
            continue
        if pattern_name in res["patterns"]:
            d = res["pattern_details"][pattern_name]
            rows.append({
                "ticker": t,
                "confidence": d["confidence"],
                "category": d.get("category", "—"),
                "close": float(df["Close"].iloc[-1]),
                "details": {k: v for k, v in d.items()
                            if k not in ("confidence", "category")},
                "last_bar": df.index[-1].strftime("%Y-%m-%d"),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("confidence", ascending=False).reset_index(drop=True)


# Exported pattern name list (for UI dropdowns)
PATTERN_NAMES: List[str] = [name for name, _, _ in PatternScanner.REGISTRY]
