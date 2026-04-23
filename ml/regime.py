"""3-state Gaussian HMM regime detector on NIFTY 50.

States (ordered by mean return):
    0 — Bull trending   (high mean / low vol)
    1 — Choppy / Range  (near-zero mean / low vol)
    2 — Risk-off        (negative mean / high vol)

Screener rule: if risk-off smoothed probability > 0.6, block ALL new
picks regardless of model probability. UI shows a red banner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

REGIME_LABELS = {0: "Bull trending", 1: "Choppy / Range", 2: "Risk-off"}
HMM_PATH_DEFAULT = Path(__file__).resolve().parent.parent / "models" / "regime_hmm.pkl"

try:
    from hmmlearn.hmm import GaussianHMM           # type: ignore
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False


@dataclass
class RegimeBundle:
    model: object = None
    state_order: Dict[int, int] = field(default_factory=dict)
    # Maps raw hmm state -> canonical label idx (0=Bull,1=Choppy,2=RiskOff)
    last_fit_range: Tuple[str, str] = ("", "")
    n_obs: int = 0

    def save(self, path: Path = HMM_PATH_DEFAULT) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path = HMM_PATH_DEFAULT) -> Optional["RegimeBundle"]:
        if not Path(path).exists():
            return None
        try:
            return joblib.load(path)
        except Exception:
            return None

    def predict_states(self, X: np.ndarray) -> np.ndarray:
        """Return canonical state indices (0/1/2)."""
        raw = self.model.predict(X)
        return np.array([self.state_order[int(s)] for s in raw], dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """n×3 posterior probs (canonical ordering)."""
        raw = self.model.predict_proba(X)
        n, k = raw.shape
        out = np.zeros_like(raw)
        for raw_idx, canon_idx in self.state_order.items():
            out[:, canon_idx] += raw[:, raw_idx]
        return out


def _prep_features(bench_df: pd.DataFrame, vol_window: int = 20) -> Tuple[pd.DataFrame, np.ndarray]:
    """Return (frame, X). Features: daily log-return, realized 20d vol."""
    close = bench_df["Close"]
    logret = np.log(close / close.shift(1))
    vol = logret.rolling(vol_window).std()
    frame = pd.DataFrame({"logret": logret, "vol20": vol}).dropna()
    X = frame.values
    return frame, X


def fit_regime(
    bench_df: pd.DataFrame,
    n_states: int = 3,
    cov_type: str = "full",
    random_state: int = 42,
    n_iter: int = 200,
) -> Optional[RegimeBundle]:
    if not HMM_AVAILABLE:
        return None
    if bench_df is None or len(bench_df) < 300:
        return None

    frame, X = _prep_features(bench_df)
    if len(X) < 200:
        return None

    hmm = GaussianHMM(n_components=n_states, covariance_type=cov_type,
                      n_iter=n_iter, random_state=random_state)
    try:
        hmm.fit(X)
    except Exception:
        return None

    means = hmm.means_[:, 0]  # mean of log-return per state
    vols = hmm.means_[:, 1]   # mean of vol per state

    # Canonical ordering: Bull (highest ret, low vol), Choppy (near-zero ret, low vol),
    # Risk-off (lowest ret, highest vol).
    order_by_ret = np.argsort(-means)      # descending
    canonical = {}
    # Pick risk-off first: highest vol among all. Then bull = highest ret of remaining.
    risk_off = int(np.argmax(vols))
    canonical[risk_off] = 2
    remaining = [i for i in range(n_states) if i != risk_off]
    bull = max(remaining, key=lambda i: means[i])
    canonical[bull] = 0
    for i in remaining:
        if i != bull:
            canonical[i] = 1
    bundle = RegimeBundle(
        model=hmm,
        state_order=canonical,
        last_fit_range=(str(frame.index[0].date()), str(frame.index[-1].date())),
        n_obs=int(len(X)),
    )
    return bundle


def regime_series(
    bundle: RegimeBundle,
    bench_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-day regime state + 3-state probabilities on benchmark."""
    if bundle is None or bench_df is None or len(bench_df) < 30:
        return pd.DataFrame()
    frame, X = _prep_features(bench_df)
    states = bundle.predict_states(X)
    probs = bundle.predict_proba(X)
    out = pd.DataFrame({
        "regime_state": states,
        "p_bull": probs[:, 0],
        "p_choppy": probs[:, 1],
        "p_riskoff": probs[:, 2],
    }, index=frame.index)
    return out


def current_regime(regime_df: pd.DataFrame) -> Dict:
    """Latest regime snapshot for the UI banner."""
    if regime_df is None or regime_df.empty:
        return {"state": -1, "label": "Unknown (HMM unavailable)",
                "p_bull": 0.0, "p_choppy": 0.0, "p_riskoff": 0.0,
                "block_new_picks": False}
    row = regime_df.iloc[-1]
    state = int(row["regime_state"])
    p_ro = float(row["p_riskoff"])
    return {
        "state": state,
        "label": REGIME_LABELS.get(state, "Unknown"),
        "p_bull": float(row["p_bull"]),
        "p_choppy": float(row["p_choppy"]),
        "p_riskoff": p_ro,
        "block_new_picks": p_ro > 0.6,
    }
