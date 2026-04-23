"""Live screener — score latest bar per ticker using the meta bundle.

Flow per ticker:
  1. build_features → latest row feature vector
  2. primary_signal → gate (skip if primary doesn't fire)
  3. Attach primary_confidence + regime probs + pattern flags
  4. meta.predict_proba → calibrated probability
  5. compute_levels + Kelly sizing
  6. composite score = 0.60·prob + 0.20·pattern + 0.20·regime
  7. SHAP drivers per pick (top 10)

Global risk-off gate: if regime.p_riskoff > 0.6, block all new picks.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import features as feat_mod
from . import risk
from .composite import compute_composite
from .meta import MetaBundle
from .pattern_features import (CONF_COLS, FIRES_COLS, PATTERN_FEATURE_COLS,
                               PATTERN_SLUGS)
from .patterns import PATTERN_NAMES, PatternScanner
from .primary import PrimaryConfig, primary_confidence, primary_signal
from .regime import current_regime
from .shap_explain import explain_pick


def _latest_pattern_features(df: pd.DataFrame) -> Dict[str, float]:
    """Run PatternScanner on the full history — take the scan's latest-bar flags."""
    row = {c: 0.0 for c in PATTERN_FEATURE_COLS}
    names: List[str] = []
    confs: List[float] = []
    if df is None or len(df) < 60:
        return {"row": row, "names": names, "avg_conf": 0.0}
    try:
        res = PatternScanner(df).scan()
    except Exception:
        return {"row": row, "names": names, "avg_conf": 0.0}
    for name, slug in zip(PATTERN_NAMES, PATTERN_SLUGS):
        if name in res["patterns"]:
            row[f"pat_{slug}_fires"] = 1.0
            conf = float(res["pattern_details"][name]["confidence"])
            row[f"pat_{slug}_conf"] = conf
            names.append(name)
            confs.append(conf)
    avg = float(np.mean(confs)) if confs else 0.0
    return {"row": row, "names": names, "avg_conf": avg}


def score_universe(
    meta_bundle: MetaBundle,
    stocks: Dict[str, pd.DataFrame],
    bench_df: Optional[pd.DataFrame] = None,
    regime_df: Optional[pd.DataFrame] = None,
    primary_cfg: Optional[PrimaryConfig] = None,
    include_shap: bool = True,
    shap_top_n: int = 10,
) -> pd.DataFrame:
    """One row per ticker that passes primary. Sorted by composite score desc.

    Columns include: ticker, prob, composite, composite components, pattern_count,
    pattern_list, regime_state, entry/stop/targets, kelly_frac, shap_drivers.

    Risk-off gate: if regime.p_riskoff > 0.6, return an empty frame
    (caller shows a red banner).
    """
    if meta_bundle is None or meta_bundle.ensemble is None or not stocks:
        return pd.DataFrame()

    # Global regime gate (latest day)
    reg_latest = current_regime(regime_df if regime_df is not None else pd.DataFrame())
    if reg_latest.get("block_new_picks"):
        return pd.DataFrame()

    upper = meta_bundle.label_params.get("upper_pct", 0.05)
    lower = meta_bundle.label_params.get("lower_pct", -0.03)
    if primary_cfg is None:
        primary_cfg = PrimaryConfig(**(meta_bundle.primary_cfg or {})) \
            if meta_bundle.primary_cfg else PrimaryConfig()

    rows: List[Dict] = []
    for t, df in stocks.items():
        if df is None or len(df) < 60:
            continue
        feats = feat_mod.build_features(df, bench_df)
        if feats.empty or feats.iloc[[-1]].isna().any(axis=1).item():
            continue

        # Stage 1: primary gate on the latest bar
        last_feats = feats.iloc[[-1]].copy()
        fires_last = primary_signal(last_feats, cfg=primary_cfg).iloc[-1]
        if not bool(fires_last):
            continue

        # Primary confidence feature
        last_feats["primary_confidence"] = float(
            primary_confidence(last_feats, cfg=primary_cfg).iloc[-1]
        )

        # Regime features (align latest)
        if regime_df is not None and not regime_df.empty:
            reg_row = regime_df.reindex([df.index[-1]], method="ffill")
            r = reg_row.iloc[-1]
            last_feats["regime_state"] = int(r.get("regime_state", 1))
            last_feats["regime_p_bull"] = float(r.get("p_bull", 0.0))
            last_feats["regime_p_choppy"] = float(r.get("p_choppy", 0.0))
            last_feats["regime_p_riskoff"] = float(r.get("p_riskoff", 0.0))
        else:
            last_feats["regime_state"] = 1
            last_feats["regime_p_bull"] = 0.0
            last_feats["regime_p_choppy"] = 0.0
            last_feats["regime_p_riskoff"] = 0.0

        # Pattern features (latest scan)
        pat = _latest_pattern_features(df)
        for c, v in pat["row"].items():
            last_feats[c] = v
        pattern_names = pat["names"]
        pattern_avg_conf = pat["avg_conf"]
        pattern_count = len(pattern_names)

        # Reindex to trained feature set (drops experimental-pattern cols that
        # weren't kept during training)
        X_row = last_feats.reindex(columns=meta_bundle.feature_names).fillna(0.0)

        try:
            prob = float(meta_bundle.predict_proba(X_row)[0])
        except Exception:
            continue

        # Levels + Kelly
        lvls = risk.compute_levels(df, upper_pct=upper, lower_pct=lower)
        if lvls is None:
            continue
        sizing = risk.size_position(capital=1_000_000, prob=prob, levels=lvls,
                                    mode="kelly")

        # Composite score (honest: never inflates beyond true prob)
        regime_state = int(last_feats["regime_state"].iloc[-1])
        comp = compute_composite(
            prob=prob,
            pattern_count=pattern_count,
            pattern_confidence_avg=pattern_avg_conf,
            regime_state=regime_state,
            upper_pct=upper,
            lower_pct=lower,
        )

        # SHAP drivers (top 10) — skipped if lib missing or model non-tree
        drivers_payload = {"available": False, "drivers": []}
        if include_shap:
            try:
                drivers_payload = explain_pick(
                    meta_bundle.ensemble, X_row,
                    meta_bundle.feature_names, top_n=shap_top_n,
                )
            except Exception:
                drivers_payload = {"available": False, "drivers": []}

        rows.append({
            "ticker": t,
            "prob": round(prob, 4),
            "prob_pct": round(prob * 100, 1),
            "composite": comp.total,
            "composite_prob": comp.prob_component,
            "composite_pattern": comp.pattern_component,
            "composite_regime": comp.regime_component,
            "pattern_count": pattern_count,
            "pattern_list": ", ".join(pattern_names) if pattern_names else "",
            "pattern_avg_conf": round(pattern_avg_conf, 3),
            "regime_state": regime_state,
            "regime_p_riskoff": float(last_feats["regime_p_riskoff"].iloc[-1]),
            "primary_confidence": float(last_feats["primary_confidence"].iloc[-1]),
            "entry": lvls.entry, "stop": lvls.stop,
            "target1": lvls.target1, "target2": lvls.target2,
            "risk_pct": lvls.risk_pct, "reward_pct": lvls.reward_pct,
            "rr": lvls.rr, "atr": lvls.atr,
            "kelly_frac": sizing.get("kelly_frac", 0.0),
            "capital_pct": sizing.get("capital_pct", 0.0),
            "last_bar": df.index[-1].strftime("%Y-%m-%d"),
            "shap_available": drivers_payload.get("available", False),
            "shap_drivers": drivers_payload.get("drivers", []),
        })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("composite", ascending=False).reset_index(drop=True)
    return out


def probability_buckets(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty or "prob" not in scored.columns:
        return pd.DataFrame()
    bins = [0.0, 0.3, 0.45, 0.55, 0.65, 0.75, 1.0]
    labels = ["<30", "30-45", "45-55", "55-65", "65-75", ">=75"]
    b = pd.cut(scored["prob"], bins=bins, labels=labels,
               include_lowest=True, right=False)
    return b.value_counts().reindex(labels).fillna(0).astype(int).to_frame("count")
