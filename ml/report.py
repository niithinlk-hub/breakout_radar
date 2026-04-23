"""Auto-generate `validation_report.md` after training.

Contents:
  - Naive vs purged CV scores (leakage quantification)
  - Per-base model performance (XGB/LGBM/CatBoost)
  - Blend vs stack comparison
  - Top 20 features by SHAP (or gain importance fallback)
  - Per-regime / per-pattern-count hit rate tables (if backtest run)
  - Honesty warnings
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPORT_PATH = Path(__file__).resolve().parent.parent / "validation_report.md"


def _fmt(x, digits: int = 4) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _table(rows: List[Dict], columns: Optional[List[str]] = None) -> str:
    if not rows:
        return "_no data_\n"
    if columns is None:
        columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |\n"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |\n"
    body = ""
    for r in rows:
        body += "| " + " | ".join(str(r.get(c, "")) for c in columns) + " |\n"
    return header + sep + body


def generate_report(
    meta_bundle,
    ensemble_result,
    backtest_result=None,
    shap_importance: Optional[pd.DataFrame] = None,
    feature_importance: Optional[pd.DataFrame] = None,
    path: Path = REPORT_PATH,
) -> Path:
    lines: List[str] = []
    lines.append(f"# Validation Report\n")
    lines.append(f"Generated: {datetime.utcnow().isoformat(timespec='seconds')} UTC\n")

    # ── Honesty banner ──
    cv = meta_bundle.cv_compare or {}
    gap = cv.get("gap", float("nan"))
    lines.append("## Honesty")
    lines.append(
        "Purged CV results are the honest numbers. Naive TimeSeriesSplit "
        "allows label-end-time bleed across the split boundary and is "
        "reported here only to quantify that leakage, not to advertise.\n"
    )
    if cv.get("leakage_warning"):
        lines.append(
            f"> ⚠️ **Leakage warning:** naive AUC exceeded purged AUC by "
            f"{_fmt(gap, 3)} (> 0.05). Older pipelines that reported only "
            "TimeSeriesSplit scores materially overstated performance.\n"
        )

    # ── CV comparison ──
    lines.append("## Cross-Validation: Naive vs Purged (XGB base)")
    lines.append(_table([{
        "metric": "AUC mean",
        "naive": _fmt(cv.get("naive_auc_mean")),
        "purged": _fmt(cv.get("purged_auc_mean")),
        "gap (naive - purged)": _fmt(cv.get("gap")),
    }]))

    # ── Per-base metrics ──
    lines.append("## Base Learner Performance (held-out tail)")
    rows = []
    for name, m in (ensemble_result.per_base_metrics or {}).items():
        rows.append({
            "learner": name,
            "AUC": _fmt(m.get("auc")),
            "Brier": _fmt(m.get("brier")),
            "LogLoss": _fmt(m.get("logloss")),
        })
    lines.append(_table(rows))

    # ── Blend vs stack ──
    lines.append("## Ensemble Strategy")
    lines.append(_table([
        {"strategy": "blend (isotonic-mean)",
         "AUC": _fmt(ensemble_result.blend_metrics.get("auc")),
         "Brier": _fmt(ensemble_result.blend_metrics.get("brier")),
         "LogLoss": _fmt(ensemble_result.blend_metrics.get("logloss"))},
        {"strategy": "stack (LogReg on OOF)",
         "AUC": _fmt(ensemble_result.stack_metrics.get("auc")),
         "Brier": _fmt(ensemble_result.stack_metrics.get("brier")),
         "LogLoss": _fmt(ensemble_result.stack_metrics.get("logloss"))},
    ]))
    lines.append(f"**Selected strategy:** `{ensemble_result.chosen}`\n")

    # ── Meta bundle summary ──
    m = meta_bundle.metrics or {}
    lines.append("## Held-out Tail Metrics (meta model, selected strategy)")
    lines.append(_table([{
        "AUC": _fmt(m.get("auc")),
        "Brier": _fmt(m.get("brier")),
        "LogLoss": _fmt(m.get("logloss")),
        "Base rate": _fmt(m.get("base_rate_tail")),
    }]))
    thr_rows = []
    for thr in (0.5, 0.6, 0.65, 0.7, 0.75):
        thr_rows.append({
            "threshold": thr,
            "precision": _fmt(m.get(f"precision@{thr}")),
            "coverage": _fmt(m.get(f"coverage@{thr}")),
            "n_signals": m.get(f"n@{thr}"),
        })
    lines.append("### Precision / coverage at probability thresholds\n")
    lines.append(_table(thr_rows))

    # ── Patterns: firing counts + experimental list ──
    lines.append("## Pattern Detectors — Historical Firings")
    exp = set(meta_bundle.experimental_patterns or [])
    p_rows = []
    for name, n in (meta_bundle.pattern_firing_counts or {}).items():
        p_rows.append({
            "pattern": name,
            "firings": n,
            "experimental (<100 firings)": "YES" if name in exp else "no",
        })
    lines.append(_table(p_rows))

    if exp:
        lines.append(
            f"\n> ⚠️ **{len(exp)} pattern detector(s)** fired < 100 times across "
            "training and are shown only as UI hints, not fed into the meta-model.\n"
        )

    # ── Feature importance / SHAP ──
    if shap_importance is not None and not shap_importance.empty:
        lines.append("## Top 20 Features by |SHAP|")
        rows = [{"feature": r["feature"], "|SHAP|": _fmt(r["importance"])}
                for _, r in shap_importance.head(20).iterrows()]
        lines.append(_table(rows))
    elif feature_importance is not None and not feature_importance.empty:
        lines.append("## Top 20 Features by Gain Importance (SHAP unavailable)")
        rows = [{"feature": r["feature"], "gain": _fmt(r["importance"])}
                for _, r in feature_importance.head(20).iterrows()]
        lines.append(_table(rows))

    # ── Backtest tables ──
    if backtest_result is not None:
        summ = backtest_result.summary or {}
        lines.append("## Walk-Forward Backtest Summary")
        lines.append(_table([{
            "folds": summ.get("folds"),
            "OOF AUC": _fmt(summ.get("auc_oof")),
            "OOF Brier": _fmt(summ.get("brier_oof")),
            "base rate": _fmt(summ.get("base_rate")),
            "n preds": summ.get("n_preds"),
        }]))
        if getattr(backtest_result, "decile_table", None) is not None and not backtest_result.decile_table.empty:
            lines.append("### Hit Rate by Probability Decile")
            rows = [{"decile": int(r["decile"]) + 1,
                     "prob_mean": _fmt(r["prob_mean"]),
                     "hit_rate": _fmt(r["hit_rate"]),
                     "n": int(r["n"])}
                    for _, r in backtest_result.decile_table.iterrows()]
            lines.append(_table(rows))
        if getattr(backtest_result, "regime_table", None) is not None and not backtest_result.regime_table.empty:
            lines.append("### Hit Rate by Regime (prob >= 0.60)")
            rows = [{"regime": r["regime"],
                     "n": int(r["n"]), "hit_rate": _fmt(r["hit_rate"])}
                    for _, r in backtest_result.regime_table.iterrows()]
            lines.append(_table(rows))
        if getattr(backtest_result, "pattern_count_table", None) is not None and not backtest_result.pattern_count_table.empty:
            lines.append("### Hit Rate by # Patterns Firing")
            rows = [{"n_patterns": int(r["n_patterns"]),
                     "n": int(r["n"]), "hit_rate": _fmt(r["hit_rate"])}
                    for _, r in backtest_result.pattern_count_table.iterrows()]
            lines.append(_table(rows))

    # ── Bundle metadata ──
    lines.append("## Model Bundle Metadata")
    lines.append(f"- trained_at: `{meta_bundle.trained_at}`")
    lines.append(f"- n_train: {meta_bundle.n_train}")
    lines.append(f"- n_tickers: {meta_bundle.n_tickers}")
    lines.append(f"- # features: {len(meta_bundle.feature_names)}")
    lines.append(f"- label_params: {meta_bundle.label_params}")
    lines.append(f"- primary_cfg: {meta_bundle.primary_cfg}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
