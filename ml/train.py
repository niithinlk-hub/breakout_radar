"""CLI: train full meta-labeling bundle offline (Phase 3+4+5).

Pipeline:
  1. Resolve tickers → bulk OHLCV + benchmark (+VIX).
  2. Fit 3-state HMM on benchmark (regime bundle).
  3. Build primary-filtered panel with regime + pattern features.
  4. Gate experimental patterns (<100 firings).
  5. Optuna-tune XGB / LGBM / CatBoost on Brier under PurgedKFold.
  6. Train calibrated ensemble (blend vs stack — pick lower Brier).
  7. Evaluate held-out tail; naive-vs-purged CV comparison.
  8. Persist: MetaBundle, RegimeBundle, PrimaryConfig, best_params.json.
  9. Generate validation_report.md.

Run:
    python -m breakout_radar.ml.train
        [--period 3y] [--tickers all|NIFTY50|PATH]
        [--upper 0.05] [--lower -0.03] [--horizon 5]
        [--trials 100] [--cv-splits 4] [--tail-frac 0.2]
        [--fresh] [--out models/meta.pkl]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml import data_loader                                         # noqa: E402
from ml.ensemble import train_ensemble                             # noqa: E402
from ml.meta import (META_PATH_DEFAULT, MetaBundle,                # noqa: E402
                     build_primary_filtered_panel,
                     gate_experimental_patterns)
from ml.optuna_tune import save_best_params, tune_learner          # noqa: E402
from ml.primary import PrimaryConfig                               # noqa: E402
from ml.regime import fit_regime, regime_series                    # noqa: E402
from ml.report import generate_report                              # noqa: E402


def _resolve_tickers(arg: str) -> List[str]:
    if arg == "NIFTY50":
        try:
            from tickers import NIFTY_50_TICKERS_NS
            return list(NIFTY_50_TICKERS_NS)
        except Exception:
            pass
    if arg == "all":
        try:
            from tickers import get_tickers_ns
            return list(get_tickers_ns())
        except Exception:
            pass
    p = Path(arg)
    if p.exists():
        return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    raise SystemExit(f"Unknown --tickers source: {arg}")


def _feature_importance(ensemble, feature_names: List[str]) -> pd.DataFrame:
    """Best-effort gain importance from first tree base (fallback for SHAP)."""
    try:
        for cal in getattr(ensemble, "calibrated_bases_", []):
            for ccb in getattr(cal, "calibrated_classifiers_", []):
                est = getattr(ccb, "estimator", None) or getattr(ccb, "base_estimator", None)
                if est is None:
                    continue
                if hasattr(est, "feature_importances_"):
                    imp = np.asarray(est.feature_importances_, dtype=float)
                    if len(imp) == len(feature_names):
                        return (pd.DataFrame({"feature": feature_names, "importance": imp})
                                .sort_values("importance", ascending=False)
                                .reset_index(drop=True))
        return pd.DataFrame(columns=["feature", "importance"])
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])


def _global_shap_importance(ensemble, X_sample: pd.DataFrame,
                            feature_names: List[str]) -> pd.DataFrame:
    try:
        from ml.shap_explain import SHAP_AVAILABLE
        if not SHAP_AVAILABLE:
            return pd.DataFrame(columns=["feature", "importance"])
        import shap                                                 # type: ignore
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])

    tree_model = None
    for cal in getattr(ensemble, "calibrated_bases_", []):
        for ccb in getattr(cal, "calibrated_classifiers_", []):
            est = getattr(ccb, "estimator", None) or getattr(ccb, "base_estimator", None)
            if est is None:
                continue
            if type(est).__name__ in ("XGBClassifier", "LGBMClassifier", "CatBoostClassifier"):
                tree_model = est
                break
        if tree_model is not None:
            break
    if tree_model is None:
        return pd.DataFrame(columns=["feature", "importance"])

    try:
        X_use = X_sample[feature_names].tail(min(500, len(X_sample)))
        explainer = shap.TreeExplainer(tree_model)
        sv = explainer.shap_values(X_use)
        if isinstance(sv, list) and len(sv) == 2:
            vals = np.abs(sv[1]).mean(axis=0)
        else:
            vals = np.abs(np.asarray(sv)).mean(axis=0)
        return (pd.DataFrame({"feature": feature_names, "importance": vals})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True))
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="3y")
    ap.add_argument("--tickers", default="all",
                    help="'all', 'NIFTY50', or a newline-delimited file path")
    ap.add_argument("--upper", type=float, default=0.05)
    ap.add_argument("--lower", type=float, default=-0.03)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--trials", type=int, default=100,
                    help="Optuna trials per learner (0 disables tuning)")
    ap.add_argument("--cv-splits", type=int, default=4)
    ap.add_argument("--tail-frac", type=float, default=0.2)
    ap.add_argument("--embargo-pct", type=float, default=0.01)
    ap.add_argument("--max-rows-per-ticker", type=int, default=None)
    ap.add_argument("--no-patterns", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--out", default=str(META_PATH_DEFAULT))
    args = ap.parse_args()

    t0 = time.time()
    tickers = _resolve_tickers(args.tickers)
    print(f"[1/9] Resolved {len(tickers)} tickers")

    print(f"[2/9] Bulk-downloading OHLCV (period={args.period}) …")
    stocks = data_loader.fetch_universe(
        tickers, period=args.period, use_cache=not args.fresh
    )
    stocks = data_loader.apply_liquidity_filter(stocks)
    print(f"      usable tickers after liquidity filter: {len(stocks)}")

    print("[3/9] Fetching benchmark (^NSEI)")
    bench = data_loader.get_benchmark(period=args.period, use_cache=not args.fresh)

    print("[4/9] Fitting 3-state Gaussian HMM regime detector")
    regime_bundle = fit_regime(bench)
    if regime_bundle is None:
        print("      HMM unavailable — regime features will be zeros")
        regime_df = pd.DataFrame()
    else:
        regime_df = regime_series(regime_bundle, bench)
        print(f"      HMM fit on {regime_bundle.n_obs} obs "
              f"({regime_bundle.last_fit_range[0]} → {regime_bundle.last_fit_range[1]})")
        try:
            regime_bundle.save()
        except Exception as e:
            print(f"      (regime save failed: {e})")

    print("[5/9] Building primary-filtered panel "
          f"(triple-barrier upper={args.upper:+.2%} lower={args.lower:+.2%} "
          f"horizon={args.horizon}d)")
    primary_cfg = PrimaryConfig()
    X, y, t1, tkr, firing_counts = build_primary_filtered_panel(
        stocks, bench,
        upper=args.upper, lower=args.lower, horizon=args.horizon,
        regime_df=regime_df,
        primary_cfg=primary_cfg,
        include_patterns=not args.no_patterns,
        max_rows_per_ticker=args.max_rows_per_ticker,
    )
    pos_rate = float(y.mean())
    print(f"      panel X={X.shape}  y_pos_rate={pos_rate:.3f}  "
          f"unique_tickers={tkr.nunique()}")

    # Drop pattern columns whose detector fired <100 times (experimental)
    X, experimental = gate_experimental_patterns(X, firing_counts)
    if experimental:
        print(f"      experimental patterns excluded from meta features "
              f"({len(experimental)}): {', '.join(experimental[:6])}"
              f"{' …' if len(experimental) > 6 else ''}")

    feature_names = list(X.columns)
    print(f"      final feature count: {len(feature_names)}")

    # Persist primary rule config (deterministic, but saved for app parity)
    try:
        primary_cfg.save()
    except Exception as e:
        print(f"      (primary save failed: {e})")

    print(f"[6/9] Optuna tuning (n_trials={args.trials} per learner, "
          f"objective=Brier under PurgedKFold)")
    best_params: Dict[str, Dict] = {}
    if args.trials > 0:
        for name in ("xgb", "lgbm", "cat"):
            try:
                p = tune_learner(name, X, y, t1,
                                 n_trials=args.trials,
                                 n_splits=args.cv_splits,
                                 embargo_pct=args.embargo_pct)
                if p is not None:
                    best_params[name] = p
            except Exception as e:
                print(f"      [{name}] tuning failed: {e}")
    else:
        print("      tuning skipped — using library defaults")
    try:
        save_best_params(best_params)
    except Exception as e:
        print(f"      (best_params save failed: {e})")

    print("[7/9] Training calibrated ensemble (XGB + LGBM + CatBoost)")
    ens = train_ensemble(
        X=X, y=y, t1=t1,
        best_params=best_params,
        tail_frac=args.tail_frac,
        cv_splits=args.cv_splits,
        embargo_pct=args.embargo_pct,
        naive_compare=True,
    )
    print(f"      base learners: {ens.base_names}")
    for name, m in ens.per_base_metrics.items():
        print(f"      {name:>5s}: AUC={m.get('auc', float('nan')):.4f}  "
              f"Brier={m.get('brier', float('nan')):.4f}  "
              f"LogLoss={m.get('logloss', float('nan')):.4f}")
    print(f"      blend: AUC={ens.blend_metrics.get('auc', float('nan')):.4f}  "
          f"Brier={ens.blend_metrics.get('brier', float('nan')):.4f}")
    print(f"      stack: AUC={ens.stack_metrics.get('auc', float('nan')):.4f}  "
          f"Brier={ens.stack_metrics.get('brier', float('nan')):.4f}")
    print(f"      CHOSEN strategy: {ens.chosen}")

    cv = ens.cv_compare or {}
    if cv:
        gap = cv.get("gap", float("nan"))
        leak = cv.get("leakage_warning")
        print(f"      CV honesty — naive AUC={cv.get('naive_auc_mean')}  "
              f"purged AUC={cv.get('purged_auc_mean')}  gap={gap}  "
              f"{'⚠ LEAKAGE' if leak else 'ok'}")

    # Held-out tail metrics for selected strategy (for meta bundle report)
    n = len(X)
    k = max(1, int(n * (1.0 - args.tail_frac)))
    X_te = X.iloc[k:]
    y_te = y.iloc[k:]
    proba_te = ens.ensemble.predict_proba(X_te)[:, 1] if ens.ensemble is not None \
        else np.full(len(X_te), 0.5)
    metrics: Dict[str, float] = {}
    try:
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        metrics["auc"] = float(roc_auc_score(y_te, proba_te)) if y_te.nunique() > 1 else float("nan")
        metrics["brier"] = float(brier_score_loss(y_te, proba_te))
        metrics["logloss"] = float(log_loss(y_te, np.clip(proba_te, 1e-6, 1 - 1e-6)))
    except Exception:
        metrics["auc"] = float("nan"); metrics["brier"] = 1.0; metrics["logloss"] = float("inf")
    metrics["base_rate_tail"] = float(y_te.mean())
    for thr in (0.5, 0.6, 0.65, 0.7, 0.75):
        sel = proba_te >= thr
        nsig = int(sel.sum())
        metrics[f"n@{thr}"] = nsig
        metrics[f"coverage@{thr}"] = float(nsig / len(y_te)) if len(y_te) else 0.0
        metrics[f"precision@{thr}"] = float(y_te[sel].mean()) if nsig > 0 else float("nan")

    print("[8/9] Persisting MetaBundle")
    bundle = MetaBundle(
        ensemble=ens.ensemble,
        feature_names=feature_names,
        label_params={"upper_pct": args.upper,
                      "lower_pct": args.lower,
                      "horizon": args.horizon},
        primary_cfg=primary_cfg.__dict__.copy(),
        pattern_firing_counts=firing_counts,
        experimental_patterns=experimental,
        metrics=metrics,
        cv_compare=cv,
        trained_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        n_train=int(n),
        n_tickers=int(tkr.nunique()),
    )
    out_path = Path(args.out)
    bundle.save(out_path)
    print(f"      saved → {out_path}")

    print("[9/9] Generating validation_report.md")
    shap_imp = _global_shap_importance(ens.ensemble, X.tail(500), feature_names)
    feat_imp = _feature_importance(ens.ensemble, feature_names) if shap_imp.empty else None
    try:
        report_path = generate_report(
            meta_bundle=bundle,
            ensemble_result=ens,
            backtest_result=None,
            shap_importance=shap_imp if not shap_imp.empty else None,
            feature_importance=feat_imp,
        )
        print(f"      report → {report_path}")
    except Exception as e:
        print(f"      (report generation failed: {e})")

    elapsed = time.time() - t0
    print("-" * 60)
    print(f"DONE in {elapsed:.1f}s")
    print(f"tail metrics: AUC={metrics.get('auc', float('nan')):.4f}  "
          f"Brier={metrics.get('brier', float('nan')):.4f}  "
          f"base_rate={metrics.get('base_rate_tail', float('nan')):.3f}")


if __name__ == "__main__":
    main()
