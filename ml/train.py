"""CLI: train ML bundle offline.

Run from repo root:
    python -m breakout_radar.ml.train
        [--period 3y] [--tickers all|NIFTY50|PATH]
        [--upper 0.05] [--lower -0.03] [--horizon 5]
        [--fresh] [--out models/ml_bundle.joblib]

Writes `models/ml_bundle.joblib` which the Streamlit app loads.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Allow running as `python breakout_radar/ml/train.py` or
# `python -m breakout_radar.ml.train` from repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml import data_loader, features, labeler, model  # noqa: E402


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


def build_training_matrix(
    stocks: Dict[str, pd.DataFrame],
    bench_df: pd.DataFrame,
    upper: float,
    lower: float,
    horizon: int,
):
    X_parts, y_parts, t1_parts, tickers = [], [], [], []
    for t, df in stocks.items():
        feats = features.build_features(df, bench_df)
        labels, t1 = labeler.triple_barrier_labels(
            df, upper_pct=upper, lower_pct=lower, horizon=horizon
        )
        combined = feats.copy()
        combined["label"] = labels
        combined["t1"] = t1
        combined["ticker"] = t
        combined = combined.dropna(subset=["label"])
        combined = combined.dropna(subset=feats.columns.tolist())
        if combined.empty:
            continue
        X_parts.append(combined[feats.columns.tolist()])
        y_parts.append(combined["label"].astype(int))
        t1_parts.append(combined["t1"])
        tickers.append(t)
    if not X_parts:
        raise SystemExit("No training rows — check data freshness.")
    X = pd.concat(X_parts).sort_index()
    y = pd.concat(y_parts).reindex(X.index)
    t1 = pd.concat(t1_parts).reindex(X.index)
    return X, y, t1, tickers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="3y")
    ap.add_argument("--tickers", default="all",
                    help="'all', 'NIFTY50', or a newline-delimited file path")
    ap.add_argument("--upper", type=float, default=0.05)
    ap.add_argument("--lower", type=float, default=-0.03)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--fresh", action="store_true",
                    help="Bypass cache, re-download")
    ap.add_argument("--out", default=str(model.BUNDLE_PATH))
    args = ap.parse_args()

    t0 = time.time()
    tickers = _resolve_tickers(args.tickers)
    print(f"[1/5] Resolving tickers → {len(tickers)} symbols")

    print(f"[2/5] Bulk-downloading OHLCV (period={args.period}) …")
    stocks = data_loader.fetch_universe(
        tickers, period=args.period, use_cache=not args.fresh
    )
    print(f"      downloaded {len(stocks)} tickers")

    stocks = data_loader.apply_liquidity_filter(stocks)
    print(f"      after liquidity filter: {len(stocks)} tickers")

    print("[3/5] Fetching benchmark (^NSEI)")
    bench = data_loader.get_benchmark(period=args.period, use_cache=not args.fresh)

    print("[4/5] Building features + triple-barrier labels")
    X, y, t1, kept = build_training_matrix(
        stocks, bench, upper=args.upper, lower=args.lower, horizon=args.horizon
    )
    pos_rate = float(y.mean())
    print(f"      X={X.shape}  y_pos_rate={pos_rate:.3f}  tickers_used={len(kept)}")

    print("[5/5] Training XGB + isotonic calibration (TSS CV)")
    bundle = model.train_bundle(
        X=X, y=y,
        label_params={"upper_pct": args.upper,
                      "lower_pct": args.lower,
                      "horizon": args.horizon},
        n_tickers=len(kept),
    )
    out_path = Path(args.out)
    bundle.save(out_path)

    elapsed = time.time() - t0
    print("-" * 60)
    print(f"Saved bundle → {out_path}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Metrics (held-out tail): {bundle.metrics}")


if __name__ == "__main__":
    main()
