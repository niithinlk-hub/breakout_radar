"""ML pipeline for NSE swing-trade screening.

Phase 1 baseline: XGBoost + isotonic calibration + TimeSeriesSplit.
Triple-barrier labels (+5% / -3% / 5-day horizon).

Future phases add PurgedKFold, Optuna, meta-labeling, HMM regime,
pattern features, ensemble, SHAP.

Training CLI:
    python -m breakout_radar.ml.train

Offline-only: Streamlit Cloud filesystem is ephemeral; ship
trained model as committed joblib artifact.
"""

__all__ = ["data_loader", "features", "labeler", "model",
           "backtest", "screener", "risk"]
