# Validation Report

Generated: 2026-04-23T19:33:07 UTC

## Honesty
Purged CV results are the honest numbers. Naive TimeSeriesSplit allows label-end-time bleed across the split boundary and is reported here only to quantify that leakage, not to advertise.

## Cross-Validation: Naive vs Purged (XGB base)
| metric | naive | purged | gap (naive - purged) |
| --- | --- | --- | --- |
| AUC mean | 0.6503 | 0.6487 | 0.0016 |

## Base Learner Performance (held-out tail)
| learner | AUC | Brier | LogLoss |
| --- | --- | --- | --- |
| xgb | 0.6837 | 0.1620 | 0.4978 |
| lgbm | 0.6859 | 0.1743 | 0.7267 |
| cat | 0.6884 | 0.1644 | 0.5046 |

## Ensemble Strategy
| strategy | AUC | Brier | LogLoss |
| --- | --- | --- | --- |
| blend (isotonic-mean) | 0.6873 | 0.1658 | 0.5093 |
| stack (LogReg on OOF) | 0.6845 | 0.1707 | 0.5268 |

**Selected strategy:** `blend`

## Held-out Tail Metrics (meta model, selected strategy)
| AUC | Brier | LogLoss | Base rate |
| --- | --- | --- | --- |
| 0.6873 | 0.1658 | 0.5093 | 0.2138 |

### Precision / coverage at probability thresholds

| threshold | precision | coverage | n_signals |
| --- | --- | --- | --- |
| 0.5 | 0.4555 | 0.1181 | 2751 |
| 0.6 | 0.4732 | 0.0961 | 2238 |
| 0.65 | 0.4775 | 0.0838 | 1952 |
| 0.7 | 0.4841 | 0.0758 | 1766 |
| 0.75 | 0.4861 | 0.0697 | 1623 |

## Pattern Detectors — Historical Firings
| pattern | firings | experimental (<100 firings) |
| --- | --- | --- |
| Bullish Engulfing | 4829 | no |
| Hammer at Support | 321 | no |
| Morning Star | 2419 | no |
| Piercing Line | 1716 | no |
| Three White Soldiers | 1268 | no |
| Bullish Harami | 5881 | no |
| Cup and Handle | 1842 | no |
| Bull Flag | 10880 | no |
| Ascending Triangle | 12843 | no |
| Inverse Head and Shoulders | 11303 | no |
| Double Bottom | 18398 | no |
| VCP (Minervini) | 94 | YES |
| Inside Bar Breakout | 9743 | no |
| 20-day High Vol Breakout | 5618 | no |
| 52-week High Proximity | 3951 | no |
| Rounding Bottom | 26815 | no |
| Pullback to 50-EMA | 8714 | no |
| Breakaway Gap | 653 | no |
| Pocket Pivot | 10945 | no |
| Golden Cross (short-term) | 4934 | no |


> ⚠️ **1 pattern detector(s)** fired < 100 times across training and are shown only as UI hints, not fed into the meta-model.

## Top 20 Features by |SHAP|
| feature | |SHAP| |
| --- | --- |
| atr_pct | 0.2228 |
| regime_p_bull | 0.1836 |
| regime_p_riskoff | 0.0171 |
| stoch_k_minus_d | 0.0159 |
| stoch_d | 0.0110 |
| rs_20d_vs_bench | 0.0098 |
| rs_60d_vs_bench | 0.0084 |
| ret_5d | 0.0072 |
| dist_ema50_in_atr | 0.0068 |
| bb_width | 0.0068 |
| gap_pct | 0.0062 |
| rsi_14_slope_5 | 0.0060 |
| rsi_14 | 0.0042 |
| wk_rsi_14 | 0.0041 |
| macd_hist_slope_3 | 0.0037 |
| regime_state | 0.0037 |
| ema50_slope_20 | 0.0034 |
| ret_1d | 0.0027 |
| close_loc_in_range | 0.0027 |
| monthly_ret | 0.0021 |

## Model Bundle Metadata
- trained_at: `2026-04-23T19:32:59Z`
- n_train: 116459
- n_tickers: 424
- # features: 88
- label_params: {'upper_pct': 0.05, 'lower_pct': -0.03, 'horizon': 5}
- primary_cfg: {'rsi_lo': 45.0, 'rsi_hi': 70.0, 'vol_ratio_min': 1.3, 'dist_20d_high_max_pct': -0.05, 'adx_min': 20.0, 'required_true': 2}
