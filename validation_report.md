# Validation Report

Generated: 2026-04-23T17:50:52 UTC

## Honesty
Purged CV results are the honest numbers. Naive TimeSeriesSplit allows label-end-time bleed across the split boundary and is reported here only to quantify that leakage, not to advertise.

## Cross-Validation: Naive vs Purged (XGB base)
| metric | naive | purged | gap (naive - purged) |
| --- | --- | --- | --- |
| AUC mean | 0.5786 | 0.5837 | -0.0051 |

## Base Learner Performance (held-out tail)
| learner | AUC | Brier | LogLoss |
| --- | --- | --- | --- |
| xgb | 0.6355 | 0.1066 | 0.3640 |
| lgbm | 0.6312 | 0.1029 | 0.3566 |
| cat | 0.6318 | 0.1042 | 0.3594 |

## Ensemble Strategy
| strategy | AUC | Brier | LogLoss |
| --- | --- | --- | --- |
| blend (isotonic-mean) | 0.6366 | 0.1031 | 0.3567 |
| stack (LogReg on OOF) | 0.6192 | 0.1041 | 0.3611 |

**Selected strategy:** `blend`

## Held-out Tail Metrics (meta model, selected strategy)
| AUC | Brier | LogLoss | Base rate |
| --- | --- | --- | --- |
| 0.6366 | 0.1031 | 0.3567 | 0.1202 |

### Precision / coverage at probability thresholds

| threshold | precision | coverage | n_signals |
| --- | --- | --- | --- |
| 0.5 | 0.4118 | 0.0052 | 17 |
| 0.6 | nan | 0.0000 | 0 |
| 0.65 | nan | 0.0000 | 0 |
| 0.7 | nan | 0.0000 | 0 |
| 0.75 | nan | 0.0000 | 0 |

## Pattern Detectors — Historical Firings
| pattern | firings | experimental (<100 firings) |
| --- | --- | --- |
| Bullish Engulfing | 695 | no |
| Hammer at Support | 94 | YES |
| Morning Star | 288 | no |
| Piercing Line | 260 | no |
| Three White Soldiers | 182 | no |
| Bullish Harami | 812 | no |
| Cup and Handle | 125 | no |
| Bull Flag | 832 | no |
| Ascending Triangle | 3446 | no |
| Inverse Head and Shoulders | 2254 | no |
| Double Bottom | 3481 | no |
| VCP (Minervini) | 9 | YES |
| Inside Bar Breakout | 1269 | no |
| 20-day High Vol Breakout | 627 | no |
| 52-week High Proximity | 848 | no |
| Rounding Bottom | 3367 | no |
| Pullback to 50-EMA | 1587 | no |
| Breakaway Gap | 56 | YES |
| Pocket Pivot | 1915 | no |
| Golden Cross (short-term) | 638 | no |


> ⚠️ **3 pattern detector(s)** fired < 100 times across training and are shown only as UI hints, not fed into the meta-model.

## Top 20 Features by |SHAP|
| feature | |SHAP| |
| --- | --- |
| atr_pct | 0.3313 |
| regime_p_bull | 0.2178 |
| ret_10d | 0.1188 |
| rs_20d_vs_bench | 0.1183 |
| ret_60d | 0.0828 |
| intraday_range_over_atr | 0.0802 |
| regime_p_riskoff | 0.0602 |
| bb_width | 0.0509 |
| bb_pct_b | 0.0496 |
| ema8_over_ema21 | 0.0492 |
| px_over_ema21 | 0.0471 |
| ret_20d | 0.0451 |
| close_loc_in_range | 0.0359 |
| px_over_ema50 | 0.0356 |
| dist_ema50_in_atr | 0.0354 |
| rsi_14_slope_5 | 0.0351 |
| vwma_20_dist_pct | 0.0329 |
| pat_inverse_head_and_shoulders_conf | 0.0308 |
| stoch_d | 0.0268 |
| ema50_over_ema200 | 0.0208 |

## Model Bundle Metadata
- trained_at: `2026-04-23T17:50:45Z`
- n_train: 16386
- n_tickers: 49
- # features: 84
- label_params: {'upper_pct': 0.05, 'lower_pct': -0.03, 'horizon': 5}
- primary_cfg: {'rsi_lo': 45.0, 'rsi_hi': 70.0, 'vol_ratio_min': 1.3, 'dist_20d_high_max_pct': -0.05, 'adx_min': 20.0, 'required_true': 2}
