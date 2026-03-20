# Training Pipeline

S&P 500 10-day direction prediction (binary: UP / DOWN).

```
Raw OHLCV → Target → Indicators → Extra Market → Stationarity Transform → ML
```

---

## 1. Target

```python
ret = (Close[t+10] - Close[t]) / Close[t]
Target = 1 (UP) if ret > 0.03, 0 (DOWN) if ret < -0.03, NaN (dead zone)
```

Dead-zone rows (|return| ≤ 3%) are dropped before training. This removes ambiguous days and pushes class balance toward ~50/50.

---

## 2. Features (~53 columns)

Target is **separated before** any gap-filling (`ffill`, `interpolate`, `bfill`) and rejoined at the end so dead-zone and prediction-tail NaN values are never corrupted.

### Technical Indicators (computed from OHLCV, High/Low/Open dropped after)

| Feature | Transform |
|---------|-----------|
| EMA 10, EMA 20, SAR | `(value - Close) / Close` |
| MACD, MACD_SIGNAL | `value / Close` |
| ADX, +DMI, -DMI, CCI, RSI, ROC, %K | unchanged (bounded/stationary) |
| BB_pband, BB_width, ATR_norm | unchanged (already normalized) |
| OBV | `diff() / Close` |
| ADOSC | unchanged |
| ret_lag{1,2,3,5,10,20}, gap_ret | unchanged (already returns) |
| vol_roll{5,10,20,60}, intraday_range | unchanged |
| Close, Volume | `pct_change()` |

### Extra Market Data (21 tickers)

- **Commodities:** GC=F, CL=F, HG=F, NG=F
- **Currencies:** EURUSD=X, GBPUSD=X, JPY=X, CNY=X
- **Indices:** ^IXIC, ^RUT, EFA, ^FTSE, ^GDAXI, ^N225, ^HSI, ^BSESN, ^MXX, ^AXJO, ^VIX
- **Rates:** ^TNX, ^IRX → `diff()` (near-zero values make `pct_change()` unstable)
- All others → `pct_change()`

### Dropped (Redundancy)

SMA 10, WMA 10, MACD_HIST, %R, %D, up/mid/low_band, SPY, ^DJI, ^FCHI, ^IBEX, ^FVX, ^TYX, SI=F — all ρ > 0.8 with kept features.

---

## 3. Rolling Window & Sample Weights

| Setting | Value | Purpose |
|---------|-------|---------|
| `training_years` | 12 | Discard data older than 12y (concept drift) |
| `sample_weight_halflife` | 0.33 | Oldest third at ≤50% weight |

Weights: `exp(linspace(-n/half_life, 0, n))`, normalized to mean ≈ 1. Applied in both Optuna CV folds and final training.

---

## 4. Data Split & Embargo

```
|<-------- 80% Train -------->|<- embargo (10d) ->|<--- 20% Test --->|
 oldest --------------------------------------------------------> newest
```

- Chronological split (`shuffle=False`)
- Embargo = `prediction_horizon_days` (10) — removes lookahead bias from forward-looking target labels

---

## 5. Hyperparameter Tuning

- **Optuna TPE**, 60 trials
- **5-fold walk-forward CV** (`TimeSeriesSplit`, `gap=10`)
- **Metric: ROC-AUC** — immune to class imbalance; "always UP" = 0.5
- `class_weight="balanced"` forced in all models
- Sample weights passed to every CV fold

---

## 6. Models

Pluggable via trainer registry. Each type gets its own registry entry: `best_{ModelName}_model`.

**ExtraTrees** (default): key params — `max_features` (0.05–1.0), `min_samples_leaf` (1–20), `max_depth` (3–40/None), `n_estimators` (100–1000).

**LightGBM**: key params — `learning_rate` (0.005–0.15), `num_leaves` (15–255), `reg_alpha/lambda`, `min_child_samples` (10–100).

```bash
make train              # ExtraTrees
make train-lgbm         # LightGBM
make train --trials 10  # fewer trials for quick test
```

---

## 7. Model Registration

Final model trained with best params + **probability calibration** (isotonic → sigmoid fallback).

| Condition | Action |
|-----------|--------|
| ROC-AUC > current best | Register new version, archive old |
| ROC-AUC ≤ current best | Log only, don't promote |
| `--force` flag | Always register |

Logged: all hyperparameters, metrics (accuracy, F1, precision, recall, ROC-AUC), top-15 feature importances, class distribution.

---

## 8. Prediction

```bash
make predict
```

Loads latest features (including rows with `Target=NaN` — future dates), validates feature count against model, predicts with calibrated probabilities. Result stored in `predictions/sp500_predictions.parquet`.

---

## Config (`params.yaml`)

```yaml
stock:
  symbol: "^GSPC"
  start_date: "2000-08-01"
  prediction_horizon_days: 10

features:
  target_threshold: 0.03
  rate_tickers: ["^TNX", "^IRX"]

training:
  training_years: 12
  sample_weight_halflife: 0.33

tuning:
  n_trials: 60
```
