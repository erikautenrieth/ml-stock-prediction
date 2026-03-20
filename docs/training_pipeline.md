# Training Pipeline Documentation

`Raw OHLCV -> Target -> Indicators -> Extra Market Data -> Stationarity Transform -> ML-ready`

---

## 1. Target Variable

```python
ret = (Close[t+10] - Close[t]) / Close[t]
Target = 1 if ret > threshold else (0 if ret < -threshold else NaN)
```

**Dead zone:** rows with |return| <= `target_threshold` (default **2%**) are dropped (NaN).
This removes ambiguous days where the market barely moved, forcing the model to learn
only clear directional signals. The dead zone improves class balance toward ~50/50.

| Parameter | Value | Location |
|-----------|-------|----------|
| `prediction_horizon_days` | 10 | `params.yaml` / `config.py` |
| `target_threshold` | 0.02 (2%) | `params.yaml` / `config.py` |

---

## 2. Features (~53 columns)

Redundant and highly correlated features have been pruned to reduce multicollinearity
and improve signal-to-noise ratio with limited training samples.

### Technical Indicators

| Feature | Category | Description | Stationarity Transform |
|---------|----------|-------------|----------------------|
| Close | Raw | Closing price | `pct_change()` -> daily return |
| Volume | Raw | Shares traded | `pct_change()` (inf->0) |
| High, Low, Open | Raw | Intraday prices | **dropped** after indicators |
| | | | |
| EMA 10 | Trend | Exponential Moving Average (10) | `(EMA - Close) / Close` |
| EMA 20 | Trend | Exponential Moving Average (20) | `(EMA - Close) / Close` |
| SAR | Trend | Parabolic Stop & Reverse | `(SAR - Close) / Close` |
| MACD | Trend | EMA(12) - EMA(26) | `value / Close` |
| MACD_SIGNAL | Trend | EMA(9) of MACD | `value / Close` |
| ADX | Trend | Average Directional Index (14) | *unchanged* (0-100) |
| +DMI / -DMI | Trend | Directional Movement (14) | *unchanged* (0-100) |
| CCI | Trend | Commodity Channel Index (14) | *unchanged* (stationary) |
| | | | |
| RSI | Momentum | Relative Strength Index (14) | *unchanged* (0-100) |
| ROC | Momentum | Rate of Change (10) | *unchanged* (stationary) |
| %K | Momentum | Stochastic Oscillator (14, 3) | *unchanged* (0-100) |
| | | | |
| BB_pband | Volatility | Position within Bollinger Bands | *unchanged* (0-1) |
| BB_width | Volatility | Bollinger Band width (normalized) | *unchanged* |
| ATR_norm | Volatility | ATR(14) / Close | *unchanged* |
| vol_roll5/10/20/60 | Volatility | Rolling std of returns (N days) | *unchanged* |
| intraday_range | Volatility | (High - Low) / Close | *unchanged* |
| | | | |
| OBV | Volume | On-Balance Volume | `diff() / Close` |
| ADOSC | Volume | Chaikin Money Flow (10) | *unchanged* |
| | | | |
| ret_lag1/2/3/5/10/20 | Lagged | Daily return shifted N days | *unchanged* |
| gap_ret | Lagged | Overnight gap return | *unchanged* |

### Extra Market Data

| Feature | Category | Transform |
|---------|----------|-----------|
| GC=F Close | Commodity (Gold) | `pct_change()` |
| CL=F Close | Commodity (Crude Oil) | `pct_change()` |
| HG=F Close | Commodity (Copper) | `pct_change()` |
| NG=F Close | Commodity (Natural Gas) | `pct_change()` |
| EURUSD=X Close | Currency (EUR/USD) | `pct_change()` |
| GBPUSD=X Close | Currency (GBP/USD) | `pct_change()` |
| JPY=X Close | Currency (USD/JPY) | `pct_change()` |
| CNY=X Close | Currency (USD/CNY) | `pct_change()` |
| ^IXIC Close | US Index (Nasdaq) | `pct_change()` |
| ^RUT Close | US Index (Russell 2000) | `pct_change()` |
| EFA Close | Intl Developed ETF | `pct_change()` |
| ^FTSE Close | Global (FTSE 100) | `pct_change()` |
| ^GDAXI Close | Global (DAX) | `pct_change()` |
| ^N225 Close | Global (Nikkei 225) | `pct_change()` |
| ^HSI Close | Global (Hang Seng) | `pct_change()` |
| ^BSESN Close | Global (BSE Sensex) | `pct_change()` |
| ^MXX Close | Global (IPC Mexico) | `pct_change()` |
| ^AXJO Close | Global (ASX 200) | `pct_change()` |
| ^VIX Close | Volatility (CBOE VIX) | `pct_change()` |
| ^TNX Close | Rates (10Y Treasury) | `diff()` |
| ^IRX Close | Rates (3M Treasury) | `diff()` |

### Dropped Features (Redundancy Reduction)

| Dropped | Reason | Kept Instead |
|---------|--------|--------------|
| SMA 10, WMA 10 | p > 0.95 with EMA 10 after `(MA-Close)/Close` | EMA 10 |
| up_band, mid_band, low_band | Redundant encoding with BB_pband + BB_width | BB_pband, BB_width |
| MACD_HIST | Linear combo: MACD - MACD_SIGNAL | MACD, MACD_SIGNAL |
| %R (Williams) | Near-inverse of RSI (both 14-period) | RSI |
| %D | 3-period SMA of %K | %K |
| SPY Close | p ~ 0.99 with ^GSPC (target) | Close |
| ^DJI Close | p > 0.95 with ^IXIC, ^GSPC | ^IXIC |
| ^FCHI Close | p > 0.90 with ^GDAXI | ^GDAXI |
| ^IBEX Close | p > 0.90 with ^GDAXI, ^FTSE | ^GDAXI, ^FTSE |
| ^FVX, ^TYX Close | p > 0.90 with ^TNX (long-end yields) | ^TNX |
| SI=F Close | p ~ 0.80 with GC=F (Gold) | GC=F |

**Result:** 67 -> ~53 features with better signal-to-noise ratio.

Final cleanup: `inf -> 0`, drop NaN rows.

---

## 3. Rolling Training Window

**Problem:** Training on all data since 2000 mixes fundamentally different market regimes
(dot-com crash, 2008 financial crisis, COVID shock, post-COVID inflation). Old patterns
often hurt more than they help due to concept drift.

**Solution:** Two-layer approach:

### Layer 1 - Rolling Window Cutoff

Controlled by `training_years` (default: **12 years**).

```
Raw data:  2000 --------------------------------------------------> today
                     | cutoff (today - 12y)
Training:            |=============================================> today
                      only these rows enter the pipeline
```

Rows older than `training_years` from today are discarded before the train/test split.
Set to `null` in `params.yaml` to keep all available data.

### Layer 2 - Exponential Sample Weights

Controlled by `sample_weight_halflife` (default: **0.33**).

Within the remaining window, exponential decay assigns higher weight to recent
observations. Half-life 0.33 = oldest third of training data receives <= 50% the
weight of the newest rows.

```
Weight
  2x |                                          /
     |                                       //
  1x |_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _/_ _ _
     |                             ///
0.5x |              /////////////
     |   ///////////
     |---------------------------------------------->
     oldest                                newest
```

**Formula:**
```python
half_life = max(int(n * halflife_fraction), 1)
weights = np.exp(np.linspace(-n / half_life, 0, n))
weights /= weights.mean()   # normalise so mean ~ 1
```

Both Optuna CV folds **and** final training use the same weights.

### Configuration (params.yaml)

```yaml
training:
  training_years: 12           # null = use all data
  sample_weight_halflife: 0.33 # oldest 1/3 at <=50% weight
```

---

## 4. Data Split

Data is split **chronologically** (`shuffle=False`). Time order is never broken.

```
|<------------ 80% Train ------------>|<- embargo ->|<---- 20% Test ---->|
 oldest ------------------------------------------------------------> newest
```

The **embargo** drops the last 10 rows of training data (= prediction horizon).
The target at row `t` uses the price at `t+10`, which leaks into the test period.
Removing these rows eliminates lookahead bias (purging, Lopez de Prado 2018).

---

## 5. Hyperparameter Tuning (Optuna)

| Setting | Value | Reason |
|---------|-------|--------|
| Search method | Optuna (TPE sampler) | Bayesian, sample-efficient |
| Trials | 60 | budget per retrain |
| CV strategy | `TimeSeriesSplit(n_splits=5, gap=10)` | walk-forward, respects time |
| Optimization metric | **ROC-AUC** (mean over 5 folds) | immune to class imbalance; "always UP" = 0.5 |
| Sample weights | exponential decay (same as final fit) | tuner matches training regime |
| Class weight | `balanced` (forced) | prevents majority-class shortcuts |
| Logged per trial | accuracy, precision, recall, ROC-AUC | all to MLflow via DagsHub |

Cross-validation folds (all within the 80% train set):

```
Fold 1: |##### train|  gap  |.. val|                          |
Fold 2: |######### train   |  gap  |.. val|                  |
Fold 3: |############# train       |  gap  |.. val|          |
Fold 4: |################# train           |  gap  |.. val| |
Fold 5: |##################### train               |  gap  |.. val|
```

Each fold trains on all data before the gap, validates on data after. No future leakage.
Sample weights are sliced per fold. Single-class validation folds default to AUC=0.5.

---

## 6. Models

Models are pluggable via the trainer registry (`backend/ml/training/__init__.py`).
Each model type gets its own DagsHub registry entry: `best_{ModelName}_model`.

### ExtraTreesClassifier

| Hyperparameter | Search Range | Default | Importance |
|---------------|-------------|---------|------------|
| n_estimators | 100-1000 (log) | 300 | low (saturates ~300) |
| max_features | 0.05-1.0 (log) | sqrt(p) | **highest** |
| min_samples_leaf | 1-20 | 1 | **high** |
| min_samples_split | 2-20 | 2 | moderate |
| max_depth | 3-40 or None | None | moderate |
| criterion | gini / entropy | gini | negligible |
| bootstrap | True / False | False | moderate |
| max_samples | 0.6-1.0 (if bootstrap) | - | low |
| class_weight | **balanced** (forced) | - | - |

### LightGBMClassifier

| Hyperparameter | Search Range | Default | Importance |
|---------------|-------------|---------|------------|
| n_estimators | 100-1500 (log) | 500 | moderate |
| learning_rate | 0.005-0.15 (log) | 0.03 | **highest** |
| num_leaves | 15-255 | 31 | **high** |
| max_depth | 3-12 | -1 | moderate (caps num_leaves) |
| min_child_samples | 10-100 | 20 | high |
| min_gain_to_split | 1e-4-1.0 (log) | 0.01 | moderate |
| subsample | 0.5-1.0 | 0.8 | moderate |
| colsample_bytree | 0.3-1.0 | 0.8 | moderate |
| reg_alpha (L1) | 1e-4-10 (log) | 0.1 | **high** |
| reg_lambda (L2) | 1e-2-50 (log) | 1.0 | **high** |
| path_smooth | 0-10 | 1.0 | moderate |
| max_bin | 63 / 127 / 255 | 127 | low |
| class_weight | **balanced** (forced) | - | - |

**CLI:**
```bash
make train            # ExtraTrees (default)
make train-lgbm       # LightGBM
make train-model MODEL=lightgbm
```

---

## 7. Model Registration

After tuning, the best params train a **final model on all training data** with:
- Exponential sample weights
- Probability calibration (isotonic -> sigmoid fallback)

Evaluated on the held-out 20% test set.

| Condition | Action |
|-----------|--------|
| New ROC-AUC > current best | Register as new version, archive old |
| Feature count changed | Force register (old model incompatible) |
| New ROC-AUC <= current best | Log only, don't promote |

Each model type has its own registry entry: `best_{ModelName}_model`

### Logged Artifacts

- All hyperparameters
- Metrics: accuracy, precision, recall, ROC-AUC
- Top-15 feature importances
- Class distribution (UP/DOWN counts + percentages)

---

## 8. End-to-End Flow Summary

```
 1. Load features from DuckDB store
 2. Rolling-window cutoff: drop rows older than training_years (12y)
 3. Drop NaN targets (dead-zone rows with |return| <= 2%)
 4. Chronological 80/20 split (shuffle=False)
 5. Embargo: remove last h=10 training rows (purge lookahead)
 6. Compute exponential sample weights (half-life = 33% of train length)
 7. Log class distribution (UP/DOWN counts + percentages)
 8. Optuna HPO: 60 trials, 5-fold walk-forward CV
    Metric: ROC-AUC | class_weight=balanced forced
 9. Train final model with best params + sample weights + calibration
10. Evaluate on held-out test set
11. Log params, metrics, and top-15 feature importances to MLflow/DagsHub
12. Register model if ROC-AUC beats current best (or force flag set)
```

---

## Current Settings (params.yaml)

```yaml
stock:
  symbol: "^GSPC"
  start_date: "2000-08-01"
  prediction_horizon_days: 10
  target_threshold: 0.02

training:
  training_years: 12
  sample_weight_halflife: 0.33

tuning:
  n_trials: 60
```

---

## References

- Geurts, P., Ernst, D., & Wehenkel, L. (2006). *Extremely randomized trees.* Machine Learning, 63(1), 3-42.
- Ke, G., Meng, Q., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Pesaran, M. H., & Timmermann, A. (2002). *Market timing and return prediction under model instability.* Journal of Empirical Finance.
- Inoue, A., Jin, L., & Rossi, B. (2017). *Rolling window selection for out-of-sample forecasting with time-varying parameters.* Journal of Econometrics.
- Probst, P., Wright, M. N., & Boulesteix, A.-L. (2018). *Hyperparameters and tuning strategies for random forest.* WIREs Data Mining and Knowledge Discovery.
- Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning.* Review of Financial Studies.
