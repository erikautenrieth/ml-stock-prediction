# Feature Engineering Pipeline

`Raw OHLCV → Target → Indicators → Extra Market Data → Stationarity Transform → ML-ready`

| Feature | Category | Description | Stationarity Transform |
|---------|----------|-------------|----------------------|
| **Target** | **Target** | `1` if Close[t+10] > Close[t], else `0` | — |
| | | | |
| Close | Raw | Closing price | `pct_change()` → daily return |
| Volume | Raw | Shares traded | `pct_change()` → daily return |
| High, Low, Open | Raw | Intraday prices | **dropped** after indicators are computed |
| | | | |
| SMA 10 | Trend | Simple Moving Average (10) | `(SMA − Close) / Close` |
| EMA 10 | Trend | Exponential Moving Average (10) | `(EMA − Close) / Close` |
| EMA 20 | Trend | Exponential Moving Average (20) | `(EMA − Close) / Close` |
| WMA 10 | Trend | Weighted Moving Average (10) | `(WMA − Close) / Close` |
| SAR | Trend | Parabolic Stop & Reverse | `(SAR − Close) / Close` |
| MACD | Trend | EMA(12) − EMA(26) | `value / Close` |
| MACD_SIGNAL | Trend | EMA(9) of MACD | `value / Close` |
| MACD_HIST | Trend | MACD − MACD_SIGNAL | `value / Close` |
| ADX | Trend | Average Directional Index (14) | *unchanged* (0–100) |
| +DMI / −DMI | Trend | Directional Movement (14) | *unchanged* (0–100) |
| CCI | Trend | Commodity Channel Index (14) | *unchanged* (stationary) |
| | | | |
| RSI | Momentum | Relative Strength Index (14) | *unchanged* (0–100) |
| ROC | Momentum | Rate of Change (10) | *unchanged* (stationary) |
| %K / %D | Momentum | Stochastic Oscillator (14, 3) | *unchanged* (0–100) |
| %R | Momentum | Williams %R (14) | *unchanged* (−100–0) |
| | | | |
| up_band | Volatility | Bollinger Upper Band (20) | `(band − Close) / Close` |
| mid_band | Volatility | Bollinger Middle Band (20) | `(band − Close) / Close` |
| low_band | Volatility | Bollinger Lower Band (20) | `(band − Close) / Close` |
| BB_pband | Volatility | Position within bands (0=lower, 1=upper) | *unchanged* (normalized) |
| BB_width | Volatility | (upper − lower) / middle | *unchanged* (normalized) |
| ATR_norm | Volatility | ATR(14) / Close | *unchanged* (normalized) |
| vol_roll5/10/20/60 | Volatility | Std of daily returns over N days | *unchanged* (pre-computed) |
| intraday_range | Volatility | (High − Low) / Close | *unchanged* (pre-computed) |
| | | | |
| OBV | Volume | On-Balance Volume | `diff() / Close` → flow rate |
| ADOSC | Volume | Chaikin Money Flow (10) | *unchanged* (stationary) |
| | | | |
| ret_lag1/2/3/5/10/20 | Lagged | Daily return shifted by N days | *unchanged* (pre-computed) |
| gap_ret | Lagged | (Open_t − Close_{t−1}) / Close_{t−1} | *unchanged* (pre-computed) |
| | | | |
| GC=F Close | Extra: Commodity | Gold | `pct_change()` |
| CL=F Close | Extra: Commodity | Crude Oil | `pct_change()` |
| SI=F Close | Extra: Commodity | Silver | `pct_change()` |
| HG=F Close | Extra: Commodity | Copper | `pct_change()` |
| NG=F Close | Extra: Commodity | Natural Gas | `pct_change()` |
| EURUSD=X Close | Extra: Currency | EUR/USD | `pct_change()` |
| GBPUSD=X Close | Extra: Currency | GBP/USD | `pct_change()` |
| JPY=X Close | Extra: Currency | USD/JPY | `pct_change()` |
| CNY=X Close | Extra: Currency | USD/CNY | `pct_change()` |
| ^IXIC Close | Extra: US Index | Nasdaq | `pct_change()` |
| ^DJI Close | Extra: US Index | Dow Jones | `pct_change()` |
| ^RUT Close | Extra: US Index | Russell 2000 | `pct_change()` |
| SPY Close | Extra: US Index | S&P 500 ETF | `pct_change()` |
| EFA Close | Extra: US Index | Intl Developed ETF | `pct_change()` |
| ^FTSE Close | Extra: Global | FTSE 100 | `pct_change()` |
| ^GDAXI Close | Extra: Global | DAX | `pct_change()` |
| ^FCHI Close | Extra: Global | CAC 40 | `pct_change()` |
| ^N225 Close | Extra: Global | Nikkei 225 | `pct_change()` |
| ^HSI Close | Extra: Global | Hang Seng | `pct_change()` |
| ^BSESN Close | Extra: Global | BSE Sensex | `pct_change()` |
| ^MXX Close | Extra: Global | IPC Mexico | `pct_change()` |
| ^AXJO Close | Extra: Global | ASX 200 | `pct_change()` |
| ^IBEX Close | Extra: Global | IBEX 35 | `pct_change()` |
| ^VIX Close | Extra: Volatility | CBOE VIX | `pct_change()` |
| ^TNX Close | Extra: Rates | 10Y Treasury Yield | `diff()` (near-zero safe) |
| ^IRX Close | Extra: Rates | 3M Treasury Yield | `diff()` (near-zero safe) |
| ^FVX Close | Extra: Rates | 5Y Treasury Yield | `diff()` (near-zero safe) |
| ^TYX Close | Extra: Rates | 30Y Treasury Yield | `diff()` (near-zero safe) |

Final cleanup: `inf → 0`, drop NaN rows.

---

# Training Pipeline

## Data Split — No Shuffling

Data is split **chronologically** (`shuffle=False`). Time order is never broken.

```
|◄──────────── 80% Train ────────────►|◄─ embargo ─►|◄──── 20% Test ────►|
 oldest ──────────────────────────────────────────────────────────► newest
```

The **embargo** drops the last 10 rows of training data (= prediction horizon).
Why: the target at row `t` uses the price at `t+10`, which leaks into the test period.
Removing these rows eliminates lookahead bias (López de Prado 2018, "purging").

## Hyperparameter Tuning (Optuna)

| Setting | Value | Reason |
|---------|-------|--------|
| Search method | Optuna (TPE sampler) | Bayesian, sample-efficient |
| Trials | 60 | budget per weekly retrain |
| CV strategy | `TimeSeriesSplit(n_splits=5, gap=10)` | walk-forward, respects time |
| Optimization metric | **F1 score** (mean over 5 folds) | robust when UP/DOWN imbalanced |
| Logged per trial | accuracy, precision, recall, F1 | all to MLflow via DagsHub |

Cross-validation folds (all within the 80% train set):

```
Fold 1: |▓▓▓▓▓ train|  gap  |░░ val|                          |
Fold 2: |▓▓▓▓▓▓▓▓▓ train   |  gap  |░░ val|                  |
Fold 3: |▓▓▓▓▓▓▓▓▓▓▓▓▓ train       |  gap  |░░ val|          |
Fold 4: |▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ train           |  gap  |░░ val| |
Fold 5: |▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ train               |  gap  |░░ val|
```

Each fold trains on all data before the gap, validates on data after. No future leakage.

## Model: ExtraTreesClassifier

| Hyperparameter | Search Range | Default | Importance |
|---------------|-------------|---------|------------|
| n_estimators | 100–1000 (log) | 300 | low (saturates ~300) |
| max_features | 0.05–1.0 (log) | sqrt(p) | **highest** (Geurts 2006) |
| min_samples_leaf | 1–20 | 1 | **high** (Probst 2018) |
| min_samples_split | 2–20 | 2 | moderate |
| max_depth | 3–40 or None | None | moderate |
| criterion | gini / entropy | gini | negligible |
| bootstrap | True / False | False | moderate |
| max_samples | 0.6–1.0 (if bootstrap) | — | low |
| class_weight | None / balanced | None | moderate |

## Model Registration

After tuning, the best params are used to train a **final model on all training data**.
Evaluated on the held-out 20% test set (accuracy, F1, precision, recall, ROC-AUC).

| Condition | Action |
|-----------|--------|
| New accuracy > current best | Register as new version, archive old |
| Feature count changed | Force register (old model incompatible) |
| New accuracy ≤ current best | Log only, don't promote |

Registered model name: `best_ExtraTreesClassifier_model`
