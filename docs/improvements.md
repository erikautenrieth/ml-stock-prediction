# Model Improvements

Improvements to increase prediction scores while keeping ExtraTreesClassifier.

## 1. Feature Selection — Remove Noise

**Problem:** ~70+ features, many highly correlated (SMA 10, EMA 10, WMA 10 measure the same thing). Noise features dilute the signal.

**Action:**
- After training, extract feature importances → drop features with importance ≈ 0
- Compute correlation matrix → if `|corr| > 0.9`, drop one of the pair
- Likely drops: WMA 10 (redundant with SMA/EMA), several global indices (move together)

**Expected impact:** High. Fewer features = less overfitting, faster training, cleaner splits.

## 2. Target Threshold — Filter the Noise Zone

**Problem:** `Target = 1 if Close[t+10] > Close[t]` — a +0.01% move counts as UP. That's noise, not signal.

**Action:**
- UP = return > +0.5%, DOWN = return < −0.5%
- Remove rows in between (dead zone) or label as a 3rd class
- Tune the threshold via Optuna as an additional hyperparameter

**How large is the dead zone?** S&P 500 10-day returns have a std of ~3-4%. A ±0.5% threshold removes roughly 10-15% of rows. That's acceptable — you lose some data but gain much cleaner labels.

**Implementation plan — 4 files to touch:**

1. **`config.py`** — add `target_threshold: float = 0.005` to `StockSettings`

2. **`preprocessing.py`** — change `calc_target()`:
   ```python
   ret = (df["Close"].shift(-horizon) - df["Close"]) / df["Close"]
   df["Target"] = np.where(ret > threshold, 1, np.where(ret < -threshold, 0, np.nan))
   ```
   Rows in the dead zone get `NaN` → dropped later by `dropna()` (already happens)

3. **`train.py`** — no change needed. Dead-zone rows are already gone before training starts.

4. **`predict.py`** — no change needed. Prediction is still binary (UP/DOWN). The model just learned from cleaner examples.

**What does NOT change:** indicators, transform_to_returns, extra market data, the split logic, Optuna tuning, model registration — all untouched. Only the label computation in `calc_target()` changes.

**Expected impact:** High. Cleaner labels → model learns real trends, not coin flips.

## 3. Rolling Training Window — Focus on Recent Regime

**Problem:** Training on data since 2000 mixes dot-com crash, 2008 crisis, COVID, post-COVID — fundamentally different market regimes. Old patterns may hurt more than help.

**Action:**
- Train only on the last 8 years instead of all 25
- Or use sample weights: exponential decay so recent data matters more

**Implementation:**

Option A — Rolling window (simplest): add `training_years` to `StockSettings`, then slice in `Trainer.prepare()`:
```python
# backend/ml/training/base.py — in prepare()
cutoff = pd.Timestamp.now() - pd.DateOffset(years=settings.stock.training_years)
df = df[df.index >= cutoff]
```

Option B — Exponential sample weights (no data loss): pass `sample_weight` to `model.fit()`:
```python
# backend/workflows/train.py — before model.fit()
n = len(x_train)
half_life = n // 3  # recent 1/3 of data gets 2× the weight
weights = np.exp(np.linspace(-n / half_life, 0, n))
model.fit(x_train, y_train, sample_weight=weights)
```
ExtraTrees supports `sample_weight` natively. Also pass weights into the Optuna CV loop.

**Expected impact:** High. The model adapts to the current market regime instead of averaging across all history.

## 4. Optimize for Precision — Quality Over Quantity

**Problem:** F1 balances precision and recall equally. For trading, a wrong signal costs money — precision matters more than recall.

**Action:**
- Switch Optuna objective from F1 to precision (or weighted F-beta with beta < 1)
- Accept fewer signals, but each signal is more reliable

**Expected impact:** Medium. Fewer but better predictions → higher P&L per trade.

## 5. Cross-Features — Help the Model See Interactions

**Problem:** ExtraTrees can learn interactions, but explicit ratio features make it much easier.

**Action:**
- `RSI × Volume_change` — volume confirms momentum signals
- `VIX_level × direction` — high-VIX markets behave differently
- `SP500_return / Nasdaq_return` — sector rotation signal
- `ATR_norm / vol_roll20` — short vs medium-term volatility ratio

**Expected impact:** Medium. Explicit interactions reduce the depth needed to capture them.

## 6. Probability Calibration

**Problem:** `predict_proba()` from ExtraTrees gives uncalibrated probabilities (proportion of trees voting, not true probability). Confidence values are unreliable.

**Action:**
- Wrap model in `CalibratedClassifierCV(method="isotonic")` after training
- Better calibrated probabilities → smarter confidence thresholds for trading

**Expected impact:** Low-Medium. Doesn't change raw accuracy, but improves confidence-based decisions.

## Priority

| # | Improvement | Effort | Impact |
|---|------------|--------|--------|
| 1 | Feature Selection | low | **high** |
| 2 | Target Threshold | low | **high** |
| 3 | Rolling Training Window | low | **high** |
| 4 | Precision as metric | minimal | medium |
| 5 | Cross-Features | medium | medium |
| 6 | Probability Calibration | low | low-medium |

# Modell-Evaluation

> **Status:** To-Do  
> **Aktuell:** `ExtraTreesClassifier` · 46 Features · 10d Horizont · Optuna-tuned (F1)

## Kandidaten

Alle sklearn-kompatibel → neuer `Trainer`-Subclass pro Modell (< 80 LOC).

| # | Modell | Install | Vorteil ggü. ExtraTrees | Prio |
|---|--------|---------|------------------------|------|
| 1 | **LightGBM** | `poetry add lightgbm` | Boosting > Bagging, L1/L2-Reg, schnell | 🔴 |
| 2 | **XGBoost** | `poetry add xgboost` | Stärkere Regularisierung (gamma/alpha/lambda) | 🔴 |
| 3 | **HistGradientBoosting** | — (sklearn) | Keine neue Dep, LightGBM-ähnlich | 🟡 |
| 4 | **CatBoost** | `poetry add catboost` | Ordered Boosting gegen temporales Leakage | 🟡 |
| 5 | **RandomForest** | — (sklearn) | Ablation: optimale vs. zufällige Splits | 🟡 |
| 6 | **Stacking** | — (sklearn) | Top-Modelle kombiniert, Meta-Learner | 🟢 |
| 7 | **LogisticRegression** | — (sklearn) | Lineare Baseline (+ `StandardScaler`) | 🟢 |

## Reihenfolge

```
Phase 1:  ① LightGBM  ② XGBoost  ③ HistGradientBoosting
Phase 2:  ④ CatBoost   ⑤ RandomForest  ⑥ LogisticRegression
Phase 3:  ⑦ Stacking (Top-2-3 aus Phase 1+2)
```

## Integration

1. `backend/ml/training/<name>.py` — erbt `Trainer`, implementiert `name()`, `default_params()`, `search_space()`, `build()`
2. `params.yaml` → `model: "lightgbm"` o.ä.
3. `poetry add <paket>` falls nötig

Optuna-Tuning, MLflow-Logging, Prediction funktionieren automatisch über das `Trainer`-Interface.
