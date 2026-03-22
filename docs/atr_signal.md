# ATR-Signal — 10-Tage-Strategie (ohne ML)

## ATR Kurzfassung

**Average True Range** (Wilder, 1978) = Volatilitätsmesser, **keine Richtung**.

```
TR  = max(High−Low, |High−Close_prev|, |Low−Close_prev|)
ATR = Wilder-Smoothing(TR, window=14)
```

| ATR steigt | Mehr Volatilität — starke Bewegung |
|---|---|
| **ATR fällt** | **Konsolidierung — Squeeze/Breakout kommt** |

Im Projekt existiert `ATR_norm = ATR / Close` (ML-Feature). Für Signal-Trading brauchen wir den **rohen ATR**.

---

## Strategie: Keltner-Breakout + ATR-Stop (10 Tage Hold)

```
1. SIGNAL   Close > EMA20 + 1.5×ATR (gestern drunter)  → BUY
            Close < EMA20 − 1.5×ATR (gestern drüber)  → SELL

2. STOP     Entry ± 2×ATR, täglich als Trailing-Stop nachgezogen

3. EXIT     Was zuerst kommt:
            → 10 Tage erreicht
            → Trailing-Stop getroffen
```

**1.5× statt 2×**: Kürzerer Horizont → früherer Entry nötig. Mehr Fehlsignale, aber schnellerer Einstieg.

### Nicht traden wenn

- ADX < 20 (kein Trend)
- ATR > 90. Perzentil (Markt überhitzt)
- Direkt nach großem Gap (Signal künstlich ausgelöst)

---

## Implementierung

```python
import pandas as pd
import ta


def atr_10d_signals(df: pd.DataFrame) -> pd.DataFrame:
    """ATR-Keltner Kauf/Verkauf-Signale für 10-Tage-Haltedauer."""
    close, high, low = df["Close"], df["High"], df["Low"]

    atr = ta.volatility.average_true_range(high, low, close, window=14)
    df["ATR"] = atr

    # Squeeze: ATR im untersten 20. Perzentil der letzten 100 Tage
    df["ATR_squeeze"] = atr.rolling(100).apply(
        lambda x: (x.iloc[-1] <= x).mean(), raw=False
    ) < 0.20

    # Keltner-Channel
    ema20 = ta.trend.ema_indicator(close, window=20)
    df["keltner_upper"] = ema20 + 1.5 * atr
    df["keltner_lower"] = ema20 - 1.5 * atr

    # Breakout-Signale (Close durchbricht Band, war gestern noch drunter/drüber)
    broke_upper = (close > df["keltner_upper"]) & (close.shift(1) <= df["keltner_upper"].shift(1))
    broke_lower = (close < df["keltner_lower"]) & (close.shift(1) >= df["keltner_lower"].shift(1))

    df["signal"] = ""
    df.loc[broke_upper, "signal"] = "BUY"
    df.loc[broke_lower, "signal"] = "SELL"

    # Optional: nur bei Squeeze traden (weniger, bessere Signale)
    # df.loc[~df["ATR_squeeze"] & (df["signal"] != ""), "signal"] = ""

    # Stop-Loss
    df["stop_loss_long"] = close - 2 * atr
    df["stop_loss_short"] = close + 2 * atr

    return df


def evaluate_trades(df: pd.DataFrame, holding_days: int = 10) -> pd.DataFrame:
    """Simuliert Trades: Entry → Stop oder 10-Tage-Exit."""
    trades = []
    i = 0

    while i < len(df):
        row = df.iloc[i]
        if row["signal"] not in ("BUY", "SELL"):
            i += 1
            continue

        entry_price = row["Close"]
        entry_date = df.index[i]
        direction = 1 if row["signal"] == "BUY" else -1
        stop = row["stop_loss_long"] if direction == 1 else row["stop_loss_short"]
        exit_reason = "HOLDING_PERIOD"

        for j in range(1, holding_days + 1):
            if i + j >= len(df):
                break
            future = df.iloc[i + j]

            # Stop-Loss prüfen
            if direction == 1 and future["Low"] <= stop:
                exit_reason, exit_price, exit_date = "STOP_LOSS", stop, df.index[i + j]
                break
            elif direction == -1 and future["High"] >= stop:
                exit_reason, exit_price, exit_date = "STOP_LOSS", stop, df.index[i + j]
                break

            # Trailing-Stop nachziehen
            new_stop = future["Close"] - 2 * future["ATR"] if direction == 1 else future["Close"] + 2 * future["ATR"]
            stop = max(stop, new_stop) if direction == 1 else min(stop, new_stop)
        else:
            exit_idx = min(i + holding_days, len(df) - 1)
            exit_price, exit_date = df.iloc[exit_idx]["Close"], df.index[exit_idx]

        trades.append({
            "entry_date": entry_date, "exit_date": exit_date,
            "direction": row["signal"], "entry_price": entry_price,
            "exit_price": exit_price,
            "return": direction * (exit_price - entry_price) / entry_price,
            "exit_reason": exit_reason,
        })
        i = df.index.get_loc(exit_date) + 1

    return pd.DataFrame(trades)
```

### Nutzung

```python
import yfinance as yf

df = yf.download("^GSPC", start="2020-01-01", end="2026-03-01")
df = atr_10d_signals(df)
trades = evaluate_trades(df, holding_days=10)

print(f"Trades: {len(trades)}, Win-Rate: {(trades['return'] > 0).mean():.1%}")
print(f"Ø Return: {trades['return'].mean():.2%}")
```

---

## Zahlenbeispiel

```
Tag 1:  Close=5000, EMA20=4950, ATR=40 → Upper=5010 → kein Signal
Tag 2:  Close=5020, EMA20=4955, ATR=42 → Upper=5018 → BUY (durchbrochen!)
        Stop = 5020 − 2×42 = 4936
Tag 5:  Close=5080, ATR=38 → neuer Stop = 5080−76 = 5004 (hochgezogen)
Tag 12: Haltedauer-Exit ODER Stop getroffen → Position schließen
```

---

## Parameter

| Parameter | Wert | Range |
|---|---|---|
| ATR Window | 14 | 7–21 |
| Keltner Multiplikator | 1.5 | 1.0–2.5 |
| Stop-Loss Multiplikator | 2.0 | 1.5–3.0 |
| Squeeze Perzentil | 20% | 5–20% |
| EMA Window | 20 | 10–50 |

---

## Nächste Schritte

- [ ] Signal-Modul als eigenständige Datei (`backend/core/signals/atr_keltner.py`)
- [ ] Backtest auf historischen S&P 500 Daten
- [ ] Position-Sizing basierend auf ATR
- [ ] Integration in Frontend (Signal-Anzeige neben ML-Prediction)
