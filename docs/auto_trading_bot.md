# Auto-Trading Bot – Anbindung an das Prediction System

## Idee

Das Prediction-System liefert pro Tag ein Signal (`prediction` + `confidence`).  
Ein Bot liest dieses Signal und handelt automatisch:

| Confidence | Label      | Aktion                          |
|-----------|------------|---------------------------------|
| ≥ 70 %    | 🟢 Stark   | Volle Positionsgröße            |
| ≥ 60 %    | 🟡 Moderat | Halbe Positionsgröße            |
| < 60 %    | 🔴 Schwach | Kein Trade / Position schließen |

- **Prediction = 1 (UP)** → Long (kaufen)
- **Prediction = 0 (DOWN)** → Short

---

## Broker-Vergleich (mit API für Algo-Trading)

### 1. Interactive Brokers (IBKR) ⭐ Empfehlung

| Eigenschaft         | Details |
|---------------------|---------|
| **Regulierung**     | BaFin, SEC, FINRA, FCA – einer der am strengsten regulierten Broker weltweit |
| **Paper Trading**   | ✅ Kostenloses Paper-Account mit identischer API |
| **Python SDK**      | `ib_insync` – bewährt, aktive Community |
| **Shorting**        | ✅ Aktien, ETFs, Futures, Optionen |
| **Konsorsbank-Transfer** | ✅ SEPA-Überweisung (EUR), Konto in Deutschland/Irland, Ein-/Auszahlung in 1-2 Tagen |
| **Kommission**      | ~1 $ pro US-Trade (Tiered), 4 € pro DE-Trade |
| **Mindesteinlage**  | 0 € (kein Minimum mehr) |
| **Assets**          | Aktien, ETFs, Futures, Optionen, Forex – weltweit |

**Warum IBKR:** Einziger ernstzunehmender Broker der sowohl BaFin-reguliert ist, SEPA-Überweisungen von/zur Consorsbank problemlos unterstützt, UND eine vollwertige Trading-API bietet. Existiert seit 1978, verwaltet >400 Mrd. $ Kundengelder.

### 2. Alpaca

| Eigenschaft         | Details |
|---------------------|---------|
| **Regulierung**     | FINRA / SIPC (nur USA) |
| **Paper Trading**   | ✅ Kostenlos, sofort aktiv, identisch zur Live-API |
| **Python SDK**      | `alpaca-py` (offiziell) – sehr einfache API |
| **Shorting**        | ✅ US-Aktien & ETFs |
| **Konsorsbank-Transfer** | ⚠️ Nur US-Banktransfer (Wire) – umständlich und teuer (~20-40 € pro Transfer) |
| **Kommission**      | 0 $ für US Aktien/ETFs |
| **Mindesteinlage**  | 0 $ |
| **Assets**          | Nur US-Aktien, ETFs, Krypto |

**Warum Alpaca:** Einfachste API überhaupt, perfekt zum Prototypen und Paper-Trading. Geld-Transfer von/nach Deutschland aber umständlich (kein SEPA).

### 3. IG Markets

| Eigenschaft         | Details |
|---------------------|---------|
| **Regulierung**     | BaFin, FCA – seit 1974 |
| **Paper Trading**   | ✅ Demo-Konto mit 10.000 € Spielgeld |
| **Python SDK**      | REST-API + `ig-markets-api-python-library` (Community) |
| **Shorting**        | ✅ Via CFDs |
| **Konsorsbank-Transfer** | ✅ SEPA, Kreditkarte, PayPal |
| **Kommission**      | Spreads (kein feste Kommission) |
| **Mindesteinlage**  | 300 € |
| **Assets**          | CFDs auf Aktien, Indizes, Forex, Krypto, Rohstoffe |

**Warum IG:** BaFin-reguliert, einfacher Geld-Transfer in DE, gute API. **Aber:** CFDs sind Hebelprodukte mit höherem Risiko, kein echtes Aktien-Owning.

---

## Zusammenfassung

| Kriterium                  | **IBKR** ⭐ | **Alpaca**  | **IG Markets** |
|----------------------------|-------------|-------------|----------------|
| Paper Trading (Spielgeld)  | ✅ Kostenlos | ✅ Kostenlos | ✅ 10k € Demo  |
| SEPA / Consorsbank         | ✅ Einfach   | ❌ Wire only | ✅ Einfach      |
| BaFin-reguliert            | ✅           | ❌ (nur US)  | ✅              |
| API-Einfachheit            | Mittel      | Sehr einfach | Mittel         |
| Commission-free            | ❌           | ✅           | ❌ (Spreads)    |
| Echte Aktien               | ✅           | ✅           | ❌ (CFDs)       |

**Empfehlung:**
- **Prototyp / Paper-Trading:** Alpaca (einfachste API, sofort loslegen)
- **Live-Trading aus DE:** Interactive Brokers (SEPA, BaFin, echte Aktien, bewährte API)

---

## Minimalbeispiel: Alpaca-Bot (Paper Trading)

```python
"""Minimal auto-trading bot – liest Prediction und handelt via Alpaca."""

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from backend.workflows.predict import predict_latest

# --- Config ---
ALPACA_KEY = "PAPER_API_KEY"         # aus .env laden!
ALPACA_SECRET = "PAPER_SECRET_KEY"   # aus .env laden!
SYMBOL = "SPY"                       # S&P500 ETF
MIN_CONFIDENCE = 0.60
FULL_CONFIDENCE = 0.70
POSITION_USD = 1000                  # Max-Position in $

client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)


def get_signal() -> tuple[int, float]:
    """Prediction + Confidence aus dem bestehenden System holen."""
    result = predict_latest()
    return result.prediction, result.confidence


def calc_qty(confidence: float) -> float:
    """Positionsgröße basierend auf Confidence."""
    if confidence >= FULL_CONFIDENCE:
        return POSITION_USD
    return POSITION_USD * 0.5


def close_position():
    """Offene Position in SPY schließen."""
    try:
        client.close_position(SYMBOL)
    except Exception:
        pass


def execute():
    """Einen Trade-Zyklus ausführen."""
    prediction, confidence = get_signal()

    if confidence < MIN_CONFIDENCE:
        close_position()
        return

    side = OrderSide.BUY if prediction == 1 else OrderSide.SELL
    notional = calc_qty(confidence)
    close_position()

    order = MarketOrderRequest(
        symbol=SYMBOL,
        notional=round(notional, 2),
        side=side,
        time_in_force=TimeInForce.DAY,
    )
    client.submit_order(order)


if __name__ == "__main__":
    execute()
```

---

## Integration

1. `poetry add alpaca-py` (oder `ib_insync` für IBKR)
2. API-Keys in `.env`
3. `make trade` nach `make predict`
4. Scheduling: Cronjob Mo-Fr nach Market Open

## Wichtige Hinweise

- **Immer erst Paper-Trading** – alle drei Broker bieten kostenlose Demo-/Paper-Konten
- **SPY statt ^GSPC** – S&P500-Index ist nicht direkt handelbar → SPY ETF nutzen
- **PDT-Regel (USA):** Bei < 25k $ max 3 Day-Trades pro 5 Tage
- **Shorting braucht Margin-Konto** – bei Paper automatisch aktiviert
