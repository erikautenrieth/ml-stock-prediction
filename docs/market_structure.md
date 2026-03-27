# Market Structure: Higher Highs / Lower Lows — Multi-Timeframe

## 1. Was ist Market Structure?

Market Structure beschreibt den Trend eines Marktes über die Abfolge von **Swing-Hochs** und **Swing-Tiefs**:

| Sequenz | Bedeutung | Trend |
|---------|-----------|-------|
| **HH** (Higher High) + **HL** (Higher Low) | Jedes neue Hoch/Tief liegt über dem vorherigen | Aufwärtstrend |
| **LH** (Lower High) + **LL** (Lower Low) | Jedes neue Hoch/Tief liegt unter dem vorherigen | Abwärtstrend |
| Gemischt (HH + LL oder LH + HL) | Kein klarer Trend | Seitwärtsmarkt / Transition |

Ein **Break of Structure (BOS)** tritt ein, wenn ein vorheriges Swing-Hoch (im Aufwärtstrend) oder Swing-Tief (im Abwärtstrend) durchbrochen wird — das bestätigt den Trend. Ein **Change of Character (CHoCH)** markiert den Wechsel: z.B. im Aufwärtstrend wird das letzte HL unterschritten.

---

## 2. Mathematische Definition der Swing-Punkte

### 2.1 Lokale Extrema (aktueller Ansatz)

Ein **Swing High** an Position $i$ ist definiert als:

$$H_i = \max(H_{i-n}, \ldots, H_{i+n})$$

wobei $H_i$ eindeutig das Maximum im Fenster $[i-n, i+n]$ ist. Analog für **Swing Low**:

$$L_i = \min(L_{i-n}, \ldots, L_{i+n})$$

Der Parameter $n$ (im Code `order`) bestimmt die Sensitivität:
- **Kleines $n$** (z.B. 3): Mehr Swing-Punkte, mehr Rauschen
- **Großes $n$** (z.B. 10): Weniger, aber signifikantere Punkte

### 2.2 Klassifikation

Gegeben die bereinigte Sequenz alternierend $(S_1, S_2, \ldots)$:

- $S_i$ ist **HH** wenn $S_i.\text{type} = \text{high}$ und $S_i.\text{price} > S_{j}.\text{price}$ (wobei $S_j$ das letzte High vor $i$)
- $S_i$ ist **HL** wenn $S_i.\text{type} = \text{low}$ und $S_i.\text{price} > S_{k}.\text{price}$ (wobei $S_k$ das letzte Low vor $i$)

---

## 3. Multi-Timeframe: Muss man das separat berechnen?

### Kurze Antwort: **Ja, zwingend.**

### Warum?

Die OHLC-Daten unterscheiden sich fundamental je nach Zeitintervall:

| Eigenschaft | 5-Min | 15-Min | Daily |
|-------------|-------|--------|-------|
| Bars pro Tag | ~78 | ~26 | 1 |
| Noise-Level | Sehr hoch | Mittel | Niedrig |
| Swing-Relevanz | Intraday-Scalps | Intraday-Swings | Positional |
| Typischer `order` | 3–5 | 5–8 | 5–15 |

Ein 5-Minuten-High hat **keine direkte Beziehung** zum Daily-High desselben Tages (es sei denn, es fällt zufällig zusammen). Das Daily-OHLC aggregiert den gesamten Tag — Intraday-Swings gehen dabei verloren.

### 3.1 Wissenschaftliche Grundlage

**Dow Theory (1900er):** Charles Dow definierte bereits drei Trend-Ebenen:
- **Primary Trend** (Monate–Jahre) → entspricht Weekly/Monthly
- **Secondary Trend** (Wochen–Monate) → entspricht Daily
- **Minor Trend** (Tage–Wochen) → entspricht Intraday

> Referenz: Hamilton, W.P. (1922). *The Stock Market Barometer*. Harper & Brothers.

**Fractal Market Hypothesis (Mandelbrot, 1997):**
Märkte zeigen **selbstähnliche Strukturen** auf verschiedenen Zeitskalen. Die gleiche HH/HL-Logik wiederholt sich fraktal — aber die konkreten Swing-Punkte sind auf jeder Skala verschieden.

> Referenz: Mandelbrot, B. & Hudson, R.L. (2004). *The (Mis)behavior of Markets*. Basic Books.

**Multi-Scale Pattern Recognition (Lo et al., 2000):**
Technische Muster (Head & Shoulders, Double Tops etc.) sind statistisch signifikant — aber **timeframe-abhängig**. Die Autoren verwenden Kernel-Regression auf verschiedenen Bandbreiten, was konzeptionell dem `order`-Parameter entspricht.

> Referenz: Lo, A.W., Mamaysky, H. & Wang, J. (2000). "Foundations of Technical Analysis." *Journal of Finance*, 55(4), 1705–1765.

**Adaptive Markets Hypothesis (Lo, 2004):**
Marktstruktur ist nicht statisch — sie passt sich an. Multi-Timeframe-Analyse fängt verschiedene Regime (institutionelle vs. retail Flows) ab.

### 3.2 Top-Down-Analyse: Wie Timeframes zusammenspielen

```
Daily:   ──── Richtung des übergeordneten Trends
              ↓ (Kontext)
15-Min:  ──── Swing-Struktur für Entry-Zone
              ↓ (Timing)
5-Min:   ──── Präziser Einstiegspunkt
```

**Regel:** Trade nur in Richtung des übergeordneten Trends. Der höhere Timeframe gibt die **Richtung** vor, der niedrigere den **Einstieg**.

| Schritt | Timeframe | Frage |
|---------|-----------|-------|
| 1 | Daily | Ist der Trend bullish (HH+HL) oder bearish (LH+LL)? |
| 2 | 15-Min | Gibt es einen Pullback in eine Demand/Supply Zone? |
| 3 | 5-Min | Zeigt der 5-Min-Chart einen CHoCH (Trendwechsel) als Entry-Trigger? |

---

## 4. Bestehende Implementierung

In `frontend/charts/candlestick.py` → `_detect_dow_swings()` existiert bereits:

- Swing-Erkennung mit `order=5`
- Deduplizierung konsekutiver gleicher Swing-Typen
- HH/HL/LH/LL-Klassifikation
- Sequenz-Zählung (3× konsekutiv → SQ Long/Short Signal)
- Zigzag-Visualisierung + S/R-Linien

**Problem:** Aktuell nur auf Daily-Daten angewandt. Kein Multi-Timeframe.

---

## 5. Implementierungsplan

### Phase 1: Swing-Detektor als eigenständiges Modul extrahieren

**Was:** `_detect_dow_swings()` aus `candlestick.py` in ein eigenes Modul verschieben, da es ein **Analyse-Algorithmus** ist, kein Chart-spezifischer Code.

**Wohin:** `backend/core/features/market_structure.py`

```
backend/core/features/market_structure.py
├── detect_swings(ohlcv, order) → DataFrame[date, price, type, label]
├── classify_structure(swings) → DataFrame[..., trend, bos, choch]
└── multi_tf_structure(data_dict: {tf: ohlcv}) → dict[tf, DataFrame]
```

### Phase 2: Multi-Timeframe Datenbeschaffung

**Was:** Intraday-Daten (5-Min, 15-Min) über Yahoo Finance laden.

**Einschränkung:** Yahoo Finance liefert Intraday-Daten nur für die letzten **60 Tage** (1-Min/5-Min) bzw. **730 Tage** (15-Min/1h). Für längere Historie bräuchte man Polygon.io, Alpha Vantage etc.

**Änderungen:**
- `params.yaml`: Neue Timeframe-Config hinzufügen
- `fetch_data.py`: Optional Intraday-Daten laden

```yaml
# params.yaml (Erweiterung)
stock:
  symbol: "^GSPC"
  timeframes:
    - interval: "1d"
      order: 5        # Swing-Sensitivität
      lookback: "max"
    - interval: "15m"
      order: 6
      lookback: "60d"
    - interval: "5m"
      order: 3
      lookback: "60d"
```

### Phase 3: Chart-Integration

**Was:** Multi-Timeframe Marktstruktur im Chart visualisieren.

**Optionen:**

| Option | Beschreibung | Komplexität |
|--------|-------------|-------------|
| A) Subplots | Separater Chart pro Timeframe | Niedrig |
| B) Overlay | Alle Timeframes im Daily-Chart (verschiedene Linienstärken) | Mittel |
| C) Dashboard | Streamlit-Tabs pro Timeframe + Summary-Panel | Mittel |

**Empfehlung:** Option C — Tabs pro Timeframe mit einem zusammenfassendem Status-Panel oben:

```
┌─────────────────────────────────────────┐
│  Market Structure:  Daily ▲  15m ▲  5m ▼ │
│  Status: Bullish — warte auf 5m CHoCH    │
└─────────────────────────────────────────┘
[Daily] [15-Min] [5-Min]   ← Tabs
```

### Phase 4: Zusätzliche Metriken (optional)

- **Break of Structure (BOS):** Markierung wo ein Swing-Level gebrochen wird
- **Change of Character (CHoCH):** Erster Bruch gegen den Trend → Frühwarnung
- **Order Blocks:** Demand/Supply Zones basierend auf Swing-Punkten
- **Fair Value Gaps (FVG):** Three-candle imbalance zones

### Phase 5: Als ML-Feature

Die Marktstruktur kann als Feature in das Modell einfließen:

| Feature | Typ | Beschreibung |
|---------|-----|-------------|
| `structure_trend` | Categorical | up / down / neutral |
| `swing_distance_high` | Float | Abstand des Close zum letzten Swing-High (%) |
| `swing_distance_low` | Float | Abstand des Close zum letzten Swing-Low (%) |
| `consecutive_bullish` | Int | Anzahl konsekutiver HH/HL |
| `consecutive_bearish` | Int | Anzahl konsekutiver LH/LL |
| `days_since_choch` | Int | Tage seit letztem Trendwechsel |

---

## 6. Empfohlene Reihenfolge

1. **Phase 1** zuerst — saubere Extraktion des Algorithmus
2. **Phase 3** (Chart) parallel — sofortiger visueller Nutzen
3. **Phase 2** (Intraday-Daten) — nur wenn Intraday-Trading relevant ist
4. **Phase 5** (ML-Features) — wenn Modell-Performance verbessert werden soll
5. **Phase 4** (BOS/CHoCH etc.) — Bonus, wenn Grundsystem steht

---

## 7. Wichtige Design-Entscheidung: `order`-Parameter pro Timeframe

Der `order` muss pro Timeframe angepasst werden, weil die **Noise-Charakteristik** sich ändert:

$$\text{SNR} \propto \sqrt{\Delta t}$$

(Signal-to-Noise Ratio skaliert mit der Wurzel des Zeitintervalls — aus der Random-Walk-Theorie)

| Timeframe | Empfohlener `order` | Begründung |
|-----------|-------------------|------------|
| 5-Min | 3–5 | Hohes Rauschen, braucht enge Fenster |
| 15-Min | 5–8 | Moderate Noise-Reduktion |
| Daily | 5–15 | Weniger Rauschen, größere Swings |
| Weekly | 3–5 | Wenige Datenpunkte, kleines Fenster nötig |

> Der `order` kann auch adaptiv gewählt werden: z.B. basierend auf ATR-Regime (hohe Vola → kleinerer order, niedrige Vola → größerer order).

---

## 8. Zusammenfassung

| Frage | Antwort |
|-------|---------|
| Muss man pro Timeframe separat berechnen? | **Ja** — verschiedene OHLC-Daten, verschiedene Swing-Punkte |
| Kann man einen einzigen `order` verwenden? | **Nein** — muss pro Timeframe kalibriert werden |
| Gibt es wissenschaftliche Basis? | **Ja** — Dow Theory, Fractal Markets, Lo et al. (2000) |
| Was existiert schon im Code? | Swing-Detektor + Zigzag im Daily-Chart |
| Nächster Schritt? | Algorithmus extrahieren → `market_structure.py` |
