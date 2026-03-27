# Chart Overlays

## Candlestick Chart
- OHLC-Kerzen (grün/rot)
- ML-Prediction-Marker: ▲ UP (grün) / ▼ DOWN (rot) mit Konfidenz-Tooltip

## Line Chart
- Close-Preis mit blauem Fill-Bereich
- 20d Moving Average (weiß, gepunktet)
- **ML-Predictions**: ▲/▼ Dreiecke (grün/rot)
- **2× ATR Days**: Dreiecke mit gelbem Rand — markiert Tage mit Range ≥ 2× ATR(14), grün=Up / rot=Down
- **Dow Theory Zigzag**: Swing-Highs/-Lows verbunden, farbcodiert nach Trend (grün=up, rot=down)
  - Labels: HH/HL (grün), LH/LL (rot) + laufende Sequenznummer
  - Horizontale S/R-Linien an jedem Swing-Punkt
  - **SQ Long / SQ Short**: Signal bei Punkt 3 einer Sequenz (3× konsekutive HH/HL bzw. LH/LL)
  - **Stop-Loss**: Gestrichelte Linie am letzten signifikanten Swing-High
- **Market Structure Signals (BOS / CHoCH)**:
  - **CHoCH (Change of Character)**: Cyan/Orange Stern ⭐ — Trendwechsel-Signal
    - ⚡ LONG: Abwärtstrend gebrochen → potentieller Einstieg Long
    - ⚡ SHORT: Aufwärtstrend gebrochen → potentieller Einstieg Short / Ausstieg Long
    - Gestrichelte Level-Linie zeigt das gebrochene Niveau
    - Vertikale gepunktete Verbindungslinie zum Breakpoint
    - Annotation-Box mit Signal-Label
  - **BOS (Break of Structure)**: Kleine Diamanten — Trendbestätigung
    - BOS ▲ (grün): Aufwärtstrend bestätigt (Hold/Add Long)
    - BOS ▼ (rot): Abwärtstrend bestätigt (Hold/Add Short)
