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
