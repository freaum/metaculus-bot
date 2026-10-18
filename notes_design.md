# Diseño MoneyBot (sobre metac-bot-template, Fall 2026)
- Ensemble: FORECAST_MODELS (var de GitHub Actions, lista LiteLLM/OpenRouter) rotando por predicción; 6 predicciones por pregunta.
- Binario: prompt outside-view; agregación = media recortada de log-odds x EXTREMIZE (1.15), clamp 2-98%.
- MC/numérico/fecha: agregación base de forecasting-tools (aggregation.py tiene versiones geométrica/cuantiles listas, aún no enchufadas).
- Pendiente: Market Pulse (re-pronosticar preguntas abiertas), research con AskNews cuando haya key, backtest real.
