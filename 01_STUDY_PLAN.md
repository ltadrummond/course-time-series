# Ten-hour study plan

The target is competence under interview pressure, not encyclopedic coverage. Every module uses the cycle **learn → recall → code → explain**.

| Time | Module | Outcome |
|---:|---|---|
| 0:00–0:50 | 1. Forecasting frame + EDA | Define the problem correctly; recognize trend, seasonality, anomalies, and leakage |
| 0:50–1:40 | 2. Time index + manipulation | Resample, align, impute, transform, and distinguish observation time from availability time |
| 1:40–2:35 | 3. Baselines + evaluation | Choose metrics and perform rolling-origin validation |
| 2:35–3:40 | 4. Regression + feature engineering | Build leakage-safe lag, rolling, calendar, and Fourier features |
| 3:40–4:50 | 5. Statistical forecasting | Explain and use ETS, ARIMA/SARIMA, stationarity, ACF/PACF, and intervals |
| 4:50–5:50 | 6. ML + multivariate series | Use trees/direct forecasting and reason about exogenous variables, panels, and hierarchy |
| 5:50–6:40 | 7. Deep learning | Understand windows, recursive/direct/multi-output prediction, RNNs, TCNs, Transformers, and when they pay off |
| 6:40–7:10 | 8. Production + diagnosis | Monitor accuracy, bias, drift, interval coverage, and retraining |
| 7:10–8:35 | Capstone | Complete an end-to-end forecast and defend choices |
| 8:35–9:35 | Theory test | Closed-book assessment |
| 9:35–10:00 | Review | Grade, revisit weak flashcards, rehearse a five-minute solution narrative |

## Rules for the sprint

- Put solutions away until you have written an answer or code attempt.
- For every model, be able to state: assumptions, fit objective, forecast strategy, validation design, and failure mode.
- Never report a model score without a naive baseline (a simple reference forecast, such as predicting that the next value will equal the most recent value) and the forecast horizon.
- If time slips, prioritize Modules 1–5 and the capstone. Deep-learning architecture details are lower yield.

## Readiness gates

You are interview-ready when you can do all of these without notes:

1. Explain why a random train/test split leaks future information.
2. Construct lag and rolling features without including the target being predicted.
3. Write naive and seasonal-naive forecasts.
4. Contrast expanding versus sliding windows and recursive versus direct multi-step forecasting.
5. Explain stationarity, differencing, ACF, ARIMA orders, and ETS components.
6. Select MAE/RMSE/MAPE/MASE for a stated business setting.
7. Describe a valid prediction interval and check its empirical coverage.
8. Diagnose residual autocorrelation and forecast bias.
