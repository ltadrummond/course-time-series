# Final assessment

Do not consult the course or solutions. Suggested passing score: **75/100**, with no critical-error cap triggered.

# Part A — theory (60 minutes, 70 points)

## Short answers (2 points each; 20 total)

1. Define forecast origin and horizon.
2. Why is a random split usually invalid for forecasting?
3. Distinguish seasonality from a cycle.
4. What does weak stationarity require?
5. State the null hypothesis of the ADF test.
6. What does ARIMA(2,1,1) mean?
7. What does MASE = 0.8 mean, assuming the denominator uses seasonal naive?
8. Which functional minimizes expected MAE: conditional mean or median?
9. Distinguish a prediction interval from a confidence interval for the mean.
10. What does a rejected Ljung–Box test on forecast residuals suggest?

## Explain and diagnose (5 points each; 30 total)

11. A daily model has excellent random-CV RMSE but fails in production. Give four time-series-specific explanations and a correct evaluation redesign.
12. A feature is `demand.rolling(28).mean()` for predicting same-row demand. Diagnose and correct it. What changes for a 14-day fixed-origin forecast?
13. Compare recursive, direct, and multi-output forecasting. Give one benefit and one risk of each.
14. Compare ETS, SARIMA, boosted trees, and deep learning for one three-year daily series with weekly seasonality and two planned covariates. Recommend an experiment order.
15. A 95% interval covers only 72% of validation outcomes, mostly missing high values. Interpret this and give three remedies.
16. Explain why realized weather should not be used to backtest forecasts that use weather predictions. What data should be used instead?

## Quantitative reasoning (10 points each; 20 total)

17. Actuals are `[0, 10, 20, 30]`, forecasts `[2, 8, 25, 25]`. Compute MAE, RMSE, bias defined as mean(forecast − actual), and explain why MAPE is unsuitable. If in-sample seasonal-naive absolute errors average 5, compute MASE.
18. A daily training series ends Sunday with the last 7 values (Mon→Sun) `[10,12,11,13,18,20,15]`. Give the 10-day seasonal-naive forecast. Then explain which true target lags are available at Sunday origin for a direct 10-day model.

# Part B — practical test (90 minutes, 30 points)

Using `make_data.build()`:

1. Reserve the final 60 rows. Construct four 14-day rolling-origin validation folds on the preceding history. (5)
2. Create a weekly seasonal-naive baseline and compute MAE, MASE, bias, and per-horizon MAE. (5)
3. Create leakage-safe target lags/rolling features and known-future calendar/business features for a fixed 14-day origin. Document illegal features. (7)
4. Compare Ridge and one nonlinear/statistical model across identical folds. Select without using test. (7)
5. Evaluate once on test, inspect residual autocorrelation or interval coverage, and write a five-sentence deployment recommendation. (6)

## Critical-error caps

- Random/shuffled validation: maximum total 55.
- Realized future target or temperature leakage: maximum 55.
- No naive baseline: maximum 70.
- Model selection on final test: maximum 70.
