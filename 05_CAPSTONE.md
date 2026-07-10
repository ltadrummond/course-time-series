# Capstone — retail demand forecasting

**Time box:** 85 minutes. **Deliverable:** notebook/script plus a five-minute spoken defense. Generate data with `python make_data.py`.

## Scenario

At the end of each Sunday, the inventory team needs daily demand forecasts for the next 14 days. Price, promotions, and holidays are planned 14 days ahead. Temperature is only observed in real time; archived weather forecasts are unavailable. The cost of a stockout is thought to exceed excess inventory cost, but the exact ratio is not yet agreed.

The file `data/retail_daily.csv` contains 900 daily observations and a small number of collection failures. There may be a structural change. Your stakeholders need a reliable first system, not a research benchmark.

## Tasks

1. **Contract (5 min):** Write target, origin, horizon, cadence, legal features, primary metric, and assumptions.
2. **Audit/EDA (10 min):** Check index integrity, missingness, target distribution, weekly/annual patterns, changing level/variance, and potential break. Include no more than four useful plots.
3. **Evaluation (10 min):** Reserve the final 60 days as untouched test. On earlier data, define at least four expanding or sliding rolling-origin folds, each with a 14-day horizon and Sunday origins. Justify window length and metrics.
4. **Baselines (10 min):** Implement naive and weekly seasonal-naive forecasts. Report MAE, RMSE, MASE, bias, and MAE by horizon.
5. **Models (25 min):** Fit at least two candidates from different families: Ridge with leakage-safe features; ETS/SARIMA; or a tree ensemble. All transformations must be trained per fold. Do not use realized future temperature.
6. **Selection/uncertainty (10 min):** Select using fold distribution, not test. Produce a reasonable 80%/95% interval or explain a concrete calibration plan. Evaluate coverage if intervals are implemented.
7. **Final (10 min):** Evaluate exactly once on the final test using a deployment-faithful simulation. Discuss errors by lead time and around the break.
8. **Delivery (5 min):** Recommend model, fallback, monitoring, retraining trigger, and next experiment.

## Critical traps intentionally present

- Missing target is not automatically zero.
- Realized temperature over the horizon is unavailable.
- A lag feature evaluated row-by-row can silently simulate daily updates instead of a fixed 14-day forecast.
- The test must not choose hyperparameters or reveal the break before selection.
- A strong seasonal naive baseline may be hard to beat at some horizons.

## Rubric (100)

| Area | Points | Full-credit evidence |
|---|---:|---|
| Forecast contract | 10 | Exact origin/horizon/cadence/availability/loss |
| Data audit | 10 | Index and missing semantics; concise structure/break evidence |
| Validation | 20 | Sunday rolling origins; 14-day blocks; transformations inside fold; untouched test |
| Baselines/metrics | 10 | Both baselines; correct MASE; horizon and bias views |
| Features/models | 20 | Past-only features; legal covariates; reproducible candidates |
| Multi-step correctness | 10 | Recursive/direct logic matches feature availability |
| Diagnostics/uncertainty | 10 | Fold stability, residuals, calibrated coverage/plan |
| Recommendation | 10 | Accuracy/complexity tradeoff, fallback, monitoring, limitations |

Automatic cap: any random split or use of realized future temperature limits the score to 55. Looking at the final test repeatedly limits it to 70.

## Worked solution outline

A strong answer uses Sunday origins ending before the last 60 days, with 14 validation days. It imputes historical collection failures from past-only values (or drops affected labels) and carries a missing flag. It starts with lag-7 seasonal naive, then compares:

- Ridge direct models (h=1,…,14) using lag 14/21/28, origin-time rolling summaries, day-of-week/Fourier/trend, and known future price/promotion/holiday; or a carefully updated recursive Ridge.
- Damped additive ETS with weekly seasonality, fit separately at each origin.
- Optionally a histogram gradient booster using the same direct design.

Temperature is excluded because archived forecasts are absent. A direct design creates one training row per historical origin/horizon; target is `demand.shift(-h)`, while all target-derived features stop at the origin and known-future covariates are aligned to origin + h. Model selection uses median/mean fold MASE plus worst fold, bias, and per-lead error. A simple residual-bootstrap interval samples 14-step residual vectors by historical origin (preserving cross-horizon dependence); quantiles give 80%/95% bounds. Coverage is checked only on validation, then once on test.

If Ridge/ETS has only a small unstable advantage, recommend seasonal naive as fallback and the simplest repeatable winner as champion. Monitor missing inputs, demand bias, MASE against live lag-7 naive, error by lead, and interval coverage. Retrain weekly or after sustained baseline-relative degradation, with a shadow backtest before promotion.

### Direct-design sketch

```python
rows = []
for origin in eligible_origins:
    history = y.loc[:origin]
    base = {"lag14": history.iloc[-14], "r28": history.iloc[-28:].mean()}
    for h in range(1, 15):
        future_date = origin + pd.Timedelta(days=h)
        rows.append({**base, "h": h, "dow": future_date.dayofweek,
                     "price": df.loc[future_date, "price"],
                     "promotion": df.loc[future_date, "promotion"],
                     "target": df.loc[future_date, "demand"]})
design = pd.DataFrame(rows)
```

Production code should add richer lags/calendar terms, entity-safe alignment, fold pipelines, assertions that every target date follows its origin, and feature-availability tests.
