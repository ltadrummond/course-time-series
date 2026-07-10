# Worked solutions

These are reference solutions, not the only valid answers. Score reasoning about information availability and validation more heavily than exact model choices.

# Chapter checkpoints

## 1. Framing and EDA

**Specification:** “At 22:00 each day, forecast each open store’s daily unit demand for each of the next 14 calendar days, refreshed daily, using transactions through 21:59 plus prices/promotions/holidays already planned for the horizon. Optimize revenue-weighted MAE, report under-forecast bias, and exclude closures from the primary latent-demand score.” This fixes unit, target, origin, horizon, cadence, available information, and loss.

1. Trend or an unremoved slow-moving level makes adjacent observations similar and creates high lag-1 ACF; that alone does not establish a stationary AR(1).
2. If “sales realized” is the target, a confirmed closed day can be zero. If estimating unconstrained demand, it is unobserved (`NaN`). State the estimand.
3. Tomorrow’s measured temperature is illegal tonight; a weather *forecast issued tonight* is legal and must be backtested using archived forecasts, not realized weather.
4. Constant weekend uplift suggests additive seasonality; seasonal swings growing in proportion to level suggest multiplicative.

## 2. Manipulation

Energy consumed is summed; temperature averaged (possibly time-weighted); tariff averaged or duration-weighted; machine status should become uptime fraction plus closing status. Status and tariff can be forward-filled if contractual/persistent and already known; energy and temperature generally cannot. Add `col_missing = col.isna().astype(int)` before a training-fold-fitted imputation. Use `np.log1p(x)`, `x.diff()`, `x.diff(7)`. A DST day has 23/25 local hours: a raw sum reflects actual duration, while an unweighted average and naive comparison with 24-hour days can mislead. Use timezone-aware indices and define business meaning.

## 3. Validation and metrics

One valid index design for 730 rows is:

```text
fold 1 train [0, 599], validate [600, 613]
fold 2 train [0, 629], validate [630, 643]
fold 3 train [0, 659], validate [660, 673]
final untouched test [674, 729]
```

It matches a 14-day horizon and roughly monthly origins; more folds are preferable if compute allows. Naive repeats `train[-1]`; seasonal naive uses the corresponding last-week values, tiled for two weeks. Metrics are in the Chapter 3 cheat sheet; per-horizon MAE pools errors at lead (h) across origins. MAPE divides by zero. After a policy break, a sliding window or post-break expanding window is defensible; compare both without using final-test outcomes.

## 4. Regression

```python
d = df.sort_index().copy()
for lag in [1, 7, 14, 28]:
    d[f"lag_{lag}"] = d.demand.shift(lag)
for w in [7, 28]:
    d[f"r{w}_mean"] = d.demand.shift(1).rolling(w).mean()
    d[f"r{w}_std"] = d.demand.shift(1).rolling(w).std()
t = np.arange(len(d))
d["trend"] = t
d["sin7"] = np.sin(2*np.pi*t/7); d["cos7"] = np.cos(2*np.pi*t/7)
d = pd.get_dummies(d.assign(dow=d.index.dayofweek), columns=["dow"])
```

At 14 days, calendar/trend/Fourier, planned promotion and price are available. Target lag 14/28 is available throughout, but lag 1/7 for later horizons partly refers to unknown future targets. Rolling features also need recursive updates or must be frozen/redefined at the origin. Measured temperature is unavailable; an archived forecast can be used. Fit scaling/Ridge separately per origin. Use direct horizon-specific models to avoid predicted lags, or recursively update lag buffers. Residual ACF means temporal information remains.

## 5. Statistical models

Candidate set: seasonal naive; damped additive Holt–Winters with period 7; SARIMA such as `(1,1,1)(1,0,1,7)` and `(0,1,1)(0,1,1,7)`. These are hypotheses, selected by rolling performance—not asserted truths. In ((p,d,q)(P,D,Q)_7), lowercase terms are nonseasonal AR order, differences, MA order; uppercase are seasonal equivalents at multiples of seven. Compute 95% coverage as `mean((y >= lo) & (y <= hi))` and width as `mean(hi-lo)`, both by horizon. Ljung–Box p < chosen alpha rejects residual white noise at the tested lag set. Increasing variance suggests log/Box–Cox, multiplicative ETS, heteroskedastic likelihood, or calibrated non-Gaussian intervals.

## 6. ML and panels

Sort by `sku, store, date`; create every lag/rolling feature with `groupby(["sku","store"])`; generate validation rows only after per-origin history cutoff. A global model is usually better for sparse SKUs because it pools information. Use SKU/store embeddings or regularized encodings fitted inside folds; metadata/category enables unseen-SKU prediction, while an arbitrary unseen ID does not. Compare all candidates at identical origins. Macro MAE averages SKU-level MAEs; revenue-weighted WAPE weights business volume; additionally show lead-time and tail results. Bottom-up summing is coherent; reconciliation can improve forecasts when aggregate models carry extra signal.

## 7. Deep learning

For batch (B): historical target `B×168×1`; historical known/observed features `B×168×F_h`; future known weather-forecast/calendar features `B×24×F_f`; metadata `B×F_s`; output quantiles `B×24×3`. Add missingness/availability masks. A direct multi-output TCN is a strong first choice: parallel, stable, and its dilations cover a week; use LSTM/Transformer only if validation supports them. Sum or average pinball loss for q=.1/.5/.9. Split on forecast origins, fit scalers on training only, and prohibit training targets/windows crossing boundaries. Early-stop on multi-origin validation pinball/MASE and compare seasonal-naive (lag 24/168), Ridge/boosting, and ETS. DL can lose through insufficient history, regime drift, weak cross-series sharing, optimization error, or because seasonality already solves the task.

## 8. Production

First verify target/feature pipeline freshness, units, timestamp alignment, and feature availability; then localize bias by origin, lead, entity, and regime; compare the live baseline; inspect shifts/breaks. Immediate mitigation: alert, fall back to seasonal naive or bias-correct within a pre-approved guardrail. Longer term: reproduce incident data, add it to rolling validation, fix pipeline or retrain/reselect, shadow-test, then release. A stakeholder summary should state impact, affected scope, current mitigation, evidence-based cause/status, next action, owner, and timing—without claiming a cause before verification.

# Runnable lab solutions

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

df = build().set_index("date").sort_index()

# Lab 1
audit = {
    "inferred_freq": pd.infer_freq(df.index),
    "duplicates": df.index.duplicated().sum(),
    "missing": df.isna().sum().to_dict(),
}
regular = df.asfreq("D")
by_dow = regular.groupby(regular.index.dayofweek).demand.mean()

# Lab 2. Forward-fill target collection gaps strictly from the past for features.
feat = regular.copy()
hist_y = feat.demand.ffill()
for lag in [1, 7, 14, 28]: feat[f"lag_{lag}"] = hist_y.shift(lag)
for w in [7, 28]: feat[f"roll{w}_mean"] = hist_y.shift(1).rolling(w).mean()
feat["roll28_std"] = hist_y.shift(1).rolling(28).std()
feat["trend"] = np.arange(len(feat))
feat["sin7"] = np.sin(2*np.pi*feat.trend/7)
feat["cos7"] = np.cos(2*np.pi*feat.trend/7)
feat = pd.get_dummies(feat.assign(dow=feat.index.dayofweek), columns=["dow"], dtype=float)

# Lab 3
train, test = feat.iloc[:-90], feat.iloc[-90:]
seasonal_pred = hist_y.shift(7).loc[test.index].to_numpy()

# Lab 4
def metrics(y, pred, insample, m=7):
    y, pred, insample = map(np.asarray, (y, pred, insample))
    ok = np.isfinite(y) & np.isfinite(pred)
    err = pred[ok] - y[ok]
    scale = np.nanmean(np.abs(insample[m:] - insample[:-m]))
    return {"mae": np.mean(abs(err)), "rmse": np.sqrt(np.mean(err**2)),
            "bias": np.mean(err), "mase": np.mean(abs(err))/scale}

# Lab 5 (historical one-step-style evaluation, not a true 90-step origin simulation)
features = [c for c in feat if c.startswith(("lag_", "roll", "dow_"))]
features += ["trend", "sin7", "cos7", "price", "promotion", "holiday"]
tr = train.dropna(subset=features + ["demand"])
te = test.dropna(subset=features)
model = Ridge(alpha=10).fit(tr[features], tr.demand)
ridge_pred = pd.Series(model.predict(te[features]), index=te.index)

# Lab 6
def rolling_origins(n, initial=600, horizon=14, step=30):
    for origin in range(initial, n-horizon+1, step):
        yield np.arange(origin), np.arange(origin, origin+horizon)
```

The Lab 5 score uses lags observed within the test period, so it answers repeated one-step forecasting with daily updates, not a single 90-day-ahead forecast. This distinction is part of the exercise.
