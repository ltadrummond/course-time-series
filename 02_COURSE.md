# Time Series Interview Sprint

Notation: (y_t) is the target at time (t); (h) is forecast horizon; \(\hat y_{t+h|t}\) is a forecast made using information available through (t). Code assumes `pandas as pd` and `numpy as np`.

---

# 1. Frame the problem and explore the series

## 1.1 Forecasting is a supervised problem with an information boundary

Before touching a model, define:

- **Unit:** one value per store-day, machine-minute, etc.
- **Target:** exactly what value is predicted.
- **Forecast origin:** last time at which information is available.
- **Horizon:** one step, next 24 hours, next 12 weeks.
- **Cadence:** how often forecasts are regenerated.
- **Availability:** when each feature becomes known, not merely when it is timestamped.
- **Decision/loss:** why errors matter. Under-forecasting stock can cost more than over-forecasting.

A covariate recorded for tomorrow is not necessarily known today. Planned price may be known; observed weather is not. This distinction is the central defense against leakage.

## 1.2 Components and dependence

A useful descriptive decomposition is

\[
y_t = T_t + S_t + R_t \quad\text{(additive)},
\]

or (y_t=T_tS_tR_t) when seasonal amplitude grows with the level. A log transform turns a positive multiplicative relationship approximately additive:

\[
\log y_t=\log T_t+\log S_t+\log R_t.
\]

- **Trend:** slow change in level.
- **Seasonality:** repeated pattern at a known/calendar frequency.
- **Cycle:** irregular longer movement, not fixed-period seasonality.
- **Noise:** unpredictable component, possibly with changing variance.
- **Structural break:** mechanism changes; old data may become harmful.

Time series violate the IID assumption because observations can be dependent. Autocovariance and autocorrelation at lag (k) are

\[
\gamma_k=\operatorname{Cov}(y_t,y_{t-k}),\qquad
\rho_k=\gamma_k/\gamma_0.
\]

Sample ACF estimates ρ across lags. Large lag-7 ACF for daily data suggests weekly repetition, but ACF is descriptive—not proof of causality.

## 1.3 Minimal EDA protocol

1. Sort by time; check duplicates, timezone, frequency, gaps, and target validity.
2. Plot the entire history, recent window, and distributions by calendar group.
3. Compare rolling mean/standard deviation; inspect seasonal plots.
4. Plot ACF of levels and, if trending, differences.
5. Mark interventions, outages, promotions, and known breaks.
6. Split chronologically *before* allowing test-period discoveries to shape modeling.

```python
df = df.sort_values("timestamp").set_index("timestamp")
assert df.index.is_monotonic_increasing
print(pd.infer_freq(df.index), df.index.duplicated().sum())

full = df["y"].asfreq("D")             # exposes missing dates
stats = full.rolling(28).agg(["mean", "std"])
by_dow = df.assign(dow=df.index.dayofweek).groupby("dow").y.agg(["mean", "median", "count"])

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(full.diff().dropna(), lags=35)
```

Correlation can be spurious when both series trend. Examine differences, domain timing, and out-of-sample incremental value.

## Checkpoint 1

A retailer asks: “Predict sales.” Write a precise forecasting specification. Then inspect a daily dataframe and answer:

1. Why might its lag-1 ACF be 0.95 even without a useful autoregressive mechanism?
2. Missing sales on a closed day: zero or `NaN`?
3. Is tomorrow’s measured temperature a legal feature for a forecast issued tonight?
4. Give one additive and one multiplicative clue.

### Flashcards

- **Q:** Seasonality versus cycle? **A:** Seasonality has a stable, known frequency; cycles have variable duration.
- **Q:** What is forecast origin? **A:** The information cutoff from which a forecast is issued.
- **Q:** Timestamp versus availability time? **A:** When an event concerns versus when the predictor actually knows it.
- **Q:** Why difference before ACF? **A:** Trend creates persistent autocorrelation that can obscure short-memory dynamics.
- **Q:** Additive versus multiplicative seasonality? **A:** Constant absolute seasonal amplitude versus amplitude proportional to level.

### Cheat sheet

```text
Frame = unit + target + origin + horizon + cadence + available inputs + loss
EDA = index integrity -> full/recent plots -> calendar groups -> rolling stats -> ACF -> breaks
Red flags = shuffled split, revised data, future rolling window, test-informed cleaning
```

---

# 2. Indexing, resampling, missingness, and transformations

## 2.1 The time index is part of the data-generating process

Use timezone-aware timestamps when daylight-saving transitions matter. Never silently assume all days have 24 hourly observations. For multiple entities, `(entity, timestamp)` must be unique.

**Resampling** changes frequency and requires a domain aggregation:

```python
daily = hourly.resample("D").agg(
    energy=("kwh", "sum"),             # a flow
    temperature=("temp", "mean"),      # an average state
    closing_inventory=("stock", "last")
)
```

`asfreq` reveals/reindexes timestamps without aggregating. `resample` groups intervals. Be explicit about interval closure and labels in financial or sensor settings.

## 2.2 Missing values are events, not merely holes

Classify a gap:

- **Structural zero:** store was open and sold none → often `0`.
- **Not applicable:** store closed → perhaps zero for demand fulfilled, but missing for latent demand.
- **Collection failure:** unknown → `NaN`, possibly impute and add a missingness flag.
- **Irregular sampling:** model elapsed time or regularize carefully.

Forward fill is legal only if the variable is a state that persists and the value was already known. Interpolation using the future is illegal for live forecasting unless done only inside historical training and the same information would exist operationally. Fit imputers separately within every training fold.

## 2.3 Variance stabilization and differencing

For positive (y), `log1p` reduces right skew and stabilizes proportional variance. Box–Cox generalizes it. Transform forecasts back carefully: `exp(E[log y])` estimates a median, not generally the mean (Jensen’s inequality). A residual-variance correction under Gaussian log errors is approximately

\[
E[Y]\approx \exp(\hat\mu + \hat\sigma^2/2).
\]

First and seasonal differences are

\[
\nabla y_t=y_t-y_{t-1},\qquad \nabla_m y_t=y_t-y_{t-m}.
\]

Differencing can remove stochastic trend/seasonality; over-differencing introduces noise and negative autocorrelation. Save initial values to invert forecasts.

## Checkpoint 2

Given hourly energy, temperature, tariff, and machine-status data:

1. Produce daily aggregations and justify each function.
2. Identify which columns may be forward-filled.
3. Add missingness indicators before imputation.
4. Create `log1p_energy`, first difference, and lag-7 seasonal difference.
5. Explain how daylight-saving time can corrupt a daily mean or sum.

### Flashcards

- **Q:** `asfreq` versus `resample`? **A:** Reindex to a frequency versus aggregate into time bins.
- **Q:** Why add missing indicator? **A:** Missingness itself may carry predictive information.
- **Q:** When is forward fill meaningful? **A:** For a persistent state whose last value was known at forecast time.
- **Q:** Why can exponentiating a log forecast be biased? **A:** A nonlinear inverse does not commute with expectation.
- **Q:** Risk of over-differencing? **A:** Inflated variance and artificial negative autocorrelation.

### Cheat sheet

```python
s.asfreq("D")
s.resample("W-MON", label="left", closed="left").sum()
s.shift(1)                 # previous observation; not index movement
s.diff(1); s.diff(7)
np.log1p(s); np.expm1(z)
```

---

# 3. Baselines, metrics, and temporal validation

## 3.1 Baselines

A model is useful only relative to a credible low-complexity alternative.

- Mean: \(\hat y_{T+h}=\bar y\)
- Naive/random walk: \(\hat y_{T+h}=y_T\)
- Drift: \(\hat y_{T+h}=y_T+h(y_T-y_1)/(T-1)\)
- Seasonal naive: \(\hat y_{T+h}=y_{T+h-m(k+1)}\), equivalently repeat the latest season.

For daily retail data, lag-7 seasonal naive is usually the minimum bar. Compare against a business incumbent too.

## 3.2 Metrics encode decisions

For errors (e_i=y_i-\hat y_i):

\[
MAE=\frac1n\sum|e_i|,\quad RMSE=\sqrt{\frac1n\sum e_i^2},\quad
MAPE=\frac{100}{n}\sum\left|\frac{e_i}{y_i}\right|.
\]

- **MAE:** interpretable, robust; optimized by conditional median.
- **MSE/RMSE:** punishes large misses; MSE optimized by conditional mean.
- **MAPE:** scale-free but undefined/explosive near zero and asymmetric.
- **WAPE:** \(\sum|e|/\sum|y|\); portfolio-friendly but aggregation-dependent.
- **MASE:** MAE divided by in-sample naive MAE; scale-free and valid with zeros. For season (m), denominator is \(\frac1{n-m}\sum_{t=m+1}^n|y_t-y_{t-m}|\). MASE < 1 beats that baseline.
- **Pinball loss** for quantile (q): \(L_q(y,\hat y)=q(y-\hat y)\) if (y\ge\hat y), else \((1-q)(\hat y-y)\).

Aggregate metrics can hide horizon, entity, or regime failures. Report by horizon and important segment. Measure signed mean error for bias.

## 3.3 Walk-forward validation

Random cross-validation leaks future regimes and destroys ordering. Use several forecast origins:

```text
fold 1: [ train -------- ][validate]
fold 2: [ train ---------------- ][validate]
fold 3: [ train ------------------------ ][validate]
```

- **Expanding window:** retains all history; good under stable mechanisms.
- **Sliding window:** retains recent history; useful under drift.
- **Gap/embargo:** space between train and validation when labels/features overlap or arrive late.
- Match validation horizon and refit cadence to deployment.
- Perform imputation, scaling, selection, feature fitting, and tuning inside each training fold.
- Keep a final untouched chronological test set.

Uncertainty across origins matters: a tiny average gain with unstable fold performance is weak evidence.

## Checkpoint 3

You have 730 daily observations, weekly seasonality, a 14-day business horizon, and monthly retraining.

1. Design three folds with exact train/validation dates or indices.
2. Implement naive and seasonal-naive forecasts.
3. Compute MAE, RMSE, MASE, bias, and per-horizon MAE.
4. Explain why MAPE fails if some products have zero sales.
5. Choose expanding or sliding validation after a pricing-policy break.

### Flashcards

- **Q:** Why multiple origins? **A:** One split confounds performance with one regime and gives a noisy estimate.
- **Q:** What does MASE < 1 mean? **A:** Lower MAE than the chosen in-sample naive baseline.
- **Q:** Which point forecast minimizes MAE/MSE? **A:** Conditional median/conditional mean.
- **Q:** Why evaluate by horizon? **A:** Error and model usefulness typically change with lead time.
- **Q:** When use an embargo? **A:** When training information overlaps or is not actually available before validation labels.

### Cheat sheet

```python
mae = np.mean(np.abs(y - pred))
rmse = np.sqrt(np.mean((y - pred)**2))
bias = np.mean(pred - y)
mase = mae / np.mean(np.abs(train[m:] - train[:-m]))
# Validation must mimic horizon, update frequency, feature availability, and latency.
```

---

# 4. Regression and leakage-safe feature engineering

## 4.1 From series to supervised table

A dynamic regression can be written

\[
y_t=\beta_0+\sum_{j\in L}\beta_jy_{t-j}+\gamma^Tx_t+\epsilon_t.
\]

OLS estimates \(\hat\beta=\arg\min_\beta\sum(y-X\beta)^2\). Autocorrelated residuals do not necessarily bias coefficients when regressors are exogenous, but make ordinary standard errors unreliable and signal missing temporal structure.

Useful features:

- Lags: 1, 2, 7, 14, 28.
- Rolling summaries of *past* values: mean/std/min/max/EWMA.
- Calendar: day of week, month, holiday, payday.
- Age/trend: elapsed time, piecewise trend.
- Fourier seasonality: for period (m), \(\sin(2\pi kt/m),\cos(2\pi kt/m)\), (k=1,\dots,K).
- Known-future inputs: planned price, promotion, holiday.
- Observed-only inputs: lag them or forecast them jointly.

Critical pattern:

```python
g = df.groupby("store", group_keys=False)
df["lag_7"] = g.y.shift(7)
df["roll28_mean"] = g.y.transform(lambda s: s.shift(1).rolling(28).mean())
df["ewm"] = g.y.transform(lambda s: s.shift(1).ewm(alpha=.2).mean())
```

`rolling(28).mean()` without `shift(1)` includes (y_t) when predicting (y_t): leakage.

Linear models offer extrapolation and interpretability. One-hot seasonal features capture arbitrary fixed seasonal effects; Fourier terms are compact and smooth. Ridge stabilizes correlated lag features; Lasso can select but is unstable among correlated predictors.

## 4.2 Multi-step strategies

- **Recursive:** train one-step model, feed predictions back. Cheap/coherent; errors accumulate and train/inference inputs differ.
- **Direct:** one model per horizon. No recursive accumulation; more models/data and forecasts may cross.
- **Multi-output:** predict all horizons jointly. Can learn cross-horizon structure; needs capable model and fixed horizon.
- **DirRec:** hybrid.

Features must match strategy. At origin (T), true (y_{T+1}) cannot be used as a lag when predicting (T+2) unless recursively replaced by its forecast.

## Checkpoint 4

Build a daily demand design matrix with lags 1/7/14/28, past-only rolling 7/28 mean and std, day-of-week, linear trend, two weekly Fourier harmonics, promotion, and price. Then:

1. Fit Ridge using walk-forward folds.
2. Compare with seasonal naive using MASE and per-horizon MAE.
3. List every feature available for a 14-day-ahead forecast.
4. Implement either recursive or direct forecasting and explain its tradeoff.
5. Inspect residual ACF and coefficients/permutation importance.

### Flashcards

- **Q:** Safe rolling feature? **A:** Shift first, then roll, within entity.
- **Q:** Fourier terms do what? **A:** Represent smooth periodic functions using sine/cosine pairs.
- **Q:** Recursive versus direct? **A:** Reuse one one-step model with feedback versus fit horizon-specific models.
- **Q:** Why Ridge for many lags? **A:** Shrinks unstable coefficients under multicollinearity.
- **Q:** Residual autocorrelation means? **A:** Predictable temporal structure remains or uncertainty assumptions are wrong.

### Cheat sheet

```python
for lag in [1, 7, 14, 28]: df[f"lag{lag}"] = df.y.shift(lag)
df["r28"] = df.y.shift(1).rolling(28).mean()
df["sin7"] = np.sin(2*np.pi*np.arange(len(df))/7)
# Pipeline preprocessing must be fit inside each fold.
```

---

# 5. ETS, stationarity, ARIMA, and uncertainty

## 5.1 Exponential smoothing / ETS

Simple exponential smoothing updates level:

\[
\ell_t=\alpha y_t+(1-\alpha)\ell_{t-1},\qquad \hat y_{t+h|t}=\ell_t.
\]

Holt adds trend; Holt–Winters adds seasonality. In additive Holt–Winters:

\[
\ell_t=\alpha(y_t-s_{t-m})+(1-\alpha)(\ell_{t-1}+b_{t-1}),
\]
\[
b_t=\beta(\ell_t-\ell_{t-1})+(1-\beta)b_{t-1},
\]
\[
s_t=\gamma(y_t-\ell_t)+(1-\gamma)s_{t-m}.
\]

Forecast: \(\hat y_{t+h|t}=\ell_t+hb_t+s_{t-m+(h\bmod m)}\). A damped trend replaces indefinite linear extrapolation with a decaying trend contribution. ETS is strong for smooth level/trend/seasonality and limited covariates.

## 5.2 Stationarity and ARIMA

Weak stationarity means constant mean and variance, and covariance depending only on lag. It matters because stable historical relationships can then estimate future dynamics. A deterministic trend can be regressed out; a stochastic unit-root trend often needs differencing.

AR((p)):

\[
y_t=c+\phi_1y_{t-1}+\cdots+\phi_py_{t-p}+\epsilon_t.
\]

MA((q)): (y_t=c+\epsilon_t+\theta_1\epsilon_{t-1}+\cdots+\theta_q\epsilon_{t-q}). ARIMA((p,d,q)) applies ARMA to (d)-times differenced data. SARIMA adds seasonal ((P,D,Q)_m).

Heuristics on a stationary series:

- AR((p)): PACF often cuts off near (p), ACF tails off.
- MA((q)): ACF often cuts off near (q), PACF tails off.
- Reality is noisy; use these to propose candidates, then validate.

ADF has a unit-root null; KPSS has a stationarity null. Tests have limited power and do not replace plots/domain reasoning. Avoid blindly differencing until a p-value passes.

AIC (=2k-2\log L) balances in-sample likelihood and parameter count; it is useful within a comparable model family, while rolling validation measures the actual forecasting task.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
fit = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,7),
              exog=X_train, enforce_stationarity=False).fit(disp=False)
fc = fit.get_forecast(steps=14, exog=X_future)
mean, interval = fc.predicted_mean, fc.conf_int(alpha=.05)
```

ARIMAX/SARIMAX assumes future exogenous values are available or separately forecast. Coefficients depend on differencing and parameterization; explain them carefully.

## 5.3 Residuals and intervals

Good residuals have near-zero mean, no remaining autocorrelation, stable variance, and no exploitable relation to available inputs. Ljung–Box tests the joint null that autocorrelations through selected lags are zero. Failure means the model/uncertainty is incomplete; passing does not prove correctness.

A 95% **prediction interval** should contain a future observation about 95% of the time under repeated use. It is wider than a confidence interval for a mean parameter. Validate coverage and average width by horizon. Gaussian intervals can fail with skew, outliers, or heteroskedasticity. Alternatives: bootstrap/simulation, quantile models, and conformal calibration under appropriate exchangeability/local-stability assumptions.

## Checkpoint 5

For a weekly seasonal daily series:

1. Fit seasonal naive, additive damped ETS, and two defensible SARIMA candidates.
2. Explain the meaning of each ARIMA order.
3. Compare rolling-origin MAE and interval coverage.
4. Plot residual ACF and run Ljung–Box at lags 7 and 14.
5. State what you would do if residual variance increases with level.

### Flashcards

- **Q:** Weak stationarity? **A:** Constant first two moments, with covariance depending only on lag.
- **Q:** ARIMA (d)? **A:** Number of ordinary differences before modeling ARMA dynamics.
- **Q:** ADF null? **A:** Unit root/nonstationarity.
- **Q:** AIC versus CV? **A:** Likelihood-complexity criterion within family versus empirical task-specific generalization.
- **Q:** Ljung–Box rejection? **A:** Residual autocorrelations are jointly inconsistent with white noise.
- **Q:** Interval calibration? **A:** Empirical coverage matches nominal coverage on future-like data.

### Cheat sheet

```text
ETS: evolving components; excellent baseline for level/trend/seasonality
ARIMA: difference to stable dynamics, then AR of values + MA of shocks
ACF/PACF: candidate hints, not automatic truth
Diagnostics: bias + residual-over-time + ACF/Ljung–Box + variance + coverage/width
```

---

# 6. Machine learning, exogenous inputs, and many series

Trees/boosting learn nonlinearities and interactions, tolerate mixed feature scales, and often excel with lag/rolling/calendar/exogenous features. But they do not naturally extrapolate trends, and random row splits remain invalid. Use the exact same temporal backtest.

For exogenous (x_t), ask:

1. Is it known for the entire horizon at origin time?
2. Is it measured without revisions?
3. Could it be caused by the target (endogeneity)?
4. Will its future distribution shift?
5. Does it improve multiple validation origins?

Scenario forecasts can be more honest than pretending uncertain covariates are known.

## 6.1 Local, global, and hierarchical models

- **Local:** one model per series; specialized but data-hungry and operationally heavy.
- **Global:** one model across entities with entity/static features; shares signal and helps sparse series.
- **Panel features:** every lag/rolling computation must group by entity and preserve order.
- **Cold start:** use metadata/global patterns; no target history means no autoregressive lags.
- **Hierarchy:** SKU forecasts should add to category/total forecasts. Bottom-up aggregates leaf forecasts; top-down allocates totals; reconciliation adjusts forecasts using estimated error structure.

Weight aggregate metrics intentionally. Micro-averaging lets large entities dominate; macro-averaging treats entities equally. State which matches the decision.

## 6.2 Choosing complexity

Compare candidates on accuracy, stability across folds, latency, maintenance, interpretability, interval quality, and sensitivity to missing inputs. A 0.5% average gain may not justify a fragile stack.

## Checkpoint 6

For 200 SKUs across 20 stores:

1. Design a leakage-safe global feature table.
2. Choose local or global modeling for sparse SKUs.
3. Explain how you will encode identity and unseen SKUs.
4. Compare a histogram gradient booster to Ridge and seasonal naive over the same folds.
5. Report macro SKU MAE, revenue-weighted WAPE, per-horizon error, and worst-decile performance.
6. Make store/SKU and total forecasts coherent.

### Flashcards

- **Q:** Why can trees fail on trend? **A:** Piecewise-constant leaves generally cannot extrapolate outside observed feature/target regimes.
- **Q:** Global model benefit? **A:** Shares statistical strength across related series.
- **Q:** Known-future covariate? **A:** Its value is available at forecast origin for the requested horizon.
- **Q:** Micro versus macro metric? **A:** Observation/volume-dominated aggregation versus equal weight per group.
- **Q:** Forecast reconciliation? **A:** Adjust forecasts so hierarchical aggregation constraints hold.

### Cheat sheet

```text
Candidates: seasonal naive -> regularized linear -> ETS/ARIMA -> boosted trees
Panel safety: sort(entity,time); groupby(entity).shift/rolling
Report: fold distribution + horizon + segment + bias + operational cost
```

---

# 7. Deep learning without the hype

Deep learning is most justified with many related series, long histories, rich covariates, complex nonlinear patterns, and enough engineering budget. For one short regular series, ETS, ARIMA, or boosted trees are usually more data-efficient and defensible.

## 7.1 Windowing

Create input context (X_t=[y_{t-L+1},\ldots,y_t]) and target vector (Y_t=[y_{t+1},\ldots,y_{t+H}]). Split by time *before* forming windows or ensure no training target crosses the split. Fit scalers only on training history. Validation windows may use past context from training, but their targets must be entirely in validation.

```python
def windows(a, lookback, horizon):
    X, Y = [], []
    for end in range(lookback, len(a)-horizon+1):
        X.append(a[end-lookback:end])
        Y.append(a[end:end+horizon])
    return np.asarray(X)[..., None], np.asarray(Y)
```

## 7.2 Architectures

- **MLP:** flattened fixed context; simple benchmark.
- **RNN/LSTM/GRU:** recurrent state summarizes sequence; gated variants ease long-dependency optimization.
- **TCN/1-D dilated CNN:** parallel, stable receptive fields; strong practical sequence baseline.
- **Transformer:** attention directly relates time positions; scalable global modeling but data/compute hungry. Positional/time encodings are essential.
- **N-BEATS/TFT-style ideas:** residual basis stacks; interpretable variable selection and multi-horizon attention. Know the concepts, not brand trivia.

Train multi-output with MSE/MAE or quantile pinball loss. Use early stopping on rolling/chronological validation. Compare against naive and classical baselines. Recursive teacher forcing can cause **exposure bias**: training sees true previous values while inference sees its own errors.

Uncertainty can use quantile outputs, distributional likelihood, ensembles, or conformal calibration. Quantiles can cross; impose/order them or repair after prediction.

## Checkpoint 7

Design a network for 10,000 hourly meter series, 168-hour context, 24-hour horizon, weather forecasts, calendar inputs, and meter metadata.

1. Specify tensor shapes and availability masks.
2. Choose direct multi-output TCN/LSTM/Transformer and justify.
3. Define loss for P10/P50/P90 forecasts.
4. Prevent window/split and scaling leakage.
5. Give stopping criteria and classical baselines.
6. Explain why it may lose to a seasonal naive model.

### Flashcards

- **Q:** Lookback versus horizon? **A:** Historical context length versus number of future steps predicted.
- **Q:** Exposure bias? **A:** Training on true past outputs but inference on model-generated ones.
- **Q:** Why positional encoding? **A:** Attention alone is permutation-invariant.
- **Q:** Pinball loss learns? **A:** A chosen conditional quantile.
- **Q:** When does DL pay? **A:** Large related data, nonlinear structure, covariates, and sufficient operational budget.

### Cheat sheet

```text
Split -> fit scaler on train -> create non-crossing windows -> train -> backtest
X: [batch, lookback, features]; Y: [batch, horizon, targets/quantiles]
Always benchmark seasonal naive and a tabular model.
```

---

# 8. Selection, production, and the interview narrative

Retrain the selected specification on all data available at deployment—not the held-out test if that test is still being used to claim performance. Version data cutoff, code, features, model, and forecast origin.

Monitor:

- input freshness, schema, missingness, range, and category drift;
- error by horizon/entity/regime; bias and baseline-relative skill;
- interval coverage and width;
- prediction distribution and business outcomes;
- latency and failure/fallback rate.

Labels arrive later, so input monitoring is immediate and performance monitoring is delayed. Always ship a fallback such as seasonal naive. Retraining can be scheduled or drift/performance-triggered, but validate the policy historically. A concept drift changes (P(y|x)); covariate drift changes (P(x)). Neither automatically proves the other.

## A five-minute interview response

1. **Frame (30s):** unit, target, origin, horizon, cadence, available features, cost.
2. **Audit (45s):** frequency/gaps/DST/duplicates/outliers, plots, seasonality, breaks.
3. **Validation (45s):** untouched final test; rolling origins matching deployment; gap if needed.
4. **Baselines (30s):** naive, seasonal naive, incumbent; metrics tied to cost.
5. **Models (60s):** regularized regression and lag features; ETS/SARIMA; boosting; DL only with justification.
6. **Diagnostics (45s):** fold/horizon/segment stability, bias, residual ACF, interval coverage.
7. **Delivery (45s):** retrain, version, monitor, fallback, retraining policy.

## Checkpoint 8

Your champion suddenly under-forecasts after launch.

1. Triage data pipeline, feature availability, target definition, and regime change in order.
2. Compare errors with seasonal naive to separate model-wide from environment-wide degradation.
3. Identify whether bias is concentrated by horizon/entity.
4. Propose immediate mitigation and a safe longer-term update.
5. Draft a one-paragraph incident summary for a nontechnical stakeholder.

### Flashcards

- **Q:** Data drift versus concept drift? **A:** Change in input distribution versus change in target mechanism conditional on inputs.
- **Q:** Why monitor a baseline live? **A:** It reveals whether model complexity still adds value under the same regime.
- **Q:** Why monitor coverage? **A:** Accurate point forecasts can still have dangerously miscalibrated uncertainty.
- **Q:** Safe fallback? **A:** A simple robust forecast that can run when features/model fail.
- **Q:** First production question? **A:** Is the model receiving the same correctly timed data it was trained on?

### Cheat sheet

```text
Ship = model + data cutoff + feature contract + fallback + monitoring + owner
Diagnose = pipeline -> availability -> definition -> segments/horizons -> drift -> model
Choose simplest model whose repeatable gain matters to the decision.
```

---

# Final memory map

```text
Define information boundary
  -> audit temporal data
  -> reserve final test
  -> rolling-origin validation
  -> naive/seasonal-naive baseline
  -> leakage-safe features
  -> linear + ETS/ARIMA + boosted candidate
  -> DL only if scale/structure justify
  -> score by horizon/segment/fold + bias
  -> diagnose residuals and interval coverage
  -> deploy with fallback and monitor
```
