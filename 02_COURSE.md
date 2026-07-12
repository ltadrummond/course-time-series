# Time Series Interview Sprint

This course assumes no previous time-series knowledge. Read it in order, and do not worry about unfamiliar words: each important word is defined where it first appears. The running example is daily ice-cream sales.

Notation used later: `y(t)` means the value we observe at time `t` (for example, sales today). `h` is how far ahead we predict. `forecast(t + h | t)` means “a forecast for a future time, made using only information available up to time `t`.” Code assumes `pandas as pd` and `numpy as np`.

## How to read the examples

Do not treat a code line as a rule to memorise. For every action, ask these three questions:

1. **What problem am I checking or solving?** For example, `asfreq("D")` checks whether dates are missing from a supposedly daily series.
2. **What mistake would happen if I skipped it?** For example, a missing collection day could be silently mistaken for a normal day.
3. **What decision do I make from the result?** For example, decide whether the missing target means zero sales, a shop closure, or unknown data.

The comments and “Why this matters in practice” sections explain these motives. You should understand the reason for a step before copying it into a model.

---

# 1. Frame the problem and explore the series

## 1.1 Forecasting is a supervised problem with an information boundary

**Forecasting** means using the past and the information available now to estimate a value that has not happened yet. For example: on Monday morning, predict ice-cream sales for Tuesday.

Before choosing a model, answer this small story in plain language:

```text
What are we predicting?       Tuesday's ice-cream sales.
When do we make the forecast? Monday morning.
How far ahead is it?          One day.
What may we use?              Sales up to Monday, the calendar, and Tuesday's weather forecast.
What may we not use?          Tuesday's actual sales or actual measured weather.
```

The last two lines are crucial. A forecast must act as though it is really Monday morning. Using information that becomes available later creates **leakage**: the model appears accurate in testing because it was accidentally allowed to see part of the future.

Before touching a model, define:

- **Unit:** what one row represents, such as one store on one day.
- **Target:** the number we want to predict, such as daily sales.
- **Forecast origin:** the “now” moment when the forecast is made.
- **Horizon:** how far into the future we predict, such as tomorrow or the next 7 days.
- **Cadence:** how often we make a new forecast, such as every morning.
- **Availability:** when a piece of information is actually known, not just the date it describes.
- **Decision/loss:** why forecast errors matter. For example, running out of stock may cost more than buying too much.

A covariate recorded for tomorrow is not necessarily known today. Planned price may be known; observed weather is not. This distinction is the central defense against leakage.

## 1.2 Components and dependence

A time-series value can often be understood as three parts:

```text
observed value = trend + seasonality + remainder
y(t)           = T(t)  + S(t)        + R(t)
```

- **Trend:** the long-term direction or slow change in level.
- **Seasonality:** a pattern that repeats at a known frequency, such as every day, week, or year.
- **Remainder/noise:** variation not explained by the trend or seasonal pattern.
- **Cycle:** a longer rise and fall whose duration is not fixed, unlike seasonality.
- **Structural break:** a change in the data-generating mechanism; after a break, older data may become less useful.

For example, consider daily ice-cream sales. Gradual business growth is the **trend**, higher sales every summer are **seasonality**, and changes caused by rain or local events are part of the **remainder**.

For example, today's ice-cream sales might be decomposed as:

```text
Observed sales:          145 ice creams
Trend:                   110 ice creams
Summer seasonal effect:  +30 ice creams
Remainder:                +5 ice creams

145 = 110 + 30 + 5
```

Here, `+30` means **30 additional ice creams sold**, not 30 days. The seasonal period is separate: the summer pattern repeats every year.

The **remainder can be positive or negative**. For example, unexpected rain might produce a remainder of `-15`, reducing sales below the trend-plus-seasonality estimate.

### Additive versus multiplicative seasonality

### Why this matters in practice

This choice tells you what kind of seasonal pattern the forecasting model should learn.

- If summer consistently adds about **30 sales**, use an additive view: forecast = underlying level + seasonal effect. A seasonal-naive or regression model can learn a fixed summer/weekday adjustment.
- If summer consistently increases sales by **30%**, use a multiplicative view: forecast = underlying level × seasonal multiplier. The seasonal swing should become larger as the business grows. A log transformation often makes this easier for regression and ARIMA-style models because multiplication becomes addition on the log scale.

You do not need to decide perfectly before modeling. Plot the series and compare models using walk-forward validation. The useful choice is the one that forecasts later unseen periods more accurately.

### How to choose

You usually cannot know with certainty in advance. Look at the seasonal swing when the series is low and when it is high:

- If the difference stays about the same size—for example, summer adds about 30 sales at both low and high sales levels—additive seasonality is a good starting point.
- If the percentage stays about the same—for example, summer is about 30% higher at both low and high sales levels—multiplicative seasonality is a good starting point.

Then test both approaches with walk-forward validation and keep the one that performs better on later unseen data. If there is no stable repeating pattern, do not force a seasonal component. Flexible ML models can often learn calendar and lag effects without requiring you to label them additive or multiplicative first.

Use an **additive** description when the seasonal change stays roughly constant in absolute size:

```text
Normal sales:   100  -> 200 ice creams/day
Summer effect:  +30  -> +30 ice creams/day
Summer sales:   130  -> 230 ice creams/day
```

The summer effect remains about `+30` as the business grows:

```text
observed value = trend + seasonality + remainder
```

Use a **multiplicative** description when the seasonal change grows with the level:

```text
Normal sales:   100  -> 200 ice creams/day
Summer effect:  x1.3 -> x1.3
Summer sales:   130  -> 260 ice creams/day
```

Here, the effect remains `+30%`, but its absolute size increases. This is described as:

```text
observed value = trend x seasonality x remainder
```

Taking the logarithm turns multiplication into addition, which can make this pattern easier to model:

```text
log(observed value) = log(trend) + log(seasonality) + log(remainder)
```

The key distinction is:

- **Additive:** the season adds about the same number of sales. For example, `100 + 30 = 130` and `200 + 30 = 230`.
- **Multiplicative:** the season increases sales by about the same percentage. For example, a 30% increase gives `100 x 1.3 = 130` and `200 x 1.3 = 260`.

With additive seasonality, the summer increase stays at `30` ice creams. With multiplicative seasonality, it grows from `30` to `60` ice creams because 30% of a larger sales level is a larger number.

### How the early concepts connect to a forecast

The concepts in this chapter are not separate facts to memorise. They answer different questions before and during modeling:

```text
Trend          -> Is the underlying level growing or shrinking?
Seasonality    -> What repeating pattern should the forecast include?
Remainder      -> What unpredictable part will remain after forecasting?
Lag/ACF        -> Which past values might help as model features?
Correlation    -> Is an extra variable genuinely useful, or merely moving with time?
```

For example, to forecast next Tuesday’s sales, a practical model may combine an estimated growth trend, the usual Tuesday effect, sales from last Tuesday (lag 7), and known inputs such as a planned promotion. It cannot reliably predict all rain, local events, and randomness; those are part of the remainder.

When considering an extra variable, do not use it only because its correlation with sales is high. Two variables can both rise over time and look related even when one does not help forecast the other. For example, electric-car sales and ice-cream sales might both increase over several years, but electric-car sales are not useful for predicting ice-cream demand. A useful variable has a believable connection, is available at forecast time, and improves forecasts on later unseen data. Temperature may pass these checks for ice-cream sales; electric-car sales probably will not.

### Dependence over time

In many ordinary machine-learning datasets, rows can be shuffled because one row does not depend on the next. In a time series, order matters: today's sales may resemble yesterday's sales or sales on the same weekday last week. This is called **dependence over time**.

A **lag** means “how many time steps back.” For daily data, lag 1 is yesterday and lag 7 is the same weekday last week. **Autocorrelation (ACF)** is a number between -1 and 1 that measures how similar a series is to its earlier values. It describes a pattern; it does not prove that an earlier value caused a later value.

- Lag 1 compares today with yesterday for daily data.
- Lag 7 compares today with the same weekday last week.
- A large lag-7 autocorrelation suggests weekly repetition.

For example, imagine these daily ice-cream sales:

| Day | Sales |
|---|---:|
| Monday, week 1 | 100 |
| Tuesday, week 1 | 105 |
| Monday, week 2 | 102 |
| Tuesday, week 2 | 107 |

For Tuesday in week 2, its **lag 1** value is Monday in week 2: `102`. Its **lag 7** value is Tuesday in week 1: `105`. If Tuesdays tend to have similar sales to earlier Tuesdays, and the same is true for the other weekdays, the series has high lag-7 autocorrelation. That is evidence of a weekly pattern, not proof that last Tuesday caused this Tuesday’s sales.

An ACF value near `+1` means the current value and earlier value move together almost perfectly. An ACF value near `-1` means they move in opposite directions almost perfectly. Exact `+1` and `-1` are rare in real data, but these simplified examples show the idea:

```text
Lag-1 ACF near +1:  10, 20, 30, 40
                    Each day is high when the previous day was high,
                    and low when the previous day was low.

Lag-1 ACF near -1:  10,  0, 10,  0, 10,  0
                    A high day is followed by a low day, and vice versa.

Lag-1 ACF near  0:  no consistent relationship between one day and the next.
```

A high positive lag-1 ACF can occur simply because the series has a trend, so it does not by itself prove a useful “yesterday causes today” mechanism.

### Why correlation can be misleading

**Correlation** says whether two quantities tend to move together. It does not automatically mean that one causes the other.

For example, over several years, both ice-cream sales and the number of electric cars sold may increase. They can have a high correlation simply because both rise over time—not because electric cars make people buy ice cream. This is a **spurious correlation**: a relationship that looks meaningful but is created by a shared trend or coincidence.

To check whether the relationship may be useful:

1. Compare **differences** as well as original values. A difference is the change from one time to the next: `sales today − sales yesterday`. This removes much of a slow upward or downward trend.
2. Check **domain timing**: does the proposed cause happen before the effect, and is there a credible business reason it could affect it?
3. Check **out-of-sample incremental value**: train a forecast with historical data, then test it on later, unseen data. The extra variable is useful only if it improves forecasts there, compared with a model without it.

## 1.3 Minimal EDA protocol

**EDA** stands for *exploratory data analysis*: looking at the data before modeling it. The goal is to find basic problems and visible patterns, not to choose the fanciest model.

1. Sort by time; check duplicates, timezone, frequency, gaps, and target validity.
2. Plot the entire history, recent window, and distributions by calendar group.
3. Compare rolling mean/standard deviation; inspect seasonal plots.
4. Plot the ACF (autocorrelation chart) of the original values and, if there is a trend, of the differences.
5. Mark interventions, outages, promotions, and known breaks.
6. Split chronologically *before* allowing test-period discoveries to shape modeling.

```python
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# This course uses generic names: `timestamp` for the time column and `y` for
# the target. For `data/retail_daily.csv`, rename `date` -> `timestamp` and
# `demand` -> `y` before running this full example.
df = df.rename(columns={"date": "timestamp", "demand": "y"})

# 1. Put the data in time order and inspect basic index/data problems.
# `.copy()` makes a separate dataframe, so the original `df` is not changed accidentally.
df = df.copy()
df["timestamp"] = pd.to_datetime(df["timestamp"])
print("duplicate timestamps:", df.duplicated("timestamp").sum())
print("timezone:", df["timestamp"].dt.tz)

df = df.sort_values("timestamp").set_index("timestamp")
# Stop with an error if dates are not in ascending order (oldest -> newest).
assert df.index.is_monotonic_increasing
print("inferred frequency:", pd.infer_freq(df.index))
print("invalid negative sales:", (df["y"] < 0).sum())  # adjust this rule for your target

# `asfreq` makes missing daily dates visible as NaN instead of silently skipping them.
# Without it, a series can look daily even though (for example) 2022-01-03 has no row.
full = df["y"].asfreq("D")
print("missing daily values:", full.isna().sum())
print("first missing dates:", full[full.isna()].index[:5].tolist())

# Goal: detect and understand missing target days; do not fill them automatically.
# 0 means “we know sales were zero”; NaN means “we do not know the sales value”.
# Investigate every gap: it may be a closure, a collection failure, or a future date to forecast.

# 2. Plot the whole history and a recent window.
fig, history_axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
full.plot(ax=history_axes[0], title="Daily sales: full history")
full.tail(90).plot(ax=history_axes[1], title="Daily sales: most recent 90 days")
plt.tight_layout()

# Compare the sales distribution for each weekday: 0=Monday, ..., 6=Sunday.
by_dow = df.assign(dow=df.index.dayofweek).groupby("dow")["y"]
print(by_dow.agg(["mean", "median", "count"]))
df.assign(dow=df.index.dayofweek).boxplot(column="y", by="dow", figsize=(8, 4))
plt.suptitle("")
plt.title("Sales distribution by weekday")

# 3. A rolling mean/std show whether level or variation changes over time.
stats = full.rolling(28).agg(["mean", "std"])
stats["mean"].plot(title="28-day rolling mean")
stats["std"].plot(title="28-day rolling standard deviation")

# 4. ACF of levels, then changes between days if the level is trending.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(full.dropna(), lags=35, ax=axes[0])
axes[0].set_title("ACF of sales levels")
plot_acf(full.diff().dropna(), lags=35, ax=axes[1])
axes[1].set_title("ACF of day-to-day changes")
plt.tight_layout()

# 5. If known, mark real-world events on the full-history plot.
# Example: history_axes[0].axvline(pd.Timestamp("2025-06-01"), color="red", label="new promotion")

# 6. Keep the latest 90 days untouched for final testing; never shuffle time rows.
cutoff = full.index.max() - pd.Timedelta(days=90)
train = full.loc[:cutoff]
test = full.loc[cutoff + pd.Timedelta(days=1):]
```

For the `asfreq` step, the point is **not** to replace every `NaN`. It is to reveal gaps so you can make the correct business decision: record `0` only when you know there were zero sales; keep `NaN` for an unknown value caused by a collection failure; and leave future targets unknown because those are the values you want to forecast. Treating all gaps as zero teaches the model false low-demand days.

If two series both trend, their correlation can be misleading. Compare their changes, check that the timing makes sense in the real world, and keep a variable only when it improves forecasts on later unseen data.

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

This chapter is about making the timeline trustworthy before modeling. A model cannot tell the difference between a real zero sale, a missing record, and a date that was accidentally skipped unless we represent them correctly.

## 2.1 The time index is part of the data-generating process

A **time index** is the date/time attached to every observation. In a daily sales table, it is the date; in hourly electricity data, it is the hour. Before doing statistics, ask: “Does every row have the correct time, in the correct order, at the expected frequency?”

**Frequency** means how often observations should occur: hourly, daily, weekly, and so on. A missing date is different from a date whose value happens to be zero.

Use timezone-aware timestamps when daylight-saving transitions matter. Never silently assume all days have 24 hourly observations. For multiple entities, `(entity, timestamp)` must be unique.

**Resampling** changes frequency. For example, it turns 24 hourly readings into one daily reading. You must choose the operation based on what the value means:

```python
daily = hourly.resample("D").agg(
    energy=("kwh", "sum"),             # a flow
    temperature=("temp", "mean"),      # an average state
    closing_inventory=("stock", "last")
)
```

`asfreq` reveals or creates the expected timestamps without combining values. For example, it can reveal that Tuesday is absent from a daily sales series. `resample` groups values into larger time intervals. Be explicit about interval closure and labels in financial or sensor settings.

### Why this matters in practice

The time index determines what one training row means. If you accidentally sum temperature instead of averaging it, or skip a date without noticing, the model learns from incorrect inputs. Before ML, choose the frequency at which decisions are made—for example, daily demand forecasts require one correct row per store-day.

## 2.2 Missing values are events, not merely holes

A missing value is not automatically zero. First ask why it is absent. For example, no recorded sale might mean “the shop sold nothing,” “the shop was closed,” or “the cash-register data failed to arrive.” Those cases need different treatment.

Classify a gap:

- **Structural zero:** store was open and sold none → often `0`.
- **Not applicable:** store closed → perhaps zero for demand fulfilled, but missing for latent demand.
- **Collection failure:** unknown → `NaN`, possibly impute and add a missingness flag.
- **Irregular sampling:** model elapsed time or regularize carefully.

**Forward fill** means replacing a missing value with the most recent earlier value. It is reasonable for a state that stays valid until changed, such as a machine-status setting. It is usually wrong for daily sales or temperature, which genuinely change every day.

**Interpolation** estimates a gap using values on both sides of it. That uses future values, so it is not allowed when simulating a live forecast—unless the same future values really would be known at forecast time. An **imputer** is any rule or model used to fill missing values; fit it using the training period only, then apply it to later validation data.

For example, this creates a missingness flag before filling a persistent machine-status column. It deliberately does **not** forward-fill sales:

```python
daily = daily.copy()

# Keep evidence that status was missing; the model may learn that this matters.
daily["machine_status_was_missing"] = daily["machine_status"].isna().astype(int)
daily["machine_status"] = daily["machine_status"].ffill()

# Leave unknown sales as NaN until you decide why they are missing.
unknown_sales = daily["sales"].isna()
print("unknown sales rows:", unknown_sales.sum())
```

### Why this matters in practice

Missingness can itself predict something: a missing machine reading may mean the sensor is failing, and a closed shop has a different meaning from zero demand. Add a missing-value flag when useful. Most importantly, never fill an old missing value with information from later dates if that information would not have existed in production; that would make validation falsely optimistic.

## 2.3 Variance stabilization and differencing

Some series become more variable as they become larger. For example, a small shop may vary by 5 sales a day, while a large shop varies by 50. A **transformation** changes the numerical scale to make such patterns easier to model; it does not change what is being predicted.

For non-negative values, `log1p(y)` means `log(1 + y)`. It compresses large values more than small ones, often reducing a long right tail and making percentage-like variation more stable. Box–Cox is a related family of transformations. When converting a log forecast back, be careful: exponentiating an average log forecast usually gives a **median-like** answer, not the arithmetic mean (this is Jensen’s inequality). A residual-variance correction under Gaussian log errors is approximately

\[
E[Y]\approx \exp(\hat\mu + \hat\sigma^2/2).
\]

A **difference** is a change rather than a level. A first difference asks “how much did sales change since yesterday?” A seasonal difference asks “how different is this Monday from last Monday?” They are

\[
\nabla y_t=y_t-y_{t-1},\qquad \nabla_m y_t=y_t-y_{t-m}.
\]

Differencing can remove a changing level or a repeating seasonal pattern, allowing models to focus on the remaining dynamics. **Over-differencing** means differencing more than needed; it adds noise and can create artificial negative autocorrelation. Save the initial values, because you need them to convert predicted changes back into predicted sales levels.

The equivalent pandas operations are:

```python
s = daily["sales"]
daily["log1p_sales"] = np.log1p(s)       # only valid when sales are >= 0
daily["change_from_yesterday"] = s.diff(1)
daily["change_from_last_week"] = s.diff(7)

# Convert a log-scale prediction back to the original sales scale.
log_forecast = 4.2
sales_forecast = np.expm1(log_forecast)
```

### Why this matters in practice

Transformations and differences are optional tools, not rituals. Use a log-like transform when forecast errors grow with the level; use differences when a model such as ARIMA needs a more stable series. Then validate whether they improve future forecasts. For many tree models, raw values plus safe features can work well without either transformation.

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

This chapter answers three practical questions: “What simple forecast must my model beat?”, “How will I measure a mistake?”, and “How can I test the model as if time were moving forward?”

## 3.1 Baselines

A **baseline** is a deliberately simple forecast. It sets the minimum standard. If a complicated model cannot beat it on future-like data, do not use the complicated model.

A model is useful only relative to a credible low-complexity alternative.

- Mean: \(\hat y_{T+h}=\bar y\)
- Naive/random walk: \(\hat y_{T+h}=y_T\)
- Drift: \(\hat y_{T+h}=y_T+h(y_T-y_1)/(T-1)\)
- Seasonal naive: \(\hat y_{T+h}=y_{T+h-m(k+1)}\), equivalently repeat the latest season.

For example, a **naive** forecast for Tuesday says “Tuesday will equal Monday.” A **seasonal-naive** forecast for Tuesday says “Tuesday will equal last Tuesday.” For daily retail data, lag-7 seasonal naive is usually the minimum bar. Compare against any forecast the business already uses too.

For a one-step-ahead daily validation period, make those baselines with `shift`:

```python
# `y` is a daily Series in chronological order.
comparison = pd.DataFrame({"actual": y})
comparison["naive"] = y.shift(1)           # yesterday's sales
comparison["seasonal_naive"] = y.shift(7)  # same weekday last week

# Drop the first rows, which do not have enough history for the baseline.
comparison = comparison.dropna()
```

### Why this matters in practice

Baselines prevent unnecessary complexity. If “same weekday last week” is already accurate, a complex ML model may add cost and failure modes without enough benefit. A production model should demonstrate a repeatable improvement over the baseline, not just a good-looking score in one split.

## 3.2 Metrics encode decisions

A **metric** turns a prediction error into a number so that forecasts can be compared. There is no universally best metric: choose one whose mistakes match the decision. Let `actual − forecast` be the error.

For errors (e_i=y_i-\hat y_i):

\[
MAE=\frac1n\sum|e_i|,\quad RMSE=\sqrt{\frac1n\sum e_i^2},\quad
MAPE=\frac{100}{n}\sum\left|\frac{e_i}{y_i}\right|.
\]

- **MAE:** average absolute mistake. If forecasts miss by 2, 5, and 3 sales, MAE is 10/3 sales. It is easy to explain and treats every unit of error equally.
- **MSE/RMSE:** squares errors before averaging, so a very large miss counts much more. Use it when large errors are especially costly; RMSE returns to the target’s units.
- **MAPE:** average percentage error. It is easy to read but fails when actual sales are zero or close to zero.
- **WAPE:** total absolute error divided by total actual volume; useful for a portfolio, but large sellers have more influence.
- **MASE:** MAE divided by the in-sample MAE of a naive baseline; it works with zeros. MASE < 1 means the model beats that baseline.
- **Pinball loss:** a metric for forecasts such as the 90th percentile, where under-forecasting and over-forecasting have intentionally different costs.

Aggregate metrics can hide horizon, entity, or regime failures. Report by horizon and important segment. Measure signed mean error for bias.

Here is a small metric function. `insample` is the training series, used only to calculate the MASE baseline scale:

```python
def forecast_metrics(actual, prediction, insample, seasonal_period=7):
    error = prediction - actual
    mae = error.abs().mean()
    rmse = np.sqrt((error**2).mean())
    bias = error.mean()                     # positive means over-forecasting
    naive_mae = insample.diff(seasonal_period).abs().dropna().mean()
    mase = mae / naive_mae
    return {"mae": mae, "rmse": rmse, "bias": bias, "mase": mase}

scores = forecast_metrics(test, seasonal_prediction, train, seasonal_period=7)
print(scores)
```

### Why this matters in practice

Metrics choose what the model is rewarded for. If stockouts are very expensive, a model with occasional large under-forecasts may be unacceptable even if its MAE is good. If there are many zero-sale products, avoid MAPE. Choose the metric before comparing models, and also inspect errors by forecast horizon and important product/store groups.

## 3.3 Walk-forward validation

**Validation** means testing a model on data it did not train on. In time series, do not randomly shuffle rows: that would allow the model to learn from the future. Instead, repeatedly pretend you are at an earlier date, train only on the past, and forecast the next period. This is **walk-forward** or **rolling-origin validation**.

Random cross-validation leaks future regimes and destroys ordering. Use several forecast origins:

```text
fold 1: [ train -------- ][validate]
fold 2: [ train ---------------- ][validate]
fold 3: [ train ------------------------ ][validate]
```

- **Expanding window:** keep all old history and add new history each time; useful when the process is stable.
- **Sliding window:** keep only a recent moving block of history; useful when old data no longer represents current behaviour.
- **Gap/embargo:** leave a time gap between training and validation when labels or features arrive late or overlap.
- Match validation horizon and refit cadence to deployment.
- Perform imputation, scaling, selection, feature fitting, and tuning inside each training fold.
- Keep a final untouched chronological test set.

Uncertainty across origins matters: a tiny average gain with unstable fold performance is weak evidence.

This function creates expanding-window folds for a one-series dataset. Each fold trains on the past and validates the next `horizon` days:

```python
def rolling_origins(n_rows, initial_train=365, horizon=14, step=30):
    for train_end in range(initial_train, n_rows - horizon + 1, step):
        train_idx = np.arange(train_end)
        valid_idx = np.arange(train_end, train_end + horizon)
        yield train_idx, valid_idx

for train_idx, valid_idx in rolling_origins(len(y)):
    train_fold = y.iloc[train_idx]
    valid_fold = y.iloc[valid_idx]
    # Fit only with train_fold; score predictions against valid_fold.
    print(train_fold.index[-1], "->", valid_fold.index[0])
```

### Why this matters in practice

Walk-forward validation is the closest offline simulation of deployment: train on what was known then, forecast what happened next, and repeat. It tells you whether a model works across different months and conditions—not merely on one convenient historical split. This is the main protection against accidentally choosing a model that only succeeded by seeing the future.

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

This chapter turns a timeline into a normal machine-learning table. Each row represents one prediction moment, and each column contains only information that was available at that moment.

## 4.1 From series to supervised table

**Feature engineering** means creating useful input columns from raw data. For daily sales, a row for 15 July might include yesterday’s sales, sales seven days ago, the weekday, whether there is a promotion, and the planned price. The **target** for that row is the sales value we want the model to learn to predict.

**Regression** learns a rule that combines these input columns to predict a numerical target. The equation below says: today’s value is estimated from past target values (lags), other inputs, and a leftover error.

A dynamic regression can be written

\[
y_t=\beta_0+\sum_{j\in L}\beta_jy_{t-j}+\gamma^Tx_t+\epsilon_t.
\]

OLS (ordinary least squares) chooses the coefficients that make squared prediction errors as small as possible. A **residual** is `actual − model prediction`. If residuals remain autocorrelated, the model has left predictable time structure unexplained. This does not automatically make every coefficient wrong, but it is a warning that the model is incomplete.

Useful features:

- **Lags:** earlier target values, such as sales 1, 7, 14, or 28 days ago.
- **Rolling summaries:** summaries of earlier values, such as the average sales over the preceding 28 days. EWMA is a rolling average that gives more weight to recent days.
- **Calendar features:** day of week, month, holiday, or payday.
- **Age/trend:** elapsed time or a simple time counter so a model can represent gradual growth or decline.
- **Fourier seasonality:** pairs of sine/cosine columns that represent a smooth repeating shape; useful when a simple weekday category is too abrupt.
- **Known-future inputs:** values already known at forecast time, such as planned price, promotion, or holiday.
- **Observed-only inputs:** values only measured once they happen, such as actual temperature; lag them or forecast them separately.

Critical pattern:

```python
g = df.groupby("store", group_keys=False)
df["lag_7"] = g.y.shift(7)
df["roll28_mean"] = g.y.transform(lambda s: s.shift(1).rolling(28).mean())
df["ewm"] = g.y.transform(lambda s: s.shift(1).ewm(alpha=.2).mean())
```

The order `shift(1).rolling(28)` matters: shift first, then calculate the average. Without the shift, a row’s rolling average includes the sales value the row is trying to predict. That is leakage.

For one daily series, this builds a small feature table and fits a Ridge model without shuffling time:

```python
from sklearn.linear_model import Ridge

feat = df.copy()
for lag in [1, 7, 14, 28]:
    feat[f"lag_{lag}"] = feat["y"].shift(lag)
feat["roll_7_mean"] = feat["y"].shift(1).rolling(7).mean()
feat["dow"] = feat.index.dayofweek
feat["trend"] = np.arange(len(feat))

# One-hot weekday columns let a linear model learn a different weekday adjustment.
feat = pd.get_dummies(feat, columns=["dow"], dtype=float)
feature_columns = [c for c in feat.columns if c != "y"]

cutoff = feat.index.max() - pd.Timedelta(days=90)
train_rows = feat.loc[:cutoff].dropna()
test_rows = feat.loc[cutoff + pd.Timedelta(days=1):].dropna()

model = Ridge(alpha=10).fit(train_rows[feature_columns], train_rows["y"])
ridge_prediction = pd.Series(
    model.predict(test_rows[feature_columns]), index=test_rows.index
)
```

Linear models are easy to inspect and can continue a simple trend outside the range seen in training. **One-hot encoding** creates a separate 0/1 column for each weekday or category. **Ridge** regression shrinks coefficients toward zero, making estimates more stable when lag columns are very similar. **Lasso** can set some coefficients exactly to zero, but its selection can be unstable when predictors are highly correlated.

### Why this matters in practice

Feature engineering is how ordinary ML models receive time information. A model cannot infer “last Tuesday’s sales” unless you make that lag column. The availability rule is the key: every feature for a historical row must contain only what would have been known at that row’s forecast origin. Safe lag and rolling features are often more valuable than switching to a more complex algorithm.

## 4.2 Multi-step strategies

A **multi-step forecast** predicts more than one future time point—for example, the next 14 days. The key question is what the model uses when it needs a future lag that has not happened yet.

- **Recursive:** predict tomorrow first, then use that prediction to help predict the day after. It needs only one model but errors can accumulate.
- **Direct:** train a separate model for each horizon—for example, one for tomorrow and one for 14 days ahead. It avoids feeding errors back but needs more models and data.
- **Multi-output:** one model predicts the whole 14-day vector at once. It can learn that the future days are related, but needs a fixed horizon and a suitable model.
- **DirRec:** a hybrid of direct and recursive approaches.

Features must match strategy. At origin (T), true (y_{T+1}) cannot be used as a lag when predicting (T+2) unless recursively replaced by its forecast.

The key recursive idea is to append each prediction to the history before predicting the next step:

```python
history = train_y.copy()  # `train_y` is the chronological training target Series.
future_predictions = []

for _ in range(14):
    # In real code, rebuild all lag/rolling features from `history` here.
    next_features = make_features_from_history(history)  # contains no true future sales
    next_prediction = model.predict(next_features)[0]
    future_predictions.append(next_prediction)
    history.loc[history.index[-1] + pd.Timedelta(days=1)] = next_prediction
```

`make_features_from_history` is deliberately a placeholder: its job is to construct the same safe feature columns used during training, using only `history` and known-future inputs.

### Why this matters in practice

Your deployment horizon determines the strategy. If the business needs a single next-day forecast, recursive/direct differences barely matter. If it needs the next 14 days, decide whether later forecasts will depend on earlier predictions. Validate using the same strategy you will deploy; otherwise the offline score does not represent reality.

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

This chapter introduces classical time-series models. They work directly with a series’ level, trend, seasonality, and previous changes, rather than requiring you to build every feature manually.

## 5.1 Exponential smoothing / ETS

**Exponential smoothing** estimates the current underlying level by averaging past observations while giving more weight to recent ones. ETS stands for **Error, Trend, Seasonal**: a family of models that explicitly tracks these components.

Simple exponential smoothing updates level:

\[
\ell_t=\alpha y_t+(1-\alpha)\ell_{t-1},\qquad \hat y_{t+h|t}=\ell_t.
\]

In the equation, \(\ell_t\) is the estimated current **level** and \(\alpha\) controls how quickly it reacts: a larger \(\alpha\) trusts the newest observation more. Holt’s method adds a trend estimate. Holt–Winters adds seasonality. In additive Holt–Winters:

\[
\ell_t=\alpha(y_t-s_{t-m})+(1-\alpha)(\ell_{t-1}+b_{t-1}),
\]
\[
b_t=\beta(\ell_t-\ell_{t-1})+(1-\beta)b_{t-1},
\]
\[
s_t=\gamma(y_t-\ell_t)+(1-\gamma)s_{t-m}.
\]

Here, \(b_t\) is the trend and \(s_t\) is the seasonal effect. A **damped trend** assumes growth will gradually slow rather than continuing forever in a straight line. ETS is often a strong choice for a smooth series with clear level, trend, and seasonality but few outside inputs.

This is a minimal additive ETS forecast for daily data with weekly seasonality:

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ets = ExponentialSmoothing(
    train,
    trend="add",
    damped_trend=True,
    seasonal="add",
    seasonal_periods=7,
).fit(optimized=True)

ets_prediction = ets.forecast(14)
```

### Why this matters in practice

ETS is a strong, fast baseline when the target’s own history carries most of the signal. It can forecast level, trend, and seasonality without manually constructing many features. Compare it with seasonal naive and ML models; it is often difficult to beat on one clean, regular series.

## 5.2 Stationarity and ARIMA

ARIMA is a family of models for patterns in past values and past surprises. Its vocabulary is compact but initially unfamiliar, so read the meanings before the equations.

**Weak stationarity** roughly means that the series behaves similarly throughout time: its average level and variability are stable, and a 7-day relationship looks the same regardless of the calendar date. This matters because ARIMA learns repeating historical relationships. A clear fixed trend can sometimes be modeled directly; a wandering, non-stable level often needs differencing first.

AR((p)):

\[
y_t=c+\phi_1y_{t-1}+\cdots+\phi_py_{t-p}+\epsilon_t.
\]

An **AR** (autoregressive) term predicts from earlier values: `p` is how many lags it uses. An **MA** (moving-average) term predicts from earlier *errors* or shocks: `q` is how many past shocks it uses. In ARIMA `(p, d, q)`, `d` is the number of differences applied before the AR/MA model. SARIMA adds equivalent seasonal terms `(P, D, Q, m)`, where `m` is the seasonal period—for daily weekly seasonality, `m = 7`.

Heuristics on a stationary series:

- **ACF** plots correlation with earlier values; **PACF** measures the direct correlation at a lag after accounting for shorter lags.
- An AR(`p`) pattern often has a PACF that becomes small after lag `p`, while its ACF fades gradually.
- An MA(`q`) pattern often has an ACF that becomes small after lag `q`, while its PACF fades gradually.
- These are hints for choosing candidate models, not automatic rules. Validate candidates on future-like data.

ADF and KPSS are statistical tests that provide evidence about stationarity. ADF starts by assuming a unit root/non-stationary series; KPSS starts by assuming stationarity. They can be wrong or inconclusive with limited data, so use plots and domain knowledge too. Do not keep differencing merely to obtain a preferred p-value.

AIC (=2k-2\log L) balances in-sample likelihood and parameter count; it is useful within a comparable model family, while rolling validation measures the actual forecasting task.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
fit = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,7),
              exog=X_train, enforce_stationarity=False).fit(disp=False)
fc = fit.get_forecast(steps=14, exog=X_future)
mean, interval = fc.predicted_mean, fc.conf_int(alpha=.05)
```

ARIMAX/SARIMAX assumes future exogenous values are available or separately forecast. Coefficients depend on differencing and parameterization; explain them carefully.

### Why this matters in practice

ARIMA is useful when the remaining series has stable short-memory patterns after trend/seasonality are handled. Its `(p, d, q)` values are not goals to tune blindly: propose a few sensible candidates using plots and knowledge of the data, then choose with rolling validation. If using price, weather, or promotion inputs, confirm their future values are actually available for the entire forecast horizon.

## 5.3 Residuals and intervals

After fitting a model, inspect what it failed to explain. A residual is the difference between the actual and forecast value. A **prediction interval** gives a range of plausible future observations, not just one best guess.

Good residuals average near zero, have no obvious repeating pattern, and do not grow much more variable at higher levels. The **Ljung–Box test** asks whether residual autocorrelations across several lags are collectively different from zero. A failure indicates remaining structure; passing does not prove the model is perfect.

A 95% **prediction interval** should contain a future observation about 95% of the time under repeated use. It is wider than a confidence interval for a mean parameter. Validate coverage and average width by horizon. Gaussian intervals can fail with skew, outliers, or heteroskedasticity. Alternatives: bootstrap/simulation, quantile models, and conformal calibration under appropriate exchangeability/local-stability assumptions.

After any model, calculate residuals and inspect whether the claimed interval coverage is realistic:

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

residuals = test - prediction
print("mean residual:", residuals.mean())
print(acorr_ljungbox(residuals.dropna(), lags=[7, 14], return_df=True))

# Lower/upper must be aligned Series for the same test dates.
coverage = ((test >= lower) & (test <= upper)).mean()
average_width = (upper - lower).mean()
print("interval coverage:", coverage, "average width:", average_width)
```

### Why this matters in practice

Point forecasts alone hide risk. Inventory planning may need a high-demand P90 forecast to reduce stockouts, while staffing may use a central P50 forecast. Check that a claimed 90% or 95% interval really covers outcomes at about that rate on unseen periods; otherwise the uncertainty estimate is not trustworthy.

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

This chapter applies familiar machine-learning ideas to time series. The rules from earlier chapters still apply: keep time order, avoid future information, and compare against a simple seasonal baseline.

**Trees** and **boosting** are machine-learning models that can learn non-linear rules and interactions. For example, they can learn that a promotion helps more on weekends than weekdays. They often work well with lag, rolling, calendar, and outside-input features. However, ordinary decision trees usually cannot sensibly continue a rising trend beyond the values they saw in training, and random row splits are still invalid. Use the same chronological backtest as before.

An **exogenous input** (also called a covariate or feature) is information outside the target series, such as price, promotion, holiday, or weather. Before using one, ask:

1. Is it known for the entire horizon at origin time?
2. Is it measured without revisions?
3. Could it be caused by the target (endogeneity)?
4. Will its future distribution shift?
5. Does it improve multiple validation origins?

If the future value of an input is uncertain, make **scenario forecasts** instead of pretending it is known. For example, forecast demand under a “hot,” “normal,” and “rainy” weather scenario.

For many store-product series, sort and group before making lag features. This prevents one product’s sales from becoming another product’s lag:

```python
panel = panel.sort_values(["store", "sku", "date"]).copy()
groups = panel.groupby(["store", "sku"], group_keys=False)
panel["lag_7"] = groups["sales"].shift(7)
panel["roll_28_mean"] = groups["sales"].transform(
    lambda s: s.shift(1).rolling(28).mean()
)

# A global model can use one-hot store/SKU columns alongside lags and calendar features.
panel = pd.get_dummies(panel, columns=["store", "sku"], dtype=float)
```

### Why this matters in practice

Extra features can help only when they are available at prediction time and continue to help on later validation periods. A planned promotion is often valid for a 14-day forecast; actual temperature next week is not. Trees and boosters can capture complicated promotion/price/calendar interactions, but they still need safe features and time-based validation.

## 6.1 Local, global, and hierarchical models

When there are many related series—such as sales for many products in many stores—you must decide whether each series gets its own model or whether one model learns from all of them.

- **Local model:** one model per series, for example one model for each store-product. It is specialized, but needs enough history for every series.
- **Global model:** one model trained across all series, with columns identifying store/product. It shares patterns and can help sparse series.
- **Panel data:** a table with many entities observed over time. Compute every lag/rolling feature separately for each entity and keep each entity’s time order.
- **Cold start:** a new product has no sales history, so lag features cannot exist. Use metadata and patterns learned from similar products.
- **Hierarchy:** SKU forecasts should add up to their category and store totals. Bottom-up adds leaf forecasts; top-down divides a total forecast among children; **reconciliation** adjusts forecasts so the totals agree.

Choose aggregate metrics intentionally. **Micro-averaging** lets large-volume products dominate the score. **Macro-averaging** gives each product equal weight. Use the one that matches the business decision, and say which you chose.

### Why this matters in practice

With many products and stores, a global model may learn common patterns that no individual product has enough history to reveal. But it must never calculate a product’s lag using another product’s history. Decide the desired aggregation too: accurate total demand may hide severe errors on small but important products, so inspect both totals and per-entity results.

## 6.2 Choosing complexity

More complex is not automatically better. A model should earn its complexity by delivering a reliable improvement that matters in practice.

### Why this matters in practice

Choose the simplest model with a stable, meaningful gain after considering accuracy, forecast speed, missing-input risk, maintenance, explainability, and uncertainty. A tiny average improvement is not valuable if the model fails when weather data arrives late or cannot be explained to the people using it.

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

Deep learning can be useful for time series, but it is not the default answer. Treat it as another model family that must beat seasonal-naive and simpler alternatives in a fair chronological evaluation.

Deep learning is most justified with many related series, long histories, rich outside inputs, complex non-linear patterns, and enough engineering budget. For one short, regular series, ETS, ARIMA, or boosted trees are usually easier to train, explain, and maintain.

## 7.1 Windowing

Neural networks expect fixed-size arrays. **Windowing** converts one long series into many training examples: choose a block of past observations as input, and the following future observations as the answer.

For example, with a 168-hour **lookback** and a 24-hour **horizon**, each example gives the model the previous 168 hours and asks it to predict the next 24. A **window** must not let a training target cross into the validation period.

### Why this matters in practice

Windowing determines precisely what the neural model is allowed to learn from. Create the chronological train/validation split first, fit scaling using training data only, and ensure validation targets are later than training targets. Otherwise the network can appear impressive by learning from future values hidden in overlapping windows.

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

An **architecture** is the structure of a neural network. You do not need to memorise every name; learn what each kind is trying to do and compare it with simpler baselines.

- **MLP:** treats a fixed past window as one flat list of numbers; use it as a simple neural baseline.
- **RNN/LSTM/GRU:** read observations in sequence and carry a memory-like state forward. LSTM/GRU gates help preserve useful longer-term information.
- **TCN / 1-D dilated CNN:** uses convolution filters across time; it can see a wide history efficiently and is a strong practical sequence baseline.
- **Transformer:** uses attention to relate different time positions directly. It can scale well across many series but needs more data and computation; time/position encodings tell it the order.
- **N-BEATS/TFT-style ideas:** specialised forecasting architectures. Know their broad ideas—decomposition, variable selection, multi-horizon attention—rather than treating brand names as a requirement.

Train multi-output forecasts with MSE/MAE or **quantile pinball loss** when predicting ranges such as P10/P50/P90. **Early stopping** stops training when chronological validation stops improving. Always compare against naive and classical baselines. In a recursive neural forecast, **teacher forcing** trains on true previous values, while deployment sees the model’s own previous predictions; this mismatch is **exposure bias**.

Uncertainty can use quantile outputs, distributional likelihood, ensembles, or conformal calibration. Quantiles can cross; impose/order them or repair after prediction.

### Why this matters in practice

Use deep learning only when the data scale and complexity justify its cost. For example, 10,000 related meter series with weather and metadata may benefit from one global neural model. A single short sales series usually will not. A neural network is successful only if it beats seasonal naive and simpler models using the same future-like validation setup.

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

A forecast is only useful if it can run safely after it leaves the notebook. This chapter covers choosing a practical model, monitoring it, and explaining the plan clearly.

After selecting a model, retrain it on all data that would genuinely be available at deployment. Do not reuse a final untouched test set to make a performance claim after training on it. Record the data cutoff date, code version, feature definitions, model version, and forecast origin so you can reproduce a forecast later.

Monitor:

- input freshness, schema, missingness, range, and category drift;
- error by horizon/entity/regime; bias and baseline-relative skill;
- interval coverage and width;
- prediction distribution and business outcomes;
- latency and failure/fallback rate.

**Labels** are the actual outcomes, such as tomorrow’s actual sales. They arrive after a forecast, so you can monitor input data immediately but can measure forecast error only later. Always provide a simple **fallback** such as seasonal naive in case the main model or an input fails.

**Covariate drift** means the inputs change distribution—for example, prices are much higher than before. **Concept drift** means the relationship between inputs and sales changes—for example, the same promotion now produces a different sales response. Retraining can be scheduled or triggered by drift/performance, but test the policy historically before relying on it.

A minimal monitoring table compares actual outcomes with both the main forecast and the seasonal-naive fallback after labels arrive:

```python
# One row per forecasted date after actual sales have arrived.
monitor = pd.DataFrame({
    "actual": actual_sales,
    "model": model_forecast,
    "seasonal_naive": seasonal_naive_forecast,
}).dropna()

monitor["model_error"] = monitor["model"] - monitor["actual"]
monitor["baseline_error"] = monitor["seasonal_naive"] - monitor["actual"]

weekly_monitor = monitor.resample("W").agg(
    model_mae=("model_error", lambda s: s.abs().mean()),
    model_bias=("model_error", "mean"),
    baseline_mae=("baseline_error", lambda s: s.abs().mean()),
)
weekly_monitor["model_beats_baseline"] = (
    weekly_monitor["model_mae"] < weekly_monitor["baseline_mae"]
)
print(weekly_monitor.tail())
```

### Why this matters in practice

A good offline model can fail after launch because inputs are missing, definitions change, or customer behaviour changes. Monitor the data first, then monitor errors when actual outcomes arrive. Keep a seasonal-naive fallback so forecasts can still be produced safely if the main pipeline breaks.

## A five-minute interview response

This is a reusable answer structure for a forecasting interview question. It is a checklist, not a script to memorise word-for-word.

### Why this matters in practice

In an interview or real project, the strongest answer is not “use model X.” It is a defensible sequence: define the decision and information boundary, inspect the timeline, test simple baselines fairly, add complexity only when it improves future forecasts, then describe how the result will operate safely after launch.

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
