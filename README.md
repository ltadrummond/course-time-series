# Time Series Interview Sprint — 10 Hours

An interview-focused course for an experienced data scientist who is new to time series. It teaches the minimum rigorous theory, the practical workflow, and the failure modes interviewers look for.

## Start here

1. Read [STUDY_PLAN.md](STUDY_PLAN.md) and reserve one uninterrupted 10-hour block (or five 2-hour sessions).
2. Work through [COURSE.md](COURSE.md) in order. Do each checkpoint without looking at the answers.
3. Run [LABS.py](LABS.py), or use the matching exercises in `COURSE.md` if you prefer paper practice.
4. Check [SOLUTIONS.md](SOLUTIONS.md) only after making a serious attempt.
5. Complete [CAPSTONE.md](CAPSTONE.md), then sit [THEORY_TEST.md](THEORY_TEST.md) closed-book.
6. Grade yourself with [ANSWER_KEY.md](ANSWER_KEY.md).

## What is included

- Eight short chapters with intuition, essential mathematics, Python patterns, exercises, flashcards, and cheat sheets.
- A realistic end-to-end forecasting capstone with rubric and worked solution.
- A 60-minute theory test plus a 90-minute practical test.
- Complete answers and explanations.
- A reproducible synthetic dataset, so no download or API key is required.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python make_data.py
jupyter lab
```

The core course does not require a deep-learning framework. To experiment beyond the architecture exercise, run `pip install -r requirements-dl.txt`. If installation time is tight, skip it; knowing when *not* to use deep learning is more valuable in most time-series interviews.

## The interview answer pattern

For almost any forecasting prompt, say this before naming a model:

> I will define the forecast origin, horizon, information available at prediction time, business loss, and deployment cadence; inspect temporal structure and data quality; establish seasonal-naive baselines; use walk-forward validation with leakage-safe feature construction; compare simple and complex candidates on several origins; inspect residuals and stability; then retrain and monitor.

In plain language, this means:

- **Forecast origin:** When is the prediction made? For example, every Monday at 09:00.
- **Horizon:** How far into the future are we predicting? For example, the next seven days.
- **Information available at prediction time:** Which data genuinely exists when the prediction is made? This prevents accidentally using future information.
- **Business loss:** Which mistakes matter most? For inventory, predicting too little demand may be more costly than predicting too much.
- **Deployment cadence:** How often will predictions be produced, and how often will the model be retrained?
- **Temporal structure and data quality:** Look for trends, seasonality, missing values, outliers, and changes in behaviour over time.
- **Seasonal-naive baseline:** Start with a simple forecast such as "next Monday will be like last Monday." A more complex model should beat it.
- **Walk-forward validation:** Simulate real use by training on the past, predicting the next period, moving forward, and repeating.
- **Leakage-safe features:** Build every feature using only information that would have been available at that historical prediction time.
- **Several forecast origins:** Compare models across many historical prediction dates, not just one convenient train/test split.
- **Residuals and stability:** Examine the errors to find periods or situations in which the model fails or becomes unreliable.
- **Retraining and monitoring:** After deployment, update the model and watch for deteriorating accuracy or changes in the data.

For example:

> Every Monday, I need to predict store demand for the next seven days. I can use sales and weather forecasts available by Monday morning, but not actual weather or sales from later in the week. Stockouts cost more than excess inventory. I will compare models against "the same weekday last week," test them by repeatedly simulating past Mondays, inspect their errors, and monitor their performance after deployment.

This process prevents common forecasting mistakes—especially future-data leakage, unrealistic evaluation, and a poorly defined business objective—before model selection even begins.
