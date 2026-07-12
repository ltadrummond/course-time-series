# Time Series Interview Sprint — 10 Hours

An interview-focused course for an experienced data scientist who is new to time series. It teaches the minimum rigorous theory, the practical workflow, and the failure modes interviewers look for.

## Start here

You do not need to understand every term on the first read. Work one chapter at a time and use this loop:

1. Read the chapter slowly. Look up or ask about any word that is unclear before continuing.
2. Answer its checkpoint in plain language. A checkpoint is practice, not an exam.
3. Do the next matching TODO in [03_LABS.py](03_LABS.py) in JupyterLab. The labs give you hands-on Python practice; the checkpoints test your reasoning.
4. Only then compare with [04_SOLUTIONS.md](04_SOLUTIONS.md). Focus on why your answer differs, rather than memorising the code.
5. Continue to the next chapter.

After the course, complete [05_CAPSTONE.md](05_CAPSTONE.md), take [06_THEORY_TEST.md](06_THEORY_TEST.md) without notes, and use [07_ANSWER_KEY.md](07_ANSWER_KEY.md) to review.

## Full learning order

Use this order from beginning to end:

1. Read one chapter of [02_COURSE.md](02_COURSE.md). Stop and ask about any unfamiliar term; understanding matters more than speed.
2. Attempt the checkpoint at the end of that chapter. Write a short answer in your own words; it is practice, not a graded test.
3. Complete the next relevant TODO in [03_LABS.py](03_LABS.py) in JupyterLab. The labs are practical Python exercises, and should be done in order from Lab 1 to Lab 6.
4. Read the matching part of [04_SOLUTIONS.md](04_SOLUTIONS.md) only after making an honest attempt. Correct your work and note the reason for any mistake.
5. Repeat until you have completed all eight course chapters.
6. Complete [05_CAPSTONE.md](05_CAPSTONE.md). This is one larger, end-to-end forecasting project that combines the ideas from the course.
7. Take [06_THEORY_TEST.md](06_THEORY_TEST.md) closed-book: do not consult the course, solutions, notes, or internet while answering.
8. Use [07_ANSWER_KEY.md](07_ANSWER_KEY.md) to grade the theory test. Review weak topics, then retry the missed questions later without notes.

If the course feels difficult, slow down rather than skipping ahead. You can finish it in the suggested 10 hours, but taking longer is completely fine.

## How the concepts connect in practice

The course follows the real workflow of building a forecast. Each chapter answers a different practical question:

| Chapter | Practical question | What it changes in your ML work |
|---|---|---|
| 1. Patterns | What is happening in the data—trend, seasonality, or noise? | Choose useful lags/calendar features and avoid misleading correlations. |
| 2. Data preparation | Are the dates, missing values, and scale trustworthy? | Build a correct model input table without inventing future information. |
| 3. Evaluation | What simple forecast must we beat, and how do we test fairly? | Select metrics and time-based validation before comparing models. |
| 4. Regression | How do we turn history into ML input columns? | Create lags, rolling features, calendar features, and known-future inputs safely. |
| 5. Classical models | Can level, trend, and seasonality be modeled directly? | Compare ETS/ARIMA with ML models; these are often strong baselines. |
| 6. ML at scale | How do external inputs and many products/stores change the design? | Choose tree/linear/global models and make grouped features safely. |
| 7. Deep learning | Is a neural network justified? | Use it only when it wins fairly over simpler models. |
| 8. Production | Will the forecast remain reliable after launch? | Monitor data, errors, drift, and fall back safely when needed. |

In short: understand the pattern → prepare time-safe data → establish a baseline → create safe features → compare models on future-like periods → deploy and monitor.

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
