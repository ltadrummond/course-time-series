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

That paragraph prevents most serious mistakes.
