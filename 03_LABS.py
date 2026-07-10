"""Runnable student labs. Search TODO; solutions are in 04_SOLUTIONS.md.

Run sections interactively (`python -i 03_LABS.py`) or paste them into a notebook.
The asserts provide quick feedback but do not reveal the implementation.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from make_data import build

df = build().set_index("date").sort_index()

# LAB 1 — AUDIT (10 min)
# TODO: expose missing dates with asfreq, report duplicates/frequency/missingness,
# and calculate mean demand by day of week.
audit = None
by_dow = None

# LAB 2 — SAFE FEATURES (15 min)
# TODO: create lag_1, lag_7, lag_14, lag_28, roll7_mean, roll28_mean,
# roll28_std, day-of-week one-hot features, trend, sin7, and cos7.
feat = df.copy()

# LAB 3 — TEMPORAL HOLDOUT + BASELINE (10 min)
# Use the last 90 observations as test. Fill historical target collection failures
# without using values after each gap. Do not fill the test target.
# TODO: create train/test and a lag-7 seasonal-naive prediction.
train = None
test = None
seasonal_pred = None

# LAB 4 — METRICS (10 min)
def metrics(y, pred, insample, m=7):
    """TODO: return dict with mae, rmse, bias (pred-y), and mase."""
    raise NotImplementedError


# LAB 5 — RIDGE (20 min)
# Assume future promotion, price, and holiday are planned and known, while measured
# future temperature is not known. Do not use temperature in this lab.
# TODO: fit Ridge(alpha=10) on complete feature rows and predict the holdout.
# For this fixed historical test, lag features are available; explain in writing why
# that is not a valid direct simulation of a forecast issued 90 days earlier.
ridge_pred = None

# LAB 6 — ROLLING ORIGINS (25 min)
def rolling_origins(n, initial=600, horizon=14, step=30):
    """TODO: yield (train_indices, validation_indices) using expanding history."""
    raise NotImplementedError


if __name__ == "__main__":
    print("Loaded", df.shape, "Search this file for TODO. Work in order.")
