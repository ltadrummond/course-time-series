"""Create a realistic daily retail forecasting dataset with no external download."""
from pathlib import Path

import numpy as np
import pandas as pd


def build(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=900, freq="D")
    t = np.arange(len(dates))
    dow = dates.dayofweek.to_numpy()
    annual = np.sin(2 * np.pi * t / 365.25)
    weekly = np.array([0, -2, -1, 1, 4, 8, 5])[dow]
    promo = rng.binomial(1, np.where(dow >= 4, .13, .07))
    price = 10 + .002 * t + .35 * np.sin(2 * np.pi * t / 90) - .7 * promo
    temperature = 14 + 10 * np.sin(2 * np.pi * (t - 110) / 365.25) + rng.normal(0, 2, len(t))
    holiday = ((dates.month == 12) & (dates.day >= 20)).astype(int)
    # Policy break at t=600 makes recent-history validation meaningful.
    level = 45 + .018 * t + np.where(t >= 600, 7, 0)
    demand = (
        level + weekly + 4 * annual + 8 * promo - 2.1 * (price - 10)
        + .13 * temperature + 6 * holiday + rng.normal(0, 3.2, len(t))
    )
    demand = np.maximum(0, np.round(demand)).astype(float)

    df = pd.DataFrame({
        "date": dates, "demand": demand, "price": price.round(2),
        "promotion": promo, "temperature": temperature.round(1),
        "holiday": holiday,
    })
    # Collection failures: targets remain unknown, not zero.
    miss = rng.choice(np.arange(30, 560), size=9, replace=False)
    df.loc[miss, "demand"] = np.nan
    return df


if __name__ == "__main__":
    out = Path("data/retail_daily.csv")
    out.parent.mkdir(exist_ok=True)
    df = build()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}; missing demand={df.demand.isna().sum()}")
