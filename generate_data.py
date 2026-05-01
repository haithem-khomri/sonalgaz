"""Generate synthetic hourly Sonelgaz-style power consumption data.

This project uses **synthetic** data (no real SCADA feed) so training and the
desktop app can run end-to-end without confidential sources.

Output
------
Writes ``sonelgaz_consumption_data.csv`` with one row per hour. Columns used
by ``train_models.py`` / ``app.py`` for learning and inference:

- Time / calendar: ``Timestamp``, ``Hour``, ``DayOfWeek``, ``Month``, ``Year``
- ``Season`` (1–4), ``IsHoliday`` (0/1 — random sample of days, simplified)
- Sensors: ``Temperature`` (°C), ``Current`` (A)
- Target: ``Consumption`` (kWh)

Notes
-----
``Year`` is kept in the CSV for possible future features but is **not** in the
model feature list today (see ``FEATURES`` in ``train_models.py``).
"""

import numpy as np
import pandas as pd

DEFAULT_START_DATE = "2024-01-01"
DEFAULT_PERIODS = 17520  # 2 years of hourly data.
DEFAULT_RANDOM_SEED = 42
OUTPUT_FILE = "sonelgaz_consumption_data.csv"


def get_season(month: int) -> int:
    """Map month to season code: 1=Winter, 2=Spring, 3=Summer, 4=Autumn."""
    if month in (12, 1, 2):
        return 1
    if month in (3, 4, 5):
        return 2
    if month in (6, 7, 8):
        return 3
    return 4


def generate_sonelgaz_data(
    start_date: str = DEFAULT_START_DATE,
    periods: int = DEFAULT_PERIODS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Generate synthetic hourly Sonelgaz consumption data."""
    rng = np.random.default_rng(random_seed)
    date_range = pd.date_range(start=start_date, periods=periods, freq="h")

    df = pd.DataFrame({"Timestamp": date_range})
    df["Hour"] = df["Timestamp"].dt.hour
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
    df["Month"] = df["Timestamp"].dt.month
    df["Year"] = df["Timestamp"].dt.year
    df["Season"] = df["Month"].apply(get_season)

    # Temperature profile: baseline + seasonal + intraday variation + noise.
    seasonal_temp = 15 * np.sin(2 * np.pi * (df["Timestamp"].dt.dayofyear / 365.25 - 0.25))
    daily_temp = 5 * np.sin(2 * np.pi * (df["Hour"] / 24 - 0.5))
    df["Temperature"] = 20 + seasonal_temp + daily_temp + rng.normal(0, 2, periods)

    # Current profile: follows seasonal/daily loads with noise.
    seasonal_current = 40 * np.sin(2 * np.pi * (df["Timestamp"].dt.dayofyear / 365.25 - 0.25))
    daily_current = 30 * np.sin(2 * np.pi * (df["Hour"] / 24 - 0.3))
    df["Current"] = 150 + seasonal_current + daily_current + rng.normal(0, 5, periods)

    # Consumption profile: combines weather, hourly usage patterns, weekdays, and seasonality.
    temp_effect = 0.5 * (df["Temperature"] - 20) ** 2  # U-shape: heating/cooling demand.
    hourly_effect = 50 * np.sin(2 * np.pi * (df["Hour"] / 24 - 0.3)) + 30 * np.sin(4 * np.pi * (df["Hour"] / 24))
    weekly_effect = df["DayOfWeek"].apply(lambda day: -40 if day in (4, 5) else 10)
    seasonal_effect = 100 * np.abs(np.sin(2 * np.pi * (df["Timestamp"].dt.dayofyear / 365.25)))
    df["Consumption"] = 500 + temp_effect + hourly_effect + weekly_effect + seasonal_effect + rng.normal(0, 15, periods)

    # Domain constraints.
    df["Consumption"] = df["Consumption"].clip(lower=0)
    df["Current"] = df["Current"].clip(lower=0)

    # Simplified holiday indicator.
    df["IsHoliday"] = 0
    holiday_dates = df["Timestamp"].dt.date.unique()
    chosen_holidays = rng.choice(holiday_dates, size=15, replace=False)
    df.loc[df["Timestamp"].dt.date.isin(chosen_holidays), "IsHoliday"] = 1

    return df


if __name__ == "__main__":
    data = generate_sonelgaz_data()
    data.to_csv(OUTPUT_FILE, index=False)
    print(f"Generated {len(data)} rows of synthetic data and saved to '{OUTPUT_FILE}'.")
    print(data.head())
