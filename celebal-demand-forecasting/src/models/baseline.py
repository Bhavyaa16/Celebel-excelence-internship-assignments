"""
Naive seasonal baseline: predicts day D's sales as the value from the same
weekday, one week earlier. This is the yardstick — if the LSTM/GRU model
can't beat this, it isn't adding value.
"""

import pandas as pd


def seasonal_naive_forecast(history: pd.DataFrame, horizon: int) -> pd.Series:
    """
    history: a single store-item series, sorted by date, with a 'sales' column
    horizon: number of days to forecast forward

    Returns a Series of length `horizon` — repeats the last 7 observed days
    forward across the forecast window.
    """
    last_week = history["sales"].values[-7:]
    reps = (horizon // 7) + 1
    forecast = list(last_week) * reps
    return pd.Series(forecast[:horizon])
