"""
Builds the feature set consumed by both the baseline and the LSTM/GRU model:
  - lag features (1, 7, 14, 28, 365 days)
  - rolling mean/std (7-day, 28-day)
  - calendar features (day-of-week, month, is-weekend)
  - integer-encoded store/item ids (for embedding lookups)

All features are computed per store-item series so no information leaks
across series.
"""

import numpy as np
import pandas as pd

LAGS = [1, 7, 14, 28, 365]
ROLLING_WINDOWS = [7, 28]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby(["store", "item"])["sales"]

    for lag in LAGS:
        df[f"lag_{lag}"] = grouped.shift(lag)

    for window in ROLLING_WINDOWS:
        shifted = grouped.shift(1)  # avoid leaking today's value into its own rolling stat
        df[f"roll_mean_{window}"] = (
            shifted.groupby([df["store"], df["item"]])
            .rolling(window)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
        df[f"roll_std_{window}"] = (
            shifted.groupby([df["store"], df["item"]])
            .rolling(window)
            .std()
            .reset_index(level=[0, 1], drop=True)
        )

    return df


def encode_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Integer-encode store and item for use as embedding indices."""
    df = df.copy()
    store_map = {int(v): i for i, v in enumerate(sorted(df["store"].unique()))}
    item_map = {int(v): i for i, v in enumerate(sorted(df["item"].unique()))}
    df["store_idx"] = df["store"].map(store_map)
    df["item_idx"] = df["item"].map(item_map)
    return df, store_map, item_map


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    df, store_map, item_map = encode_ids(df)
    # earliest rows won't have a full 365-day lag — drop them, they can't be used for training anyway
    df = df.dropna().reset_index(drop=True)
    return df, store_map, item_map


FEATURE_COLUMNS = (
    [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"roll_std_{w}" for w in ROLLING_WINDOWS]
    + ["day_of_week", "month", "is_weekend", "day_of_year"]
)
