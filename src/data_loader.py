"""
Loads the raw store-item daily sales data and returns a clean,
sorted DataFrame ready for feature engineering.

Expected input file: data/train.csv with columns [date, store, item, sales]
(Kaggle "Store Item Demand Forecasting Challenge" format).
"""

import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "train.csv"


def load_sales_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Download train.csv from the Kaggle "
            "'Store Item Demand Forecasting Challenge' and place it in data/."
        )

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)

    # sanity checks — fail loudly rather than silently forecasting on bad data
    assert df["sales"].isna().sum() == 0, "Found missing sales values"
    assert (df["sales"] >= 0).all(), "Found negative sales values"

    return df


def series_id(df: pd.DataFrame) -> pd.Series:
    """A single id per store-item pair, e.g. 's3_i12', used for indexing."""
    return "s" + df["store"].astype(str) + "_i" + df["item"].astype(str)


if __name__ == "__main__":
    data = load_sales_data()
    print(f"Loaded {len(data):,} rows | "
          f"{data['store'].nunique()} stores x {data['item'].nunique()} items | "
          f"{data['date'].min().date()} to {data['date'].max().date()}")
