"""
Converts the flat feature table into (lookback-window, horizon-target)
sequence samples per store-item series, for feeding into the LSTM/GRU.
"""

import numpy as np
import pandas as pd

from src.feature_engineering import FEATURE_COLUMNS

SEQ_FEATURE_COLUMNS = ["sales"] + FEATURE_COLUMNS


def build_sequences(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
    val_holdout_days: int,
    stride: int = 1,
):
    """
    Returns train and validation sets:
      X_seq, X_store, X_item, y   (for both train and val)

    The last `val_holdout_days` of each series form the validation targets;
    everything before that is available for training windows. This keeps
    validation strictly out-of-time, as it would be in production use.
    """
    seq_arrays, store_arrays, item_arrays, targets = [], [], [], []
    val_seq, val_store, val_item, val_targets = [], [], [], []

    for (store, item), group in df.groupby(["store", "item"]):
        group = group.sort_values("date").reset_index(drop=True)
        feats = group[SEQ_FEATURE_COLUMNS].values
        sales = group["sales"].values
        store_idx = group["store_idx"].iloc[0]
        item_idx = group["item_idx"].iloc[0]

        n = len(group)
        cutoff = n - val_holdout_days  # last val_holdout_days reserved for validation targets

        # training windows: every `stride`-th window whose full horizon lands before cutoff
        for start in range(0, cutoff - lookback - horizon + 1, stride):
            end = start + lookback
            seq_arrays.append(feats[start:end])
            store_arrays.append(store_idx)
            item_arrays.append(item_idx)
            targets.append(sales[end:end + horizon])

        # one validation sample: last `lookback` days before cutoff predict the holdout window
        if cutoff - lookback >= 0 and cutoff + horizon <= n:
            val_seq.append(feats[cutoff - lookback:cutoff])
            val_store.append(store_idx)
            val_item.append(item_idx)
            val_targets.append(sales[cutoff:cutoff + horizon])

    train = (
        np.array(seq_arrays, dtype=np.float32),
        np.array(store_arrays, dtype=np.int32),
        np.array(item_arrays, dtype=np.int32),
        np.array(targets, dtype=np.float32),
    )
    val = (
        np.array(val_seq, dtype=np.float32),
        np.array(val_store, dtype=np.int32),
        np.array(val_item, dtype=np.int32),
        np.array(val_targets, dtype=np.float32),
    )
    return train, val
