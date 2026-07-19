"""
End-to-end training run:
  load data -> build features -> build sequences -> train LSTM/GRU -> save artifacts

Usage:
    python -m src.train
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data_loader import load_sales_data
from src.feature_engineering import build_feature_frame
from src.sequence_builder import build_sequences, SEQ_FEATURE_COLUMNS
from src.models.lstm_gru import build_model

LOOKBACK = 90
HORIZON = 90          # max horizon; dashboard slices this down to 7/15/30/60/90
VAL_HOLDOUT_DAYS = 90  # last 3 months held out, matches the Kaggle test window
TRAIN_STRIDE = 14      # thin overlapping windows — neighboring windows are highly
                       # redundant, so this keeps training tractable without losing signal
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    print("Loading raw data...")
    raw = load_sales_data()

    print("Building features...")
    feats, store_map, item_map = build_feature_frame(raw)

    print("Building train/validation sequences (this is the slow step)...")
    (X_seq, X_store, X_item, y), (Xv_seq, Xv_store, Xv_item, yv) = build_sequences(
        feats, lookback=LOOKBACK, horizon=HORIZON, val_holdout_days=VAL_HOLDOUT_DAYS,
        stride=TRAIN_STRIDE,
    )
    print(f"Train windows: {len(X_seq):,} | Validation windows: {len(Xv_seq):,}")

    model = build_model(
        lookback=LOOKBACK,
        n_seq_features=len(SEQ_FEATURE_COLUMNS),
        n_stores=len(store_map),
        n_items=len(item_map),
        horizon=HORIZON,
        rnn_type="lstm",
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            str(ARTIFACTS_DIR / "lstm_model.keras"), save_best_only=True
        ),
    ]

    model.fit(
        [X_seq, X_store, X_item],
        y,
        validation_data=([Xv_seq, Xv_store, Xv_item], yv),
        epochs=5,
        batch_size=256,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(ARTIFACTS_DIR / "lstm_model.keras")

    with open(ARTIFACTS_DIR / "id_maps.json", "w") as f:
        json.dump({"store_map": store_map, "item_map": item_map}, f)

    np.savez(
        ARTIFACTS_DIR / "validation_arrays.npz",
        Xv_seq=Xv_seq, Xv_store=Xv_store, Xv_item=Xv_item, yv=yv,
    )

    print(f"Done. Artifacts saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
