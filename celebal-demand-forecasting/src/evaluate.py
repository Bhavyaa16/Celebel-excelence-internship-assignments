"""
Loads the trained model + saved validation arrays, generates predictions,
and computes WAPE (primary) and RMSE (secondary) both overall and per
store-item series. Results are saved to artifacts/metrics.json for the
dashboard and RAG assistant to read directly, with no retraining needed.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sum(np.abs(y_true - y_pred)) / max(np.sum(np.abs(y_true)), 1e-8))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    model = tf.keras.models.load_model(ARTIFACTS_DIR / "lstm_model.keras")
    arrays = np.load(ARTIFACTS_DIR / "validation_arrays.npz")
    Xv_seq, Xv_store, Xv_item, yv = (
        arrays["Xv_seq"], arrays["Xv_store"], arrays["Xv_item"], arrays["yv"]
    )

    with open(ARTIFACTS_DIR / "id_maps.json") as f:
        id_maps = json.load(f)
    # id_maps.json stores keys as strings (JSON forces this) — cast back to
    # int here so downstream sorting/display in the dashboard is numeric,
    # not alphabetical (which would put store 10 right after store 1).
    store_lookup = {v: int(k) for k, v in id_maps["store_map"].items()}
    item_lookup = {v: int(k) for k, v in id_maps["item_map"].items()}

    preds = model.predict([Xv_seq, Xv_store, Xv_item], verbose=0)

    overall = {"wape": wape(yv, preds), "rmse": rmse(yv, preds)}

    per_series = []
    for i in range(len(preds)):
        per_series.append({
            "store": store_lookup[int(Xv_store[i])],
            "item": item_lookup[int(Xv_item[i])],
            "wape": wape(yv[i], preds[i]),
            "rmse": rmse(yv[i], preds[i]),
            "actual": yv[i].tolist(),
            "forecast": preds[i].tolist(),
        })

    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump({"overall": overall, "per_series": per_series}, f)

    print(f"Overall WAPE: {overall['wape']:.3f} | RMSE: {overall['rmse']:.2f}")
    print(f"Saved metrics for {len(per_series)} series to {ARTIFACTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
