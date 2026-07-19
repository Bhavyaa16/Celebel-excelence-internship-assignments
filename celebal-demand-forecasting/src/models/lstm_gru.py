"""
Global sequence model: one LSTM/GRU trained across all store-item series.

Inputs:
  - seq_input:   (lookback, n_seq_features) — daily sales + engineered features
                 over the lookback window
  - store_input: () int — store id, looked up in a learned embedding
  - item_input:  () int — item id, looked up in a learned embedding

The recurrent layer captures each series' own temporal dynamics
(trend, autocorrelation); the embeddings let the single global model
still tell series apart and share statistical strength across all of them,
which is the whole point of going global instead of one model per series.

Output: a vector of length `horizon` — direct multi-step forecast,
avoiding the compounding error of feeding predictions back in recursively.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_model(
    lookback: int,
    n_seq_features: int,
    n_stores: int,
    n_items: int,
    horizon: int,
    rnn_type: str = "lstm",
    rnn_units: int = 64,
    embedding_dim: int = 8,
) -> Model:
    seq_input = layers.Input(shape=(lookback, n_seq_features), name="seq_input")
    store_input = layers.Input(shape=(1,), name="store_input")
    item_input = layers.Input(shape=(1,), name="item_input")

    store_emb = layers.Embedding(n_stores, embedding_dim, name="store_embedding")(store_input)
    item_emb = layers.Embedding(n_items, embedding_dim, name="item_embedding")(item_input)
    store_emb = layers.Flatten()(store_emb)
    item_emb = layers.Flatten()(item_emb)

    rnn_layer = layers.LSTM if rnn_type.lower() == "lstm" else layers.GRU
    x = rnn_layer(rnn_units, return_sequences=True)(seq_input)
    x = rnn_layer(rnn_units // 2)(x)

    merged = layers.Concatenate()([x, store_emb, item_emb])
    merged = layers.Dense(64, activation="relu")(merged)
    merged = layers.Dropout(0.2)(merged)
    output = layers.Dense(horizon, activation="relu", name="forecast")(merged)

    model = Model(inputs=[seq_input, store_input, item_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber", metrics=["mae"])
    return model
