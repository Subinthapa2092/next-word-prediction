"""Model definitions. Only the recurrent layer changes between models."""
from __future__ import annotations

import config


def build_model(
    kind: str,
    vocab_size: int,
    seq_len: int = config.SEQ_LEN,
    embed_dim: int = config.EMBED_DIM,
    units: int = config.UNITS,
    dropout: float = config.DROPOUT,
):
    """Embedding -> (SimpleRNN | LSTM | GRU) -> Dropout -> Dense softmax.

    The final Dense layer is the "linear layer" in the design sketch and the
    softmax turns its scores into one probability per vocabulary word.
    """
    import keras
    from keras import layers

    kind = kind.lower()
    recurrent = {"rnn": layers.SimpleRNN, "lstm": layers.LSTM, "gru": layers.GRU}
    if kind not in recurrent:
        raise ValueError(f"Unknown model kind '{kind}'. Choose from {sorted(recurrent)}.")

    model = keras.Sequential(
        [
            keras.Input(shape=(seq_len,), dtype="int32"),
            layers.Embedding(vocab_size, embed_dim, mask_zero=True, name="embedding"),
            layers.Dropout(dropout, name="embed_dropout"),
            recurrent[kind](units, name=kind),
            layers.Dropout(dropout, name="rnn_dropout"),
            layers.Dense(vocab_size, activation="softmax", name="next_word_probs"),
        ],
        name=f"next_word_{kind}",
    )
    return model


def compile_model(model, learning_rate: float = config.LEARNING_RATE):
    import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="top1_accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )
    return model
