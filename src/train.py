"""Train one model and record its results.

    python -m src.train --model lstm
    python -m src.train --model gru --epochs 20 --data data/raw/my_corpus.txt
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import config
from src.dataset import load_corpus, prepare_data, save_stats, to_tf_dataset
from src.model import build_model, compile_model
from src.utils import plot_history, set_seed, update_metrics


def train(
    kind: str,
    corpus_path: Path | str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    seq_len: int | None = None,
    embed_dim: int | None = None,
    units: int | None = None,
    dropout: float | None = None,
    learning_rate: float | None = None,
    models_dir: Path | str | None = None,
    outputs_dir: Path | str | None = None,
    verbose: int = 2,
) -> dict:
    """Train one model. Any argument left as None is read from `config` at CALL time
    (not import time), so editing config.py / env vars and re-running a notebook cell just works."""
    import keras

    corpus_path = corpus_path or config.DEFAULT_CORPUS
    epochs = config.EPOCHS if epochs is None else epochs
    batch_size = config.BATCH_SIZE if batch_size is None else batch_size
    seq_len = config.SEQ_LEN if seq_len is None else seq_len
    embed_dim = config.EMBED_DIM if embed_dim is None else embed_dim
    units = config.UNITS if units is None else units
    dropout = config.DROPOUT if dropout is None else dropout
    learning_rate = config.LEARNING_RATE if learning_rate is None else learning_rate
    models_dir = models_dir or config.MODELS_DIR
    outputs_dir = outputs_dir or config.OUTPUTS_DIR

    set_seed()
    models_dir, outputs_dir = Path(models_dir), Path(outputs_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    splits = prepare_data(load_corpus(corpus_path), seq_len=seq_len)
    splits.vocab.save(models_dir / "tokenizer.json")
    save_stats(splits, config.PROCESSED_DIR / "stats.json")
    print("Data:", json.dumps(splits.stats()))

    model = compile_model(
        build_model(kind, len(splits.vocab), seq_len, embed_dim, units, dropout), learning_rate
    )
    model.summary()

    train_ds = to_tf_dataset(splits.x_train, splits.y_train, batch_size, shuffle=True)
    val_ds = to_tf_dataset(splits.x_val, splits.y_val, batch_size)
    test_ds = to_tf_dataset(splits.x_test, splits.y_test, batch_size)

    ckpt = models_dir / f"{kind}.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5),
        keras.callbacks.CSVLogger(outputs_dir / f"{kind}_training_log.csv"),
    ]

    start = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=verbose,
                        shuffle=False)  # shuffling is done inside tf.data
    train_seconds = time.perf_counter() - start

    test = model.evaluate(test_ds, return_dict=True, verbose=0)
    val_losses = history.history["val_loss"]
    record = {
        "model": kind,
        "parameters": int(model.count_params()),
        "top1_accuracy": round(float(test["top1_accuracy"]), 4),
        "top5_accuracy": round(float(test["top5_accuracy"]), 4),
        "test_loss": round(float(test["loss"]), 4),
        "perplexity": round(math.exp(float(test["loss"])), 2),
        "train_seconds": round(train_seconds, 1),
        "epochs_run": len(val_losses),
        "best_epoch": int(val_losses.index(min(val_losses)) + 1),
        "hyperparameters": {
            "seq_len": seq_len, "embed_dim": embed_dim, "units": units, "dropout": dropout,
            "batch_size": batch_size, "learning_rate": learning_rate, **splits.stats(),
        },
    }
    update_metrics(kind, record, outputs_dir / "metrics.json")
    (outputs_dir / f"{kind}_history.json").write_text(json.dumps(history.history), encoding="utf-8")
    plot_history(history.history, kind.upper(), outputs_dir / f"{kind}_training_curves.png")
    print(json.dumps(record, indent=2))
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=config.MODEL_KINDS, default="lstm")
    p.add_argument("--data", default=str(config.DEFAULT_CORPUS), help="path to a plain-text corpus")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--seq-len", type=int, default=config.SEQ_LEN)
    p.add_argument("--embed-dim", type=int, default=config.EMBED_DIM)
    p.add_argument("--units", type=int, default=config.UNITS)
    p.add_argument("--dropout", type=float, default=config.DROPOUT)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    a = p.parse_args()
    train(a.model, a.data, a.epochs, a.batch_size, a.seq_len, a.embed_dim, a.units, a.dropout, a.lr)


if __name__ == "__main__":
    main()
