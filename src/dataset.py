"""Build (previous words -> next word) training pairs with sliding windows."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from src.preprocess import Vocabulary, split_sentences


@dataclass
class Splits:
    vocab: Vocabulary
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    def stats(self) -> dict:
        return {
            "vocab_size": len(self.vocab),
            "train_samples": int(len(self.y_train)),
            "val_samples": int(len(self.y_val)),
            "test_samples": int(len(self.y_test)),
            "seq_len": int(self.x_train.shape[1]),
        }


def make_windows(id_sentences: list[list[int]], seq_len: int = config.SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows inside each sentence, left padded with <pad>.

    "i want to go" with seq_len=3 gives
        [0 0 i]    -> want
        [0 i want] -> to
        [i want to]-> go
    so the model also learns to predict from very short contexts.
    """
    xs, ys = [], []
    for ids in id_sentences:
        n = len(ids)
        if n < 2:
            continue
        arr = np.asarray(ids, dtype=np.int32)
        padded = np.concatenate([np.zeros(seq_len, dtype=np.int32), arr])
        windows = np.lib.stride_tricks.sliding_window_view(padded, seq_len)  # (n + 1, seq_len)
        xs.append(windows[1:n])
        ys.append(arr[1:])
    if not xs:
        return np.empty((0, seq_len), dtype=np.int32), np.empty((0,), dtype=np.int32)
    return np.concatenate(xs), np.concatenate(ys)


def split_sentence_list(sentences: list, val_fraction: float, test_fraction: float, seed: int):
    """Shuffle whole sentences, then cut into train / val / test."""
    order = np.random.default_rng(seed).permutation(len(sentences))
    n_test = int(len(sentences) * test_fraction)
    n_val = int(len(sentences) * val_fraction)
    test = [sentences[i] for i in order[:n_test]]
    val = [sentences[i] for i in order[n_test:n_test + n_val]]
    train = [sentences[i] for i in order[n_test + n_val:]]
    return train, val, test


def prepare_data(
    text: str,
    seq_len: int = config.SEQ_LEN,
    max_vocab: int = config.MAX_VOCAB,
    min_freq: int = config.MIN_FREQ,
    val_fraction: float = config.VAL_FRACTION,
    test_fraction: float = config.TEST_FRACTION,
    seed: int = config.RANDOM_STATE,
) -> Splits:
    """Full data pipeline. The vocabulary is fit on training sentences only."""
    sentences = split_sentences(text)
    if len(sentences) < 10:
        raise ValueError("Corpus is too small: fewer than 10 usable sentences.")
    train, val, test = split_sentence_list(sentences, val_fraction, test_fraction, seed)
    vocab = Vocabulary.build(train, max_size=max_vocab, min_freq=min_freq)

    def encode(part):
        return [vocab.encode(s) for s in part]

    x_tr, y_tr = make_windows(encode(train), seq_len)
    x_va, y_va = make_windows(encode(val), seq_len)
    x_te, y_te = make_windows(encode(test), seq_len)
    return Splits(vocab, x_tr, y_tr, x_va, y_va, x_te, y_te)


def load_corpus(path: Path | str = config.DEFAULT_CORPUS) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {path}. Run `python scripts/download_data.py` "
            "or pass --data path/to/your.txt"
        )
    return path.read_text(encoding="utf-8", errors="ignore")


def save_stats(splits: Splits, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits.stats(), indent=2), encoding="utf-8")


def to_tf_dataset(x: np.ndarray, y: np.ndarray, batch_size: int = config.BATCH_SIZE, shuffle: bool = False):
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(min(len(y), 100_000), seed=config.RANDOM_STATE, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
