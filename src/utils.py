"""Small helpers: seeding, metrics file, plots."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

import config


def set_seed(seed: int = config.RANDOM_STATE) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:  # tests for preprocessing do not need TF
        pass


def update_metrics(kind: str, record: dict, path: Path | str | None = None) -> dict:
    """Merge one model's results into outputs/metrics.json."""
    path = Path(path or config.OUTPUTS_DIR / "metrics.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[kind] = record
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def plot_history(history: dict, title: str, path: Path | str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["loss"]) + 1)
    axes[0].plot(epochs, history["loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set(title=f"{title}: loss", xlabel="epoch", ylabel="cross-entropy")
    axes[1].plot(epochs, history["top1_accuracy"], label="top-1 train")
    axes[1].plot(epochs, history["val_top1_accuracy"], label="top-1 val")
    axes[1].plot(epochs, history["top5_accuracy"], "--", label="top-5 train")
    axes[1].plot(epochs, history["val_top5_accuracy"], "--", label="top-5 val")
    axes[1].set(title=f"{title}: accuracy", xlabel="epoch")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
