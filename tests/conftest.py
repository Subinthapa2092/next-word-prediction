import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "tests" / "fixtures" / "sample_corpus.txt"


@pytest.fixture(scope="session")
def corpus_text() -> str:
    return CORPUS.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    """Train a tiny LSTM once and share model + tokenizer paths across tests."""
    from src.train import train

    out = tmp_path_factory.mktemp("artifacts")
    train("lstm", CORPUS, epochs=25, batch_size=64, seq_len=4, embed_dim=16, units=32,
          models_dir=out, outputs_dir=out, verbose=0)
    return {"model": out / "lstm.keras", "vocab": out / "tokenizer.json", "dir": out}


@pytest.fixture(scope="session")
def predictor(artifacts):
    from src.predict import Predictor

    return Predictor.load(artifacts["model"], artifacts["vocab"])
