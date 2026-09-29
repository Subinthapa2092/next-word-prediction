"""Central configuration: paths, data settings and hyperparameters.

Every value can be overridden with an environment variable, which is how the
Docker container and the tests point the project at different folders.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---- Paths ---------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", BASE_DIR / "outputs"))

# Built by scripts/download_data.py from the SwiftKey/HC Corpora dataset:
# a shuffled sample of blogs + news + twitter text (see that script's docstring).
DEFAULT_CORPUS = RAW_DIR / "corpus.txt"
VOCAB_PATH = MODELS_DIR / "tokenizer.json"

# Lines sampled from EACH of blogs / news / twitter when building corpus.txt.
# 15k x 3 keeps CPU training to minutes. For a full-scale run: CORPUS_LINES_PER_SOURCE=150000
CORPUS_LINES_PER_SOURCE = int(os.getenv("CORPUS_LINES_PER_SOURCE", 15_000))

# ---- Reproducibility -----------------------------------------------------
RANDOM_STATE = 42

# ---- Text / vocabulary ---------------------------------------------------
SEQ_LEN = int(os.getenv("SEQ_LEN", 5))        # words of context the model sees
MAX_VOCAB = int(os.getenv("MAX_VOCAB", 10_000))  # includes <pad> and <unk>
MIN_FREQ = int(os.getenv("MIN_FREQ", 2))      # rarer words become <unk>
PAD_TOKEN = "<pad>"   # id 0, used for left padding and masked by the model
UNK_TOKEN = "<unk>"   # id 1, any word outside the vocabulary
PAD_ID = 0
UNK_ID = 1

# ---- Data splits (done on sentences, before windowing, to avoid leakage) ---
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10

# Defaults below are the "quick run" profile (fast on a CPU). Override per run with env
# vars or CLI flags, e.g. UNITS=256 EMBED_DIM=128 EPOCHS=30 python -m src.train --model gru
# ---- Model ---------------------------------------------------------------
EMBED_DIM = int(os.getenv("EMBED_DIM", 64))
UNITS = int(os.getenv("UNITS", 64))
DROPOUT = float(os.getenv("DROPOUT", 0.3))
MODEL_KINDS = ("rnn", "lstm", "gru")

# ---- Training ------------------------------------------------------------
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 256))
EPOCHS = int(os.getenv("EPOCHS", 8))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-3))
EARLY_STOPPING_PATIENCE = 3

# ---- Serving -------------------------------------------------------------
## PRODUCTION_MODEL = os.getenv("PRODUCTION_MODEL", "gru")  # which models/<name>.keras the app loads
PRODUCTION_MODEL = os.getenv("PRODUCTION_MODEL", "gru")  # which models/<name>.keras the app loads — gru had the lowest perplexity, see README Results
TOP_K = 5
MAX_INPUT_CHARS = 500
