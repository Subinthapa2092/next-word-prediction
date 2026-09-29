# Next-Word Prediction (SimpleRNN vs LSTM vs GRU)

A phone-keyboard-style next-word predictor: type a sentence, get five ranked suggestions for what comes next, or for how to finish the word you're mid-typing. Built to compare three recurrent architectures on the exact same data and pick a winner for production.

Dataset: **SwiftKey / HC Corpora** — blogs, news and Twitter text (en_US), the real corpus SwiftKey released for exactly this task, via Coursera's Data Science Specialization capstone. Mixing three registers (long-form blogs, edited news, short noisy tweets) matters here: a model trained only on 19th-century prose or only on one register won't generalize to how people actually type.

> **License note:** this dataset is free and widely used for research, learning and portfolio projects like this one. Its terms for a shipped commercial product aren't spelled out as a permissive license (e.g. MIT/CC-BY), so if this ever becomes a commercial app, swap in something with clearer commercial terms — WikiText-103, or your own collected user text.

This is the companion project to the article series *[RNN / LSTM / GRU From First Principles](#)*: theory in the articles, this repo is the implementation.

## What it does

Given text like `"i want to go to the "`, the model reads the last few words and ranks every word in its vocabulary by how likely it is to come next. The app shows the top 5. If you're mid-word (`"i want to g"`), it instead ranks completions that start with `g`.

```
"i want to go to the"
        │
   clean + tokenize
        │
  ["i","want","to","go","to","the"]
        │
   token ids (vocab lookup)
        │
      Embedding
        │
    LSTM / GRU / SimpleRNN
        │
     Dense (linear layer)
        │
      softmax
        │
  probability per word  →  top 5 suggestions
```

## Results

Run `notebooks/05_comparison.ipynb` to fill this in with your own numbers — do not trust numbers you haven't produced yourself.

| Model | Parameters | Top-1 accuracy | Top-5 accuracy | Perplexity | Training time |
|---|---|---|---|---|---|
| SimpleRNN | | | | | |
| LSTM | | | | | |
| GRU | | | | | |

The notebook also writes this table to `outputs/results.md` and saves comparison charts to `outputs/model_comparison.png` and `outputs/validation_curves_comparison.png`.

`config.PRODUCTION_MODEL` (env var `PRODUCTION_MODEL`) selects which trained model the app serves. Perplexity is the deciding metric; top-5 accuracy is what users feel; training time and parameter count break ties.

## Project structure

```
next-word-prediction/
  data/
    raw/              downloaded corpus (gitignored)
    processed/        cached train/val/test windows + stats.json (gitignored)
  models/              tokenizer.json + <kind>.keras checkpoints (gitignored)
  notebooks/
    01_eda.ipynb              corpus stats, sentence lengths, Zipf's law, vocab coverage
    02_preprocessing.ipynb    clean → tokenize → ids → sliding windows, step by step
    03_lstm.ipynb             train + evaluate the LSTM
    04_gru.ipynb              train + evaluate the GRU
    05_comparison.ipynb       SimpleRNN vs LSTM vs GRU, side by side
  outputs/             training curves, metrics.json, comparison charts (gitignored)
  src/
    preprocess.py      cleaning, tokenizing, the Vocabulary class
    dataset.py         sliding-window pair construction, train/val/test splits
    model.py           build_model(kind, ...) — the only thing that changes is the recurrent layer
    train.py           training script / CLI, writes metrics + training curves
    predict.py         Predictor — the inference class the app and notebooks both use
    utils.py           seeding, metrics file, plotting
  static/              style.css, app.js for the web UI
  templates/           index.html
  tests/               pytest suite (36 tests): preprocessing, model, predictor, API
  scripts/
    download_data.py   downloads the SwiftKey archive, extracts it, builds data/raw/corpus.txt
    build_corpus.py    samples + mixes blogs/news/twitter without re-downloading
  app.py               FastAPI app (web UI + JSON API)
  config.py            every path and hyperparameter, overridable via env vars
  Dockerfile, docker-compose.yml
  requirements.txt, requirements-dev.txt
  .github/workflows/ci.yml   lint + test + docker build on every push
```

## Setup

```bash
git clone <your-repo-url>
cd next-word-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python scripts/download_data.py        # ~560 MB download, builds data/raw/corpus.txt
```

By default this samples 150,000 lines from each of blogs/news/twitter (450k lines total) — enough for a real model without an all-day CPU training run. Adjust with `--lines-per-source`:

```bash
python scripts/download_data.py --lines-per-source 400000   # bigger corpus, longer training
python scripts/download_data.py --skip-download             # already have the zip; just re-mix
```

If the automatic download is blocked in your environment (some networks or sandboxes disallow direct file downloads), grab the archive yourself from the URL printed in the error message, save it to `data/raw/Coursera-SwiftKey.zip`, and re-run with `--skip-download`.

To use your own text instead, drop any plain-text file in `data/raw/` and pass `--data` to training, or point `config.DEFAULT_CORPUS` at it — no code changes needed, since `split_sentences`/`clean_text` handle both plain prose and social-media noise (URLs, @mentions, hashtags).

## Train

```bash
python -m src.train --model lstm
python -m src.train --model gru
python -m src.train --model rnn
```

Each run saves `models/<kind>.keras`, updates `models/tokenizer.json`, appends to `outputs/metrics.json`, and writes `outputs/<kind>_training_curves.png`. Or run the notebooks in order (`01` → `05`) for the full walkthrough with plots and commentary.

Key hyperparameters (see `config.py`, all overridable by environment variable):

| Variable | Default | Meaning |
|---|---|---|
| `SEQ_LEN` | 6 | words of context the model sees |
| `MAX_VOCAB` | 10000 | vocabulary cap, including `<pad>` and `<unk>` |
| `MIN_FREQ` | 2 | words rarer than this become `<unk>` |
| `EMBED_DIM` | 128 | embedding size |
| `UNITS` | 256 | recurrent layer width |
| `EPOCHS` | 30 | upper bound; early stopping usually stops sooner |

## Run the app

```bash
uvicorn app:app --reload
# open http://localhost:8000
```

or with Docker:

```bash
docker compose up --build
```

`docker-compose.yml` mounts `./models` read-only, so retraining a model and restarting the container picks up new weights without rebuilding the image.

### API

`POST /api/predict`

```bash
curl -X POST localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "i want to go to the ", "k": 5}'
```

```json
{
  "suggestions": [
    {"word": "station", "probability": 0.18},
    {"word": "house", "probability": 0.11}
  ],
  "mode": "next",
  "latency_ms": 4.2
}
```

`mode` is `"next"` when the text ends in whitespace (suggest the next word) or `"complete"` when it ends mid-word (rank completions that start with that prefix). `GET /health` reports whether a model is loaded.

## Tests

```bash
pytest -q          # 36 tests: preprocessing, model shapes, predictor behaviour, API
ruff check .        # lint
```

CI (`.github/workflows/ci.yml`) runs both on every push, plus a Docker build, so a broken build is caught before it reaches `main`.

## Design notes

- **No data leakage**: sentences are split into train/val/test *before* the vocabulary is built and before windowing, so the vocabulary and the windows never see test sentences.
- **Left padding + masking**: short contexts are padded with `<pad>` (id 0) on the left; `Embedding(mask_zero=True)` tells the recurrent layer to ignore those steps rather than learn from them.
- **`<unk>` and `<pad>` are never suggested**: both are zeroed out of the probability vector before ranking.
- **Social-media noise is stripped, not learned**: URLs, `@mentions` and a leading `RT @user:` are removed and `#hashtags` are unwrapped to plain words before tokenizing, so the model spends its capacity on language, not on memorizing usernames and links.
- **Reservoir sampling for the corpus build**: `scripts/build_corpus.py` samples lines from each 200+ MB source file without loading it fully into memory, then shuffles the mixed result so no epoch is dominated by one register.
- **Deterministic**: one `RANDOM_STATE` seeds NumPy, Python's `random`, and TensorFlow; splits and vocabulary are reproducible from the same corpus.
- **One model function**: `build_model(kind, ...)` in `src/model.py` is the single place that defines SimpleRNN, LSTM and GRU, so the comparison is a fair one — only the recurrent layer changes.

## License

MIT — see [LICENSE](LICENSE).
#   n e x t - w o r d - p r e d i c t i o n  
 