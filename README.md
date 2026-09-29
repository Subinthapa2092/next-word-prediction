# Next-Word Prediction (SimpleRNN vs LSTM vs GRU)

A phone-keyboard-style next-word predictor: type a sentence, get five ranked suggestions for what comes next, or for how to finish the word you're mid-typing. Built to compare three recurrent architectures on the exact same data and pick a winner for production.

Dataset: **SwiftKey / HC Corpora**, which is blogs, news and Twitter text (en_US). SwiftKey released this corpus for exactly this task, via Coursera's Data Science Specialization capstone. Mixing three registers (long-form blogs, edited news, short noisy tweets) matters here: a model trained only on 19th-century prose or only on one register won't generalize to how people actually type.

> **License note:** this dataset is free and widely used for research, learning and portfolio projects like this one. Its terms for a shipped commercial product aren't spelled out as a permissive license (e.g. MIT or CC-BY), so if this ever becomes a commercial app, swap in something with clearer commercial terms, such as WikiText-103 or your own collected user text.

This is the companion project to the article series *[RNN / LSTM / GRU From First Principles](#)*: theory in the articles, and this repo is the implementation.

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

Real numbers from an actual run (see `outputs/results.md`), on a training set of about 977k windows built from a ~45,000-line sample of the corpus, with a reduced model (`embed_dim=64`, `units=64`, `seq_len=5`, 8 epochs). It is sized for a CPU-only laptop rather than a GPU box. See [Full-scale run](#full-scale-run-vs-this-quick-run) below for what a production-sized run looks like and how to reproduce it.

| Model | Parameters | Top-1 accuracy | Top-5 accuracy | Perplexity | Training time |
|---|---|---|---|---|---|
| **GRU** | 1,314,960 | 16.01% | 33.37% | **253.01** | 25m 35s |
| SimpleRNN | 1,298,256 | 15.77% | 33.02% | 262.11 | 25m 24s |
| LSTM | 1,323,024 | 15.74% | 32.81% | 265.92 | 23m 08s |

**GRU wins on perplexity** (the deciding metric here) and is close to the fastest to train, so `config.PRODUCTION_MODEL` is set to `gru`. Sample completions from the trained GRU and LSTM:

```
'i want to go to the '   -> world, same, next, end, house
'holmes was '            -> a, the, not, in, on
'what do you '           -> have, do, know, are, want
'i want to g'            -> get, go, give, grow, grab
'it was a very d'        -> difficult, different, dangerous, dark, delicious
```

**Being honest about what these numbers mean:** SimpleRNN scoring essentially tied with LSTM (and even slightly ahead on some metrics) isn't the textbook result. LSTM and GRU's advantage is remembering *long-range* dependencies, and `seq_len=5` gives them almost no room to show it. At this scale, the extra gating machinery costs capacity without much payoff. This table reflects a fast, CPU-feasible run, not a claim that GRU and LSTM don't matter. See below for the settings that would let them show their real advantage.

The comparison notebook also saves `outputs/model_comparison.png` (accuracy, perplexity, time and parameter bar charts) and `outputs/validation_curves_comparison.png` (loss and top-5 accuracy per epoch, all three models overlaid).

### Full-scale run vs. this quick run

`01_eda.ipynb` and `02_preprocessing.ipynb` in this repo can be run against the **full 450,000-line corpus** that `scripts/download_data.py` builds by default. The training run above used a **smaller ~45,000-line sample** of it instead, purely so a full LSTM, GRU and SimpleRNN comparison would finish in under an hour on a CPU rather than take many hours per model.

If you have more time or a GPU (for example Google Colab's free tier), reproduce this at full scale, where LSTM and GRU should meaningfully separate from SimpleRNN, with:

```bash
python scripts/download_data.py --lines-per-source 150000   # back to the full 450k-line corpus
python -m src.train --model lstm --seq-len 10 --units 256 --embed-dim 128 --epochs 30
python -m src.train --model gru  --seq-len 10 --units 256 --embed-dim 128 --epochs 30
python -m src.train --model rnn  --seq-len 10 --units 256 --embed-dim 128 --epochs 30
```

A longer `seq_len` (8 to 10 or more words instead of 5) matters most here, because it is what actually exercises the long-range memory that LSTM and GRU are built for.

## Project structure

```
next-word-prediction/
  data/
    raw/              downloaded corpus (gitignored, created by scripts/download_data.py)
    processed/        cached train/val/test windows + stats.json (gitignored)
  models/              trained checkpoints (gitignored, created by training)
    tokenizer.json     the vocabulary; loaded by the app and every notebook
    gru.keras           production model (lowest perplexity, see Results)
    lstm.keras, rnn.keras   the other two, kept for comparison or for swapping PRODUCTION_MODEL
  notebooks/
    01_eda.ipynb              corpus stats, sentence lengths, Zipf's law, vocab coverage
    02_preprocessing.ipynb    clean, tokenize, ids, sliding windows, step by step
    03_lstm.ipynb             train + evaluate the LSTM
    04_gru.ipynb              train + evaluate the GRU
    05_comparison.ipynb       SimpleRNN vs LSTM vs GRU, side by side, with conclusions
  outputs/             generated by the notebooks (gitignored, regenerate by re-running them)
    eda_*.png                  sentence-length histogram, top words, Zipf plot
    <kind>_training_curves.png, <kind>_history.json, <kind>_training_log.csv   per model
    <kind>_example_prediction.png   sample top-10 probability bar chart per model
    model_comparison.png, validation_curves_comparison.png   the three models side by side
    metrics.json, results.md   raw + markdown comparison table (paste results.md into this README)
  src/
    preprocess.py      cleaning, tokenizing, the Vocabulary class
    dataset.py         sliding-window pair construction, train/val/test splits
    model.py           build_model(kind, ...), where the only change is the recurrent layer
    train.py           training script / CLI, writes metrics + training curves
    predict.py         Predictor, the inference class the app and notebooks both use
    utils.py           seeding, metrics file, plotting
  static/              style.css, app.js for the web UI
  templates/           index.html
  tests/               pytest suite: preprocessing, model, predictor, API
  scripts/
    download_data.py   downloads the SwiftKey archive, extracts it, builds data/raw/corpus.txt
    build_corpus.py    samples + mixes blogs/news/twitter without re-downloading
  app.py               FastAPI app (web UI + JSON API)
  config.py            every path and hyperparameter, overridable via env vars (PRODUCTION_MODEL=gru)
  Dockerfile, docker-compose.yml    containerized deployment; mounts ./models read-only
  requirements.txt, requirements-dev.txt
  .github/workflows/ci.yml   lint + test + docker build on every push
```

## Data

The `data/` folder is not stored in this repo. It is listed in `.gitignore` because the files are large (the archive alone is about 560 MB) and can be regenerated with the commands below. Each folder keeps only a `.gitkeep` file so the directory structure exists after cloning.

After setup and training, the folder looks like this:

```
data/
├── raw/
│   ├── Coursera-SwiftKey.zip     original downloaded archive (~560 MB)
│   ├── en_US.blogs.txt           blog posts: long-form, informal writing
│   ├── en_US.news.txt            news articles: formal, edited writing
│   ├── en_US.twitter.txt         tweets: short, noisy, social text
│   └── corpus.txt                shuffled sample of blogs + news + twitter used for training
└── processed/
    ├── train.npz                 training set (80% of sentences)
    ├── val.npz                   validation set (10% of sentences)
    ├── test.npz                  test set (10% of sentences)
    └── stats.json                vocabulary size, sample counts and sequence length
```

### raw/

This folder holds the original text exactly as downloaded, plus the training corpus built from it.

The three `en_US.*.txt` files are extracted from `Coursera-SwiftKey.zip`. They contain roughly 900k blog lines, 1M news lines and 2.4M tweets. Each is far too large to train on comfortably, so `scripts/build_corpus.py` samples the same number of lines from each one (150,000 by default), shuffles them together, and writes the result to `corpus.txt`. Shuffling keeps the three writing styles evenly mixed in every training batch. The sampling uses reservoir sampling, so the huge source files are never fully loaded into memory.

### processed/

This folder holds the model-ready data that `src/preprocess.py` and `src/dataset.py` produce from `corpus.txt`.

Sentences are split into train, validation and test sets before the vocabulary is built and before windowing, so no test text leaks into training. Each `.npz` file stores the sliding-window pairs: an input array of context word IDs and a target array of the next word ID. `stats.json` records the numbers behind them, such as vocabulary size, the number of samples in each split, and the context length, so results stay reproducible.

### What preprocessing does

- Strips URLs, `@mentions`, retweet markers (`RT @user:`) and `#` symbols (the hashtag word is kept).
- Lowercases, then tokenizes to words, keeping apostrophes so `don't` stays one token.
- Splits into sentences and drops sentences shorter than 2 tokens.
- Builds a vocabulary of the 10,000 most frequent words (`MAX_VOCAB`). Words seen fewer than 2 times (`MIN_FREQ`) and anything outside the vocabulary become `<unk>`. `<pad>` is id 0 and `<unk>` is id 1.
- Splits sentences 80% train, 10% validation and 10% test.
- Creates sliding-window pairs: `SEQ_LEN` context words followed by the next word, left-padded with `<pad>`.

### Rebuilding the data

To recreate the whole folder from scratch:

```bash
python scripts/download_data.py          # fills data/raw/
python -m src.train --model gru          # fills data/processed/ and models/
```

Use `--skip-download` if you already have `Coursera-SwiftKey.zip` in `data/raw/`, and `--lines-per-source N` to make the corpus smaller or larger.

## Setup

```bash
git clone https://github.com/Subinthapa2092/next-word-prediction.git
cd next-word-prediction
python -m venv venv && source venv/bin/activate   # Windows PowerShell: venv\Scripts\activate
pip install -r requirements-dev.txt

python scripts/download_data.py        # ~560 MB download, builds data/raw/corpus.txt
```

**Windows note:** TensorFlow ships wheels for Python 3.10 to 3.13 only. If `pip install` can't find a `tensorflow` version, check `python --version` first, because Python 3.14 (or newer) has no TensorFlow wheel yet. Use the `py` launcher to create the venv with a supported version instead, for example `py -3.11 -m venv venv`. Also note that native Windows TensorFlow (2.11 and later) cannot use a GPU at all, CUDA or not. Only WSL2 and Linux get GPU acceleration, so a CPU-only Windows machine should expect the "quick run" settings in the Train section rather than the full-scale ones.

By default this samples 150,000 lines from each of blogs, news and twitter (450k lines total). That is enough for a real model, but on a CPU-only Windows machine expect training on the full corpus to take multiple hours **per model**. Adjust with `--lines-per-source`:

```bash
python scripts/download_data.py --lines-per-source 400000   # bigger corpus, longer training
python scripts/download_data.py --skip-download             # already have the zip; just re-mix
python scripts/download_data.py --skip-download --lines-per-source 15000   # small/fast corpus (~45k lines)
```

If the automatic download is blocked in your environment (some networks or sandboxes disallow direct file downloads), grab the archive yourself from the URL printed in the error message, save it to `data/raw/Coursera-SwiftKey.zip`, and re-run with `--skip-download`.

To use your own text instead, drop any plain-text file in `data/raw/` and pass `--data` to training, or point `config.DEFAULT_CORPUS` at it. No code changes are needed, since `split_sentences` and `clean_text` handle both plain prose and social-media noise (URLs, @mentions, hashtags).

## Train

```bash
python -m src.train --model lstm
python -m src.train --model gru
python -m src.train --model rnn
```

For a quick CPU run (matching the numbers in [Results](#results) above), pass smaller settings explicitly rather than relying on the defaults in `config.py`:

```bash
python -m src.train --model gru --epochs 8 --batch-size 256 --seq-len 5 --embed-dim 64 --units 64
```

**Notebook users:** the same caveat applies inside `notebooks/03_lstm.ipynb`, `04_gru.ipynb` and `05_comparison.ipynb`. Pass hyperparameters as explicit arguments to `train(...)` rather than setting `os.environ[...]` before calling it. The parameter defaults of `train()` (`epochs=config.EPOCHS`, etc.) are bound once when `src/train.py` is first imported in a kernel session, so an environment variable set afterwards has no effect unless you fully restart the kernel before setting it. Explicit arguments always work regardless of import history:

```python
result = train("gru", config.DEFAULT_CORPUS, epochs=8, batch_size=256, seq_len=5, embed_dim=64, units=64, verbose=2)
```

Each run saves `models/<kind>.keras`, updates `models/tokenizer.json`, appends to `outputs/metrics.json`, and writes `outputs/<kind>_training_curves.png`. You can also run the notebooks in order (`01` to `05`) for the full walkthrough with plots and commentary.

Key hyperparameters (see `config.py`, all overridable by environment variable or CLI flag):

| Variable | Default | Used for the Results run above | Meaning |
|---|---|---|---|
| `SEQ_LEN` | 6 | 5 | words of context the model sees |
| `MAX_VOCAB` | 10000 | 10000 | vocabulary cap, including `<pad>` and `<unk>` |
| `MIN_FREQ` | 2 | 2 | words rarer than this become `<unk>` |
| `EMBED_DIM` | 128 | 64 | embedding size |
| `UNITS` | 256 | 64 | recurrent layer width |
| `EPOCHS` | 30 | 8 | upper bound; early stopping usually stops sooner |
| `BATCH_SIZE` | 128 | 256 | samples per training step |

## Run the app

```bash
uvicorn app:app --reload
# open http://localhost:8000
```

`config.PRODUCTION_MODEL` (default `"gru"`, matching the [Results](#results) above) picks which `models/<kind>.keras` file gets served. Override it per run with an environment variable if you want to compare models live:

```bash
# PowerShell
$env:PRODUCTION_MODEL = "lstm"; uvicorn app:app --reload

# bash
PRODUCTION_MODEL=lstm uvicorn app:app --reload
```

The app's status bar shows which model and vocab size are actually loaded, so you can always confirm which one is live.

### Docker

```bash
docker compose up --build
# open http://localhost:8000
```

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) running first. `docker-compose.yml` mounts `./models` read-only into the container, so retraining a model and restarting the container picks up new weights without rebuilding the image. `PRODUCTION_MODEL` is set to `gru` there too, matching `config.py`.

The first build takes a few minutes (installing TensorFlow and everything else inside the image). A successful run ends with:

```
web-1  | INFO:     Application startup complete.
web-1  | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

A `WARNING ... Could not find cuda drivers` or `CUDA error: Failed call to cuInit` in the logs is expected and harmless. The image is CPU-only by design, and this is just TensorFlow confirming no GPU is present. Stop the container with `Ctrl+C`, then run `docker compose down` to remove it.

**Troubleshooting**, in the order to try them:

- `failed to connect to the docker API at npipe://...`: Docker Desktop isn't running. Open it from the Start menu and wait 30 to 60 seconds for it to fully start before retrying.
- `dial tcp: lookup registry-1.docker.io: no such host`: Docker can't reach the internet to download the base image. Confirm with `docker pull hello-world`. If that also fails, restart Docker's networking with `wsl --shutdown`, then fully quit and reopen Docker Desktop. This is usually a transient issue right after Docker Desktop starts.

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
pytest -q          # preprocessing, model shapes, predictor behaviour, API
ruff check .       # lint
```

CI (`.github/workflows/ci.yml`) runs both on every push, plus a Docker build, so a broken build is caught before it reaches `main`.

## Design notes

- **No data leakage:** sentences are split into train, validation and test *before* the vocabulary is built and before windowing, so the vocabulary and the windows never see test sentences.
- **Left padding and masking:** short contexts are padded with `<pad>` (id 0) on the left. `Embedding(mask_zero=True)` tells the recurrent layer to ignore those steps rather than learn from them.
- **`<unk>` and `<pad>` are never suggested:** both are zeroed out of the probability vector before ranking.
- **Social-media noise is stripped, not learned:** URLs, `@mentions` and a leading `RT @user:` are removed, and `#hashtags` are unwrapped to plain words before tokenizing. The model spends its capacity on language, not on memorizing usernames and links.
- **Reservoir sampling for the corpus build:** `scripts/build_corpus.py` samples lines from each 200+ MB source file without loading it fully into memory, then shuffles the mixed result so no epoch is dominated by one register.
- **Deterministic:** one `RANDOM_STATE` seeds NumPy, Python's `random`, and TensorFlow. Splits and vocabulary are reproducible from the same corpus.
- **One model function:** `build_model(kind, ...)` in `src/model.py` is the single place that defines SimpleRNN, LSTM and GRU, so the comparison is a fair one. Only the recurrent layer changes.
- **Jupyter and `os.environ` is a trap:** the default arguments of `train()` (`epochs=config.EPOCHS`, etc.) are evaluated once, when `src/train.py` is first imported into a kernel. Setting `os.environ["EPOCHS"]` in a later cell does nothing unless you restart the kernel *before* re-importing, and it's easy to think you restarted when you didn't. Passing hyperparameters as explicit arguments to `train(...)` sidesteps this entirely and is what the notebooks and CLI examples in this README do.

## License

MIT, see [LICENSE](LICENSE).