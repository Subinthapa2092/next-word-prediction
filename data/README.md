## Data Folder

The `data/` folder is not stored in this repo, because it is listed in `.gitignore`. The files are large and can be regenerated. After you run the setup commands, it looks like this:

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

The three `en_US.*.txt` files are extracted from `Coursera-SwiftKey.zip`. They contain roughly 900k blog lines, 1M news lines and 2.4M tweets. Each is far too large to train on comfortably, so `scripts/build_corpus.py` samples the same number of lines from each one (150,000 by default), shuffles them together, and writes the result to `corpus.txt`. Shuffling keeps the three writing styles evenly mixed in every training batch.

Create it with:

```bash
python scripts/download_data.py
```

### processed/

This folder holds the model-ready data that `src/preprocess.py` and `src/dataset.py` produce from `corpus.txt`.

Sentences are split into train, validation and test sets before the vocabulary is built and before windowing, so no test text leaks into training. Each `.npz` file stores the sliding-window pairs: an input array of context word IDs and a target array of the next word ID. `stats.json` records the numbers behind them, such as vocabulary size, the number of samples in each split, and the context length, so results stay reproducible.

These files are regenerated automatically when you train:

```bash
python -m src.train --model gru
```

### Rebuilding the data

To recreate the whole folder from scratch:

```bash
python scripts/download_data.py          # fills data/raw/
python -m src.train --model gru          # fills data/processed/ and models/
```

Use `--skip-download` if you already have `Coursera-SwiftKey.zip` in `data/raw/`, and `--lines-per-source N` to make the corpus smaller or larger.

### Using your own text

Put any plain `.txt` file in `data/raw/` and pass it to training with `--data`:

```bash
python -m src.train --model gru --data data/raw/my_text.txt
```

Nothing else needs to change. Cleaning already handles URLs, @mentions, hashtags and retweet markers.