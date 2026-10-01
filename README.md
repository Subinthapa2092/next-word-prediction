# Next Word Prediction

### SimpleRNN vs LSTM vs GRU

A neural language modeling system that predicts the next word from user text and returns ranked suggestions.

This project implements and compares three recurrent neural network architectures: **SimpleRNN, LSTM, and GRU**.

The same dataset, preprocessing pipeline, vocabulary, training configuration, and evaluation process are used for the three architectures.

The trained model is integrated into a FastAPI application that provides real time next word prediction and word completion. The API returns five suggestions by default (`k` can be 1 to 10), and the web page asks for six.

## Live Demo

Frontend (Vercel):
[https://next-word-prediction-frontend.vercel.app/](https://next-word-prediction-frontend.vercel.app/)

Backend API (Render):
[https://next-word-prediction-f3tv.onrender.com/](https://next-word-prediction-f3tv.onrender.com/)

The frontend is a separately hosted static site that calls the backend API over CORS. The backend runs on Render's free tier, which spins down after periods of inactivity. The first request after idle time can take significantly longer than normal while the service wakes up; subsequent requests return at normal speed.

## Overview

The system supports two prediction modes.

### Next Word Prediction

When the input ends with a space, the model predicts the most likely next words.

```text
Input:
i want to go to the

Suggestions (example from the quick-run model):
world
same
next
end
house
```

### Word Completion

When the user is typing an incomplete word, the system switches to completion mode.

```text
Input:
i want to g

Suggestions (example from the quick-run model):
get
go
give
grow
grab
```

The predictor ranks the vocabulary by predicted probability and returns the top suggestions. The live site serves the larger GRU described below, so its suggestions can differ from these examples.

## How It Works

```text
User Text
    ↓
Text Cleaning
    ↓
Tokenization
    ↓
Token IDs
    ↓
Embedding
    ↓
SimpleRNN / LSTM / GRU
    ↓
Dense Layer
    ↓
Softmax
    ↓
Vocabulary Probabilities
    ↓
Top Suggestions (5 by default)
```

The same pipeline is used for all three recurrent architectures so their behavior can be compared under the same experimental conditions.

## Dataset

The project uses the SwiftKey / HC Corpora dataset containing English blog posts, news articles, and Twitter text.

The three sources represent different writing styles.

| Source | Description |
|---|---|
| Blogs | Longer and informal writing |
| News | Structured and edited language |
| Twitter | Short and noisy conversational text |

### Dataset Configuration

The default full corpus configuration uses:

```text
150,000 blog lines
150,000 news lines
150,000 Twitter lines

Total: 450,000 lines
```

All reported experiments in this README use a smaller sample of 15,000 lines per source (about 45,000 lines in total). After cleaning, that sample holds 90,279 sentences. The sentence level split gives:

```text
Train:       976,755 windows
Validation:  120,778 windows
Test:        121,282 windows
```

Training on the full 450,000-line corpus has not been run yet.

Reservoir sampling is used to sample large source files without loading the complete files into memory.

> **Dataset note:** The SwiftKey / HC Corpora dataset has separate usage considerations and is used here for experimentation and portfolio purposes. A commercial system should use data with clearly documented commercial usage rights.

## Model Comparison

The project evaluates three recurrent architectures:

1. SimpleRNN
2. LSTM
3. GRU

All three models use the same experimental configuration.

```text
Embedding dimension: 64
Recurrent units:     64
Sequence length:     5
Epochs:              8
Vocabulary size:     10,000
```

The reported experiment was intentionally kept lightweight so the complete comparison could run on a CPU only machine.

### Quick CPU Experiment Results

| Model | Parameters | Top 1 Accuracy | Top 5 Accuracy | Perplexity | Training Time |
|---|---:|---:|---:|---:|---:|
| GRU | 1,314,960 | 16.01% | 33.37% | 253.01 | 25m 35s |
| SimpleRNN | 1,298,256 | 15.77% | 33.02% | 262.11 | 25m 24s |
| LSTM | 1,323,024 | 15.74% | 32.81% | 265.92 | 23m 08s |

GRU had the lowest perplexity in this comparison, so it was chosen as the model served by the app. The served model is the larger GRU described below (10-word context, 128-dimensional embeddings, 256 units), which improved on the quick GRU.

```python
config.PRODUCTION_MODEL = "gru"
```

The complete raw comparison from the quick experiment is available in:

```text
outputs/results.md
outputs/metrics.json
```

### Larger Model Experiment Results

A larger run was also completed for GRU, using a longer context window and a larger model. It uses the same corpus sample as the quick experiment, so the difference comes from the model configuration and not from more data:

```text
Sequence length:     10
Embedding dimension: 128
Recurrent units:     256
Batch size:          256
Dropout:             0.3
Learning rate:       0.001 (halved when validation loss stalls)
Epochs (configured): 30
Epochs (run):        11 (best epoch: 8)
Training samples:    976,755
Vocabulary size:     10,000
```

| Model | Parameters | Top 1 Accuracy | Top 5 Accuracy | Test Loss | Perplexity | Training Time |
|---|---:|---:|---:|---:|---:|---:|
| GRU (larger) | 4,146,448 | 17.37% | 35.78% | 5.3808 | 217.19 | 1h 13m 54s |

This is an improvement over the quick experiment (perplexity 217.19 versus 253.01). Context length, embedding size, and recurrent units were all changed together, so this run cannot show which change contributed most. LSTM and SimpleRNN have not yet been run at this larger configuration, so a fair three way comparison at this size is still open. The commands to reproduce this for all three models are in the Larger Model Training section below.

## Interpreting the Results

These results represent this particular experiment configuration.

In the quick comparison the sequence length is five words and the models use a relatively small embedding dimension and recurrent width. This makes the experiment suitable for a CPU friendly comparison, but it is not intended to establish that one recurrent architecture is universally superior. Each result comes from a single run with a single seed, and the gaps between the three architectures are small (0.3 points of top 1 accuracy).

In the quick comparison, the recurrent layer is only 0.6% to 2.5% of each model's parameters. The rest is the embedding table and the output layer, which are identical across the three models.

LSTM and GRU are designed to handle longer dependencies through their gating mechanisms, and a five word window leaves little distance for that to matter. The larger GRU result performed better, but the improvement may come from the longer context, the larger model, or both. Testing the long range memory argument properly needs all three architectures at the larger configuration.

### Why the accuracy numbers look low

Next word prediction is ambiguous: many words are valid continuations, but the test counts only the one that followed in the original text. A perplexity of 217 over a 10,000 word vocabulary is far better than the 10,000 of random guessing. Top 5 accuracy is the metric closest to the product, since the app shows a short list of suggestions.

One caveat: the most common target in the data is `<unk>` (7.6% of training targets), and the accuracy metrics count it like any other word. The app never suggests `<unk>`, so the accuracy a user experiences is probably somewhat lower than the reported numbers. This has not been measured yet.

## Example Predictions

The quick-run models produced examples such as:

```text
"i want to go to the "
→ world, same, next, end, house

"holmes was "
→ a, the, not, in, on

"what do you "
→ have, do, know, are, want

"i want to g"
→ get, go, give, grow, grab

"it was a very d"
→ difficult, different, dangerous, dark, delicious
```

## Project Structure

The repository is organized into data, trained models, notebooks, generated outputs, reusable source code, frontend files, tests, scripts, and deployment configuration.

```text
next-word-prediction-main/
│
├── .github/
│
├── data/
│   ├── processed/
│   │   ├── stats.json
│   │   ├── test.npz
│   │   ├── train.npz
│   │   └── val.npz
│   │
│   └── raw/
│       ├── corpus.txt
│       ├── Coursera-SwiftKey.zip
│       ├── en_US.blogs.txt
│       ├── en_US.news.txt
│       └── en_US.twitter.txt
│
├── frontend/
│   ├── app.js
│   ├── index.html
│   └── style.css
│
├── models/
│   ├── gru.keras
│   ├── lstm.keras
│   ├── rnn.keras
│   └── tokenizer.json
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_lstm.ipynb
│   ├── 04_gru.ipynb
│   └── 05_comparison.ipynb
│
├── outputs/
│   ├── eda_sentence_lengths.png
│   ├── eda_top_words.png
│   ├── eda_zipf.png
│   ├── gru_example_prediction.png
│   ├── gru_history.json
│   ├── gru_training_curves.png
│   ├── gru_training_log.csv
│   ├── lstm_example_prediction.png
│   ├── lstm_history.json
│   ├── lstm_training_curves.png
│   ├── lstm_training_log.csv
│   ├── metrics.json
│   ├── model_comparison.png
│   ├── results.md
│   ├── rnn_history.json
│   ├── rnn_training_curves.png
│   ├── rnn_training_log.csv
│   └── validation_curves_comparison.png
│
├── scripts/
│   ├── __init__.py
│   ├── build_corpus.py
│   └── download_data.py
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── train.py
│   └── utils.py
│
├── static/
│   ├── app.js
│   └── style.css
│
├── templates/
│   └── index.html
│
├── tests/
│   ├── fixtures/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_predict.py
│   └── test_preprocess.py
│
├── .dockerignore
├── .gitignore
├── app.py
├── config.py
├── docker-compose.yml
├── Dockerfile
├── LICENSE
├── pyproject.toml
├── pytest.ini
├── README.md
├── requirements-dev.txt
└── requirements.txt
```

`data/` and `outputs/` are generated locally and are not stored in the repository (see `.gitignore`). The `.venv` or `venv` environment and Python cache directories are also local development files.

`static/` and `templates/` are the combined FastAPI application used for local development and Docker (`app.py` serves the page directly). `frontend/` is a separate, independently hosted static site (plain HTML, CSS, and JavaScript) used for the live Vercel deployment. It calls the backend `/api/predict` and `/health` endpoints over CORS, using the same JSON contract described in the API section below.

## Output Files

The `outputs/` directory contains the generated experiment artifacts.

### Exploratory Data Analysis Images

```text
outputs/eda_sentence_lengths.png
outputs/eda_top_words.png
outputs/eda_zipf.png
```

### GRU Outputs

```text
outputs/gru_example_prediction.png
outputs/gru_history.json
outputs/gru_training_curves.png
outputs/gru_training_log.csv
```

### LSTM Outputs

```text
outputs/lstm_example_prediction.png
outputs/lstm_history.json
outputs/lstm_training_curves.png
outputs/lstm_training_log.csv
```

### SimpleRNN Outputs

```text
outputs/rnn_history.json
outputs/rnn_training_curves.png
outputs/rnn_training_log.csv
```

### Comparison Outputs

```text
outputs/model_comparison.png
outputs/validation_curves_comparison.png
outputs/metrics.json
outputs/results.md
```

The JSON and CSV files contain the experiment histories, metrics, and training logs.

> **Note:** training the larger GRU overwrote the GRU files in `outputs/` (history, training log, and training curves), so those now describe the larger run. The comparison images made before that run still show the quick experiment.

## Data Pipeline

The preprocessing pipeline performs the following operations:

1. URLs are removed.
2. User mentions are removed.
3. Retweet markers are removed.
4. Hashtag symbols are removed while preserving the hashtag word.
5. Text is converted to lowercase.
6. Text is tokenized into words.
7. Apostrophes are preserved.
8. Sentences shorter than two tokens are removed.
9. Sentences are divided into training, validation, and test sets.
10. The vocabulary is built from the training sentences only, limited to 10,000 entries including `<pad>` and `<unk>` (so 9,998 real words).
11. Rare words and words outside the vocabulary are mapped to `<unk>`.
12. `<pad>` receives ID `0`.
13. `<unk>` receives ID `1`.
14. Sliding windows are generated using the sequence length.
15. Short contexts are left padded with `<pad>`.

Example:

```text
Context:
i want to go to

Target:
the
```

The model therefore learns:

```text
Previous Context → Next Token
```

## Preventing Data Leakage

The dataset is divided into training, validation, and test sentences before vocabulary construction and window generation.

This prevents test sentences from influencing the vocabulary or appearing inside the training windows.

The resulting evaluation is therefore performed on text that was not used during training.

## Training

### Train GRU

```bash
python -m src.train --model gru
```

### Train LSTM

```bash
python -m src.train --model lstm
```

### Train SimpleRNN

```bash
python -m src.train --model rnn
```

The defaults in `config.py` match the quick CPU experiment below.

## Quick CPU Experiment

To reproduce the configuration used for the reported quick comparison:

```bash
python -m src.train --model gru --epochs 8 --batch-size 256 --seq-len 5 --embed-dim 64 --units 64
```

The same configuration can be used for all three models:

```bash
python -m src.train --model rnn --epochs 8 --batch-size 256 --seq-len 5 --embed-dim 64 --units 64

python -m src.train --model lstm --epochs 8 --batch-size 256 --seq-len 5 --embed-dim 64 --units 64

python -m src.train --model gru --epochs 8 --batch-size 256 --seq-len 5 --embed-dim 64 --units 64
```

Training produces:

```text
models/rnn.keras
models/lstm.keras
models/gru.keras
models/tokenizer.json
```

Training metrics and generated artifacts are stored under:

```text
outputs/
```

## Larger Model Training

The larger experiment uses a longer context window and a wider model on the same corpus sample as the quick experiment:

```bash
python -m src.train --model lstm --seq-len 10 --units 256 --embed-dim 128 --epochs 30

python -m src.train --model gru --seq-len 10 --units 256 --embed-dim 128 --epochs 30

python -m src.train --model rnn --seq-len 10 --units 256 --embed-dim 128 --epochs 30
```

The GRU run using this configuration is reported above in Larger Model Experiment Results. On a CPU only machine it took about 1 hour and 14 minutes. It stopped after epoch 11 because validation loss had not improved for three epochs, and it restored the weights from epoch 8, the best one. **This is the model the live site serves.**

Training writes to the same `models/` and `outputs/` files as the quick experiment, so back up those folders before running it again if you want to keep the earlier results.

GPU environments such as Linux, WSL2, or Google Colab can significantly reduce training time compared with CPU only training.

### Full 450,000-Line Corpus (Not Yet Run)

To build the full corpus of 150,000 lines per source and train on it:

```bash
python scripts/download_data.py --lines-per-source 150000
```

Then run any of the training commands above. This has not been run yet, so no results are reported for it.

## Dataset Setup

Download and prepare the dataset:

```bash
python scripts/download_data.py
```

To reuse an existing downloaded archive:

```bash
python scripts/download_data.py --skip-download
```

To create the smaller development dataset used for the reported experiments:

```bash
python scripts/download_data.py --skip-download --lines-per-source 15000
```

The downloaded archive is placed under:

```text
data/raw/Coursera-SwiftKey.zip
```

The processed dataset is stored under:

```text
data/processed/
```

## Setup

Clone the repository:

```bash
git clone https://github.com/Subinthapa2092/next-word-prediction.git
cd next-word-prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

### Windows PowerShell

```powershell
venv\Scripts\activate
```

### Linux or macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements-dev.txt
```

Download and prepare the dataset:

```bash
python scripts/download_data.py
```

## Run the Application

Start the FastAPI application:

```bash
uvicorn app:app --reload
```

Open:

```text
http://localhost:8000
```

The application loads the model specified by:

```python
config.PRODUCTION_MODEL
```

The default model is:

```python
PRODUCTION_MODEL = "gru"
```

To run the application using LSTM:

### Windows PowerShell

```powershell
$env:PRODUCTION_MODEL = "lstm"
uvicorn app:app --reload
```

### Linux or macOS

```bash
PRODUCTION_MODEL=lstm uvicorn app:app --reload
```

The application status line displays the currently loaded model and vocabulary size.

## API

The prediction endpoint is:

```text
POST /api/predict
```

Example request (mid word, six suggestions):

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "My name is Subin Thapa. I love Stock Mar", "k": 6}'
```

Example response from the live app (probabilities rounded; latency varies by machine):

```json
{
  "suggestions": [
    {"word": "market", "probability": 0.24},
    {"word": "marketing", "probability": 0.11},
    {"word": "markets", "probability": 0.10},
    {"word": "marathon", "probability": 0.077},
    {"word": "marks", "probability": 0.061},
    {"word": "markers", "probability": 0.049}
  ],
  "mode": "complete",
  "latency_ms": 1295.27
}
```

`k` is the number of suggestions and can be 1 to 10. It defaults to 5.

### Prediction Modes

When the input ends with whitespace:

```text
"i want to go to the "
```

the API uses:

```text
mode = next
```

When the input ends inside a word:

```text
"i want to g"
```

the API uses:

```text
mode = complete
```

In completion mode the system keeps only the vocabulary words that begin with the typed prefix, renormalizes their probabilities, and ranks them.

## Health Check

The application exposes:

```text
GET /health
```

This endpoint reports whether a model is loaded and ready to serve predictions, and the vocabulary size.

## Docker

The project includes Docker support for reproducible deployment.

Build and start the application:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8000
```

Stop the containers:

```bash
docker compose down
```

The compose file mounts the local `models/` directory into the container as read only, so a newly trained model can be used after restarting the container without rebuilding the image. The `Dockerfile` also copies `models/gru.keras` and `models/tokenizer.json` into the image, which is what makes the model available on hosts that cannot mount a folder.

## Deployment

The live demo is split into two services:

| Part | Host | Details |
|---|---|---|
| Backend and model | Render (Docker) | Built from the `Dockerfile` in this repository |
| Frontend | Vercel | Static files from the `frontend/` folder (Root Directory `frontend`, Framework Preset `Other`) |

The frontend finds the backend through `window.API_BASE`, set in `frontend/index.html`. The backend only accepts browser requests from the addresses listed in the `ALLOWED_ORIGINS` environment variable (comma separated, no trailing slashes). If the live page shows an offline or waking up status for more than a minute or two, check that variable and that `window.API_BASE` points at the Render address.

The backend also limits TensorFlow to one thread, because free tier containers receive only a fraction of a CPU core. This setting has not been benchmarked on its own.

This is a demonstration deployment on free tiers. It has no monitoring, rate limiting, or authentication.

## Testing

Run the complete test suite (45 tests):

```bash
pytest -q
```

The tests cover:

1. Preprocessing
2. Model construction
3. Model shapes
4. Predictor behavior
5. API behavior
6. CORS behavior

Run linting:

```bash
ruff check .
```

## Continuous Integration

GitHub Actions runs automated checks whenever changes are pushed.

The workflow performs:

1. Test suite
2. Linting
3. Docker build

The workflow is defined in:

```text
.github/workflows/ci.yml
```

## Design Decisions

### Shared Model Construction

All three recurrent architectures are implemented through the same model construction function:

```python
build_model(kind, ...)
```

The primary architectural difference is the recurrent layer:

```text
SimpleRNN
LSTM
GRU
```

This keeps the comparison controlled.

### Padding and Masking

Short contexts are padded on the left using:

```text
<pad>
```

The padding token uses ID `0`.

The embedding layer uses:

```python
mask_zero=True
```

This tells the recurrent layer to ignore padding positions instead of treating them as meaningful language tokens.

### Special Tokens

The predictor uses:

```text
<pad>
<unk>
```

These tokens are never returned as user suggestions.

Their probabilities are removed before the final ranking is generated.

### Social Media Cleaning

The preprocessing pipeline removes common social media noise.

```text
URLs
@mentions
RT markers
```

Hashtags are converted into their underlying words.

For example:

```text
#python
```

becomes:

```text
python
```

### Reservoir Sampling

The corpus builder uses reservoir sampling to select lines from large source files without loading the complete files into memory.

Each source is sampled independently.

The resulting samples are mixed and shuffled before being written to:

```text
data/raw/corpus.txt
```

### Reproducible Experiments

The project uses a single `RANDOM_STATE` to seed:

```text
Python random
NumPy
TensorFlow
```

This helps produce reproducible dataset splits and vocabulary construction.

### Separately Hosted Frontend

The live deployment splits the application into two independently hosted pieces. The backend (FastAPI, in `app.py`) is deployed to Render and exposes `/api/predict` and `/health`. The frontend (in `frontend/`) is a plain HTML, CSS, and JavaScript site deployed to Vercel that calls the backend over CORS. The backend's `CORSMiddleware` allow list is controlled by the `ALLOWED_ORIGINS` environment variable, which includes the live Vercel URL.

The editor draws the top suggestion as faded text after the typed text. A hidden mirror element with identical typography sits behind the textarea, and the suggestion only shows while it still matches the current text, so pressing Tab never accepts a stale guess.

## Notebook Workflow

The notebooks provide the experimental workflow with explanations, plots, and intermediate results.

Run them in this order:

```text
01_eda.ipynb
02_preprocessing.ipynb
03_lstm.ipynb
04_gru.ipynb
05_comparison.ipynb
```

The notebooks cover exploratory data analysis, preprocessing, model training, and architecture comparison. They are configured for the quick CPU experiment, so re-running `04_gru.ipynb` trains the small GRU and overwrites `models/gru.keras`.

## Hyperparameters

The main configuration values are defined in `config.py`.

| Parameter | Default | Quick Experiment | Larger Model Experiment | Purpose |
|---|---:|---:|---:|---|
| `SEQ_LEN` | 5 | 5 | 10 | Number of previous words used as context |
| `MAX_VOCAB` | 10,000 | 10,000 | 10,000 | Maximum vocabulary size |
| `MIN_FREQ` | 2 | 2 | 2 | Minimum word frequency |
| `EMBED_DIM` | 64 | 64 | 128 | Word embedding dimension |
| `UNITS` | 64 | 64 | 256 | Recurrent layer width |
| `EPOCHS` | 8 | 8 | 30 | Maximum training epochs |
| `BATCH_SIZE` | 256 | 256 | 256 | Samples processed per batch |

Major configuration values can be overridden using environment variables or command line arguments.

## Reproducibility

The quick comparison can be reproduced with the following workflow.

Download the dataset:

```bash
python scripts/download_data.py
```

Create the reduced corpus used for the reported experiments:

```bash
python scripts/download_data.py --skip-download --lines-per-source 15000
```

Train the three models:

```bash
python -m src.train --model rnn --epochs 8 --batch-size 256 --seq-len 5 --embed-dim 64 --units 64

python -m src.train --model lstm --epochs 8 --batch-size 256 --seq-len 5 --embed-dim 64 --units 64

python -m src.train --model gru --epochs 8 --batch-size 256 --seq-len 5 --embed-dim 64 --units 64
```

Run the comparison notebook:

```text
notebooks/05_comparison.ipynb
```

To reproduce the larger GRU that the live site serves:

```bash
python -m src.train --model gru --seq-len 10 --units 256 --embed-dim 128 --epochs 30
```

The resulting metrics and plots are written to:

```text
outputs/
```

The trained checkpoints are written to:

```text
models/
```

## What This Project Demonstrates

This project covers the complete machine learning to application workflow:

```text
Raw Language Data
        ↓
Corpus Construction
        ↓
Exploratory Data Analysis
        ↓
Text Cleaning
        ↓
Train / Validation / Test Split
        ↓
Tokenization and Vocabulary Construction
        ↓
Sliding Window Dataset
        ↓
Model Training
        ↓
Evaluation
        ↓
Architecture Comparison
        ↓
Model Selection
        ↓
FastAPI Inference
        ↓
Web Application
        ↓
Docker and Cloud Deployment
        ↓
Automated Testing
        ↓
Continuous Integration
```

The project brings together machine learning, natural language processing, model evaluation, backend development, web application development, testing, and deployment.

## Future Improvements

Possible directions for extending the system include:

1. Training on the full 450,000-line corpus
2. Larger LSTM and SimpleRNN runs, with several seeds each, to complete a fair three way comparison at the longer context window
3. Baselines (most frequent word and a bigram model) to show how much the neural network adds
4. An evaluation that excludes `<unk>` targets, matching what the app can actually suggest
5. Keystroke savings as an additional metric
6. A proper latency benchmark, with and without the single thread setting
7. Smaller model files (saving checkpoints without optimizer state)
8. Beam search based decoding
9. Improved prefix completion ranking
10. Transformer based language models
11. Model quantization
12. Monitoring and rate limiting for a real deployment

## License

The source code for this project is released under the MIT License.

See [`LICENSE`](LICENSE) for the complete license text.

The dataset has separate usage considerations described in the dataset section above.