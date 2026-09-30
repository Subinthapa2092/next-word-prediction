# Next Word Prediction

### SimpleRNN vs LSTM vs GRU

A neural language modeling system that predicts the next word from user text and generates five ranked suggestions.

This project implements and compares three recurrent neural network architectures: **SimpleRNN, LSTM, and GRU**.

The same dataset, preprocessing pipeline, vocabulary, training configuration, and evaluation process are used for the three architectures.

The trained model is integrated into a FastAPI application that provides real time next word prediction and word completion.

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

Suggestions:
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

Suggestions:
get
go
give
grow
grab
```

The predictor generates the top five suggestions using the predicted vocabulary probabilities.

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
Top 5 Suggestions
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

All reported experiments in this README use a smaller sample of 15,000 lines per source (about 45,000 lines in total), which produces 976,755 training windows. Training on the full 450,000-line corpus has not been run yet.

Reservoir sampling is used to sample large source files without loading the complete files into memory.

> **Dataset note:** The SwiftKey / HC Corpora dataset has separate usage considerations and is used here for experimentation and portfolio purposes. A production commercial system should use data with clearly documented commercial usage rights.

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

GRU had the lowest perplexity in this comparison, so it was chosen as the production model. The served model is the larger GRU described below (10-word context, 128-dimensional embeddings, 256 units), which improved on the quick GRU.

```python
config.PRODUCTION_MODEL = "gru"
```

The complete raw comparison is available in:

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
Epochs (configured): 30
Epochs (run):        11, stopped early at best epoch 8
Training samples:    976,755
Vocabulary size:     10,000
```

| Model | Parameters | Top 1 Accuracy | Top 5 Accuracy | Test Loss | Perplexity | Training Time |
|---|---:|---:|---:|---:|---:|---:|
| GRU (larger) | 4,146,448 | 17.37% | 35.78% | 5.3808 | 217.19 | 1h 13m 54s |

This is a real improvement over the quick experiment (perplexity 217.19 versus 253.01), consistent with the expectation that a longer context window and larger recurrent width let GRU's gating mechanism do more useful work. Context length, embedding size, and recurrent units were all changed together, so this run cannot show which change contributed most. LSTM and SimpleRNN have not yet been run at this same larger configuration, so a fair three way comparison at this size is still open. The commands to reproduce this for all three models are in the Larger Model Training section below.

## Interpreting the Results

These results represent this particular experiment configuration.

In the quick comparison the sequence length is five words and the models use a relatively small embedding dimension and recurrent width. This makes the experiment suitable for a CPU friendly comparison, but it is not intended to establish that one recurrent architecture is universally superior. Each result comes from a single run with a single seed.

LSTM and GRU are designed to handle longer dependencies through their gating mechanisms. A longer context window and larger training configuration would provide a more meaningful test of their long range memory capabilities. The larger GRU result above is consistent with this: performance improved once the context window and model size increased.

## Example Predictions

The trained models produce examples such as:

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
│   ├── eda_zip.png
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

The `.venv` or `venv` environment and Python cache directories are local development files and are not part of the core project structure.

`static/` and `templates/` are the combined FastAPI application used for local development and Docker (`app.py` serves the page directly). `frontend/` is a separate, independently hosted static site (plain HTML, CSS, and JavaScript) used for the live Vercel deployment. It calls the backend `/api/predict` and `/health` endpoints over CORS, using the same JSON contract described in the API section below.

## Output Files

The `outputs/` directory contains the complete set of generated experiment artifacts.

### Exploratory Data Analysis Images

```text
outputs/eda_sentence_lengths.png
outputs/eda_top_words.png
outputs/eda_zip.png
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

### All Generated Image Files

The project currently contains these image artifacts:

```text
eda_sentence_lengths.png
eda_top_words.png
eda_zip.png

gru_example_prediction.png
gru_training_curves.png

lstm_example_prediction.png
lstm_training_curves.png

rnn_training_curves.png

model_comparison.png
validation_curves_comparison.png
```

The JSON and CSV files contain the experiment histories, metrics, and training logs.

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
9. The vocabulary is limited to 10,000 entries, including `<pad>` and `<unk>` (so 9,998 real words).
10. Rare words are mapped to `<unk>`.
11. `<pad>` receives ID `0`.
12. `<unk>` receives ID `1`.
13. Sentences are divided into training, validation, and test sets.
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

The larger experiment uses a context window of ten words. The GRU run using this exact configuration is reported above in Larger Model Experiment Results; on a CPU only machine it took approximately 1 hour and 14 minutes and stopped early at epoch 8 of 30 due to early stopping on validation loss.

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

To create a smaller development dataset:

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

The default production model is:

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

The application status bar displays the currently loaded model and vocabulary size.

## API

The prediction endpoint is:

```text
POST /api/predict
```

Example request:

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "i want to go to the ", "k": 5}'
```

Example response (shape shown; probability and latency values shortened):

<!-- TODO: replace with a real response copied from your running app -->

```json
{
  "suggestions": [
    {
      "word": "world",
      "probability": 0.0
    },
    {
      "word": "same",
      "probability": 0.0
    }
  ],
  "mode": "next",
  "latency_ms": 0.0
}
```

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

The system filters vocabulary candidates beginning with the provided prefix and ranks the possible completions.

## Health Check

The application exposes:

```text
GET /health
```

This endpoint reports whether a model is loaded and ready to serve predictions.

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

The Docker configuration mounts the local `models/` directory into the container as read only.

This means a newly trained model can be used after restarting the container without rebuilding the image.

## Testing

Run the complete test suite:

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

The notebooks cover exploratory data analysis, preprocessing, model training, and architecture comparison.

## Hyperparameters

The main configuration values are defined in `config.py`.

| Parameter | Default | Quick Experiment | Larger Model Experiment | Purpose |
|---|---:|---:|---:|---|
| `SEQ_LEN` | 6 | 5 | 10 | Number of previous words used as context |
| `MAX_VOCAB` | 10,000 | 10,000 | 10,000 | Maximum vocabulary size |
| `MIN_FREQ` | 2 | 2 | 2 | Minimum word frequency |
| `EMBED_DIM` | 128 | 64 | 128 | Word embedding dimension |
| `UNITS` | 256 | 64 | 256 | Recurrent layer width |
| `EPOCHS` | 30 | 8 | 30 | Maximum training epochs |
| `BATCH_SIZE` | 128 | 256 | 128 | Samples processed per batch |

Major configuration values can be overridden using environment variables or command line arguments.

## Reproducibility

A complete experiment can be reproduced using the following workflow.

Download the dataset:

```bash
python scripts/download_data.py
```

Create the reduced corpus for a faster experiment:

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
Tokenization
        ↓
Vocabulary Construction
        ↓
Train / Validation / Test Split
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
Docker Deployment
        ↓
Automated Testing
        ↓
Continuous Integration
```

The project brings together machine learning, natural language processing, model evaluation, backend development, web application development, testing, and deployment.

## Future Improvements

Possible directions for extending the system include:

1. Training on the full 450,000-line corpus
2. Longer context windows
3. Larger recurrent architectures
4. Improved text normalization
5. Beam search based decoding
6. Transformer based language models
7. Improved prefix completion ranking
8. Model quantization
9. GPU optimized training
10. Production scale inference optimization
11. Larger model training for LSTM and SimpleRNN, run with several seeds, to complete a fair three way comparison at the longer context window

## License

The source code for this project is released under the MIT License.

See [`LICENSE`](LICENSE) for the complete license text.

The dataset has separate usage considerations described in the dataset section above.