# Next-Word Prediction: GRU

A phone-keyboard-style next-word predictor built with a GRU neural network.

Given a sentence, the model predicts the most likely words that come next. It also supports mid-word completion, similar to a mobile keyboard.

This project is part of my RNN / LSTM / GRU learning series, where the concepts are explained from first principles and then implemented here as a complete machine-learning project.

## What it does

For a prompt such as:

```text
i want to go to the
```

the model uses the previous words as context and returns the top 5 predicted next words.

It also supports incomplete words:

```text
i want to g
```

In this case, it returns word completions beginning with `g`.

The prediction pipeline is:

```text
Input text
    │
    ▼
Clean + tokenize
    │
    ▼
Token IDs
    │
    ▼
Embedding
    │
    ▼
GRU
    │
    ▼
Dropout
    │
    ▼
Dense layer
    │
    ▼
Softmax
    │
    ▼
Word probabilities
    │
    ▼
Top 5 suggestions
```

## Current model

The current implementation uses:

```text
Embedding
    ↓
GRU
    ↓
Dropout
    ↓
Dense
    ↓
Softmax
```

The GRU uses update and reset gates to control how information is retained and forgotten over the sequence.

Compared with an LSTM, a GRU combines the cell and hidden state and uses fewer gates, giving it a simpler architecture and generally fewer parameters.

## Dataset

The project uses the **SwiftKey / HC Corpora** dataset from the Coursera Data Science Specialization capstone.

The corpus contains three different types of English text:

* Blogs
* News
* Twitter

Using multiple registers makes the task more representative of real-world typing than training on a single style of text.

The raw dataset is not committed to this repository.

### Dataset terms

The dataset is widely used for learning and research projects, but its terms should be checked carefully before using it in a commercial product.

For a commercial deployment, use a dataset with clearly defined commercial-use permissions, such as an appropriately licensed public dataset or your own collected text with the required consent and permissions.

## Project structure

```text
next-word-prediction/
│
├── data/
│   ├── raw/                  # downloaded corpus (gitignored)
│   └── processed/            # processed datasets and statistics (gitignored)
│
├── models/
│   ├── tokenizer.json
│   └── gru.keras             # trained GRU checkpoint (gitignored)
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_lstm.ipynb
│   ├── 04_gru.ipynb
│   └── 05_comparison.ipynb
│
├── outputs/
│   ├── gru_training_curves.png
│   ├── gru_example_prediction.png
│   ├── metrics.json
│   └── ...
│
├── src/
│   ├── preprocess.py         # cleaning, tokenization, vocabulary
│   ├── dataset.py            # windows and train/validation/test splits
│   ├── model.py              # model architecture
│   ├── train.py              # training pipeline
│   ├── predict.py            # inference and prediction
│   └── utils.py              # utilities and metrics
│
├── static/
│   ├── style.css
│   └── app.js
│
├── templates/
│   └── index.html
│
├── tests/
│   └── ...
│
├── scripts/
│   ├── download_data.py
│   └── build_corpus.py
│
├── app.py                    # FastAPI application
├── config.py                 # paths and hyperparameters
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
└── .github/
    └── workflows/
        └── ci.yml
```

## Setup

Clone the repository:

```bash
git clone <your-repo-url>
cd next-word-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements-dev.txt
```

Download and prepare the dataset:

```bash
python scripts/download_data.py
```

By default, the corpus-building script samples a subset of the blogs, news and Twitter data so that training remains practical on a normal machine.

You can increase the number of lines per source:

```bash
python scripts/download_data.py --lines-per-source 400000
```

If the dataset has already been downloaded:

```bash
python scripts/download_data.py --skip-download
```

## Train the GRU

The GRU can be trained directly from the command line:

```bash
python -m src.train --model gru
```

The training pipeline uses:

* Early stopping
* Learning-rate reduction on validation-loss plateaus
* A fixed random seed for reproducibility
* Validation and test evaluation
* Automatic checkpoint saving

The trained model is saved as:

```text
models/gru.keras
```

Training metrics are written to:

```text
outputs/metrics.json
```

Training curves are saved to:

```text
outputs/gru_training_curves.png
```

## GRU notebook

The complete GRU walkthrough is available in:

```text
notebooks/04_gru.ipynb
```

The notebook covers:

1. Loading the prepared corpus
2. Training the GRU
3. Visualizing training curves
4. Evaluating the model on the test set
5. Generating next-word predictions
6. Generating mid-word completions
7. Visualizing the probability distribution behind the predictions

Example prompts include:

```text
i want to go to the
it is a
holmes was
what do you
```

And for word completion:

```text
i want to g
it was a very d
```

## Evaluation metrics

The model reports:

### Top-1 accuracy

Whether the correct next word is the model's first prediction.

### Top-5 accuracy

Whether the correct next word appears anywhere in the five suggestions.

This is particularly useful for a keyboard-style application because users see multiple suggestions rather than only one prediction.

### Perplexity

Perplexity is calculated from the test-set cross-entropy:

```text
perplexity = exp(cross_entropy)
```

Lower perplexity means the model assigns higher probability to the actual next words in the test set.

## Current GRU configuration

The GRU notebook currently uses:

| Hyperparameter      |            Value |
| ------------------- | ---------------: |
| Sequence length     |                5 |
| Embedding dimension |               64 |
| GRU units           |               64 |
| Batch size          |              256 |
| Maximum epochs      |                8 |
| Random seed         | From `config.py` |

These values are intentionally kept small enough to make experimentation practical.

The project configuration can be changed through `config.py` and the training interface.

## Prediction

The same predictor used by the web application can be loaded directly:

```python
from src.predict import Predictor

predictor = Predictor.load(
    config.MODELS_DIR / "gru.keras",
    config.VOCAB_PATH
)
```

Then:

```python
predictor.predict_next(
    "i want to go to the ",
    k=5
)
```

returns ranked suggestions containing the predicted word and probability.

For an incomplete word:

```python
predictor.predict_next(
    "i want to g",
    k=5
)
```

the predictor switches to completion mode and filters predictions according to the typed prefix.

## Web application

Run the FastAPI application:

```bash
uvicorn app:app --reload
```

Then open:

```text
http://localhost:8000
```

The API exposes:

```text
POST /api/predict
```

Example:

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "i want to go to the ", "k": 5}'
```

Example response:

```json
{
  "suggestions": [
    {
      "word": "station",
      "probability": 0.18
    },
    {
      "word": "house",
      "probability": 0.11
    }
  ],
  "mode": "next",
  "latency_ms": 4.2
}
```

The API supports two prediction modes:

```text
"next"
```

when the input ends with whitespace, and:

```text
"complete"
```

when the user is currently typing a word.

Health status is available at:

```text
GET /health
```

## Docker

Build and run the application with:

```bash
docker compose up --build
```

The Docker setup mounts the trained models read-only, so updated model weights can be picked up by restarting the container without rebuilding the image.

## Reproducibility and design decisions

### No data leakage

The corpus is split into train, validation and test sentences before vocabulary construction and window generation.

This prevents test sentences from influencing the vocabulary or training windows.

### Left padding and masking

Short contexts are left-padded with:

```text
<pad>
```

The embedding layer uses masking so the recurrent network does not learn from padding tokens.

### Special tokens

`<pad>` and `<unk>` are never returned as user-facing predictions.

### Text cleaning

The preprocessing pipeline removes or normalizes common social-media noise such as:

* URLs
* @mentions
* Retweet prefixes
* Hashtags

This allows the model to focus more on language patterns rather than memorizing usernames and URLs.

### Deterministic experiments

A fixed random seed is used for:

* Python
* NumPy
* TensorFlow

This makes experiments reproducible under the same environment and data.

## RNN / LSTM / GRU comparison

The project is designed to compare three recurrent architectures on the same task:

```text
SimpleRNN
LSTM
GRU
```

The comparison notebook is:

```text
notebooks/05_comparison.ipynb
```

The goal is to keep the comparison fair:

* Same dataset
* Same train/validation/test split
* Same vocabulary
* Same sequence length
* Same embedding dimension
* Same recurrent width
* Same batch size
* Same training configuration
* Same random seed

The main architectural difference is the recurrent layer itself.

Run the LSTM and GRU notebooks first:

```text
03_lstm.ipynb
04_gru.ipynb
```

Then run:

```text
05_comparison.ipynb
```

The comparison notebook trains any missing model, collects the metrics, and produces a table containing:

| Model     | Parameters | Top-1 accuracy | Top-5 accuracy | Perplexity | Training time |
| --------- | ---------: | -------------: | -------------: | ---------: | ------------: |
| SimpleRNN |          — |              — |              — |          — |             — |
| LSTM      |          — |              — |              — |          — |             — |
| GRU       |          — |              — |              — |          — |             — |

The actual values are generated from the experiments and should not be hard-coded into this README.

The notebook also generates:

```text
outputs/results.md
outputs/model_comparison.png
outputs/validation_curves_comparison.png
```

## Production model

After all three models have been evaluated, `05_comparison.ipynb` identifies the model with the lowest test-set perplexity.

The production model can then be selected using:

```text
PRODUCTION_MODEL
```

or the corresponding setting in:

```text
config.py
```

For example:

```bash
PRODUCTION_MODEL=gru
```

The comparison is based on measured test-set performance rather than assuming beforehand that RNN, LSTM or GRU will perform best.

## Tests

Run the test suite:

```bash
pytest -q
```

Run linting:

```bash
ruff check .
```

CI runs the tests, linting and Docker build on every push.

## Project status

Current workflow:

```text
01_eda.ipynb
      ↓
02_preprocessing.ipynb
      ↓
03_lstm.ipynb
      ↓
04_gru.ipynb
      ↓
05_comparison.ipynb
      ↓
Production model
```

The GRU implementation and prediction pipeline are complete.

The final step is the controlled comparison of **SimpleRNN vs LSTM vs GRU** using `05_comparison.ipynb`.

## License

This project is released under the MIT License.

See:

```text
LICENSE
```

for details.
