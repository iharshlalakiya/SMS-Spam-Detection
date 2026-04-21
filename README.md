# 🛡️ SMS Spam Detection

An end-to-end machine learning project that detects SMS spam with a **Linear SVM** classifier and a reproducible MLOps workflow built around **DVC**, **MLflow**, **Streamlit**, **pytest**, and **GitHub Actions**.

The repository is organized so you can do three things cleanly:

1. Train the model end-to-end from raw data.
2. Serve predictions through an interactive web app.
3. Reproduce, track, and redeploy the pipeline on Azure.

---

## What this project does

The pipeline starts with the UCI SMS Spam Collection dataset and ends with a trained spam detector and a deployed Streamlit app.

Raw CSV data is downloaded or reused, cleaned, tokenized, vectorized with TF-IDF, trained with a linear SVM, evaluated, and then logged to MLflow while DVC keeps the pipeline reproducible.

The web app in [app.py](app.py) loads the saved model artifacts from [models/](models/) and lets you classify a single SMS message with confidence scores and a recent-history panel.

---

## Project at a glance

```text
data/raw/spam.csv
	↓
sms_spam.data.preprocessing
	↓
sms_spam.features.extraction
	↓
sms_spam.models.svm
	↓
sms_spam.evaluation.evaluate
	↓
results/metrics/svm_results.json
models/svm.pkl
models/tfidf_vectorizer.pkl
	↓
app.py (Streamlit inference UI)
```

The same flow is also captured in [dvc.yaml](dvc.yaml), so you can reproduce only the stages that changed instead of rerunning everything manually.

---

## Model performance

The latest checked-in metrics in [results/metrics/svm_results.json](results/metrics/svm_results.json) report:

| Metric | Score |
|---|---|
| Accuracy | 97.94% |
| Precision | 97.73% |
| Recall | 86.58% |
| F1 Score | 91.81% |
| ROC-AUC | 98.37% |

These scores come from a model trained on the SMS Spam Collection dataset and evaluated on the held-out test split.

---

## Repository structure

```text
SMS-Spam-Detection/
├── app.py                      # Streamlit inference app
├── main.py                     # One-command full pipeline entry point
├── params.yaml                 # DVC-tracked hyperparameters and paths
├── dvc.yaml                    # DVC pipeline definition
├── requirements.txt            # Python dependencies
├── scripts/
│   └── download_dataset.py     # Standalone dataset download helper
├── automation/
│   ├── watcher.py              # Watches raw CSV files and retrains
│   ├── install_watcher.sh      # Installs the watcher on Azure VM
│   ├── vm_setup.sh             # VM bootstrap script
│   └── sms-watcher.service     # systemd service for the watcher
├── sms_spam/
│   ├── data/
│   │   ├── download.py         # DVC stage 1: download raw data
│   │   └── preprocessing.py    # Cleaning, preprocessing, dataset split
│   ├── features/
│   │   └── extraction.py       # TF-IDF feature engineering
│   ├── models/
│   │   └── svm.py              # Linear SVM wrapper
│   ├── evaluation/
│   │   ├── evaluate.py         # Metrics, plots, saved artifacts
│   │   └── metrics.py          # Metric and plot helpers
│   ├── train/
│   │   └── train.py            # Training stage entry point
│   ├── mlflow/
│   │   ├── mlflow_tracker.py    # MLflow logging wrapper
│   │   └── mlflow_registry.py  # Model registry helpers
│   └── logs/
│       └── logger.py           # Shared logging setup
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_model.py
├── data/
│   └── raw/spam.csv            # Raw dataset location after download
├── models/
│   ├── svm.pkl                 # Trained classifier
│   └── tfidf_vectorizer.pkl    # Fitted vectorizer
└── results/
	├── metrics/svm_results.json
	├── confusion_matrices/
	└── plots/
```

---

## Tech stack

| Layer | Tools |
|---|---|
| Machine learning | scikit-learn, NumPy, pandas |
| Text processing | NLTK |
| Experiment tracking | MLflow |
| Pipeline versioning | DVC |
| Web app | Streamlit |
| Automation | systemd, Python watcher script |
| CI/CD | GitHub Actions |
| Testing | pytest |

---

## How the pipeline runs

The main pipeline in [main.py](main.py) is the clearest way to understand the project.

It executes these steps:

1. Check whether `data/raw/spam.csv` already exists.
2. Download the dataset if needed.
3. Load and preprocess the SMS text.
4. Split the data into an 80/20 stratified train/test split.
5. Fit TF-IDF features with up to 5,000 terms and 1-2 grams.
6. Train a linear SVM classifier.
7. Evaluate predictions and save metrics, plots, and model artifacts.
8. Log parameters, metrics, and environment details to MLflow.

That logic is mirrored in DVC stages so you can run the workflow either as a single Python entry point or as a reproducible pipeline.

---

## Local setup

### 1. Clone the repository

```bash
git clone https://github.com/iharshlalakiya/SMS-Spam-Detection.git
cd SMS-Spam-Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you plan to use Kaggle downloads, install the optional Kaggle package as well:

```bash
pip install kaggle
```

---

## How to run it

### Option 1: Run the full pipeline

```bash
python main.py
```

This is the simplest end-to-end entry point. It will attempt to download the dataset automatically if `data/raw/spam.csv` is missing.

Useful flags:

```bash
python main.py --skip-download
python main.py --data-path data/raw/spam.csv
python main.py --models-dir models
python main.py --results-dir results
```

### Option 2: Reproduce with DVC

```bash
dvc repro
dvc dag
dvc metrics show
```

Use DVC when you want reproducibility and incremental stage execution driven by `params.yaml` and tracked pipeline outputs.

### Option 3: Launch the Streamlit app

```bash
streamlit run app.py
```

Open the app in your browser and paste an SMS message. The app returns:

* spam or ham prediction,
* spam/ham confidence bars,
* example messages in the sidebar,
* recent prediction history.

If the model files are missing, the app will ask you to train the pipeline first.

### Option 4: View MLflow locally

```bash
mlflow ui --port 5000
```

Then open `http://localhost:5000`.

### Option 5: Run the tests

```bash
pytest tests/ -v
```

---

## Dataset handling

The project uses the **UCI SMS Spam Collection** dataset.

There are three ways to get it into `data/raw/spam.csv`:

1. Let [main.py](main.py) download it automatically.
2. Run [scripts/download_dataset.py](scripts/download_dataset.py).
3. Place the CSV manually in `data/raw/`.

If you download manually, the code expects a dataset with `ham` and `spam` labels and the message text in the standard SMS Spam Collection format.

---

## Pipeline details

### Text preprocessing

Implemented in [sms_spam/data/preprocessing.py](sms_spam/data/preprocessing.py):

* lowercase normalization,
* URL, email, and phone-like pattern removal,
* tokenization,
* stopword removal,
* lemmatization,
* cleaned text stored as `processed_text`.

### Feature extraction

Implemented in [sms_spam/features/extraction.py](sms_spam/features/extraction.py):

| Parameter | Value |
|---|---|
| max_features | 5,000 |
| ngram_range | (1, 2) |
| stop_words | english |

The fitted vectorizer is saved to [models/tfidf_vectorizer.pkl](models/tfidf_vectorizer.pkl).

### Model

Implemented in [sms_spam/models/svm.py](sms_spam/models/svm.py):

| Parameter | Value |
|---|---|
| kernel | linear |
| C | 1.0 |
| max_iter | 1000 |
| probability | true |

The trained classifier is saved to [models/svm.pkl](models/svm.pkl).

### Evaluation outputs

Implemented in [sms_spam/evaluation/evaluate.py](sms_spam/evaluation/evaluate.py):

* confusion matrix saved to `results/confusion_matrices/svm_cm.png`,
* ROC curve saved to `results/plots/svm_roc.png`,
* metrics JSON saved to `results/metrics/svm_results.json`,
* model and vectorizer logged through MLflow.

---

## DVC stages

The repository’s [dvc.yaml](dvc.yaml) contains five stages:

1. `download`
2. `preprocess`
3. `featurize`
4. `train`
5. `evaluate`

You can compare changes over time with:

```bash
dvc metrics diff
dvc params diff
```

---

## MLflow tracking

MLflow tracking is wired through [sms_spam/mlflow/mlflow_tracker.py](sms_spam/mlflow/mlflow_tracker.py).

Each run logs:

* parameters from `params.yaml`,
* evaluation metrics,
* confusion matrix and ROC artifacts,
* environment snapshot for reproducibility,
* model registration settings when enabled.

The default local tracking URI is `mlruns`, but GitHub Actions can override it with environment variables.

---

## Automation and deployment

This repo includes two automation paths.

### 1. Azure VM watcher

The watcher in [automation/watcher.py](automation/watcher.py) monitors `data/raw/*.csv`. When a file changes, it runs the pipeline again and updates the model artifacts.

Install it with:

```bash
bash automation/install_watcher.sh
```

The watcher is registered as the `sms-watcher` systemd service using [automation/sms-watcher.service](automation/sms-watcher.service).

### 2. VM bootstrap script

[automation/vm_setup.sh](automation/vm_setup.sh) installs Python, creates a virtual environment, installs dependencies, registers the watcher, and creates a Streamlit systemd service if one does not already exist.

This script is meant for the Azure Ubuntu VM described in the workflow files.

### 3. GitHub Actions

The repository includes two workflows:

* [.github/workflows/ci.yml](.github/workflows/ci.yml) for test + reproduce + deploy on pushes to `main`.
* [.github/workflows/retrain.yml](.github/workflows/retrain.yml) for retraining when `data/raw/*.csv` or `params.yaml` changes.

Required secrets in GitHub Actions:

| Secret | Purpose |
|---|---|
| `VM_HOST` | Azure VM public IP |
| `VM_USER` | SSH user, usually `azureuser` |
| `VM_SSH_KEY` | Private SSH key content |
| `MLFLOW_TRACKING_URI` | Remote MLflow URI |
| `MLFLOW_TRACKING_USERNAME` | MLflow username |
| `MLFLOW_TRACKING_PASSWORD` | MLflow password or token |

---

## Tests

The tests in [tests/](tests/) are small smoke tests that validate the main building blocks:

* [tests/test_preprocessing.py](tests/test_preprocessing.py) checks cleaning and preprocessing.
* [tests/test_features.py](tests/test_features.py) checks TF-IDF extraction and persistence.
* [tests/test_model.py](tests/test_model.py) checks the SVM wrapper, save/load, and predictions.

Run them with:

```bash
pytest tests/ -v
```

---

## Common run sequence

If you want the shortest path from a fresh clone to a working app, use this order:

```bash
pip install -r requirements.txt
python main.py
streamlit run app.py
```

If you want full reproducibility and tracked artifacts, run:

```bash
dvc repro
mlflow ui --port 5000
streamlit run app.py
```

---

## Notes

* The app expects the trained artifacts in [models/](models/).
* The pipeline assumes the raw dataset is available at `data/raw/spam.csv`.
* NLTK resources are downloaded automatically by the preprocessing module when needed.
* The repo currently ships without a separate `monitoring/` folder, so MLflow is the main experiment-tracking surface.