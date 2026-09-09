# Final Dissertation Software

## Project
**Explainable Deep Learning for Early Depression Detection from Social Media Data Using NLP and Temporal Behavioural Analysis**

This repository contains **one integrated final software system**. Development scripts, benchmark evaluation, the final hybrid DistilBERT model, temporal analysis, LIME explanations, and the Streamlit interface all belong to the same codebase.

## Final architecture

```text
Kaggle Reddit depression dataset
        |
        +--> validation + text cleaning
        |       - title + body
        |       - malformed rows removed
        |       - URLs/user mentions cleaned
        |       - explicit r/depression and r/SuicideWatch cues masked
        |
        +--> temporal analysis over the full valid dataset
        |       - hour of day
        |       - day of week
        |       - calendar trends
        |       - upvotes/comments/text length (descriptive only)
        |
        +--> balanced reproducible training subset
                |
                +--> TF-IDF + Logistic Regression benchmark
                |
                +--> FINAL MODEL
                     DistilBERT text representation
                         +
                     posting-time features
                     (hour/day cyclic encoding)
                         |
                     binary source-class prediction
                         |
                     LIME explanation
                         |
                     Streamlit application
```

## Important research interpretation

The Kaggle target is a **source/subreddit proxy**, not a clinical diagnosis. The software therefore reports **depression-associated source/content** rather than claiming that a person has depression.

The dataset includes `created_utc` but does not contain a stable `author`/`user_id`. Consequently, temporal findings are **population-level posting patterns**, not longitudinal tracking of an individual user.

Upvotes and comment counts are useful for descriptive behavioural analysis, but they are not included in the live classifier because they are not known at the instant a new post is published. The final classifier uses only posting-time features that are available immediately.

## 1. Put the dataset in the project

Place the downloaded Kaggle ZIP here without extracting it:

```text
data/raw/reddit_depression_dataset-compressed.zip
```

The scripts read the ZIP directly, so the 1+ GB CSV does not need to be manually extracted.

## 2. Create environment

Python 3.11 is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Prepare the data

```bash
python prepare_data.py
```

Default behaviour:

- scans the full raw dataset in chunks;
- validates the schema;
- removes malformed rows;
- creates population-level temporal aggregate tables from the full dataset;
- creates a reproducible balanced subset of 25,000 examples per class;
- removes duplicate model text;
- creates 80% train / 10% validation / 10% test splits;
- saves temporal feature preprocessing metadata.

For a smaller quick run:

```bash
python prepare_data.py --per-class 5000
```

## 4. Train the conventional benchmark

```bash
python train_baseline.py
```

This is **not a second version of the software**. It is a benchmark experiment required to critically evaluate whether the final deep-learning approach improves on a conventional NLP baseline.

Outputs are saved in `artifacts/results/`.

## 5. Train the final model

The final deployed classifier is a **single hybrid DistilBERT model** combining:

1. contextual text representation from `distilbert-base-uncased`; and
2. posting-time features: hour-of-day and day-of-week encoded cyclically.

```bash
python train_model.py
```

Defaults:

- max sequence length: 256 tokens
- batch size: 16
- learning rate: 2e-5
- maximum epochs: 3
- validation F1 early stopping
- 10% linear warm-up
- AdamW optimization
- GPU mixed precision when CUDA is available

### Recommended: Google Colab GPU

Upload the project folder and dataset ZIP to Google Drive or the Colab runtime, install requirements, then run:

```bash
!python prepare_data.py
!python train_baseline.py
!python train_model.py --batch-size 16
```

If GPU memory is limited:

```bash
!python train_model.py --batch-size 8
```

Do not use `--max-train-rows` for the final dissertation experiment; it exists only for debugging.

## 6. Run the final Streamlit system

```bash
streamlit run app.py
```

The application contains:

- **Prediction & LIME** — final model output, confidence and local explanation;
- **Temporal Analysis** — hour/day/month and engagement patterns;
- **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, confusion matrix and model comparison;
- **Project & Ethics** — data manifest, scope and responsible-use limitations.

## Folder structure

```text
depression_detection_final/
|
|-- app.py
|-- config.py
|-- prepare_data.py
|-- train_baseline.py
|-- train_model.py
|-- requirements.txt
|-- README.md
|
|-- data/
|   `-- raw/
|       `-- reddit_depression_dataset-compressed.zip   # user supplies
|
|-- src/
|   |-- data_utils.py
|   |-- features.py
|   |-- model.py
|   |-- inference.py
|   |-- explain.py
|   `-- metrics_utils.py
|
|-- artifacts/
|   |-- data/       # generated splits and temporal aggregates
|   |-- model/      # final trained model bundle
|   `-- results/    # metrics, confusion matrix, history
|
`-- tests/
    `-- test_data_utils.py
```

## Dissertation evidence generated automatically

The pipeline creates reusable evidence for the report:

- `manifest.json` — data preparation and split details;
- `source_label_audit.csv` — evidence of source-label construction;
- temporal aggregate CSV files — evidence for temporal analysis;
- `baseline_metrics.json` — conventional benchmark results;
- `hybrid_metrics.json` — final model results;
- `hybrid_confusion_matrix.png` — test confusion matrix;
- `training_history.csv` — epoch-level train/validation history;
- `test_predictions.csv` — final held-out predictions.

These outputs should be preserved exactly after the final experiment so that the dissertation reports the actual software results rather than reconstructed numbers.

## Ethical boundary

This software is for academic research only. It is not a medical device, screening service, crisis service, or clinical decision system. Predictions should never be used to diagnose an individual or determine treatment.
