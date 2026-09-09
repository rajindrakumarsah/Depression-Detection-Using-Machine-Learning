from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import DATA_DIR, RESULTS_DIR, RANDOM_SEED
from src.metrics_utils import classification_metrics, save_confusion_matrix, save_metrics


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=60_000,
            ngram_range=(1, 2),
            min_df=2,
            strip_accents="unicode",
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            class_weight="balanced",
        )),
    ])
    pipeline.fit(train["text"].astype(str), train["label"])
    pred = pipeline.predict(test["text"].astype(str))
    prob = pipeline.predict_proba(test["text"].astype(str))[:, 1]
    metrics = classification_metrics(test["label"], pred, prob)
    save_metrics(metrics, RESULTS_DIR / "baseline_metrics.json")
    save_confusion_matrix(test["label"], pred, RESULTS_DIR / "baseline_confusion_matrix.png")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()
