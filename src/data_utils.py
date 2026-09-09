from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = {
    "subreddit",
    "title",
    "body",
    "upvotes",
    "created_utc",
    "num_comments",
    "label",
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_USER_RE = re.compile(r"(?<!\w)(?:/?u/)[A-Za-z0-9_-]+", flags=re.IGNORECASE)
# Remove explicit source-community names to reduce label leakage from subreddit mentions in text.
_SOURCE_SUB_RE = re.compile(
    r"(?<!\w)/?r/(?:depression|suicidewatch)(?!\w)", flags=re.IGNORECASE
)
_WS_RE = re.compile(r"\s+")


def validate_columns(columns: Iterable[str]) -> None:
    missing = EXPECTED_COLUMNS.difference(columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")


def clean_text(title: object, body: object) -> str:
    """Combine title/body while removing URLs, user mentions, and direct source-label cues.

    We deliberately do not stem, lemmatize, or remove stop words because the final
    classifier uses a pretrained transformer tokenizer and benefits from natural context.
    """
    title = "" if pd.isna(title) else str(title)
    body = "" if pd.isna(body) else str(body)
    text = f"{title}. {body}".strip()
    text = _URL_RE.sub(" [URL] ", text)
    text = _USER_RE.sub(" [USER] ", text)
    text = _SOURCE_SUB_RE.sub(" [COMMUNITY] ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_chunk(chunk: pd.DataFrame, build_text: bool = True) -> pd.DataFrame:
    """Validate and clean one raw CSV chunk.

    Set build_text=False for fast full-dataset aggregation; expensive regex text cleaning
    is then applied only to sampled model-training rows.
    """
    validate_columns(chunk.columns)
    df = chunk.copy()

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
    df["upvotes"] = pd.to_numeric(df["upvotes"], errors="coerce").fillna(0)
    df["num_comments"] = pd.to_numeric(df["num_comments"], errors="coerce").fillna(0)

    # Drop malformed rows and keep the documented binary labels only.
    df = df[df["label"].isin([0, 1]) & df["created_utc"].notna()].copy()
    df["label"] = df["label"].astype("int8")

    df["title"] = df["title"].fillna("")
    df["body"] = df["body"].fillna("")
    df["subreddit"] = df["subreddit"].fillna("unknown").astype(str)
    if build_text:
        df["text"] = [clean_text(t, b) for t, b in zip(df["title"], df["body"])]
        df = df[df["text"].str.len() >= 5].copy()

    ts = pd.to_datetime(df["created_utc"], unit="s", utc=True, errors="coerce")
    df["datetime"] = ts
    df = df[df["datetime"].notna()].copy()
    df["year"] = df["datetime"].dt.year.astype("int16")
    df["month"] = df["datetime"].dt.month.astype("int8")
    df["hour"] = df["datetime"].dt.hour.astype("int8")
    df["day_of_week"] = df["datetime"].dt.dayofweek.astype("int8")
    df["year_month"] = df["datetime"].dt.strftime("%Y-%m")

    # Cyclical representations avoid treating 23:00 and 00:00 as far apart.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0).astype("float32")
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0).astype("float32")
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0).astype("float32")

    if build_text:
        df["text_length"] = df["text"].str.len().astype("int32")
    else:
        df["text_length"] = (df["title"].str.len() + df["body"].str.len() + 2).astype("int32")
    return df


def read_in_chunks(path: str | Path, chunksize: int = 100_000):
    path = Path(path)
    compression = "zip" if path.suffix.lower() == ".zip" else "infer"
    yield from pd.read_csv(path, compression=compression, chunksize=chunksize, low_memory=False)
