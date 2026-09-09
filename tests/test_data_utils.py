from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.data_utils import clean_text, normalize_chunk
from src.features import temporal_features_from_datetime


def test_clean_text_keeps_semantic_depression_word_but_removes_subreddit_cue():
    text = clean_text("I am worried about depression", "see r/depression and u/example")
    assert "depression" in text.lower()
    assert "r/depression" not in text.lower()
    assert "u/example" not in text.lower()


def test_temporal_features_shape_and_finiteness():
    x = temporal_features_from_datetime(datetime(2026, 8, 11, 1, 0, tzinfo=timezone.utc))
    assert x.shape == (4,)
    assert np.isfinite(x).all()


def test_normalize_chunk():
    raw = pd.DataFrame({
        "subreddit": ["depression", "DeepThoughts"],
        "title": ["hello", "world"],
        "body": ["sample", None],
        "upvotes": [2, 3],
        "created_utc": [1_700_000_000, 1_700_000_100],
        "num_comments": [1, None],
        "label": [1, 0],
    })
    out = normalize_chunk(raw)
    assert len(out) == 2
    assert set(out["label"].tolist()) == {0, 1}
    assert all(c in out.columns for c in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"])
