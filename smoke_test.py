"""Fast checks that do not require a trained Transformer model."""
from datetime import datetime, timezone

import pandas as pd

from src.data_utils import normalize_chunk
from src.features import temporal_features_from_datetime

raw = pd.DataFrame({
    "subreddit": ["depression", "DeepThoughts"],
    "title": ["I feel exhausted", "A philosophical question"],
    "body": ["Nothing feels enjoyable lately", "What makes an idea meaningful?"],
    "upvotes": [3, 5],
    "created_utc": [1700000000, 1700001000],
    "num_comments": [1, 2],
    "label": [1, 0],
})

clean = normalize_chunk(raw)
assert len(clean) == 2
assert temporal_features_from_datetime(datetime.now(timezone.utc)).shape == (4,)
print("Smoke test passed.")
