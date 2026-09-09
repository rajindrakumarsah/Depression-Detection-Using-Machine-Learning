from __future__ import annotations

from datetime import datetime, timezone
import numpy as np


def temporal_features_from_datetime(dt: datetime | None = None) -> np.ndarray:
    """Return the four posting-time features used by the deployed model."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    hour = dt.hour
    dow = dt.weekday()
    return np.array(
        [
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
        ],
        dtype=np.float32,
    )
