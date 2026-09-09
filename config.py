from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = ARTIFACT_DIR / "data"
MODEL_DIR = ARTIFACT_DIR / "model"
RESULTS_DIR = ARTIFACT_DIR / "results"

DEFAULT_DATASET = RAW_DATA_DIR / "reddit_depression_dataset-compressed.zip"
BASE_MODEL = "distilbert-base-uncased"
RANDOM_SEED = 42
MAX_LENGTH = 256

# Only features known at posting time are used in the deployed classifier.
# Upvotes/comments are analyzed in the temporal dashboard but not used for prediction,
# avoiding post-publication information leakage.
TEMPORAL_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

CLASS_NAMES = ["Control / non-depression-source", "Depression-associated source"]
