from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import DATA_DIR, DEFAULT_DATASET, RANDOM_SEED, TEMPORAL_FEATURES
from src.data_utils import clean_text, normalize_chunk, read_in_chunks


def _combine_grouped(parts, group_cols, value_cols):
    if not parts:
        return pd.DataFrame(columns=[*group_cols, *value_cols])
    df = pd.concat(parts, ignore_index=True)
    return df.groupby(group_cols, as_index=False)[value_cols].sum()


def main():
    parser = argparse.ArgumentParser(description="Prepare the final dissertation dataset and temporal aggregates.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--per-class", type=int, default=25_000,
                        help="Final balanced sample size per class before train/val/test split.")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.input.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.input}\n"
            "Place reddit_depression_dataset-compressed.zip in data/raw/ or pass --input."
        )

    rng = np.random.default_rng(RANDOM_SEED)
    pool_target = max(args.per_class, int(args.per_class * 1.25))
    pools = {0: pd.DataFrame(), 1: pd.DataFrame()}

    hour_parts, dow_parts, month_parts, source_parts, engagement_parts = [], [], [], [], []
    raw_rows = valid_rows = 0

    reader = read_in_chunks(args.input, chunksize=args.chunk_size)
    for raw in tqdm(reader, desc="Scanning raw dataset"):
        raw_rows += len(raw)
        df = normalize_chunk(raw, build_text=False)
        valid_rows += len(df)
        if df.empty:
            continue

        # Uniform priority sampling: keep the globally smallest random priorities per class.
        df = df.copy()
        df["_priority"] = rng.random(len(df))
        keep_cols = [
            "subreddit", "title", "body", "created_utc", "datetime", "label", "upvotes", "num_comments",
            "year", "month", "hour", "day_of_week", "year_month", "text_length",
            *TEMPORAL_FEATURES, "_priority"
        ]
        for label in (0, 1):
            candidate = df[df["label"].eq(label)][keep_cols]
            if candidate.empty:
                continue
            candidate = candidate.nsmallest(min(pool_target, len(candidate)), "_priority")
            merged = pd.concat([pools[label], candidate], ignore_index=True)
            pools[label] = merged.nsmallest(min(pool_target, len(merged)), "_priority")

        # Full-dataset aggregate statistics; no raw post text is retained for these tables.
        hour_parts.append(df.groupby(["label", "hour"]).size().rename("posts").reset_index())
        dow_parts.append(df.groupby(["label", "day_of_week"]).size().rename("posts").reset_index())
        month_parts.append(df.groupby(["label", "year_month"]).size().rename("posts").reset_index())
        source_parts.append(df.groupby(["label", "subreddit"]).size().rename("posts").reset_index())

        e = df.assign(
            upvotes_sum=df["upvotes"].clip(lower=0),
            comments_sum=df["num_comments"].clip(lower=0),
            text_length_sum=df["text_length"],
            rows=1,
        ).groupby(["label", "year_month"], as_index=False)[
            ["upvotes_sum", "comments_sum", "text_length_sum", "rows"]
        ].sum()
        engagement_parts.append(e)

    combined = pd.concat([pools[0], pools[1]], ignore_index=True)
    combined = combined.drop(columns="_priority").copy()
    combined["text"] = [clean_text(t, b) for t, b in zip(combined["title"], combined["body"])]
    combined = combined[combined["text"].str.len() >= 5].drop_duplicates(subset=["text"]).copy()
    combined["text_length"] = combined["text"].str.len().astype("int32")

    # Re-balance after deduplication.
    final_parts = []
    for label in (0, 1):
        part = combined[combined["label"].eq(label)]
        if len(part) < args.per_class:
            raise RuntimeError(
                f"Only {len(part)} unique rows available for label {label}; requested {args.per_class}."
            )
        final_parts.append(part.sample(n=args.per_class, random_state=RANDOM_SEED))
    balanced = pd.concat(final_parts, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    train, temp = train_test_split(
        balanced, test_size=0.20, stratify=balanced["label"], random_state=RANDOM_SEED
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["label"], random_state=RANDOM_SEED
    )

    # Features are already bounded cyclic encodings, but fitting a scaler keeps the pipeline explicit
    # and allows future temporal features to be added safely.
    scaler = StandardScaler()
    scaler.fit(train[TEMPORAL_FEATURES])
    joblib.dump(scaler, args.out_dir / "temporal_scaler.joblib")

    save_cols = ["text", "label", "created_utc", "subreddit", "upvotes", "num_comments", *TEMPORAL_FEATURES]
    train[save_cols].to_csv(args.out_dir / "train.csv", index=False)
    val[save_cols].to_csv(args.out_dir / "val.csv", index=False)
    test[save_cols].to_csv(args.out_dir / "test.csv", index=False)

    hour = _combine_grouped(hour_parts, ["label", "hour"], ["posts"])
    dow = _combine_grouped(dow_parts, ["label", "day_of_week"], ["posts"])
    month = _combine_grouped(month_parts, ["label", "year_month"], ["posts"])
    source = _combine_grouped(source_parts, ["label", "subreddit"], ["posts"])
    engagement = _combine_grouped(
        engagement_parts, ["label", "year_month"],
        ["upvotes_sum", "comments_sum", "text_length_sum", "rows"]
    )
    engagement["mean_upvotes"] = engagement["upvotes_sum"] / engagement["rows"].clip(lower=1)
    engagement["mean_comments"] = engagement["comments_sum"] / engagement["rows"].clip(lower=1)
    engagement["mean_text_length"] = engagement["text_length_sum"] / engagement["rows"].clip(lower=1)

    # Within-class normalized distributions are safer to interpret than raw class prevalence,
    # because the Kaggle labels are constructed from source subreddits and collection sizes differ.
    hour["within_class_share"] = hour["posts"] / hour.groupby("label")["posts"].transform("sum")
    dow["within_class_share"] = dow["posts"] / dow.groupby("label")["posts"].transform("sum")
    month["within_class_share"] = month["posts"] / month.groupby("label")["posts"].transform("sum")

    hour.to_csv(args.out_dir / "temporal_hour.csv", index=False)
    dow.to_csv(args.out_dir / "temporal_day_of_week.csv", index=False)
    month.to_csv(args.out_dir / "temporal_month.csv", index=False)
    engagement.to_csv(args.out_dir / "temporal_engagement.csv", index=False)
    source.to_csv(args.out_dir / "source_label_audit.csv", index=False)

    manifest = {
        "input_file": args.input.name,
        "raw_rows_scanned": raw_rows,
        "valid_rows_after_schema_cleaning": valid_rows,
        "balanced_rows": len(balanced),
        "per_class": args.per_class,
        "train_rows": len(train),
        "validation_rows": len(val),
        "test_rows": len(test),
        "random_seed": RANDOM_SEED,
        "temporal_features_used_by_classifier": TEMPORAL_FEATURES,
        "note": (
            "Labels are subreddit/source proxies, not clinical diagnoses. Temporal analysis is "
            "population-level because the dataset has no author/user identifier."
        ),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nPreparation complete")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
