from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import (
    BASE_MODEL,
    CLASS_NAMES,
    DATA_DIR,
    MAX_LENGTH,
    MODEL_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    TEMPORAL_FEATURES,
)
from src.metrics_utils import classification_metrics, save_confusion_matrix, save_metrics
from src.model import HybridDistilBertClassifier


torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class RedditDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, scaler):
        self.texts = frame["text"].fillna("").astype(str).tolist()
        self.labels = frame["label"].astype(int).to_numpy()
        self.temporal = scaler.transform(frame[TEMPORAL_FEATURES]).astype("float32")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.temporal[idx], int(self.labels[idx])


def make_collate(tokenizer, max_length):
    def collate(batch):
        texts, temporal, labels = zip(*batch)
        enc = tokenizer(
            list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "temporal_features": torch.tensor(np.stack(temporal), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    return collate


def run_epoch(model, loader, device, optimizer=None, scheduler=None, use_amp=False):
    training = optimizer is not None
    model.train(training)
    losses, true_all, pred_all, prob_all = [], [], [], []
    scaler = torch.amp.GradScaler("cuda", enabled=training and use_amp)

    iterator = tqdm(loader, desc="train" if training else "eval", leave=False)
    for batch in iterator:
        batch = {k: v.to(device) for k, v in batch.items()}
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(**batch)
                loss = out["loss"]
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

        probs = torch.softmax(out["logits"].detach(), dim=1)
        preds = probs.argmax(dim=1)
        losses.append(float(loss.detach().cpu()))
        true_all.extend(batch["labels"].detach().cpu().tolist())
        pred_all.extend(preds.cpu().tolist())
        prob_all.extend(probs[:, 1].cpu().tolist())

    return {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "f1": float(f1_score(true_all, pred_all, zero_division=0)),
        "y_true": np.array(true_all),
        "y_pred": np.array(pred_all),
        "y_prob": np.array(prob_all),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the final hybrid DistilBERT model.")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-train-rows", type=int, default=0,
                        help="Debug only: limit training rows. 0 uses the full prepared split.")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    if args.max_train_rows:
        train = train.sample(n=min(args.max_train_rows, len(train)), random_state=RANDOM_SEED)

    scaler = joblib.load(DATA_DIR / "temporal_scaler.joblib")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    collate = make_collate(tokenizer, args.max_length)

    train_loader = DataLoader(
        RedditDataset(train, scaler), batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=0
    )
    val_loader = DataLoader(
        RedditDataset(val, scaler), batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=collate, num_workers=0
    )
    test_loader = DataLoader(
        RedditDataset(test, scaler), batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=collate, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Training on: {device}")

    model = HybridDistilBertClassifier(
        args.base_model,
        n_temporal_features=len(TEMPORAL_FEATURES),
        temporal_hidden=32,
        dropout=0.2,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=max(1, total_steps),
    )

    best_f1 = -1.0
    patience_left = args.patience
    history = []
    best_path = RESULTS_DIR / "best_hybrid_state.pt"

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, optimizer, scheduler, use_amp)
        va = run_epoch(model, val_loader, device, use_amp=use_amp)
        row = {"epoch": epoch, "train_loss": tr["loss"], "train_f1": tr["f1"],
               "val_loss": va["loss"], "val_f1": va["f1"]}
        history.append(row)
        print(json.dumps(row, indent=2))

        if va["f1"] > best_f1:
            best_f1 = va["f1"]
            patience_left = args.patience
            torch.save(model.state_dict(), best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_result = run_epoch(model, test_loader, device, use_amp=use_amp)
    metrics = classification_metrics(
        test_result["y_true"], test_result["y_pred"], test_result["y_prob"]
    )
    save_metrics(metrics, RESULTS_DIR / "hybrid_metrics.json")
    save_confusion_matrix(
        test_result["y_true"], test_result["y_pred"], RESULTS_DIR / "hybrid_confusion_matrix.png"
    )
    pd.DataFrame(history).to_csv(RESULTS_DIR / "training_history.csv", index=False)
    pd.DataFrame({
        "true_label": test_result["y_true"],
        "predicted_label": test_result["y_pred"],
        "depression_probability": test_result["y_prob"],
    }).to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    metadata = {
        "base_model": args.base_model,
        "max_length": args.max_length,
        "n_temporal_features": len(TEMPORAL_FEATURES),
        "temporal_features": TEMPORAL_FEATURES,
        "temporal_hidden": 32,
        "dropout": 0.2,
        "num_labels": 2,
        "class_names": CLASS_NAMES,
        "random_seed": RANDOM_SEED,
        "deployed_model": "Hybrid DistilBERT + posting-time features",
    }
    model.save_bundle(MODEL_DIR, tokenizer, metadata)
    # The fitted scaler is part of the deployed artifact bundle.
    joblib.dump(scaler, MODEL_DIR / "temporal_scaler.joblib")
    if best_path.exists():
        best_path.unlink()

    print("\nFinal test metrics:")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print(f"Saved final model to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
