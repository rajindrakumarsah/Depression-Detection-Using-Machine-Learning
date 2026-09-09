from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import torch
from transformers import AutoTokenizer

from .data_utils import clean_text
from .features import temporal_features_from_datetime
from .model import HybridDistilBertClassifier


class DepressionPredictor:
    def __init__(self, model_dir: str | Path, device: str | None = None):
        model_dir = Path(model_dir)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.model, self.metadata = HybridDistilBertClassifier.load_bundle(
            model_dir,
            map_location=self.device,
        )

        self.model.to(self.device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir / "tokenizer"
        )

        self.temporal_scaler = joblib.load(
            model_dir / "temporal_scaler.joblib"
        )

        self.max_length = int(
            self.metadata.get("max_length", 256)
        )

        self.class_names = self.metadata.get(
            "class_names",
            [
                "Control / non-depression-source",
                "Depression-associated source",
            ],
        )

    def _scaled_temporal_from_datetime(
        self,
        dt: datetime | None,
    ) -> np.ndarray:

        if dt is None:
            dt = datetime.now(timezone.utc)

        temp = temporal_features_from_datetime(dt)

        return self.temporal_scaler.transform(
            temp.reshape(1, -1)
        )[0].astype(np.float32)

    @torch.inference_mode()
    def _predict_proba_with_scaled_temporal(
        self,
        texts: Iterable[str],
        scaled_temporal: np.ndarray,
        batch_size: int = 32,
    ) -> np.ndarray:

        texts = [clean_text(t, "") for t in texts]

        scaled_temporal = np.asarray(
            scaled_temporal,
            dtype=np.float32,
        ).reshape(-1)

        all_probs = []

        for start in range(0, len(texts), batch_size):

            batch_texts = texts[start:start + batch_size]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            encoded = {
                k: v.to(self.device)
                for k, v in encoded.items()
            }

            temp_tensor = torch.tensor(
                np.repeat(
                    scaled_temporal[None, :],
                    len(batch_texts),
                    axis=0,
                ),
                dtype=torch.float32,
                device=self.device,
            )

            logits = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                temporal_features=temp_tensor,
            )["logits"]

            probs = torch.softmax(
                logits,
                dim=1,
            ).cpu().numpy()

            all_probs.append(probs)

        if all_probs:
            return np.vstack(all_probs)

        return np.empty(
            (0, 2),
            dtype=np.float32,
        )

    @torch.inference_mode()
    def predict_proba(
        self,
        texts: Iterable[str],
        dt: datetime | None = None,
        batch_size: int = 32,
    ) -> np.ndarray:

        scaled_temp = self._scaled_temporal_from_datetime(dt)

        return self._predict_proba_with_scaled_temporal(
            texts,
            scaled_temp,
            batch_size=batch_size,
        )

    def predict_one(
        self,
        text: str,
        dt: datetime | None = None,
    ) -> dict:

        probs = self.predict_proba(
            [text],
            dt=dt,
        )[0]

        pred = int(np.argmax(probs))

        return {
            "label_id": pred,
            "label": self.class_names[pred],
            "confidence": float(probs[pred]),
            "probabilities": {
                self.class_names[i]: float(probs[i])
                for i in range(len(probs))
            },
        }

    def temporal_effect(
        self,
        text: str,
        dt: datetime | None = None,
    ) -> dict:
        """
        Estimate the local influence of posting time.

        The actual posting time is compared with the average
        standardized temporal context while keeping the post
        text unchanged.

        This is a model-sensitivity estimate rather than
        a causal or clinical explanation.
        """

        if dt is None:
            dt = datetime.now(timezone.utc)

        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=timezone.utc
            )

        dt = dt.astimezone(
            timezone.utc
        )

        actual_scaled = self._scaled_temporal_from_datetime(dt)

        # Zero represents the training-set mean
        # in standardized temporal-feature space.
        average_scaled = np.zeros_like(
            actual_scaled,
            dtype=np.float32,
        )

        actual_probs = (
            self._predict_proba_with_scaled_temporal(
                [text],
                actual_scaled,
            )[0]
        )

        average_probs = (
            self._predict_proba_with_scaled_temporal(
                [text],
                average_scaled,
            )[0]
        )

        pred = int(
            np.argmax(actual_probs)
        )

        return {
            "label_id": pred,

            "timestamp_utc": dt.strftime(
                "%A, %Y-%m-%d %H:%M UTC"
            ),

            "actual_probability":
                float(actual_probs[pred]),

            "average_temporal_probability":
                float(average_probs[pred]),

            "displayed_class_delta":
                float(
                    actual_probs[pred]
                    - average_probs[pred]
                ),

            "depression_probability_actual":
                float(actual_probs[1]),

            "depression_probability_average_temporal":
                float(average_probs[1]),

            "depression_probability_delta":
                float(
                    actual_probs[1]
                    - average_probs[1]
                ),
        }