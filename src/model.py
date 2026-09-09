from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel


class HybridDistilBertClassifier(nn.Module):
    """DistilBERT text encoder + small temporal-feature branch.

    The final deployed model combines contextual NLP representations with posting-time
    features (hour/day encoded cyclically). This is one integrated model, not a separate
    software version.
    """

    def __init__(
        self,
        transformer_source: str | Path,
        n_temporal_features: int = 4,
        temporal_hidden: int = 32,
        dropout: float = 0.2,
        num_labels: int = 2,
    ):
        super().__init__()
        self.transformer_source = str(transformer_source)
        self.transformer = AutoModel.from_pretrained(self.transformer_source)
        hidden_size = int(self.transformer.config.hidden_size)

        self.temporal_net = nn.Sequential(
            nn.Linear(n_temporal_features, temporal_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size + temporal_hidden, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temporal_features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT's first token representation is used as the sentence representation.
        text_repr = outputs.last_hidden_state[:, 0]
        temp_repr = self.temporal_net(temporal_features.float())
        fused = torch.cat([text_repr, temp_repr], dim=1)
        logits = self.classifier(self.dropout(fused))
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

    def save_bundle(self, output_dir: str | Path, tokenizer, metadata: dict) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        transformer_dir = output_dir / "transformer"
        tokenizer_dir = output_dir / "tokenizer"
        self.transformer.save_pretrained(transformer_dir)
        tokenizer.save_pretrained(tokenizer_dir)
        torch.save({
            "temporal_net": self.temporal_net.state_dict(),
            "classifier": self.classifier.state_dict(),
        }, output_dir / "hybrid_head.pt")
        (output_dir / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    @classmethod
    def load_bundle(cls, model_dir: str | Path, map_location="cpu"):
        model_dir = Path(model_dir)
        metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
        model = cls(
            transformer_source=model_dir / "transformer",
            n_temporal_features=metadata["n_temporal_features"],
            temporal_hidden=metadata["temporal_hidden"],
            dropout=metadata["dropout"],
            num_labels=metadata["num_labels"],
        )
        state = torch.load(model_dir / "hybrid_head.pt", map_location=map_location)
        model.temporal_net.load_state_dict(state["temporal_net"])
        model.classifier.load_state_dict(state["classifier"])
        return model, metadata
