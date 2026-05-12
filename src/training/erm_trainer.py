from typing import Optional
import torch
import torch.nn as nn
from .base_trainer import BaseTrainer


class ERMTrainer(BaseTrainer):
    def __init__(self, backbone, classifier, lr, weight_decay, device,
                 pos_weight: Optional[torch.Tensor] = None):
        super().__init__(backbone, classifier, lr, weight_decay, device)
        pw = pos_weight.to(device) if pos_weight is not None else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    def train_epoch(self, loader, **kwargs) -> float:
        self.backbone.train()
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        self.classifier.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            input_values = batch["input_values"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch["label"].float().to(self.device)

            self.optimizer.zero_grad()
            features = self.backbone(input_values, attention_mask=attention_mask)
            logits = self.classifier(features).squeeze(-1)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        mean_loss = total_loss / max(n_batches, 1)
        self.last_stats = {"label_loss": mean_loss}
        return mean_loss
