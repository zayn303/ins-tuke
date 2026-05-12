import torch
import torch.nn as nn
from .base_trainer import BaseTrainer


class MixupTrainer(BaseTrainer):
    def __init__(self, backbone, classifier, lr, weight_decay, device):
        super().__init__(backbone, classifier, lr, weight_decay, device)
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, loader, **kwargs) -> float:
        self.backbone.train()
        self.classifier.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            waveform = batch["waveform"].squeeze(1).to(self.device)  # [B, T]
            labels = batch["label"].float().to(self.device)          # [B] soft labels

            self.optimizer.zero_grad()
            features = self.backbone(waveform)
            logits = self.classifier(features).squeeze(-1)  # [B]
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)
