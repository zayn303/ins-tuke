from typing import List
import torch
import torch.nn as nn
from .base_trainer import BaseTrainer


def _cov(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    if n <= 1:
        return torch.zeros(x.shape[1], x.shape[1], device=x.device, dtype=x.dtype)
    x_c = x - x.mean(dim=0, keepdim=True)
    return (x_c.T @ x_c) / (n - 1)


def coral_loss(feature_list: List[torch.Tensor]) -> torch.Tensor:
    loss = torch.tensor(0.0, device=feature_list[0].device, requires_grad=True)
    d = feature_list[0].shape[1]
    for i in range(len(feature_list)):
        for j in range(i + 1, len(feature_list)):
            c_i = _cov(feature_list[i])
            c_j = _cov(feature_list[j])
            diff = c_i - c_j
            loss = loss + torch.sum(diff * diff) / (4 * d * d)
    return loss


class CORALTrainer(BaseTrainer):
    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        lr: float,
        weight_decay: float,
        lambda_coral: float,
        device: torch.device,
    ):
        super().__init__(backbone, classifier, lr, weight_decay, device)
        self.lambda_coral = lambda_coral
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, loader, **kwargs) -> float:
        self.backbone.train()
        self.classifier.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            waveform = batch["waveform"].squeeze(1).to(self.device)
            labels = batch["label"].float().to(self.device)
            domain_ids = batch["domain_id"].to(self.device)

            self.optimizer.zero_grad()

            backbone_out = self.backbone(waveform)
            features = self.classifier.get_features(backbone_out)
            logits = self.classifier.head(features).squeeze(-1)

            label_loss = self.criterion(logits, labels)

            unique_domains = domain_ids.unique()
            if len(unique_domains) > 1:
                domain_features = [
                    features[domain_ids == d] for d in unique_domains
                ]
                c_loss = coral_loss(domain_features)
            else:
                c_loss = torch.tensor(0.0, device=self.device)

            loss = label_loss + self.lambda_coral * c_loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)
