from typing import List
import torch
import torch.nn as nn
from .base_trainer import BaseTrainer
from src.models.domain_discriminator import DomainDiscriminator


class DIFLTrainer(BaseTrainer):
    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        source_domain_ids: List[int],
        lr: float,
        weight_decay: float,
        lambda_grl_max: float,
        total_epochs: int,
        device: torch.device,
    ):
        super().__init__(backbone, classifier, lr, weight_decay, device)
        n_domains = len(source_domain_ids)
        # Map global domain_id → local class index {0, ..., n_domains-1}
        self._domain_id_map = {gid: lid for lid, gid in enumerate(sorted(source_domain_ids))}
        self.domain_discriminator = DomainDiscriminator(
            feature_dim=512, n_domains=n_domains, alpha=1.0
        ).to(device)
        self.lambda_grl_max = lambda_grl_max
        self.total_epochs = total_epochs
        self.label_criterion = nn.BCEWithLogitsLoss()
        self.domain_criterion = nn.CrossEntropyLoss()

        self.optimizer.add_param_group(
            {"params": self.domain_discriminator.parameters(), "lr": lr, "weight_decay": weight_decay}
        )

    def _get_lambda(self, epoch: int, total_epochs: int) -> float:
        return self.lambda_grl_max * epoch / max(total_epochs - 1, 1)

    def train_epoch(self, loader, epoch: int = 0, **kwargs) -> float:
        self.backbone.train()
        self.classifier.train()
        self.domain_discriminator.train()

        lam = self._get_lambda(epoch, self.total_epochs)
        self.domain_discriminator.set_alpha(lam)

        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            waveform = batch["waveform"].squeeze(1).to(self.device)
            labels = batch["label"].float().to(self.device)
            global_domain_ids = batch["domain_id"]
            local_domain_ids = torch.tensor(
                [self._domain_id_map[d.item()] for d in global_domain_ids],
                dtype=torch.long, device=self.device,
            )

            self.optimizer.zero_grad()

            backbone_out = self.backbone(waveform)
            features = self.classifier.get_features(backbone_out)
            logits = self.classifier.head(features).squeeze(-1)
            domain_logits = self.domain_discriminator(features)

            label_loss = self.label_criterion(logits, labels)
            domain_loss = self.domain_criterion(domain_logits, local_domain_ids)
            loss = label_loss + lam * domain_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)
