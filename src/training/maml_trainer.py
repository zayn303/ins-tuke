import random
from typing import Dict, List, Any
import torch
import torch.nn as nn
import higher
from .base_trainer import BaseTrainer


class _FullModel(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        features = self.backbone(input_values)
        return self.classifier(features).squeeze(-1)  # [B] logit


class MAMLTrainer(BaseTrainer):
    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        lr: float,
        weight_decay: float,
        inner_lr: float,
        n_inner_steps: int,
        n_episodes_per_epoch: int,
        device: torch.device,
    ):
        super().__init__(backbone, classifier, lr, weight_decay, device)
        self.inner_lr = inner_lr
        self.n_inner_steps = n_inner_steps
        self.n_episodes_per_epoch = n_episodes_per_epoch
        self.full_model = _FullModel(backbone, classifier)
        self.criterion = nn.BCEWithLogitsLoss()

    def _get_batch(self, domain_batches: List[Dict]) -> Dict:
        return random.choice(domain_batches)

    def train_epoch(self, domain_loaders: Dict[int, List[Dict]], **kwargs) -> float:
        self.backbone.train()
        self.classifier.train()

        domain_ids = list(domain_loaders.keys())
        inner_opt = torch.optim.SGD(self.full_model.parameters(), lr=self.inner_lr)

        total_loss = 0.0

        for _ in range(self.n_episodes_per_epoch):
            # Support: N-1 domains, Query: remaining domain
            random.shuffle(domain_ids)
            query_domain = domain_ids[-1]
            support_domains = domain_ids[:-1]

            self.optimizer.zero_grad()

            with higher.innerloop_ctx(
                self.full_model, inner_opt, copy_initial_weights=False
            ) as (fmodel, diffopt):

                # Inner loop on support domains
                for _ in range(self.n_inner_steps):
                    support_loss = torch.tensor(0.0, device=self.device)
                    for d in support_domains:
                        batch = self._get_batch(domain_loaders[d])
                        wav = batch["waveform"].squeeze(1).to(self.device)
                        labels = batch["label"].float().to(self.device)
                        logits = fmodel(wav)
                        support_loss = support_loss + self.criterion(logits, labels)
                    support_loss = support_loss / len(support_domains)
                    diffopt.step(support_loss)

                # Query loss on held-out domain
                query_batch = self._get_batch(domain_loaders[query_domain])
                query_wav = query_batch["waveform"].squeeze(1).to(self.device)
                query_labels = query_batch["label"].float().to(self.device)
                query_logits = fmodel(query_wav)
                query_loss = self.criterion(query_logits, query_labels)

                query_loss.backward()

            self.optimizer.step()
            total_loss += query_loss.item()

        return total_loss / max(self.n_episodes_per_epoch, 1)
