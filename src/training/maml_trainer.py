import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import higher
from .base_trainer import BaseTrainer


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
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__(backbone, classifier, lr, weight_decay, device)
        self.inner_lr = inner_lr
        self.n_inner_steps = n_inner_steps
        self.n_episodes_per_epoch = n_episodes_per_epoch
        pw = pos_weight.to(device) if pos_weight is not None else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    @staticmethod
    def _cyclic(loader: DataLoader):
        while True:
            yield from loader

    def train_epoch(self, domain_loaders: Dict[int, DataLoader], **kwargs) -> float:
        self.backbone.train()
        self.classifier.train()

        domain_ids = list(domain_loaders.keys())
        domain_iters = {d: self._cyclic(loader) for d, loader in domain_loaders.items()}
        # Only wrap classifier — backbone is frozen, patching it with higher
        # causes 38+ GB VRAM usage from the unrolled computation graph.
        inner_opt = torch.optim.SGD(self.classifier.parameters(), lr=self.inner_lr)

        total_loss = 0.0

        for _ in range(self.n_episodes_per_epoch):
            random.shuffle(domain_ids)
            query_domain = domain_ids[-1]
            support_domains = domain_ids[:-1]

            self.optimizer.zero_grad()

            with higher.innerloop_ctx(
                self.classifier, inner_opt, copy_initial_weights=False
            ) as (fclassifier, diffopt):

                for _ in range(self.n_inner_steps):
                    support_loss = torch.tensor(0.0, device=self.device)
                    for d in support_domains:
                        batch = next(domain_iters[d])
                        wav = batch["waveform"].squeeze(1).to(self.device)
                        labels = batch["label"].float().to(self.device)
                        with torch.no_grad():
                            features = self.backbone(wav)
                        logits = fclassifier(features).squeeze(-1)
                        support_loss = support_loss + self.criterion(logits, labels)
                    support_loss = support_loss / len(support_domains)
                    diffopt.step(support_loss)

                query_batch = next(domain_iters[query_domain])
                query_wav = query_batch["waveform"].squeeze(1).to(self.device)
                query_labels = query_batch["label"].float().to(self.device)
                with torch.no_grad():
                    query_features = self.backbone(query_wav)
                query_logits = fclassifier(query_features).squeeze(-1)
                query_loss = self.criterion(query_logits, query_labels)

                query_loss.backward()

            self.optimizer.step()
            total_loss += query_loss.item()

        mean_loss = total_loss / max(self.n_episodes_per_epoch, 1)
        self.last_stats = {"query_loss": mean_loss}
        return mean_loss
