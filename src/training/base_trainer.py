from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class BaseTrainer(ABC):
    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        lr: float,
        weight_decay: float,
        device: torch.device,
        lr_schedule: str = "none",
        lr_warmup_epochs: int = 2,
        lr_min: float = 1e-6,
        total_epochs: int = 40,
    ):
        self.backbone = backbone.to(device)
        self.classifier = classifier.to(device)
        self.device = device
        self.last_stats: Dict[str, Any] = {}

        trainable = [p for p in backbone.parameters() if p.requires_grad] + list(classifier.parameters())
        self.optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)

        if lr_schedule == "cosine":
            start_factor = max(lr_min / max(lr, 1e-10), 1e-10)
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=start_factor,
                end_factor=1.0,
                total_iters=lr_warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(total_epochs - lr_warmup_epochs, 1),
                eta_min=lr_min,
            )
            self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = (
                torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[lr_warmup_epochs],
                )
            )
        else:
            self.scheduler = None

    @abstractmethod
    def train_epoch(self, loader, **kwargs) -> float:
        """Run one training epoch. Returns mean loss."""
        pass

    def eval_epoch(self, loader) -> Dict[str, float]:
        from src.evaluation.evaluator import run_eval
        return run_eval(self.backbone, self.classifier, loader, self.device)

    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "backbone_state": self.backbone.state_dict(),
            "classifier_state": self.classifier.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": metrics,
        }
        if self.scheduler is not None:
            ckpt["scheduler_state"] = self.scheduler.state_dict()
        torch.save(ckpt, path)

    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        ckpt = torch.load(Path(path), map_location=self.device)
        self.backbone.load_state_dict(ckpt["backbone_state"])
        self.classifier.load_state_dict(ckpt["classifier_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if self.scheduler is not None and "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        return ckpt
