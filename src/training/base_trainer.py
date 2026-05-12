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
    ):
        self.backbone = backbone.to(device)
        self.classifier = classifier.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            list(backbone.parameters()) + list(classifier.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

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
        torch.save({
            "epoch": epoch,
            "backbone_state": self.backbone.state_dict(),
            "classifier_state": self.classifier.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": metrics,
        }, path)

    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        ckpt = torch.load(Path(path), map_location=self.device)
        self.backbone.load_state_dict(ckpt["backbone_state"])
        self.classifier.load_state_dict(ckpt["classifier_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        return ckpt
