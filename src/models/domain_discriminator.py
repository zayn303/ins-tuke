import torch
import torch.nn as nn
from .grl import GradientReversalLayer

class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim: int = 512, n_domains: int = 2, alpha: float = 1.0):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=alpha)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_domains),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        reversed_features = self.grl(features)
        return self.classifier(reversed_features)

    def set_alpha(self, alpha: float) -> None:
        self.grl.set_alpha(alpha)
