import torch
import torch.nn as nn

class PDClassifier(nn.Module):
    def __init__(self, hidden_dim: int = 768, proj_dim: int = 512):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(proj_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.projection(features))

    def get_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)
