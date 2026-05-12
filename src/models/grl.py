import torch
import torch.nn as nn

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(alpha)
        return x.clone()

    @staticmethod
    def backward(ctx, grads: torch.Tensor):
        alpha, = ctx.saved_tensors
        return -alpha * grads, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float) -> None:
        self.alpha.fill_(alpha)
