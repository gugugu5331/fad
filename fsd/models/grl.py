"""Gradient reversal layer (GRL)."""

from __future__ import annotations

import torch
from torch import nn


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):  # type: ignore[override]
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


class GradientReversal(nn.Module):
    def __init__(self, lambd: float = 1.0) -> None:
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return grad_reverse(x, self.lambd)
