"""Core type aliases and small data-containers.

Mirrors ``nl_sqp.types``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch

Tensor: TypeAlias = torch.Tensor


@dataclass(frozen=True)
class ValueQuad:
    """Quadratic value function  V(x) = 0.5 x^T P x + r^T x + c.

    Shapes
    ------
    P : (dx, dx)
    r : (dx,)
    c : ()
    """

    P: Tensor
    r: Tensor
    c: Tensor

    def evaluate(self, x: Tensor) -> Tensor:
        return 0.5 * x @ (self.P @ x) + self.r @ x + self.c
