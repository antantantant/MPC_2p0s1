# core/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

Tensor = torch.Tensor
NodeId = int
EdgeId = int


@dataclass
class ValueQuad:
    """
    Quadratic value representation for a node in the public game tree.

    We use the convention
        V(x) = 0.5 * x^T P x + r^T x + c

    where:
        - P is a (dx × dx) symmetric matrix,
        - r is a (dx,) vector,
        - c is a scalar.

    This structure is used by the tree-structured Riccati recursion to represent
    node values compactly for any x ∈ R^dx.
    """

    P: Tensor  # (dx, dx)
    r: Tensor  # (dx,)
    c: Tensor  # ()

    def evaluate(self, x: Tensor) -> Tensor:
        """
        Evaluate the quadratic value at one or more states.

        Parameters
        ----------
        x:
            State tensor of shape (..., dx).

        Returns
        -------
        Tensor
            Value tensor of shape (...,).
        """
        # Ensure last dimension matches
        if x.shape[-1] != self.P.shape[-1]:
            raise ValueError(
                f"Incompatible shapes: x has last dim {x.shape[-1]}, "
                f"P is {self.P.shape}"
            )
        # 0.5 x^T P x
        Px = torch.matmul(x, self.P)  # (..., dx)
        quad = 0.5 * (Px * x).sum(dim=-1)  # (...)
        # r^T x
        lin = torch.matmul(x, self.r)  # (...)
        return quad + lin + self.c

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "ValueQuad":
        """
        Move this ValueQuad to a different device and/or dtype.

        Parameters
        ----------
        device:
            Target device or device string (e.g. "cpu", "cuda").
        dtype:
            Target dtype, e.g. torch.float32.

        Returns
        -------
        ValueQuad
            A new ValueQuad on the requested device/dtype.
        """
        P = self.P
        r = self.r
        c = self.c
        if device is not None or dtype is not None:
            P = P.to(device=device, dtype=dtype if dtype is not None else P.dtype)
            r = r.to(device=device, dtype=dtype if dtype is not None else r.dtype)
            c = c.to(device=device, dtype=dtype if dtype is not None else c.dtype)
        return ValueQuad(P=P, r=r, c=c)

    @classmethod
    def zeros(
        cls,
        dx: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "ValueQuad":
        """
        Construct a zero ValueQuad in R^dx.

        Parameters
        ----------
        dx:
            State dimension.
        device:
            Target device; if None, uses PyTorch default.
        dtype:
            Tensor dtype (default: float32).

        Returns
        -------
        ValueQuad
            V(x) ≡ 0 for all x.
        """
        P = torch.zeros(dx, dx, device=device, dtype=dtype)
        r = torch.zeros(dx, device=device, dtype=dtype)
        c = torch.zeros((), device=device, dtype=dtype)
        return cls(P=P, r=r, c=c)


@runtime_checkable
class HasTo(Protocol):
    """
    Small protocol capturing `.to(device, dtype)` behavior.

    This is useful for generic helper functions that should work with both
    torch.Tensor and lightweight containers like ValueQuad that support a
    `.to(...)` method.
    """

    def to(self, *args, **kwargs):  # pragma: no cover - protocol stub
        ...