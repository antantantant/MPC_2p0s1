# core/tensor_ops.py
from __future__ import annotations

from typing import Tuple

import torch

from .types import Tensor


def batch_solve(A: Tensor, B: Tensor) -> Tensor:
    """
    Solve a batch of linear systems A x = B.

    Parameters
    ----------
    A:
        Tensor of shape (..., n, n) representing a batch of square matrices.
    B:
        Tensor of shape (..., n) or (..., n, k) representing right-hand sides.

    Returns
    -------
    Tensor
        Solution tensor with the same batch shape as B.

    Notes
    -----
    This is a thin wrapper around `torch.linalg.solve` that enforces a
    consistent API and handles the common vector RHS case.
    """
    if A.shape[-1] != A.shape[-2]:
        raise ValueError(f"batch_solve: A must be square on last two dims, got {A.shape}")
    if A.shape[:-2] != B.shape[:-1] and A.shape[:-2] != B.shape[:-2]:
        raise ValueError(
            f"batch_solve: batch shapes of A {A.shape[:-2]} and B {B.shape[:-1]} "
            "are not compatible"
        )

    # If B is (..., n), unsqueeze to (..., n, 1) and squeeze after solve
    vector_rhs = False
    if B.ndim >= 1 and B.shape[-1] == A.shape[-1] and (B.ndim == A.ndim - 1):
        vector_rhs = True
        B_expanded = B.unsqueeze(-1)
    else:
        B_expanded = B

    X = torch.linalg.solve(A, B_expanded)
    if vector_rhs:
        X = X.squeeze(-1)
    return X


def symmetrize(M: Tensor) -> Tensor:
    """
    Symmetrize a matrix or batch of matrices: (M + M^T) / 2.

    Parameters
    ----------
    M:
        Tensor of shape (..., n, n).

    Returns
    -------
    Tensor
        Symmetric tensor of the same shape as M.
    """
    return 0.5 * (M + M.transpose(-1, -2))


def ensure_posdef(M: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Make a (batch of) symmetric matrices numerically positive definite.

    Parameters
    ----------
    M:
        Tensor of shape (..., n, n). Typically already symmetric.
    eps:
        Diagonal jitter added to improve conditioning: M + eps * I.

    Returns
    -------
    Tensor
        Adjusted tensor that is safer to use with Cholesky or solve.

    Notes
    -----
    This function does not *guarantee* positive definiteness in a strict
    mathematical sense, but in practice it is often sufficient to handle
    numerical issues in Riccati recursions and local LQ solves.
    """
    n = M.shape[-1]
    eye = torch.eye(n, device=M.device, dtype=M.dtype)
    return symmetrize(M) + eps * eye