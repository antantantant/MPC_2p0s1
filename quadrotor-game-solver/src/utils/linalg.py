"""Small linear-algebra helpers.

Mirrors ``core/tensor_ops.py`` in the parent LQ project.
"""

from __future__ import annotations

import torch
from torch import Tensor


def symmetrize(M: Tensor) -> Tensor:
    """Return (M + M^T) / 2  (last two dims)."""
    return 0.5 * (M + M.transpose(-1, -2))
