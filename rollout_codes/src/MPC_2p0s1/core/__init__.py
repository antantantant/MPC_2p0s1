# core/__init__.py
"""
Core utilities for the 2p0s1 LQ differential game solver.

This subpackage provides:
- Basic typed aliases and small data structures (e.g., quadratic value representation).
- General utilities for seeding, timing, and debugging.
- Simple box-constrained action-space helpers.
- Small tensor utilities for batched linear algebra.

These pieces are intentionally lightweight and independent of any particular game
instance so they can be reused across the primal/dual solvers, Hexner’s game, and
larger applications such as the football case study.  [oai_citation:0‡16010_Solving_Football_by_Expl-2.pdf](sediment://file_000000001b0871f58cd1c91b483083b4)
"""

from .types import Tensor, NodeId, EdgeId, ValueQuad
from .utils import (
    set_random_seeds,
    Timer,
    count_parameters,
)
from .action_spaces import BoxActionSpace
from .tensor_ops import (
    batch_solve,
    symmetrize,
    ensure_posdef,
)

__all__ = [
    "Tensor",
    "NodeId",
    "EdgeId",
    "ValueQuad",
    "set_random_seeds",
    "Timer",
    "count_parameters",
    "BoxActionSpace",
    "batch_solve",
    "symmetrize",
    "ensure_posdef",
]