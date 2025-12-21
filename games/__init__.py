# games/__init__.py
"""
Game definitions for the 2p0s1 LQ differential game solver.

This subpackage provides:
- An abstract base class for linear–quadratic two-player zero-sum games with
  one-sided payoff information.
- A concrete implementation of Hexner’s game, which serves as a minimal but
  nontrivial benchmark with an analytical NE and a clear “conceal then reveal”
  structure.  [oai_citation:0‡16010_Solving_Football_by_Expl-2.pdf](sediment://file_000000001b0871f58cd1c91b483083b4)
"""

from .base_lq_game import BaseLQGame
from .hexner_game import HexnerGame, HexnerParams

__all__ = [
    "BaseLQGame",
    "HexnerGame",
    "HexnerParams",
]