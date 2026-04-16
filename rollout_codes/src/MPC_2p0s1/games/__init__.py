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
from .hexner_mod_3d_game import HexnerMod3DGame, HexnerMod3DParams
from .hexner_mod_3d_game_original import (
    HexnerMod3DOriginalGame,
    HexnerMod3DOriginalParams,
)
from .hexner_mod_game import HexnerModGame, HexnerModParams

__all__ = [
    "BaseLQGame",
    "HexnerGame",
    "HexnerParams",
    "HexnerModGame",
    "HexnerModParams",
    "HexnerMod3DGame",
    "HexnerMod3DParams",
    "HexnerMod3DOriginalGame",
    "HexnerMod3DOriginalParams",
]
