# __init__.py
"""
Top-level package for solving 2p0s1 linear–quadratic differential games with one-sided
payoff information using tree-structured Riccati recursions and gradient-based
optimization over belief-splitting parameters.

The implementation follows the primal/dual and atomic-equilibrium structure described
in “Solving Football by Exploiting Equilibrium Structure of 2P0S Differential Games
with One-Sided Information”.  [oai_citation:0‡16010_Solving_Football_by_Expl-2.pdf](sediment://file_000000001b0871f58cd1c91b483083b4)
"""

from .config.base_config import GameConfig, TrainingConfig, PathsConfig
from .config.hexner_config import (
    make_hexner_game_config,
    make_hexner_training_config,
    make_hexner_paths_config,
)

__all__ = [
    "GameConfig",
    "TrainingConfig",
    "PathsConfig",
    "make_hexner_game_config",
    "make_hexner_training_config",
    "make_hexner_paths_config",
]

__version__ = "0.1.0"