# config/__init__.py
"""
Configuration utilities for the 2p0s1 LQ differential game solver.

This subpackage provides:
- Generic dataclasses for game and training configuration.
- Predefined configurations for benchmark environments such as Hexner’s game.
"""

from .base_config import GameConfig, TrainingConfig, PathsConfig, resolve_device
from .hexner_config import (
    make_hexner_game_config,
    make_hexner_training_config,
    make_hexner_paths_config,
)

__all__ = [
    "GameConfig",
    "TrainingConfig",
    "PathsConfig",
    "resolve_device",
    "make_hexner_game_config",
    "make_hexner_training_config",
    "make_hexner_paths_config",
]