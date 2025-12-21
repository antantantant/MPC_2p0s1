# outer_opt/checkpointing.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from ..config.base_config import GameConfig, TrainingConfig, PathsConfig
from .alpha_param import AlphaParam


@dataclass
class CheckpointMeta:
    """
    Metadata stored alongside model and optimizer state.

    Attributes
    ----------
    step:
        Outer-loop iteration index at which the checkpoint was saved.
    loss:
        Scalar loss value at that iteration (P1's value).
    game_config:
        Plain-Python dictionary describing the GameConfig used.
    training_config:
        Plain-Python dictionary describing the TrainingConfig used.
    """

    step: int
    loss: float
    game_config: Dict[str, Any]
    training_config: Dict[str, Any]


def _checkpoint_path(paths_cfg: PathsConfig, step: int) -> str:
    return os.path.join(paths_cfg.run_dir, f"checkpoint_step_{step:06d}.pt")


def save_checkpoint(
    paths_cfg: PathsConfig,
    step: int,
    loss: float,
    alpha_module: AlphaParam,
    optimizer: torch.optim.Optimizer,
    game_cfg: GameConfig,
    train_cfg: TrainingConfig,
) -> str:
    """
    Save a checkpoint containing α/logits, optimizer state, and metadata.

    Parameters
    ----------
    paths_cfg:
        PathsConfig describing where to save.
    step:
        Current optimization iteration.
    loss:
        Current scalar loss (float).
    alpha_module:
        AlphaParam module whose state_dict will be saved.
    optimizer:
        Optimizer whose state_dict will be saved.
    game_cfg:
        GameConfig used to construct the game instance.
    train_cfg:
        TrainingConfig used for this run.

    Returns
    -------
    str
        Path to the saved checkpoint file.
    """
    paths_cfg.ensure_exists()
    ckpt_path = _checkpoint_path(paths_cfg, step)

    meta = CheckpointMeta(
        step=step,
        loss=float(loss),
        game_config=game_cfg.as_dict(),
        training_config=train_cfg.as_dict(),
    )

    payload = {
        "meta": meta.__dict__,
        "alpha_state_dict": alpha_module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(payload, ckpt_path)

    # Also write/overwrite a "latest.pt" symlink or file for convenience.
    latest_path = os.path.join(paths_cfg.run_dir, "latest.pt")
    try:
        # Overwrite regular file or symlink
        if os.path.islink(latest_path) or os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(os.path.basename(ckpt_path), latest_path)
    except OSError:
        # Symlinks may not be available on all platforms; fall back to copy.
        torch.save(payload, latest_path)

    return ckpt_path


def load_checkpoint(
    ckpt_path: str,
    alpha_module: AlphaParam,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> Tuple[CheckpointMeta, Optional[torch.optim.Optimizer]]:
    """
    Load a checkpoint and restore α/logits (and optionally optimizer).

    Parameters
    ----------
    ckpt_path:
        Path to the checkpoint file.
    alpha_module:
        AlphaParam instance into which the α state_dict will be loaded.
    optimizer:
        Optional optimizer instance into which the optimizer state_dict
        will be loaded. If None, only α will be restored.
    map_location:
        Argument forwarded to torch.load for device mapping.

    Returns
    -------
    (meta, optimizer)
        - meta: CheckpointMeta with step, loss, and config dicts.
        - optimizer: the same optimizer instance passed in, after state
          restoration (or None if no optimizer was provided).
    """
    payload = torch.load(ckpt_path, map_location=map_location)

    meta_dict = payload.get("meta", {})
    meta = CheckpointMeta(
        step=int(meta_dict.get("step", 0)),
        loss=float(meta_dict.get("loss", 0.0)),
        game_config=meta_dict.get("game_config", {}),
        training_config=meta_dict.get("training_config", {}),
    )

    alpha_state = payload.get("alpha_state_dict", {})
    alpha_module.load_state_dict(alpha_state)

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    return meta, optimizer