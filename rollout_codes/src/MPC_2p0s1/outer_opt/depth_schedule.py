# outer_opt/depth_schedule.py
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..config.base_config import TrainingConfig
from ..tree.indexing import FullIaryTreeIndexer


@dataclass
class DepthSchedule:
    """
    Simple depth-wise schedule for staged optimization of α.

    The schedule controls which depths k ∈ {0, ..., K-1} are “unfrozen”
    (i.e., allowed to update) at a given optimization iteration.

    Parameters
    ----------
    K:
        Number of time steps (tree depth minus one).
    initial_unfrozen_depth:
        Maximum depth unfrozen at step 0 (inclusive). For example,
        0 means only the root logits are trainable initially.
    depth_increment:
        Number of extra depths to unfreeze when the schedule advances.
    depth_increment_every:
        Number of iterations between depth increments.
    max_unfrozen_depth:
        Optional cap on the maximum depth ever unfrozen. If None,
        defaults to K-1 (all depths eventually).
    """

    K: int
    initial_unfrozen_depth: int = 0
    depth_increment: int = 1
    depth_increment_every: int = 200
    max_unfrozen_depth: int | None = None

    def __post_init__(self) -> None:
        if self.max_unfrozen_depth is None:
            self.max_unfrozen_depth = self.K - 1
        self.initial_unfrozen_depth = max(0, min(self.initial_unfrozen_depth, self.K - 1))
        self.max_unfrozen_depth = max(0, min(self.max_unfrozen_depth, self.K - 1))

    def current_unfrozen_depth(self, iteration: int) -> int:
        """
        Compute the maximum unfrozen depth at a given iteration.

        Parameters
        ----------
        iteration:
            0-based outer-loop iteration counter.

        Returns
        -------
        int
            Maximum depth k (inclusive) for which logits are trainable.
        """
        if iteration < 0:
            iteration = 0

        increments = iteration // max(self.depth_increment_every, 1)
        k_max = self.initial_unfrozen_depth + increments * self.depth_increment
        return max(0, min(k_max, self.max_unfrozen_depth))

    def make_mask(
        self,
        logits: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        """
        Create a depth-wise mask tensor for the logits.

        Parameters
        ----------
        logits:
            Tensor of shape (K, max_nodes, I, I), i.e., alpha_module.logits.
        iteration:
            Current optimization iteration.

        Returns
        -------
        torch.Tensor
            Mask tensor of the same shape as logits, with entries in {0, 1}.
            Depths k ≤ current_unfrozen_depth receive mask=1; deeper
            depths receive mask=0.
        """
        if logits.ndim != 4 or logits.shape[0] != self.K:
            raise ValueError(
                f"DepthSchedule.make_mask: expected logits shape (K={self.K}, "
                f"max_nodes, I, I), got {tuple(logits.shape)}"
            )

        cur_max_depth = self.current_unfrozen_depth(iteration)
        mask = torch.zeros_like(logits)

        if cur_max_depth >= 0:
            mask[: cur_max_depth + 1, :, :, :] = 1.0

        return mask


def make_default_depth_schedule(
    train_cfg: TrainingConfig,
    indexer: FullIaryTreeIndexer,
) -> DepthSchedule | None:
    """
    Construct a DepthSchedule from TrainingConfig, or return None if
    staged depth-wise optimization is disabled.

    Parameters
    ----------
    train_cfg:
        Training configuration specifying schedule parameters.
    indexer:
        Tree indexer used to obtain K.

    Returns
    -------
    DepthSchedule or None
        A depth schedule if `train_cfg.staged_depth` is True, otherwise None.
    """
    if not train_cfg.staged_depth:
        return None

    K = indexer.K
    max_unfrozen_depth = (
        train_cfg.max_unfrozen_depth if train_cfg.max_unfrozen_depth is not None else K - 1
    )

    return DepthSchedule(
        K=K,
        initial_unfrozen_depth=train_cfg.initial_unfrozen_depth,
        depth_increment=train_cfg.depth_increment,
        depth_increment_every=train_cfg.depth_increment_every,
        max_unfrozen_depth=max_unfrozen_depth,
    )