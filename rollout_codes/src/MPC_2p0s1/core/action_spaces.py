# core/action_spaces.py
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..config.base_config import GameConfig
from .types import Tensor


@dataclass
class BoxActionSpace:
    """
    Simple box-constrained action spaces for both players.

    This class centralizes the treatment of control bounds so that the rest of the
    solver (Riccati recursion, belief tree, etc.) can work with unconstrained
    action prototypes and then optionally clamp them into a feasible range.

    We intentionally keep this minimal; if you later need non-rectangular constraints
    or different bounds per time step, those extensions can be added here without
    touching the rest of the codebase.
    """

    u_min: Tensor  # (du,)
    u_max: Tensor  # (du,)
    v_min: Tensor  # (dv,)
    v_max: Tensor  # (dv,)

    @classmethod
    def from_config(cls, cfg: GameConfig) -> "BoxActionSpace":
        """
        Build a BoxActionSpace from a GameConfig.

        Scalar bounds in GameConfig are expanded to vectors of appropriate
        dimension; for more complex setups, you can construct this class
        directly with per-dimension tensors.
        """
        device = cfg.device_resolved
        dtype = cfg.dtype

        u_min = torch.full((cfg.du,), cfg.u_min, device=device, dtype=dtype)
        u_max = torch.full((cfg.du,), cfg.u_max, device=device, dtype=dtype)
        v_min = torch.full((cfg.dv,), cfg.v_min, device=device, dtype=dtype)
        v_max = torch.full((cfg.dv,), cfg.v_max, device=device, dtype=dtype)

        return cls(u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max)

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "BoxActionSpace":
        """
        Move bounds to a target device and/or dtype.

        Returns a new BoxActionSpace; the original is unchanged.
        """
        if device is None and dtype is None:
            return self

        u_min = self.u_min.to(device=device, dtype=dtype or self.u_min.dtype)
        u_max = self.u_max.to(device=device, dtype=dtype or self.u_max.dtype)
        v_min = self.v_min.to(device=device, dtype=dtype or self.v_min.dtype)
        v_max = self.v_max.to(device=device, dtype=dtype or self.v_max.dtype)
        return BoxActionSpace(u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max)

    # --- P1 ----------------------------------------------------------------- #

    def clip_u(self, u: Tensor) -> Tensor:
        """
        Clip Player 1's action into the feasible box.

        Parameters
        ----------
        u:
            Tensor of shape (..., du) representing (possibly unconstrained)
            actions for Player 1.

        Returns
        -------
        Tensor
            Clipped actions with the same shape as `u`.
        """
        if u.shape[-1] != self.u_min.shape[-1]:
            raise ValueError(
                f"clip_u: expected last dim {self.u_min.shape[-1]}, got {u.shape[-1]}"
            )
        return torch.max(torch.min(u, self.u_max), self.u_min)

    def sample_u(self, batch_shape: tuple[int, ...] = ()) -> Tensor:
        """
        Sample random P1 actions uniformly from the box.

        Useful for debugging and stochastic initializations.
        """
        shape = batch_shape + self.u_min.shape
        rand = torch.rand(shape, device=self.u_min.device, dtype=self.u_min.dtype)
        return self.u_min + rand * (self.u_max - self.u_min)

    # --- P2 ----------------------------------------------------------------- #

    def clip_v(self, v: Tensor) -> Tensor:
        """
        Clip Player 2's action into the feasible box.

        Parameters
        ----------
        v:
            Tensor of shape (..., dv) representing (possibly unconstrained)
            actions for Player 2.

        Returns
        -------
        Tensor
            Clipped actions with the same shape as `v`.
        """
        if v.shape[-1] != self.v_min.shape[-1]:
            raise ValueError(
                f"clip_v: expected last dim {self.v_min.shape[-1]}, got {v.shape[-1]}"
            )
        return torch.max(torch.min(v, self.v_max), self.v_min)

    def sample_v(self, batch_shape: tuple[int, ...] = ()) -> Tensor:
        """
        Sample random P2 actions uniformly from the box.

        Useful for debugging and stochastic initializations.
        """
        shape = batch_shape + self.v_min.shape
        rand = torch.rand(shape, device=self.v_min.device, dtype=self.v_min.dtype)
        return self.v_min + rand * (self.v_max - self.v_min)