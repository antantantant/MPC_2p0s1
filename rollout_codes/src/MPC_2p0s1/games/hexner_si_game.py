# games/hexner_si_game.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from ..config.base_config import GameConfig
from ..core.types import Tensor
from .base_lq_game import BaseLQGame


@dataclass
class HexnerSIParams:
    """
    Parameterization of a position-only (single-integrator) Hexner variant.

    Each player has 2D position state and 2D velocity control:
      x_{j,k+1} = x_{j,k} + tau * u_{j,k}.
    """

    theta_values: Iterable[float] = (-1.0, 1.0)
    target_z: Optional[Iterable[float]] = None

    R1_scale: float = 1.0
    R2_scale: float = 1.0
    K1_scale: float = 1.0
    K2_scale: float = 1.0

    # speed_bound: float = 0.2
    # enforce_speed_bound: bool = True


class HexnerSIGame(BaseLQGame):
    """
    Position-only SI Hexner game for direct Robotarium-style velocity control.
    """

    def __init__(
        self,
        cfg: GameConfig,
        params: Optional[HexnerSIParams] = None,
        prior: Optional[Tensor] = None,
    ) -> None:
        if cfg.dx1 != 2 or cfg.dx2 != 2 or cfg.du != 2 or cfg.dv != 2:
            raise ValueError(
                "HexnerSIGame expects dx1=dx2=2 (2D positions) and du=dv=2 (2D SI velocities). "
                f"Got dx1={cfg.dx1}, dx2={cfg.dx2}, du={cfg.du}, dv={cfg.dv}."
            )
        super().__init__(cfg)

        if params is None:
            params = HexnerSIParams()

        # if params.enforce_speed_bound:
        #     bound = float(params.speed_bound)
        #     if bound <= 0.0:
        #         raise ValueError(f"HexnerSIGame: speed_bound must be positive, got {bound}.")
        #     cfg.u_min = -bound
        #     cfg.u_max = bound
        #     cfg.v_min = -bound
        #     cfg.v_max = bound

        device = cfg.device_resolved
        dtype = cfg.dtype

        theta_vals = torch.as_tensor(list(params.theta_values), device=device, dtype=dtype)
        if theta_vals.numel() != cfg.I:
            raise ValueError(
                f"HexnerSIGame: cfg.I={cfg.I} but got {theta_vals.numel()} theta values."
            )

        if params.target_z is None:
            z = torch.zeros(cfg.dx1, device=device, dtype=dtype)
            z[1] = 1.0
        else:
            z = torch.as_tensor(params.target_z, device=device, dtype=dtype)
            if z.shape != (cfg.dx1,):
                raise ValueError(
                    f"HexnerSIGame: target_z must have shape ({cfg.dx1},), got {tuple(z.shape)}"
                )

        self.register_buffer("theta_vals", theta_vals)
        self.register_buffer("target_z", z)

        tau = cfg.tau
        A_player = torch.eye(2, device=device, dtype=dtype)
        B_player = tau * torch.eye(2, device=device, dtype=dtype)

        A = torch.block_diag(A_player, A_player)  # (4, 4)

        B1 = torch.zeros(cfg.dx, cfg.du, device=device, dtype=dtype)
        B2 = torch.zeros(cfg.dx, cfg.dv, device=device, dtype=dtype)
        B1[: cfg.dx1, :] = B_player
        B2[cfg.dx1 :, :] = B_player

        self.register_buffer("_A", A)
        self.register_buffer("_B1", B1)
        self.register_buffer("_B2", B2)

        R1_base = torch.diag(torch.tensor([0.5, 0.5], device=device, dtype=dtype))
        R2_base = torch.diag(torch.tensor([0.5, 1.5], device=device, dtype=dtype))

        R1 = params.R1_scale * R1_base
        R2 = params.R2_scale * R2_base

        R_stack = torch.stack([2.0 * R1 for _ in range(cfg.I)], dim=0)
        S_stack = torch.stack([2.0 * R2 for _ in range(cfg.I)], dim=0)

        self.register_buffer("_R", R_stack)
        self.register_buffer("_S", S_stack)

        K_base = torch.diag(torch.tensor([1.0, 1.0], device=device, dtype=dtype))
        K1 = params.K1_scale * K_base
        K2 = params.K2_scale * K_base

        Q_block = torch.block_diag(K1, -K2)
        Q_i = 2.0 * Q_block

        zK1z = torch.dot(z, K1 @ z)
        zK2z = torch.dot(z, K2 @ z)

        Q_stack = torch.empty(cfg.I, cfg.dx, cfg.dx, device=device, dtype=dtype)
        q_stack = torch.empty(cfg.I, cfg.dx, device=device, dtype=dtype)
        c_stack = torch.empty(cfg.I, device=device, dtype=dtype)

        for idx, theta in enumerate(theta_vals):
            q1 = -2.0 * theta * (K1.T @ z)
            q2 = 2.0 * theta * (K2.T @ z)
            q_full = torch.cat([q1, q2], dim=0)
            c_i = (theta * theta) * (zK1z - zK2z)

            Q_stack[idx] = Q_i
            q_stack[idx] = q_full
            c_stack[idx] = c_i

        self.register_buffer("_Q", Q_stack)
        self.register_buffer("_q", q_stack)
        self.register_buffer("_c", c_stack)

        if prior is not None:
            if prior.shape != (cfg.I,):
                raise ValueError(
                    f"HexnerSIGame: prior must have shape ({cfg.I},), got {tuple(prior.shape)}"
                )
            p0 = prior.to(device=device, dtype=dtype).clone()
            p0 = p0 / p0.sum()
        else:
            p0 = torch.full((cfg.I,), 1.0 / cfg.I, device=device, dtype=dtype)
        self.register_buffer("_p0_default", p0)

    def default_initial_state(self) -> Tensor:
        x0 = torch.zeros(self.dx, device=self.device_resolved, dtype=self.dtype)
        x0[0] = -1.0
        x0[1] = 0.0
        x0[2] = 1.0
        x0[3] = 0.0
        return x0

    def default_prior(self) -> Tensor:
        return self._p0_default.clone()

    def type_targets(self) -> Tensor:
        return self.theta_vals.view(-1, 1) * self.target_z.view(1, -1)
