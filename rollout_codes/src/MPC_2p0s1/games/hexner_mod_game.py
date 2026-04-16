from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from ..config.base_config import GameConfig
from ..core.types import Tensor
from .base_lq_game import BaseLQGame


@dataclass
class HexnerModParams:
    """
    Parameters for a Hexner game variant with stronger type/state sensitivity.

    By default, for I=2:
    - target_z = (1, 1, 0, 0), so type targets are (-1, -1) and (+1, +1)
    - type-1 uses diag([1, 20, 0, 0])  (y penalized much more than x)
    - type-2 uses diag([20, 1, 0, 0])  (x penalized much more than y)
    - R2_scale = 0.4 to make informed response pressure stronger
    """

    theta_values: Iterable[float] = (-1.0, 1.0)
    target_z: Optional[Iterable[float]] = None

    R1_scale: float = 1.0
    R2_scale: float = 0.4
    K1_scale: float = 1.0
    K2_scale: float = 1.0

    # One diagonal vector per type, each length dx1.
    # Example for I=2, dx1=4:
    # ((1, 20, 0, 0), (20, 1, 0, 0))
    type_k_diags: Optional[Iterable[Iterable[float]]] = None


class HexnerModGame(BaseLQGame):
    """
    Discrete-time Hexner game with type-dependent terminal penalties.
    """

    def __init__(
        self,
        cfg: GameConfig,
        params: Optional[HexnerModParams] = None,
        prior: Optional[Tensor] = None,
    ) -> None:
        if cfg.dx1 != 4 or cfg.dx2 != 4 or cfg.du != 2 or cfg.dv != 2:
            raise ValueError(
                "HexnerModGame expects dx1=dx2=4 (2D pos+vel) and du=dv=2 (2D accel). "
                f"Got dx1={cfg.dx1}, dx2={cfg.dx2}, du={cfg.du}, dv={cfg.dv}."
            )
        super().__init__(cfg)

        if params is None:
            params = HexnerModParams()

        device = cfg.device_resolved
        dtype = cfg.dtype

        theta_vals = torch.as_tensor(list(params.theta_values), device=device, dtype=dtype)
        if theta_vals.numel() != cfg.I:
            raise ValueError(
                f"HexnerModGame: cfg.I={cfg.I} but got {theta_vals.numel()} theta values."
            )

        if params.target_z is None:
            z = torch.zeros(cfg.dx1, device=device, dtype=dtype)
            # Default extreme setting: diagonal target direction in position space.
            z[0] = 1.0
            z[1] = 1.0
        else:
            z = torch.as_tensor(params.target_z, device=device, dtype=dtype)
            if z.shape != (cfg.dx1,):
                raise ValueError(
                    f"HexnerModGame: target_z must have shape ({cfg.dx1},), got {tuple(z.shape)}"
                )

        self.register_buffer("theta_vals", theta_vals)
        self.register_buffer("target_z", z)

        tau = cfg.tau
        A_player = torch.tensor(
            [
                [1.0, 0.0, tau, 0.0],
                [0.0, 1.0, 0.0, tau],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )
        B_player = torch.tensor(
            [
                [0.5 * tau * tau, 0.0],
                [0.0, 0.5 * tau * tau],
                [tau, 0.0],
                [0.0, tau],
            ],
            device=device,
            dtype=dtype,
        )

        A = torch.block_diag(A_player, A_player)

        B1 = torch.zeros(cfg.dx, cfg.du, device=device, dtype=dtype)
        B2 = torch.zeros(cfg.dx, cfg.dv, device=device, dtype=dtype)
        B1[: cfg.dx1, :] = B_player
        B2[cfg.dx1 :, :] = B_player

        self.register_buffer("_A", A)
        self.register_buffer("_B1", B1)
        self.register_buffer("_B2", B2)

        R1_base = torch.diag(torch.tensor([0.05, 0.025], device=device, dtype=dtype))
        R2_base = torch.diag(torch.tensor([0.05, 0.10], device=device, dtype=dtype))

        R1 = params.R1_scale * R1_base
        R2 = params.R2_scale * R2_base

        R_stack = torch.stack([2.0 * R1 for _ in range(cfg.I)], dim=0)
        S_stack = torch.stack([2.0 * R2 for _ in range(cfg.I)], dim=0)

        self.register_buffer("_R", R_stack)
        self.register_buffer("_S", S_stack)

        if params.type_k_diags is None:
            if cfg.I != 2:
                raise ValueError(
                    "HexnerModGame: default type_k_diags are only defined for I=2. "
                    "Please provide params.type_k_diags with one diagonal per type."
                )
            type_k_diags = torch.tensor(
                [[1.0, 20.0, 0.0, 0.0], [20.0, 1.0, 0.0, 0.0]],
                device=device,
                dtype=dtype,
            )
        else:
            type_k_diags = torch.as_tensor(
                list(params.type_k_diags), device=device, dtype=dtype
            )

        if type_k_diags.shape != (cfg.I, cfg.dx1):
            raise ValueError(
                "HexnerModGame: type_k_diags must have shape "
                f"({cfg.I}, {cfg.dx1}), got {tuple(type_k_diags.shape)}"
            )

        K1_stack = params.K1_scale * torch.diag_embed(type_k_diags)
        K2_stack = params.K2_scale * torch.diag_embed(type_k_diags)
        self.register_buffer("_K1_type", K1_stack)
        self.register_buffer("_K2_type", K2_stack)

        Q_stack = torch.empty(cfg.I, cfg.dx, cfg.dx, device=device, dtype=dtype)
        q_stack = torch.empty(cfg.I, cfg.dx, device=device, dtype=dtype)
        c_stack = torch.empty(cfg.I, device=device, dtype=dtype)

        for idx, theta in enumerate(theta_vals):
            K1_i = K1_stack[idx]
            K2_i = K2_stack[idx]

            Q_block = torch.block_diag(K1_i, -K2_i)
            Q_stack[idx] = 2.0 * Q_block

            q1 = -2.0 * theta * (K1_i.T @ z)
            q2 = 2.0 * theta * (K2_i.T @ z)
            q_stack[idx] = torch.cat([q1, q2], dim=0)

            zK1z = torch.dot(z, K1_i @ z)
            zK2z = torch.dot(z, K2_i @ z)
            c_stack[idx] = (theta * theta) * (zK1z - zK2z)

        self.register_buffer("_Q", Q_stack)
        self.register_buffer("_q", q_stack)
        self.register_buffer("_c", c_stack)

        if prior is not None:
            if prior.shape != (cfg.I,):
                raise ValueError(
                    f"HexnerModGame: prior must have shape ({cfg.I},), got {tuple(prior.shape)}"
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
        x0[2] = 0.0
        x0[3] = 0.0
        x0[4] = 1.0
        x0[5] = 0.0
        x0[6] = 0.0
        x0[7] = 0.0
        return x0

    def default_prior(self) -> Tensor:
        return self._p0_default.clone()

    def type_targets(self) -> Tensor:
        return self.theta_vals.view(-1, 1) * self.target_z.view(1, -1)
