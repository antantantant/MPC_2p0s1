from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from ..config.base_config import GameConfig
from ..core.types import Tensor
from .base_lq_game import BaseLQGame


@dataclass
class HexnerMod3DOriginalParams:
    """
    Parameters for the earlier target-based 3D Hexner-mod extension.

    By default, for I=2:
    - target_z = (1, 1, 0, 0, 0, 0), so type targets are (-1, -1, 0, 0, 0, 0)
      and (+1, +1, 0, 0, 0, 0)
    - type-0 uses diag([1, 20, extra_position_scale, terminal_velocity_scale, ...])
    - type-1 uses diag([20, 1, extra_position_scale, terminal_velocity_scale, ...])
    - R2_scale = 0.4 to make informed response pressure stronger
    """

    theta_values: Iterable[float] = (-1.0, 1.0)
    target_z: Optional[Iterable[float]] = None

    R1_scale: float = 1.0
    R2_scale: float = 0.4
    K1_scale: float = 1.0
    K2_scale: float = 1.0

    extra_position_scale: float = 20.0
    terminal_velocity_scale: float = 1.0
    type_k_diags: Optional[Iterable[Iterable[float]]] = None

    default_x0: Optional[Iterable[float]] = None


class HexnerMod3DOriginalGame(BaseLQGame):
    """
    Earlier target-based 3D Hexner-mod game with per-player double-integrator dynamics.

    State and controls:

        x = [x1, x2] in R^12
        x1 = [p1_x, p1_y, p1_z, v1_x, v1_y, v1_z] in R^6
        x2 = [p2_x, p2_y, p2_z, v2_x, v2_y, v2_z] in R^6
        u in R^3  (P1 control)
        v in R^3  (P2 control)

    Terminal cost for type i:

        (x1_T - theta_i z)^T K1_i (x1_T - theta_i z)
      - (x2_T - theta_i z)^T K2_i (x2_T - theta_i z)

    with default type targets on the z=0 landing plane and zero terminal velocity.
    """

    def __init__(
        self,
        cfg: GameConfig,
        params: Optional[HexnerMod3DOriginalParams] = None,
        prior: Optional[Tensor] = None,
    ) -> None:
        if cfg.dx1 != 6 or cfg.dx2 != 6 or cfg.du != 3 or cfg.dv != 3:
            raise ValueError(
                "HexnerMod3DOriginalGame expects dx1=dx2=6 and du=dv=3. "
                f"Got dx1={cfg.dx1}, dx2={cfg.dx2}, du={cfg.du}, dv={cfg.dv}."
            )
        super().__init__(cfg)

        if params is None:
            params = HexnerMod3DOriginalParams()

        device = cfg.device_resolved
        dtype = cfg.dtype

        theta_vals = torch.as_tensor(list(params.theta_values), device=device, dtype=dtype)
        if theta_vals.numel() != cfg.I:
            raise ValueError(
                f"HexnerMod3DOriginalGame: cfg.I={cfg.I} but got {theta_vals.numel()} theta values."
            )

        if params.target_z is None:
            z = torch.zeros(cfg.dx1, device=device, dtype=dtype)
            z[0] = 1.0
            z[1] = 1.0
        else:
            z = torch.as_tensor(params.target_z, device=device, dtype=dtype)
            if z.shape != (cfg.dx1,):
                raise ValueError(
                    "HexnerMod3DOriginalGame: target_z must have shape "
                    f"({cfg.dx1},), got {tuple(z.shape)}"
                )

        self.register_buffer("theta_vals", theta_vals)
        self.register_buffer("target_z", z)

        tau = cfg.tau
        eye3 = torch.eye(3, device=device, dtype=dtype)
        zero3 = torch.zeros(3, 3, device=device, dtype=dtype)

        A_player = torch.cat(
            [
                torch.cat([eye3, tau * eye3], dim=1),
                torch.cat([zero3, eye3], dim=1),
            ],
            dim=0,
        )
        B_player = torch.cat([0.5 * tau * tau * eye3, tau * eye3], dim=0)

        A = torch.block_diag(A_player, A_player)
        B1 = torch.zeros(cfg.dx, cfg.du, device=device, dtype=dtype)
        B2 = torch.zeros(cfg.dx, cfg.dv, device=device, dtype=dtype)
        B1[: cfg.dx1, :] = B_player
        B2[cfg.dx1 :, :] = B_player

        self.register_buffer("_A", A)
        self.register_buffer("_B1", B1)
        self.register_buffer("_B2", B2)

        r1_base = torch.diag(torch.tensor([0.05, 0.025, 0.05], device=device, dtype=dtype))
        r2_base = torch.diag(torch.tensor([0.05, 0.10, 0.05], device=device, dtype=dtype))

        r1 = params.R1_scale * r1_base
        r2 = params.R2_scale * r2_base
        self.register_buffer("_R", torch.stack([2.0 * r1 for _ in range(cfg.I)], dim=0))
        self.register_buffer("_S", torch.stack([2.0 * r2 for _ in range(cfg.I)], dim=0))

        if params.type_k_diags is None:
            if cfg.I != 2:
                raise ValueError(
                    "HexnerMod3DOriginalGame: default type_k_diags are only defined for I=2. "
                    "Please provide params.type_k_diags with one diagonal per type."
                )
            type_k_diags = torch.tensor(
                [
                    [
                        1.0,
                        20.0,
                        float(params.extra_position_scale),
                        float(params.terminal_velocity_scale),
                        float(params.terminal_velocity_scale),
                        float(params.terminal_velocity_scale),
                    ],
                    [
                        20.0,
                        1.0,
                        float(params.extra_position_scale),
                        float(params.terminal_velocity_scale),
                        float(params.terminal_velocity_scale),
                        float(params.terminal_velocity_scale),
                    ],
                ],
                device=device,
                dtype=dtype,
            )
        else:
            type_k_diags = torch.as_tensor(
                list(params.type_k_diags), device=device, dtype=dtype
            )

        if type_k_diags.shape != (cfg.I, cfg.dx1):
            raise ValueError(
                "HexnerMod3DOriginalGame: type_k_diags must have shape "
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
                    f"HexnerMod3DOriginalGame: prior must have shape ({cfg.I},), "
                    f"got {tuple(prior.shape)}"
                )
            p0 = prior.to(device=device, dtype=dtype).clone()
            p0 = p0 / p0.sum()
        else:
            p0 = torch.full((cfg.I,), 1.0 / cfg.I, device=device, dtype=dtype)
        self.register_buffer("_p0_default", p0)

        if params.default_x0 is None:
            x0 = torch.zeros(cfg.dx, device=device, dtype=dtype)
            x0[0] = -1.0
            x0[2] = 1.0
            x0[cfg.dx1 + 0] = 1.0
            x0[cfg.dx1 + 2] = 1.0
        else:
            x0 = torch.as_tensor(list(params.default_x0), device=device, dtype=dtype)
            if x0.shape != (cfg.dx,):
                raise ValueError(
                    f"HexnerMod3DOriginalParams.default_x0 must have shape ({cfg.dx},), "
                    f"got {tuple(x0.shape)}"
                )
        self.register_buffer("_x0_default", x0)

    def default_initial_state(self) -> Tensor:
        return self._x0_default.clone()

    def default_prior(self) -> Tensor:
        return self._p0_default.clone()

    def type_targets(self) -> Tensor:
        return self.theta_vals.view(-1, 1) * self.target_z.view(1, -1)
