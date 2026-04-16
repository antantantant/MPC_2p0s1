from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from ..config.base_config import GameConfig
from ..core.types import Tensor
from .base_lq_game import BaseLQGame


_DEFAULT_TERMINAL_K = (
    (19.67, -6.81, 5.84, 0.0, 4.20, -6.60),
    (-6.81, 7.93, -6.40, -4.20, 0.0, 0.0),
    (5.84, -6.40, 14.97, 6.60, 0.0, 0.0),
    (0.0, -4.20, 6.60, 12.49, 0.0, 0.0),
    (4.20, 0.0, 0.0, 0.0, 12.49, 0.0),
    (-6.60, 0.0, 0.0, 0.0, 0.0, 13.69),
)


@dataclass
class HexnerMod3DParams:
    """
    Parameters for a two-player 3D zero-sum LQ landing game.

    The current solver supports control running costs and a terminal quadratic
    state penalty. We therefore encode the proposed coupled matrix K as a
    type-dependent terminal cost

        g_i(x_T) = (x_{1,T} - z_i)^T K (x_{1,T} - z_i)
                 - (x_{2,T} - z_i)^T K (x_{2,T} - z_i)

    via BaseLQGame's convention

        g_i(x_T) = 0.5 x_T^T Q_i x_T + q_i^T x_T + c_i.

    The default type targets are diagonal x/y targets with shared z=0 and
    zero terminal velocity:

        z_i = theta_i * [1, 1, 0, 0, 0, 0].
    """

    theta_values: Iterable[float] = (-1.0, 1.0)
    target_state: Optional[Iterable[float]] = None
    R_diag: Iterable[float] = (1.30, 0.55, 1.80)
    S_diag: Iterable[float] = (3.60, 2.40, 3.00)
    terminal_K: Optional[Iterable[Iterable[float]]] = None
    terminal_scale: float = 1.0
    default_x0: Optional[Iterable[float]] = None


class HexnerMod3DGame(BaseLQGame):
    """
    Two-player 3D landing game with per-player double-integrator dynamics.

    State and controls:

        x = [x1, x2] in R^12
        x1 = [p1_x, p1_y, p1_z, v1_x, v1_y, v1_z] in R^6
        x2 = [p2_x, p2_y, p2_z, v2_x, v2_y, v2_z] in R^6
        u in R^3  (P1 control)
        v in R^3  (P2 control)

    Dynamics:

        x1_{k+1} = A_player x1_k + B_player u_k
        x2_{k+1} = A_player x2_k + B_player v_k

    Running cost:

        0.5 u^T R u - 0.5 v^T S v

    Terminal cost:

        (x1_T - z_i)^T K (x1_T - z_i) - (x2_T - z_i)^T K (x2_T - z_i)

    The original request described a hard terminal constraint x(T)=0. The
    current tree Riccati solver does not support hard terminal equality
    constraints directly, so this implementation uses the coupled quadratic K
    as a terminal penalty instead.
    """

    def __init__(
        self,
        cfg: GameConfig,
        params: Optional[HexnerMod3DParams] = None,
        prior: Optional[Tensor] = None,
    ) -> None:
        if cfg.dx1 != 6 or cfg.dx2 != 6 or cfg.du != 3 or cfg.dv != 3:
            raise ValueError(
                "HexnerMod3DGame expects dx1=dx2=6 and du=dv=3. "
                f"Got dx1={cfg.dx1}, dx2={cfg.dx2}, du={cfg.du}, dv={cfg.dv}."
            )
        super().__init__(cfg)

        if params is None:
            params = HexnerMod3DParams()

        device = cfg.device_resolved
        dtype = cfg.dtype

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

        theta_vals = torch.as_tensor(list(params.theta_values), device=device, dtype=dtype)
        if theta_vals.numel() != cfg.I:
            raise ValueError(
                f"HexnerMod3DGame: cfg.I={cfg.I} but got {theta_vals.numel()} theta values."
            )

        if params.target_state is None:
            target_state = torch.zeros(cfg.dx1, device=device, dtype=dtype)
            target_state[0] = 1.0
            target_state[1] = 1.0
        else:
            target_state = torch.as_tensor(params.target_state, device=device, dtype=dtype)
            if target_state.shape != (cfg.dx1,):
                raise ValueError(
                    f"HexnerMod3DParams.target_state must have shape ({cfg.dx1},), "
                    f"got {tuple(target_state.shape)}"
                )

        self.register_buffer("theta_vals", theta_vals)
        self.register_buffer("target_state", target_state)

        R_diag = torch.as_tensor(list(params.R_diag), device=device, dtype=dtype)
        S_diag = torch.as_tensor(list(params.S_diag), device=device, dtype=dtype)
        if R_diag.shape != (cfg.du,):
            raise ValueError(
                f"HexnerMod3DParams.R_diag must have shape ({cfg.du},), got {tuple(R_diag.shape)}"
            )
        if S_diag.shape != (cfg.dv,):
            raise ValueError(
                f"HexnerMod3DParams.S_diag must have shape ({cfg.dv},), got {tuple(S_diag.shape)}"
            )

        R = torch.diag(R_diag)
        S = torch.diag(S_diag)
        self.register_buffer("_R", R.unsqueeze(0).repeat(cfg.I, 1, 1))
        self.register_buffer("_S", S.unsqueeze(0).repeat(cfg.I, 1, 1))

        if params.terminal_K is None:
            K = torch.tensor(_DEFAULT_TERMINAL_K, device=device, dtype=dtype)
        else:
            K = torch.as_tensor(list(params.terminal_K), device=device, dtype=dtype)
        if K.shape != (cfg.dx1, cfg.dx1):
            raise ValueError(
                f"HexnerMod3DParams.terminal_K must have shape ({cfg.dx1}, {cfg.dx1}), "
                f"got {tuple(K.shape)}"
            )

        K = 0.5 * (K + K.T)
        eigvals = torch.linalg.eigvalsh(K)
        if torch.any(eigvals < -1e-7):
            raise ValueError(
                "HexnerMod3DGame expects terminal_K to be positive semidefinite. "
                f"Minimum eigenvalue was {float(eigvals.min().item()):.6f}."
            )

        K_term = float(params.terminal_scale) * K
        self.register_buffer("_terminal_K", K_term)
        type_targets = theta_vals.view(-1, 1) * target_state.view(1, -1)
        self.register_buffer("_type_targets", type_targets)

        Q_block = torch.block_diag(K_term, -K_term)
        Q_stack = (2.0 * Q_block).unsqueeze(0).repeat(cfg.I, 1, 1)
        q_stack = torch.empty(cfg.I, cfg.dx, device=device, dtype=dtype)
        c_stack = torch.empty(cfg.I, device=device, dtype=dtype)
        for idx in range(cfg.I):
            z_i = type_targets[idx]
            q1 = -2.0 * (K_term @ z_i)
            q2 = 2.0 * (K_term @ z_i)
            q_stack[idx] = torch.cat([q1, q2], dim=0)
            c_stack[idx] = torch.dot(z_i, K_term @ z_i) - torch.dot(z_i, K_term @ z_i)

        self.register_buffer("_Q", Q_stack)
        self.register_buffer("_q", q_stack)
        self.register_buffer("_c", c_stack)

        if prior is not None:
            if prior.shape != (cfg.I,):
                raise ValueError(
                    f"HexnerMod3DGame: prior must have shape ({cfg.I},), got {tuple(prior.shape)}"
                )
            p0 = prior.to(device=device, dtype=dtype).clone()
            p0 = p0 / p0.sum()
        else:
            p0 = torch.full((cfg.I,), 1.0 / cfg.I, device=device, dtype=dtype)
        self.register_buffer("_p0_default", p0)

        if params.default_x0 is None:
            x0_default = torch.tensor(
                [1.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                device=device,
                dtype=dtype,
            )
        else:
            x0_default = torch.as_tensor(list(params.default_x0), device=device, dtype=dtype)
            if x0_default.shape != (cfg.dx,):
                raise ValueError(
                    f"HexnerMod3DParams.default_x0 must have shape ({cfg.dx},), "
                    f"got {tuple(x0_default.shape)}"
                )
        self.register_buffer("_x0_default", x0_default)

    @property
    def terminal_matrix(self) -> Tensor:
        return self._terminal_K

    def type_targets(self) -> Tensor:
        return self._type_targets.clone()

    def default_initial_state(self) -> Tensor:
        return self._x0_default.clone()

    def default_prior(self) -> Tensor:
        return self._p0_default.clone()
