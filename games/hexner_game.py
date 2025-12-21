# games/hexner_game.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn

from ..config.base_config import GameConfig
from ..core.types import Tensor
from .base_lq_game import BaseLQGame


@dataclass
class HexnerParams:
    """
    Parameterization of Hexner’s game (2D double integrator variant).

    The continuous-time game is described in Hexner (1979) and revisited in the
    2p0s1 paper as a canonical example with an analytical NE and a critical
    “reveal time” t_r.  [oai_citation:1‡16010_Solving_Football_by_Expl-2.pdf](sediment://file_000000001b0871f58cd1c91b483083b4)

    This dataclass gathers the numeric choices for a discrete-time approximation
    used as a minimal test case in this codebase.

    Attributes
    ----------
    theta_values:
        Vector of payoff types θ ∈ R^I. For the standard two-target version,
        θ = (-1, +1), so the two targets are z·θ_1 and z·θ_2.
    target_z:
        Base target vector z ∈ R^{dx1}. The actual target for type θ_i is
        z_i = θ_i * z. We usually concentrate z on position coordinates.
    R1_scale, R2_scale:
        Scalars controlling the running costs:
            ∥u1∥^2_{R1} - ∥u2∥^2_{R2}.
        We internally translate this into ℓ_i(u, v) = 0.5 u^T R_i u - 0.5 v^T S_i v
        with R_i = 2 R1, S_i = 2 R2.
    K1_scale, K2_scale:
        Scalars controlling the terminal costs:
            ∥x1(T) - zθ∥^2_{K1} - ∥x2(T) - zθ∥^2_{K2}.
        We use K1 = K1_scale * I_{dx1}, K2 = K2_scale * I_{dx1} by default.
    """

    theta_values: Iterable[float] = (-1.0, 1.0)
    target_z: Optional[Iterable[float]] = None

    R1_scale: float = 1.0
    R2_scale: float = 1.0
    K1_scale: float = 10.0
    K2_scale: float = 10.0


class HexnerGame(BaseLQGame):
    """
    Discrete-time LQ implementation of Hexner’s game in 2D.

    Structure (following the 2p0s1 paper’s formulation of Hexner’s game):  [oai_citation:2‡16010_Solving_Football_by_Expl-2.pdf](sediment://file_000000001b0871f58cd1c91b483083b4)

    - Two players j = 1,2 with double-integrator dynamics in 2D:
        state x_j = (pos_x, pos_y, vel_x, vel_y) ∈ R^4,
        control u_j = (acc_x, acc_y) ∈ R^2.

      Continuous-time dynamics:
        d/dt pos = vel,
        d/dt vel = acc.

      We discretize with time step τ = T / K and constant controls per step,
      resulting in standard double-integrator matrices:

        x_{j,k+1} = A_player x_{j,k} + B_player u_{j,k},

      where:

        A_player = [[1, 0, τ, 0],
                    [0, 1, 0, τ],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]],

        B_player = [[0.5 τ^2,       0],
                    [0,       0.5 τ^2],
                    [τ,             0],
                    [0,             τ]].

      The global state is x = (x_1, x_2) ∈ R^8, and we set

        A = diag(A_player, A_player),
        B1 = [B_player; 0],
        B2 = [0; B_player].

    - Payoff types and targets:
        Nature draws θ ∈ Θ with |Θ| = I from prior p0. P1 observes θ, P2 only
        sees the public belief. The target for P1 is zθ, where z ∈ R^{dx1} is
        fixed and public. P2 tries to match or beat P1’s proximity to zθ.

    - Running cost:
        ∫ (∥u1(t)∥^2_{R1} - ∥u2(t)∥^2_{R2}) dt,

      which we encode via ℓ_i(u, v) = 0.5 u^T R_i u - 0.5 v^T S_i v with
      R_i = 2 R1, S_i = 2 R2, identical across types.

    - Terminal cost:
        ∥x1(T) - zθ∥^2_{K1} - ∥x2(T) - zθ∥^2_{K2},

      expanded into 0.5 x^T Q_i x + q_i^T x + c_i for each type θ_i.

    This implementation is meant as a small but representative test case. The
    numerical choices (scales, exact z) are configurable via HexnerParams.
    """

    def __init__(self, cfg: GameConfig, params: Optional[HexnerParams] = None) -> None:
        if cfg.dx1 != 4 or cfg.dx2 != 4 or cfg.du != 2 or cfg.dv != 2:
            raise ValueError(
                "HexnerGame expects dx1=dx2=4 (2D pos+vel) and du=dv=2 (2D accel). "
                f"Got dx1={cfg.dx1}, dx2={cfg.dx2}, du={cfg.du}, dv={cfg.dv}."
            )
        super().__init__(cfg)

        if params is None:
            params = HexnerParams()

        device = cfg.device_resolved
        dtype = cfg.dtype

        # ------------------------------------------------------------------ #
        # Payoff types θ and targets zθ                                      #
        # ------------------------------------------------------------------ #
        theta_vals = torch.as_tensor(
            list(params.theta_values), device=device, dtype=dtype
        )
        if theta_vals.numel() != cfg.I:
            raise ValueError(
                f"HexnerGame: cfg.I={cfg.I} but got {theta_vals.numel()} theta values."
            )

        if params.target_z is None:
            # Default: target along y-axis in position space only.
            # For dx1=4 (pos_x, pos_y, vel_x, vel_y), we set z = (0, 1, 0, 0).
            z = torch.zeros(cfg.dx1, device=device, dtype=dtype)
            z[1] = 1.0
        else:
            z = torch.as_tensor(params.target_z, device=device, dtype=dtype)
            if z.shape != (cfg.dx1,):
                raise ValueError(
                    f"HexnerGame: target_z must have shape ({cfg.dx1},), "
                    f"got {tuple(z.shape)}"
                )

        self.register_buffer("theta_vals", theta_vals)
        self.register_buffer("target_z", z)

        # ------------------------------------------------------------------ #
        # Dynamics matrices A, B1, B2                                       #
        # ------------------------------------------------------------------ #
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

        A = torch.block_diag(A_player, A_player)  # (8, 8)

        B1 = torch.zeros(cfg.dx, cfg.du, device=device, dtype=dtype)
        B2 = torch.zeros(cfg.dx, cfg.dv, device=device, dtype=dtype)
        # P1 controls player 1, P2 controls player 2
        B1[: cfg.dx1, :] = B_player
        B2[cfg.dx1 :, :] = B_player

        # Register as buffers so they follow the module's device/dtype
        self.register_buffer("_A", A)
        self.register_buffer("_B1", B1)
        self.register_buffer("_B2", B2)

        # ------------------------------------------------------------------ #
        # Running cost matrices R_i, S_i                                    #
        # ------------------------------------------------------------------ #
        R1 = params.R1_scale * torch.eye(cfg.du, device=device, dtype=dtype)
        R2 = params.R2_scale * torch.eye(cfg.dv, device=device, dtype=dtype)

        # To match ℓ_i(u,v) = ∥u1∥^2_{R1} - ∥v∥^2_{R2} when plugged into 0.5 u^T R u
        # we set R_i = 2 R1, S_i = 2 R2.
        R_stack = torch.stack([2.0 * R1 for _ in range(cfg.I)], dim=0)  # (I, du, du)
        S_stack = torch.stack([2.0 * R2 for _ in range(cfg.I)], dim=0)  # (I, dv, dv)

        self.register_buffer("_R", R_stack)
        self.register_buffer("_S", S_stack)

        # ------------------------------------------------------------------ #
        # Terminal cost matrices Q_i, q_i, c_i                              #
        # ------------------------------------------------------------------ #
        K1 = params.K1_scale * torch.eye(cfg.dx1, device=device, dtype=dtype)
        K2 = params.K2_scale * torch.eye(cfg.dx1, device=device, dtype=dtype)

        # Base quadratic form for g_i(x):
        #   g_i(x) = ||x1 - z θ_i||^2_{K1} - ||x2 - z θ_i||^2_{K2}.
        #
        # Expand:
        #   = x1^T K1 x1 - x2^T K2 x2
        #     - 2 θ_i z^T K1 x1 + 2 θ_i z^T K2 x2
        #     + θ_i^2 (z^T K1 z - z^T K2 z).
        #
        # We encode:
        #   g_i(x) = 0.5 x^T Q_i x + q_i^T x + c_i,
        # where x = (x1, x2) ∈ R^{dx1+dx2}.
        #
        # Quadratic term:
        #   0.5 x^T Q_i x = x^T diag(K1, -K2) x
        # so Q_i = 2 * diag(K1, -K2) (independent of θ_i).
        Q_block = torch.block_diag(K1, -K2)  # (dx, dx)
        Q_i = 2.0 * Q_block                   # same for all i

        # Precompute z^T K1 z and z^T K2 z
        zK1z = torch.dot(z, K1 @ z)
        zK2z = torch.dot(z, K2 @ z)

        Q_stack = torch.empty(cfg.I, cfg.dx, cfg.dx, device=device, dtype=dtype)
        q_stack = torch.empty(cfg.I, cfg.dx, device=device, dtype=dtype)
        c_stack = torch.empty(cfg.I, device=device, dtype=dtype)

        for idx, theta in enumerate(theta_vals):
            # Linear term:
            #   q_i^T x = -2 θ_i z^T K1 x1 + 2 θ_i z^T K2 x2
            # so
            #   q1_i = -2 θ_i K1^T z, q2_i =  2 θ_i K2^T z.
            q1 = -2.0 * theta * (K1.T @ z)  # (dx1,)
            q2 = 2.0 * theta * (K2.T @ z)   # (dx1,)
            q_full = torch.cat([q1, q2], dim=0)  # (dx1+dx2,)

            # Constant term:
            #   c_i = θ_i^2 (z^T K1 z - z^T K2 z)
            c_i = (theta * theta) * (zK1z - zK2z)

            Q_stack[idx] = Q_i
            q_stack[idx] = q_full
            c_stack[idx] = c_i

        self.register_buffer("_Q", Q_stack)
        self.register_buffer("_q", q_stack)
        self.register_buffer("_c", c_stack)

        # ------------------------------------------------------------------ #
        # Default prior p0 (uniform over θ)                                  #
        # ------------------------------------------------------------------ #
        p0 = torch.full((cfg.I,), 1.0 / cfg.I, device=device, dtype=dtype)
        self.register_buffer("_p0_default", p0)

    # ---------------------------------------------------------------------- #
    # Convenience methods                                                    #
    # ---------------------------------------------------------------------- #

    def default_initial_state(self) -> Tensor:
        """
        Default initial state x0 for Hexner’s game.

        We place both players at the origin with zero velocity:

            x1 = (0, 0, 0, 0),  x2 = (0, 0, 0, 0),

        so x0 ∈ R^8 is all zeros.
        """
        return torch.zeros(self.dx, device=self.device_resolved, dtype=self.dtype)

    def default_prior(self) -> Tensor:
        """
        Default prior p0 over types: uniform over θ-values."""
        return self._p0_default.clone()

    # Optional helpers for visualization / analysis ------------------------ #

    def type_targets(self) -> Tensor:
        """
        Return the type-dependent target states z θ_i for P1.

        Returns
        -------
        Tensor
            Tensor of shape (I, dx1) where entry i is z θ_i.
        """
        # (I, 1) * (dx1,) -> (I, dx1) via broadcasting
        return self.theta_vals.view(-1, 1) * self.target_z.view(1, -1)