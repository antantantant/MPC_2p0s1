"""3-D Hexner game with nonlinear 6-DoF quadrotor dynamics.

This is the faithful 3-D extension of the 2-D LQ Hexner game.  The game
structure is *identical*:

  - 2 players, zero-sum, asymmetric information.
  - P1 (minimiser, informed) knows the payoff type θ ∈ {θ₁,…,θ_I}.
  - P2 (maximiser, uninformed) knows only the prior p₀.
  - Signaling via α-parameterisation (same as LQ code).

What changes:
  - Dynamics: each player is a full 6-DoF quadrotor instead of a 2-D double
    integrator.
  - Costs: position-based Hexner terminal cost extended to 3-D; running cost
    is still control-effort only (R u, S v).
  - The inner Riccati step is used inside an SQP loop that re-linearises the
    dynamics at each iteration.

State layout (per player, 12-D):
    x_j = [px, py, pz, vx, vy, vz, phi, theta, psi, wx, wy, wz]

Joint state (24-D):
    x = [x_P1 (12), x_P2 (12)]

Control (per player, 4-D):
    u_j = [f, tau_x, tau_y, tau_z]

Terminal cost:
    g_i(x) = ||pos1 − z·θ_i||²_{K1}  −  ||pos2 − z·θ_i||²_{K2}

    where z ∈ R^{dx_single} is the target direction (default: z-axis position),
    K1, K2 ∈ R^{dx_single × dx_single} penalise position only
    (K_base = diag(1,1,1, 0,…,0)  for the 3 position coordinates).

Running cost:
    ℓ(u,v) = (τ/2) u^T R u  −  (τ/2) v^T S v

    Belief-averaged versions R̄, S̄ when R,S are per-type (here type-independent
    by default, exactly like the original Hexner game).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from ..utils.config import GameConfig, QuadrotorParams
from ..dynamics.quadrotor_model import (
    DX_SINGLE, DX_JOINT, DU, DV,
    step_dynamics, linearize_dynamics,
)


class Hexner3DQuadrotorGame:
    """Nonlinear 3-D Hexner game with quadrotor dynamics.

    Mirrors the interface of ``DroneSignalingGame`` in ``nl_sqp/drone_game.py``
    but with:
      - proper 6-DoF quadrotor dynamics (instead of kinematic model)
      - Hexner-style costs (terminal = position tracking, running = control only)
    """

    dx: int = DX_JOINT   # 24
    du: int = DU          # 4
    dv: int = DV          # 4
    dx_single: int = DX_SINGLE  # 12

    def __init__(
        self,
        cfg: GameConfig,
        *,
        prior: Optional[Tensor] = None,
    ) -> None:
        self.cfg = cfg
        self.I = cfg.I
        self.device = cfg.device_resolved
        self.dtype = cfg.dtype
        self.linearized_mode = cfg.linearized_mode
        self.control_cost_mode = cfg.control_cost_mode

        # ── Payoff types θ ───────────────────────────────────────────
        theta_vals = torch.as_tensor(
            list(cfg.theta_values), device=self.device, dtype=self.dtype
        )
        if theta_vals.numel() != self.I:
            raise ValueError(
                f"cfg.I={self.I} but got {theta_vals.numel()} theta values."
            )
        self.theta_vals = theta_vals

        # ── Target direction z ∈ R^{dx_single} ──────────────────────
        #   In 2-D Hexner:  z = (0,1,0,0)   →  target along y-position.
        #   In 3-D Hexner:  z = (0,0,1, 0,…,0) →  target along z-position.
        if cfg.target_z is None:
            z = torch.zeros(self.dx_single, device=self.device, dtype=self.dtype)
            z[2] = 1.0   # z-position
        else:
            z = torch.as_tensor(cfg.target_z, device=self.device, dtype=self.dtype)
            if z.shape != (self.dx_single,):
                raise ValueError(
                    f"target_z must have shape ({self.dx_single},), "
                    f"got {tuple(z.shape)}"
                )
        self.target_z = z

        # ── Running cost matrices R_i, S_i ───────────────────────────
        #   Convention:  ℓ = ½ u^T R u − ½ v^T S v
        #   In original Hexner: R = 2·R1_base, S = 2·R2_base  (type-indep.)
        R1 = torch.diag(torch.tensor(cfg.R1_diag, device=self.device, dtype=self.dtype))
        R2 = torch.diag(torch.tensor(cfg.R2_diag, device=self.device, dtype=self.dtype))

        #  R_i = 2 R1,  S_i = 2 R2   (same for all types, as in Hexner)
        self._R = (2.0 * R1).unsqueeze(0).expand(self.I, self.du, self.du).clone()
        self._S = (2.0 * R2).unsqueeze(0).expand(self.I, self.dv, self.dv).clone()

        # ── Terminal cost matrices Q_i, q_i, c_i ────────────────────
        #   K_base = diag(1,1,1, 0,…,0)  for the 3 position dims.
        K_base = torch.zeros(self.dx_single, self.dx_single,
                             device=self.device, dtype=self.dtype)
        K_base[0, 0] = 1.0
        K_base[1, 1] = 1.0
        K_base[2, 2] = 1.0

        K1 = cfg.K1_scale * K_base
        K2 = cfg.K2_scale * K_base

        #   g_i(x) = ||x1_pos − z θ_i||²_{K1} − ||x2_pos − z θ_i||²_{K2}
        #          = x^T diag(K1,−K2) x  − 2θ_i [K1 z ; −K2 z]^T x
        #            + θ_i² (z^T K1 z − z^T K2 z)
        #   Stored as  g_i = ½ x^T Q_i x + q_i^T x + c_i
        #   ⟹  Q_i = 2 diag(K1,−K2),  q_i = −2θ_i [K1 z ; −K2 z]

        Q_block = torch.block_diag(K1, -K2)       # (dx, dx)
        Q_i = 2.0 * Q_block                        # same for all types

        zK1z = torch.dot(z, K1 @ z)
        zK2z = torch.dot(z, K2 @ z)

        Q_stack = torch.empty(self.I, self.dx, self.dx,
                              device=self.device, dtype=self.dtype)
        q_stack = torch.empty(self.I, self.dx,
                              device=self.device, dtype=self.dtype)
        c_stack = torch.empty(self.I, device=self.device, dtype=self.dtype)

        for idx, theta in enumerate(theta_vals):
            q1 = -2.0 * theta * (K1.T @ z)
            q2 =  2.0 * theta * (K2.T @ z)
            q_full = torch.cat([q1, q2], dim=0)

            c_i = (theta * theta) * (zK1z - zK2z)

            Q_stack[idx] = Q_i
            q_stack[idx] = q_full
            c_stack[idx] = c_i

        self._Q = Q_stack
        self._q = q_stack
        self._c = c_stack

        # ── Default prior ────────────────────────────────────────────
        if prior is not None:
            if prior.shape != (self.I,):
                raise ValueError(
                    f"prior must have shape ({self.I},), got {tuple(prior.shape)}"
                )
            p0 = prior.to(device=self.device, dtype=self.dtype).clone()
            p0 = p0 / p0.sum()
        else:
            p0 = torch.full((self.I,), 1.0 / self.I,
                            device=self.device, dtype=self.dtype)
        self._p0_default = p0

        # ── Linearized mode: precompute frozen A, B1, B2 at hover ────────
        if self.linearized_mode:
            x_hover = self.default_initial_state().unsqueeze(0)  # (1, 24)
            u_hover, v_hover = self.default_hover_control()
            u_hover = u_hover.unsqueeze(0)  # (1, 4)
            v_hover = v_hover.unsqueeze(0)  # (1, 4)

            # Linearize once at hover (call dynamics module directly to avoid recursion)
            A_lin, B1_lin, B2_lin, d_lin = linearize_dynamics(
                x_hover, u_hover, v_hover, self.cfg.tau, self.cfg.quad_params, self.cfg.integrator
            )

            # Store as cached tensors (squeeze batch dim)
            self._A_cached = A_lin.squeeze(0)      # (24, 24)
            self._B1_cached = B1_lin.squeeze(0)    # (24, 4)
            self._B2_cached = B2_lin.squeeze(0)    # (24, 4)

            # Zero out d to make it purely linear (not affine) for true LQ problem
            # This is acceptable for testing since we just want to verify SQP works
            self._d_cached = torch.zeros(self.dx, device=self.device, dtype=self.dtype)
        else:
            self._A_cached = None
            self._B1_cached = None
            self._B2_cached = None
            self._d_cached = None

    # ─── Properties ──────────────────────────────────────────────────────

    @property
    def R(self) -> Tensor:
        """Per-type P1 running cost: (I, du, du)."""
        return self._R

    @property
    def S(self) -> Tensor:
        """Per-type P2 running cost: (I, dv, dv)."""
        return self._S

    @property
    def Q(self) -> Tensor:
        """Per-type terminal quadratic: (I, dx, dx)."""
        return self._Q

    @property
    def q(self) -> Tensor:
        """Per-type terminal linear: (I, dx)."""
        return self._q

    @property
    def c(self) -> Tensor:
        """Per-type terminal constant: (I,)."""
        return self._c

    # ─── Defaults ────────────────────────────────────────────────────────

    def default_initial_state(self) -> Tensor:
        """Default x0: both drones hovering, facing each other.

        P1 at (-2, 0, 1) with zero velocity/angles, thrust = mg.
        P2 at (+2, 0, 1) with zero velocity/angles, thrust = mg.
        """
        x0 = torch.zeros(self.dx, device=self.device, dtype=self.dtype)
        # P1
        x0[0] = -2.0   # px
        x0[1] =  0.0   # py
        x0[2] =  1.0   # pz
        # velocities, angles, angular rates = 0
        # P2
        x0[12] =  2.0  # px
        x0[13] =  0.0  # py
        x0[14] =  1.0  # pz
        return x0

    def hover_physical_control(self) -> Tuple[Tensor, Tensor]:
        """Physical controls that keep the drones hovering (f = mg, τ = 0)."""
        mg = self.cfg.quad_params.mass * self.cfg.quad_params.g
        u0 = torch.zeros(self.du, device=self.device, dtype=self.dtype)
        u0[0] = mg
        v0 = torch.zeros(self.dv, device=self.device, dtype=self.dtype)
        v0[0] = mg
        return u0, v0

    def control_bias(self) -> Tuple[Tensor, Tensor]:
        """Additive bias mapping optimization controls to physical controls."""
        if self.control_cost_mode == "hover_relative":
            return self.hover_physical_control()
        z_u = torch.zeros(self.du, device=self.device, dtype=self.dtype)
        z_v = torch.zeros(self.dv, device=self.device, dtype=self.dtype)
        return z_u, z_v

    def to_physical_controls(self, u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """Convert optimization controls to physical controls."""
        u_bias, v_bias = self.control_bias()
        return u + u_bias, v + v_bias

    def default_hover_control(self) -> Tuple[Tensor, Tensor]:
        """Default optimization controls for hover equilibrium.

        In ``hover_relative`` mode these are zero deltas.
        In ``absolute`` mode these are the physical hover controls.
        """
        if self.control_cost_mode == "hover_relative":
            z_u = torch.zeros(self.du, device=self.device, dtype=self.dtype)
            z_v = torch.zeros(self.dv, device=self.device, dtype=self.dtype)
            return z_u, z_v
        return self.hover_physical_control()

    def default_prior(self) -> Tensor:
        return self._p0_default.clone()

    # ─── Dynamics ────────────────────────────────────────────────────────

    def step(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """One-step dynamics (nonlinear or frozen linearized).

        If linearized_mode=True, uses frozen linear dynamics: x⁺ = Ax + B1u + B2v + d
        Otherwise uses full nonlinear dynamics.
        """
        u_phys, v_phys = self.to_physical_controls(u, v)

        if self.linearized_mode:
            # Use frozen linearized dynamics: x⁺ = Ax + B1u + B2v + d
            # x: (..., 24), u: (..., 4), v: (..., 4)
            Ax = torch.matmul(x, self._A_cached.T)          # (..., 24)
            B1u = torch.matmul(u_phys, self._B1_cached.T)   # (..., 24)
            B2v = torch.matmul(v_phys, self._B2_cached.T)   # (..., 24)
            return Ax + B1u + B2v + self._d_cached

        # Otherwise use full nonlinear dynamics (with optional small-angle approx)
        return step_dynamics(
            x, u_phys, v_phys, self.cfg.tau, self.cfg.quad_params, self.cfg.integrator,
            small_angle=self.cfg.small_angle_approx
        )

    def linearize(
        self, x: Tensor, u: Tensor, v: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Linearize dynamics:  x⁺ ≈ A x + B1 u + B2 v + d.

        Parameters
        ----------
        x : (B, 24)
        u : (B, 4)
        v : (B, 4)

        Returns A (B,24,24), B1 (B,24,4), B2 (B,24,4), d (B,24).
        """
        u_phys, v_phys = self.to_physical_controls(u, v)

        # If in linearized mode, return frozen matrices cached at hover
        if self.linearized_mode:
            B = x.shape[0]
            A = self._A_cached.unsqueeze(0).expand(B, -1, -1)   # (B, 24, 24)
            B1 = self._B1_cached.unsqueeze(0).expand(B, -1, -1) # (B, 24, 4)
            B2 = self._B2_cached.unsqueeze(0).expand(B, -1, -1) # (B, 24, 4)
            d = self._d_cached.unsqueeze(0).expand(B, -1)       # (B, 24)
        else:
            # Otherwise, compute linearization normally (with optional small-angle approx)
            A, B1, B2, d = linearize_dynamics(
                x, u_phys, v_phys, self.cfg.tau, self.cfg.quad_params, self.cfg.integrator,
                small_angle=self.cfg.small_angle_approx
            )

        # Map affine term from physical-control coordinates to optimization-control coordinates.
        if self.control_cost_mode == "hover_relative":
            u_bias, v_bias = self.control_bias()
            d = (
                d
                + (B1 @ u_bias.view(1, self.du, 1)).squeeze(-1)
                + (B2 @ v_bias.view(1, self.dv, 1)).squeeze(-1)
            )

        return A, B1, B2, d

    # ─── Belief-averaged cost helpers (same API as nl_sqp) ───────────────

    def running_cost_mats(
        self, belief: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Belief-averaged (R̄, S̄).

        Parameters
        ----------
        belief : (..., I)

        Returns
        -------
        R_bar : (..., du, du)
        S_bar : (..., dv, dv)
        """
        R_bar = torch.einsum("...i, iab -> ...ab", belief, self._R)
        S_bar = torch.einsum("...i, iab -> ...ab", belief, self._S)
        return R_bar, S_bar

    def terminal_cost_quad(
        self, belief: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Belief-averaged terminal (Q̄, q̄, c̄).

        Parameters
        ----------
        belief : (..., I)

        Returns
        -------
        Q_bar : (..., dx, dx)
        q_bar : (..., dx)
        c_bar : (...,)
        """
        Q_bar = torch.einsum("...i, iab -> ...ab", belief, self._Q)
        q_bar = torch.einsum("...i, ia  -> ...a",  belief, self._q)
        c_bar = torch.einsum("...i, i   -> ...",   belief, self._c)
        return Q_bar, q_bar, c_bar

    def stage_cost_mats_batch(
        self, beliefs: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Vectorised running-cost data for a batch of beliefs.

        Parameters
        ----------
        beliefs : (B, I)

        Returns
        -------
        Q   : (dx, dx)    — zero (no state cost in running cost)
        q   : (B, dx)     — zero
        c   : (B,)        — zero
        R_bar : (B, du, du)
        S_bar : (B, dv, dv)
        """
        B = beliefs.shape[0]
        Q = torch.zeros(self.dx, self.dx, device=self.device, dtype=self.dtype)
        q = torch.zeros(B, self.dx, device=self.device, dtype=self.dtype)
        c = torch.zeros(B, device=self.device, dtype=self.dtype)
        R_bar, S_bar = self.running_cost_mats(beliefs)
        return Q, q, c, R_bar, S_bar

    def terminal_value_quad_batch(
        self, beliefs: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Vectorised terminal cost for a batch of beliefs.

        Parameters
        ----------
        beliefs : (B, I)

        Returns
        -------
        P   : (B, dx, dx)   or broadcastable
        r   : (B, dx)
        c   : (B,)
        """
        return self.terminal_cost_quad(beliefs)

    # ─── Targets for visualisation ───────────────────────────────────────

    def type_targets(self) -> Tensor:
        """Return type-dependent target states z·θ_i.

        Returns shape (I, dx_single).
        """
        return self.theta_vals.view(-1, 1) * self.target_z.view(1, -1)

    def type_target_positions(self) -> Tensor:
        """Return (I, 3) target *positions* for plotting."""
        targets = self.type_targets()    # (I, dx_single)
        return targets[:, :3]            # (I, 3)
