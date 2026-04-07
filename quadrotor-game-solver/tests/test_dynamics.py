"""Tests for the 6-DoF quadrotor dynamics model.

Verifies:
  1. Hover equilibrium (f = mg, zero torques → no acceleration)
  2. Output shapes
  3. Euler vs RK4 agreement for small dt
  4. Linearisation Jacobians (jacfwd) vs finite differences
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.utils.config import QuadrotorParams
from src.dynamics.quadrotor_model import (
    DX_SINGLE,
    DX_JOINT,
    DU,
    DV,
    _single_quadrotor_dynamics,
    _joint_continuous_dynamics,
    step_euler,
    step_rk4,
    step_dynamics,
    linearize_dynamics,
)


PARAMS = QuadrotorParams()
DTYPE = torch.float64
DEVICE = torch.device("cpu")


def _hover_state() -> Tensor:
    """Joint state with both drones hovering at (0,0,1)."""
    x = torch.zeros(DX_JOINT, dtype=DTYPE, device=DEVICE)
    x[2] = 1.0    # P1 pz
    x[14] = 1.0   # P2 pz
    return x


def _hover_control() -> tuple[Tensor, Tensor]:
    """Hover controls: thrust = mg, zero torques."""
    mg = PARAMS.mass * PARAMS.g
    u = torch.zeros(DU, dtype=DTYPE, device=DEVICE)
    u[0] = mg
    v = torch.zeros(DV, dtype=DTYPE, device=DEVICE)
    v[0] = mg
    return u, v


# ── Test: hover equilibrium ────────────────────────────────────────────────

class TestHoverEquilibrium:

    def test_single_drone_hover_zero_derivative(self):
        """At hover (f=mg, zero angles/rates), xdot should be ≈ 0."""
        x_single = torch.zeros(DX_SINGLE, dtype=DTYPE)
        x_single[2] = 1.0  # pz
        u_single = torch.zeros(4, dtype=DTYPE)
        u_single[0] = PARAMS.mass * PARAMS.g

        xdot = _single_quadrotor_dynamics(x_single, u_single, PARAMS)
        assert xdot.shape == (DX_SINGLE,)
        torch.testing.assert_close(
            xdot, torch.zeros_like(xdot), atol=1e-12, rtol=0
        )

    def test_joint_hover_zero_derivative(self):
        """Joint continuous dynamics at hover should give zero derivative."""
        x = _hover_state()
        u, v = _hover_control()
        xdot = _joint_continuous_dynamics(x, u, v, PARAMS)
        assert xdot.shape == (DX_JOINT,)
        torch.testing.assert_close(
            xdot, torch.zeros_like(xdot), atol=1e-12, rtol=0
        )

    def test_euler_hover_stays(self):
        """One Euler step at hover should keep the state unchanged."""
        x = _hover_state()
        u, v = _hover_control()
        dt = 0.2
        x_next = step_euler(x, u, v, dt, PARAMS)
        torch.testing.assert_close(x_next, x, atol=1e-10, rtol=0)

    def test_rk4_hover_stays(self):
        """One RK4 step at hover should keep the state unchanged."""
        x = _hover_state()
        u, v = _hover_control()
        dt = 0.2
        x_next = step_rk4(x, u, v, dt, PARAMS)
        torch.testing.assert_close(x_next, x, atol=1e-10, rtol=0)


# ── Test: output shapes ───────────────────────────────────────────────────

class TestShapes:

    def test_step_shapes(self):
        x = _hover_state()
        u, v = _hover_control()
        x_next = step_dynamics(x, u, v, 0.1, PARAMS, "euler")
        assert x_next.shape == (DX_JOINT,)

    def test_step_batch_shapes(self):
        B = 5
        x = _hover_state().unsqueeze(0).expand(B, -1)
        u, v = _hover_control()
        u = u.unsqueeze(0).expand(B, -1)
        v = v.unsqueeze(0).expand(B, -1)
        x_next = step_dynamics(x, u, v, 0.1, PARAMS, "euler")
        assert x_next.shape == (B, DX_JOINT)

    def test_linearize_shapes(self):
        B = 3
        x = _hover_state().unsqueeze(0).expand(B, -1).clone()
        u, v = _hover_control()
        u = u.unsqueeze(0).expand(B, -1).clone()
        v = v.unsqueeze(0).expand(B, -1).clone()

        A, B1, B2, d = linearize_dynamics(x, u, v, 0.1, PARAMS, "euler")
        assert A.shape == (B, DX_JOINT, DX_JOINT)
        assert B1.shape == (B, DX_JOINT, DU)
        assert B2.shape == (B, DX_JOINT, DV)
        assert d.shape == (B, DX_JOINT)


# ── Test: Euler ≈ RK4 for small dt ────────────────────────────────────────

class TestEulerVsRK4:

    def test_close_for_small_dt(self):
        """For very small dt both integrators should give nearly identical results."""
        x = _hover_state()
        # Add a small perturbation
        x[3] = 0.1   # vx for P1
        x[6] = 0.05  # phi for P1
        u, v = _hover_control()

        dt = 1e-4
        x_euler = step_euler(x, u, v, dt, PARAMS)
        x_rk4   = step_rk4(x, u, v, dt, PARAMS)
        torch.testing.assert_close(x_euler, x_rk4, atol=1e-8, rtol=1e-6)


# ── Test: linearisation vs finite differences ─────────────────────────────

class TestLinearization:

    def test_jacobian_A_fd(self):
        """A from jacfwd should match finite-difference Jacobian."""
        x = _hover_state().unsqueeze(0)
        x[0, 3] = 0.1
        x[0, 6] = 0.05
        u, v = _hover_control()
        u = u.unsqueeze(0)
        v = v.unsqueeze(0)

        dt = 0.1
        A_jac, _, _, _ = linearize_dynamics(x, u, v, dt, PARAMS, "euler")

        # Finite-difference A
        eps = 1e-5
        A_fd = torch.zeros(DX_JOINT, DX_JOINT, dtype=DTYPE)
        x0 = x.squeeze(0)
        u0 = u.squeeze(0)
        v0 = v.squeeze(0)
        f0 = step_euler(x0, u0, v0, dt, PARAMS)
        for j in range(DX_JOINT):
            xp = x0.clone()
            xp[j] += eps
            fp = step_euler(xp, u0, v0, dt, PARAMS)
            A_fd[:, j] = (fp - f0) / eps

        torch.testing.assert_close(
            A_jac.squeeze(0), A_fd, atol=1e-4, rtol=1e-3
        )

    def test_jacobian_B1_fd(self):
        """B1 from jacfwd should match finite-difference Jacobian."""
        x = _hover_state().unsqueeze(0)
        u, v = _hover_control()
        u = u.unsqueeze(0)
        v = v.unsqueeze(0)

        dt = 0.1
        _, B1_jac, _, _ = linearize_dynamics(x, u, v, dt, PARAMS, "euler")

        eps = 1e-5
        B1_fd = torch.zeros(DX_JOINT, DU, dtype=DTYPE)
        x0 = x.squeeze(0)
        u0 = u.squeeze(0)
        v0 = v.squeeze(0)
        f0 = step_euler(x0, u0, v0, dt, PARAMS)
        for j in range(DU):
            up = u0.clone()
            up[j] += eps
            fp = step_euler(x0, up, v0, dt, PARAMS)
            B1_fd[:, j] = (fp - f0) / eps

        torch.testing.assert_close(
            B1_jac.squeeze(0), B1_fd, atol=1e-4, rtol=1e-3
        )

    def test_affine_residual(self):
        """d = f(x,u,v) - A x - B1 u - B2 v at the linearisation point."""
        x = _hover_state().unsqueeze(0)
        x[0, 3] = 0.2
        u, v = _hover_control()
        u = u.unsqueeze(0)
        v = v.unsqueeze(0)

        dt = 0.15
        A, B1, B2, d = linearize_dynamics(x, u, v, dt, PARAMS, "euler")

        x_next_true = step_euler(x.squeeze(0), u.squeeze(0), v.squeeze(0),
                                  dt, PARAMS)
        x_next_lin = (
            (A @ x.unsqueeze(-1)).squeeze(-1)
            + (B1 @ u.unsqueeze(-1)).squeeze(-1)
            + (B2 @ v.unsqueeze(-1)).squeeze(-1)
            + d
        ).squeeze(0)

        torch.testing.assert_close(
            x_next_lin, x_next_true, atol=1e-10, rtol=0
        )


# ── Test: free-fall under zero thrust ──────────────────────────────────────

class TestFreeFall:

    def test_vertical_freefall(self):
        """With zero thrust, drone should accelerate downward at g."""
        x = torch.zeros(DX_SINGLE, dtype=DTYPE)
        x[2] = 10.0  # start at z=10
        u = torch.zeros(4, dtype=DTYPE)  # no thrust

        xdot = _single_quadrotor_dynamics(x, u, PARAMS)
        # dvz should be -g
        assert abs(float(xdot[5]) - (-PARAMS.g)) < 1e-12
        # dpz should be 0 (no velocity yet)
        assert abs(float(xdot[2])) < 1e-12
