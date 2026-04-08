"""Nonlinear 6-DoF quadrotor dynamics (Euler-angle parameterisation).

Each drone has a 12-D state and 4-D control:

    x_j = [px, py, pz, vx, vy, vz, phi, theta, psi, wx, wy, wz]
    u_j = [f, tau_x, tau_y, tau_z]

Continuous-time equations of motion
------------------------------------

Position derivatives:
    ṗ = v

Velocity derivatives (world frame, ZYX rotation convention):
    v̇ = R(phi,theta,psi) · [0, 0, f/m]^T − [0, 0, g]^T

Euler-angle derivatives (body-rate → Euler-rate mapping):
    φ̇   = wx + (wy sin(φ) + wz cos(φ)) tan(θ)
    θ̇   = wy cos(φ) − wz sin(φ)
    ψ̇   = (wy sin(φ) + wz cos(φ)) / cos(θ)

Angular-rate derivatives (Euler's rigid-body equations):
    ẇx = (tau_x − (Izz − Iyy) wy wz) / Ixx
    ẇy = (tau_y − (Ixx − Izz) wx wz) / Iyy
    ẇz = (tau_z − (Iyy − Ixx) wx wy) / Izz

We discretize with forward Euler or RK4.

Convention:
    - Joint state of the two-player game:  x = [x_P1, x_P2] ∈ R^24.
    - P1 controls: u ∈ R^4.
    - P2 controls: v ∈ R^4.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..utils.config import QuadrotorParams

# ── Dimensions ──────────────────────────────────────────────────────────────

DX_SINGLE: int = 12    # per-player state dimension
DU_SINGLE: int = 4     # per-player control dimension
DX_JOINT: int = 24     # joint state  (two players)
DU: int = 4            # P1 control dimension
DV: int = 4            # P2 control dimension


# ── Continuous-time dynamics for one quadrotor ──────────────────────────────

def _single_quadrotor_dynamics(
    state: Tensor,
    control: Tensor,
    params: QuadrotorParams,
    small_angle: bool = False,
) -> Tensor:
    """Continuous-time state derivative for one quadrotor.

    Parameters
    ----------
    state   : (..., 12)
    control : (..., 4)
    small_angle : bool
        If True, use polynomial approximations: sin(x) ≈ x - x³/6, cos(x) ≈ 1 - x²/2

    Returns
    -------
    xdot : (..., 12)
    """
    # Unpack state
    vx, vy, vz = state[..., 3], state[..., 4], state[..., 5]
    phi   = state[..., 6]
    theta = state[..., 7]
    psi   = state[..., 8]
    wx    = state[..., 9]
    wy    = state[..., 10]
    wz    = state[..., 11]

    # Unpack control
    f     = control[..., 0]
    tau_x = control[..., 1]
    tau_y = control[..., 2]
    tau_z = control[..., 3]

    m   = params.mass
    Ixx = params.Ixx
    Iyy = params.Iyy
    Izz = params.Izz
    g   = params.g

    # Trigonometric helpers (exact or small-angle approximation)
    if small_angle:
        # Small-angle approximation: sin(x) ≈ x - x³/6, cos(x) ≈ 1 - x²/2
        sphi = phi - (phi**3) / 6.0
        cphi = 1.0 - (phi**2) / 2.0
        sth  = theta - (theta**3) / 6.0
        cth  = 1.0 - (theta**2) / 2.0
        spsi = psi - (psi**3) / 6.0
        cpsi = 1.0 - (psi**2) / 2.0
    else:
        # Exact trig functions
        cphi = torch.cos(phi);   sphi = torch.sin(phi)
        cth  = torch.cos(theta); sth  = torch.sin(theta)
        cpsi = torch.cos(psi);   spsi = torch.sin(psi)

    # ── Position derivative: ṗ = v ──
    dpx = vx
    dpy = vy
    dpz = vz

    # ── Velocity derivative (world-frame thrust + gravity) ──
    #   Third column of R_ZYX(psi,theta,phi):
    #     [cpsi sth cphi + spsi sphi,
    #      spsi sth cphi − cpsi sphi,
    #      cth cphi]
    thrust_per_mass = f / m
    dvx = thrust_per_mass * (cpsi * sth * cphi + spsi * sphi)
    dvy = thrust_per_mass * (spsi * sth * cphi - cpsi * sphi)
    dvz = thrust_per_mass * (cth * cphi) - g

    # ── Euler-angle derivatives ──
    eps = torch.as_tensor(1e-6, device=cth.device, dtype=cth.dtype)
    cth_safe = torch.where(cth.abs() < eps, torch.where(cth >= 0, eps, -eps), cth)
    tth = sth / cth_safe

    dphi   = wx + (wy * sphi + wz * cphi) * tth
    dtheta = wy * cphi - wz * sphi
    dpsi   = (wy * sphi + wz * cphi) / cth_safe

    # ── Angular-rate derivatives (Euler's equation) ──
    dwx = (tau_x - (Izz - Iyy) * wy * wz) / Ixx
    dwy = (tau_y - (Ixx - Izz) * wx * wz) / Iyy
    dwz = (tau_z - (Iyy - Ixx) * wx * wy) / Izz

    return torch.stack([dpx, dpy, dpz,
                        dvx, dvy, dvz,
                        dphi, dtheta, dpsi,
                        dwx, dwy, dwz], dim=-1)


# ── Joint dynamics for two drones ──────────────────────────────────────────

def _joint_continuous_dynamics(
    x: Tensor,
    u: Tensor,
    v: Tensor,
    params: QuadrotorParams,
    small_angle: bool = False,
) -> Tensor:
    """Continuous-time derivative for the joint (24-D) state.

    Parameters
    ----------
    x : (..., 24)
    u : (..., 4)   P1 control
    v : (..., 4)   P2 control
    small_angle : bool
        If True, use polynomial approximations for trig functions

    Returns
    -------
    xdot : (..., 24)
    """
    x1 = x[..., :DX_SINGLE]
    x2 = x[..., DX_SINGLE:]
    xdot1 = _single_quadrotor_dynamics(x1, u, params, small_angle=small_angle)
    xdot2 = _single_quadrotor_dynamics(x2, v, params, small_angle=small_angle)
    return torch.cat([xdot1, xdot2], dim=-1)


# ── Discretization ──────────────────────────────────────────────────────────

def step_euler(
    x: Tensor,
    u: Tensor,
    v: Tensor,
    dt: float,
    params: QuadrotorParams,
    small_angle: bool = False,
) -> Tensor:
    """Forward-Euler integration of the joint dynamics."""
    return x + dt * _joint_continuous_dynamics(x, u, v, params, small_angle=small_angle)


def step_rk4(
    x: Tensor,
    u: Tensor,
    v: Tensor,
    dt: float,
    params: QuadrotorParams,
    small_angle: bool = False,
) -> Tensor:
    """Classical RK4 integration (zero-order hold on controls)."""
    k1 = _joint_continuous_dynamics(x, u, v, params, small_angle=small_angle)
    k2 = _joint_continuous_dynamics(x + 0.5 * dt * k1, u, v, params, small_angle=small_angle)
    k3 = _joint_continuous_dynamics(x + 0.5 * dt * k2, u, v, params, small_angle=small_angle)
    k4 = _joint_continuous_dynamics(x + dt * k3, u, v, params, small_angle=small_angle)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def step_dynamics(
    x: Tensor,
    u: Tensor,
    v: Tensor,
    dt: float,
    params: QuadrotorParams,
    integrator: str = "euler",
    small_angle: bool = False,
) -> Tensor:
    """Dispatch to Euler or RK4."""
    if integrator == "rk4":
        return step_rk4(x, u, v, dt, params, small_angle=small_angle)
    return step_euler(x, u, v, dt, params, small_angle=small_angle)


# ── Linearization via torch.func ────────────────────────────────────────────

def linearize_dynamics(
    x: Tensor,
    u: Tensor,
    v: Tensor,
    dt: float,
    params: QuadrotorParams,
    integrator: str = "euler",
    small_angle: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Linearize the discrete dynamics about a nominal trajectory point.

    Returns the affine approximation:
        x_next ≈ A x + B1 u + B2 v + d

    Uses ``torch.func.jacfwd`` + ``vmap`` for exact, efficient Jacobians.

    Parameters
    ----------
    x  : (B, 24)
    u  : (B, 4)
    v  : (B, 4)
    small_angle : bool
        If True, use polynomial approximations for trig functions

    Returns
    -------
    A  : (B, 24, 24)
    B1 : (B, 24, 4)
    B2 : (B, 24, 4)
    d  : (B, 24)
    """
    step_fn = step_rk4 if integrator == "rk4" else step_euler

    def _f(xi: Tensor, ui: Tensor, vi: Tensor) -> Tensor:
        return step_fn(xi, ui, vi, dt, params, small_angle=small_angle)

    from torch.func import jacfwd, vmap

    A  = vmap(jacfwd(_f, argnums=0))(x, u, v)   # (B, 24, 24)
    B1 = vmap(jacfwd(_f, argnums=1))(x, u, v)   # (B, 24, 4)
    B2 = vmap(jacfwd(_f, argnums=2))(x, u, v)   # (B, 24, 4)

    # torch.func may return Jacobians in a promoted dtype; keep everything in
    # the same dtype/device as the primal tensors so float32 runs remain valid.
    A = A.to(dtype=x.dtype, device=x.device)
    B1 = B1.to(dtype=x.dtype, device=x.device)
    B2 = B2.to(dtype=x.dtype, device=x.device)

    # Affine residual  d = f(x,u,v) − A x − B1 u − B2 v
    x_next = step_fn(x, u, v, dt, params, small_angle=small_angle).to(dtype=x.dtype, device=x.device)
    Ax  = torch.bmm(A,  x.unsqueeze(-1)).squeeze(-1)
    B1u = torch.bmm(B1, u.unsqueeze(-1)).squeeze(-1)
    B2v = torch.bmm(B2, v.unsqueeze(-1)).squeeze(-1)
    d = x_next - Ax - B1u - B2v

    return A, B1, B2, d
