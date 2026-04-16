# games/drone_game.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from ..core.types import Tensor
from ..config.base_config import GameConfig


def _euler_to_rot(angles: Tensor) -> Tensor:
    """
    angles: (...,3) roll, pitch, yaw
    returns: (...,3,3) rotation matrix body->world
    """
    roll = angles[..., 0]
    pitch = angles[..., 1]
    yaw = angles[..., 2]

    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)

    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr

    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr

    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    R = torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )
    return R


def _euler_rates(angles: Tensor, omega_body: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Map body rates (p,q,r) to Euler angle derivatives (phi_dot, theta_dot, psi_dot).
    angles: (...,3) roll, pitch, yaw
    omega_body: (...,3) p,q,r
    """
    phi = angles[..., 0]
    theta = angles[..., 1]
    p = omega_body[..., 0]
    q = omega_body[..., 1]
    r = omega_body[..., 2]

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    tan_t = sin_t / (cos_t + eps)

    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)

    phi_dot = p + q * sin_phi * tan_t + r * cos_phi * tan_t
    theta_dot = q * cos_phi - r * sin_phi
    psi_dot = q * sin_phi / (cos_t + eps) + r * cos_phi / (cos_t + eps)

    return torch.stack([phi_dot, theta_dot, psi_dot], dim=-1)


@dataclass
class DroneParams:
    """
    Parameters for the 2-drone nonlinear game.

    Cost convention (zero-sum):
      P1 minimizes:  sum 0.5 u^T R u  - 0.5 v^T S v  + terminal(x)
      P2 maximizes the same objective.
    """
    mass: float = 1.0
    gravity: float = 9.81
    inertia_diag: Tuple[float, float, float] = (0.02, 0.02, 0.04)

    # Type-dependent control penalties
    # R_types: (I, du, du), S_types: (I, dv, dv)
    R_types: Tensor | None = None
    S_types: Tensor | None = None

    # Type-dependent terminal quadratic:
    # Qf_types: (I, dx, dx), x_goal_types: (I, dx)
    Qf_types: Tensor | None = None
    x_goal_types: Tensor | None = None


class DroneGame:
    """
    Two-drone nonlinear dynamics game with belief-averaged quadratic costs.
    """

    def __init__(self, cfg: GameConfig, params: DroneParams):
        self.cfg = cfg
        self.params = params

        # Dimensions
        self.I = int(cfg.I)
        self.du = int(cfg.du)
        self.dv = int(cfg.dv)
        self.dx = int(cfg.dx)

        if self.dx != 24:
            raise ValueError("DroneGame expects dx=24 (12 per drone).")
        if self.du != 4 or self.dv != 4:
            raise ValueError("DroneGame expects du=dv=4 (thrust + 3 torques).")

        self.device_resolved = cfg.device_resolved
        self.dtype = cfg.dtype

        # Default cost tensors if not provided
        device = self.device_resolved
        dtype = self.dtype

        if params.R_types is None:
            self.R_types = torch.eye(self.du, device=device, dtype=dtype).unsqueeze(0).repeat(self.I, 1, 1)
        else:
            self.R_types = params.R_types.to(device=device, dtype=dtype)

        if params.S_types is None:
            self.S_types = torch.eye(self.dv, device=device, dtype=dtype).unsqueeze(0).repeat(self.I, 1, 1)
        else:
            self.S_types = params.S_types.to(device=device, dtype=dtype)

        if params.Qf_types is None:
            # Default: penalize positions heavily, mild velocity, ignore angles/omega.
            Q = torch.zeros(self.dx, self.dx, device=device, dtype=dtype)

            # indices: drone1 pos(0:3), vel(3:6), angles(6:9), omega(9:12)
            #          drone2 pos(12:15), ...
            w_pos = 50.0
            w_vel = 2.0

            for base in (0, 12):
                Q[base + 0:base + 3, base + 0:base + 3] = w_pos * torch.eye(3, device=device, dtype=dtype)
                Q[base + 3:base + 6, base + 3:base + 6] = w_vel * torch.eye(3, device=device, dtype=dtype)

            self.Qf_types = Q.unsqueeze(0).repeat(self.I, 1, 1)  # same for all types by default
        else:
            self.Qf_types = params.Qf_types.to(device=device, dtype=dtype)

        if params.x_goal_types is None:
            # Default: type-specific P1 goal positions; P2 goal stays at origin.
            goals = torch.zeros(self.I, self.dx, device=device, dtype=dtype)
            # Drone2 position goal fixed at origin; type affects Drone1 pos goal:
            for i in range(self.I):
                goals[i, 0:3] = torch.tensor([2.0 * i, 0.0, 1.0], device=device, dtype=dtype)
                goals[i, 12:15] = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
            self.x_goal_types = goals
        else:
            self.x_goal_types = params.x_goal_types.to(device=device, dtype=dtype)

        if self.R_types.shape != (self.I, self.du, self.du):
            raise ValueError(f"R_types shape must be {(self.I, self.du, self.du)}, got {tuple(self.R_types.shape)}")
        if self.S_types.shape != (self.I, self.dv, self.dv):
            raise ValueError(f"S_types shape must be {(self.I, self.dv, self.dv)}, got {tuple(self.S_types.shape)}")
        if self.Qf_types.shape != (self.I, self.dx, self.dx):
            raise ValueError(f"Qf_types shape must be {(self.I, self.dx, self.dx)}, got {tuple(self.Qf_types.shape)}")
        if self.x_goal_types.shape != (self.I, self.dx):
            raise ValueError(f"x_goal_types shape must be {(self.I, self.dx)}, got {tuple(self.x_goal_types.shape)}")

        # precompute terminal constants per type: 0.5 xg^T Q xg
        self._c_term_types = 0.5 * torch.einsum("id,idd,ie->i", self.x_goal_types, self.Qf_types, self.x_goal_types)

    def default_prior(self) -> Tensor:
        p = torch.ones(self.I, device=self.device_resolved, dtype=self.dtype)
        return p / p.sum()

    def default_initial_state(self) -> Tensor:
        # hover-ish initial condition for both drones
        x0 = torch.zeros(self.dx, device=self.device_resolved, dtype=self.dtype)
        x0[2] = 1.0      # drone1 z
        x0[14] = 1.0     # drone2 z
        return x0

    # -------------------------- Dynamics -------------------------------- #

    def _single_drone_step(self, x: Tensor, u: Tensor) -> Tensor:
        """
        x: (...,12) = [pos3 vel3 euler3 omega3]
        u: (...,4)  = [thrust, tau_x, tau_y, tau_z]
        returns: (...,12)
        """
        dt = float(self.cfg.tau)
        m = float(self.params.mass)
        g = float(self.params.gravity)

        I_diag = torch.tensor(self.params.inertia_diag, device=x.device, dtype=x.dtype)  # (3,)

        pos = x[..., 0:3]
        vel = x[..., 3:6]
        angles = x[..., 6:9]
        omega = x[..., 9:12]

        thrust = u[..., 0]          # (...)
        tau = u[..., 1:4]           # (...,3)

        R = _euler_to_rot(angles)   # (...,3,3)
        f_body = torch.stack(
            [torch.zeros_like(thrust), torch.zeros_like(thrust), thrust],
            dim=-1,
        )                            # (...,3)
        f_world = (R @ f_body.unsqueeze(-1)).squeeze(-1)  # (...,3)

        acc = f_world / m - torch.tensor([0.0, 0.0, g], device=x.device, dtype=x.dtype)

        angles_dot = _euler_rates(angles, omega)

        Iomega = omega * I_diag
        omega_cross_Iomega = torch.cross(omega, Iomega, dim=-1)
        omega_dot = (tau - omega_cross_Iomega) / I_diag

        pos_next = pos + dt * vel
        vel_next = vel + dt * acc
        angles_next = angles + dt * angles_dot
        omega_next = omega + dt * omega_dot

        return torch.cat([pos_next, vel_next, angles_next, omega_next], dim=-1)

    def step_dynamics(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """
        Joint dynamics for both drones.

        x: (...,24)  -> [x1(12), x2(12)]
        u: (...,4)   -> control for drone1
        v: (...,4)   -> control for drone2
        """
        x1 = x[..., 0:12]
        x2 = x[..., 12:24]

        x1_next = self._single_drone_step(x1, u)
        x2_next = self._single_drone_step(x2, v)

        return torch.cat([x1_next, x2_next], dim=-1)

    # -------------------------- Costs ----------------------------------- #

    def running_cost_mats(self, belief: Tensor) -> Tuple[Tensor, Tensor]:
        """
        belief: (..., I)
        returns:
          R_bar: (..., du, du)
          S_bar: (..., dv, dv)
        """
        # Weighted sum over types
        R_bar = torch.einsum("...i,iab->...ab", belief, self.R_types)
        S_bar = torch.einsum("...i,iab->...ab", belief, self.S_types)
        return R_bar, S_bar

    def terminal_value_quads(self, belief: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Terminal value in quadratic form:
          V(x) = 0.5 x^T P x + r^T x + c
        with belief-averaging over payoff types.

        belief: (..., I)
        returns:
          P: (..., dx, dx)
          r: (..., dx)
          c: (...,)
        """
        P = torch.einsum("...i,iab->...ab", belief, self.Qf_types)  # (...,dx,dx)
        # b = sum_i p_i Q_i x_goal_i  -> r = -b
        b = torch.einsum("...i,iab,ib->...a", belief, self.Qf_types, self.x_goal_types)  # (...,dx)
        r = -b
        c = torch.einsum("...i,i->...", belief, self._c_term_types)  # (...,)
        return P, r, c
