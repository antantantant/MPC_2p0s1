"""Discrete-time jerk-integrator interception dynamics from Mueller-D'Andrea.

This model follows the paper:

    Mark W. Mueller and Raffaello D'Andrea,
    "A Model Predictive Controller for Quadrocopter State Interception", ECC 2013.

The translational motion on each axis is modeled as a triple integrator with
jerk input. For one player, the state is

    x_j = [px, py, pz, vx, vy, vz, ax, ay, az]

and the control is

    u_j = [jx, jy, jz].

The discrete-time dynamics with zero-order hold over dt are exact:

    p⁺ = p + dt v + 0.5 dt² a + (1/6) dt³ j
    v⁺ = v + dt a + 0.5 dt² j
    a⁺ = a + dt j

The joint two-player state is the concatenation of the two single-player
states. The resulting dynamics are linear time invariant, so the "linearize"
step returns constant A, B1, B2 and zero affine residual d.
"""

from __future__ import annotations

import torch
from torch import Tensor


DX_SINGLE: int = 9
DU_SINGLE: int = 3
DX_JOINT: int = 18
DU: int = 3
DV: int = 3


def _single_player_step(state: Tensor, jerk: Tensor, dt: float) -> Tensor:
    """Exact discrete-time triple-integrator update for one player."""
    p = state[..., 0:3]
    v = state[..., 3:6]
    a = state[..., 6:9]

    dt_t = torch.as_tensor(dt, dtype=state.dtype, device=state.device)
    dt2 = dt_t * dt_t
    dt3 = dt2 * dt_t

    p_next = p + dt_t * v + 0.5 * dt2 * a + (dt3 / 6.0) * jerk
    v_next = v + dt_t * a + 0.5 * dt2 * jerk
    a_next = a + dt_t * jerk
    return torch.cat([p_next, v_next, a_next], dim=-1)


def step_dynamics(
    x: Tensor,
    u: Tensor,
    v: Tensor,
    dt: float,
) -> Tensor:
    """Exact discrete-time joint update."""
    x1 = x[..., :DX_SINGLE]
    x2 = x[..., DX_SINGLE:]
    x1_next = _single_player_step(x1, u, dt)
    x2_next = _single_player_step(x2, v, dt)
    return torch.cat([x1_next, x2_next], dim=-1)


def _constant_linearization(
    *,
    dt: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Constant discrete-time linear dynamics matrices."""
    dt_t = torch.as_tensor(dt, dtype=dtype, device=device)
    eye3 = torch.eye(3, dtype=dtype, device=device)
    zero33 = torch.zeros(3, 3, dtype=dtype, device=device)

    A_single = torch.cat(
        [
            torch.cat([eye3, dt_t * eye3, 0.5 * (dt_t ** 2) * eye3], dim=1),
            torch.cat([zero33, eye3, dt_t * eye3], dim=1),
            torch.cat([zero33, zero33, eye3], dim=1),
        ],
        dim=0,
    )
    B_single = torch.cat(
        [
            (dt_t ** 3 / 6.0) * eye3,
            0.5 * (dt_t ** 2) * eye3,
            dt_t * eye3,
        ],
        dim=0,
    )

    A = torch.block_diag(A_single, A_single)
    zero93 = torch.zeros(DX_SINGLE, DU_SINGLE, dtype=dtype, device=device)
    B1 = torch.cat([B_single, zero93], dim=0)
    B2 = torch.cat([zero93, B_single], dim=0)
    return A, B1, B2


def linearize_dynamics(
    x: Tensor,
    u: Tensor,
    v: Tensor,
    dt: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return the exact affine linearization x⁺ = A x + B1 u + B2 v + d."""
    batch = x.shape[0]
    dtype = x.dtype
    device = x.device

    A_single, B1_single, B2_single = _constant_linearization(
        dt=dt, dtype=dtype, device=device
    )
    A = A_single.unsqueeze(0).expand(batch, -1, -1)
    B1 = B1_single.unsqueeze(0).expand(batch, -1, -1)
    B2 = B2_single.unsqueeze(0).expand(batch, -1, -1)
    d = torch.zeros(batch, DX_JOINT, dtype=dtype, device=device)
    return A, B1, B2, d
