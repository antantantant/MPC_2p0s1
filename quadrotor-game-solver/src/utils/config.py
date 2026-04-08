"""Configuration dataclasses for the 3-D Hexner game with quadrotor dynamics.

Mirrors ``config/hexner_config.py`` and ``config/base_config.py`` in the parent
LQ project, but with quadrotor-specific parameters.

State layout (per player, 12-D):
    x_j = [px, py, pz, vx, vy, vz, phi, theta, psi, wx, wy, wz]

Joint state (24-D):
    x = [x_P1 (12), x_P2 (12)]

Control (per player, 4-D):
    u_j = [f, tau_x, tau_y, tau_z]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


# ── Physical parameters ──────────────────────────────────────────────────

@dataclass
class QuadrotorParams:
    """Physical parameters of a single quadrotor drone."""

    mass: float = 1.0
    Ixx: float = 0.01
    Iyy: float = 0.01
    Izz: float = 0.02
    g: float = 9.81
    arm_length: float = 0.25


# ── Game configuration ───────────────────────────────────────────────────

@dataclass
class GameConfig:
    """Configuration for the 3-D Hexner game with quadrotor dynamics.

    Cost convention (same as original Hexner):

        Running cost:  ℓ(u,v) = (τ/2) u^T R u  −  (τ/2) v^T S v
            where R = 2·diag(R1_diag),  S = 2·diag(R2_diag)  (factor 2 from the
            ½ u^T R u convention matching ||u||²_{R1}).

        Terminal cost:
            g_i(x) = ||pos1 − z·θ_i||²_{K1}  −  ||pos2 − z·θ_i||²_{K2}
            expanded into  ½ x^T Q_i x + q_i^T x + c_i.
    """

    # ── Information structure ────────────────────────────────────────
    I: int = 2                          # number of payoff types

    # ── Time horizon ─────────────────────────────────────────────────
    T: float = 2.0                      # continuous-time horizon (s)
    K: int = 10                         # number of discrete steps

    # ── Physics ──────────────────────────────────────────────────────
    quad_params: QuadrotorParams = field(default_factory=QuadrotorParams)
    dynamics_model: str = "rigid_body"  # "rigid_body" or "interception"
    payoff_model: str = "hexner"       # "hexner" or "hexner_mod"

    # ── Integrator ───────────────────────────────────────────────────
    integrator: str = "euler"           # "euler" or "rk4"

    # ── Linearization mode ───────────────────────────────────────────
    linearized_mode: bool = False       # If True, freeze linearization at hover
                                         # (makes problem LQ for debugging)

    # ── Small-angle approximation ────────────────────────────────────
    small_angle_approx: bool = False    # If True, use polynomial approx for trig:
                                         # sin(x) ≈ x - x³/6, cos(x) ≈ 1 - x²/2
                                         # (still nonlinear but smoother)

    # ── Control representation / cost convention ─────────────────────
    # "hover_relative": optimization controls represent deltas around hover.
    #                   Running-cost penalty is on these deltas.
    # "absolute": optimization controls are physical controls directly.
    control_cost_mode: str = "hover_relative"

    # ── SQP line-search safety default ───────────────────────────────
    # If False, keep previous iterate when all tested line-search steps
    # worsen the current nominal cost.
    line_search_accept_worse: bool = False

    # ── Device / dtype ───────────────────────────────────────────────
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

    # ── Running cost diagonals  (before ×2) ──────────────────────────
    #   u = [f, tau_x, tau_y, tau_z]
    R1_diag: Tuple[float, ...] = (0.05, 0.025, 0.025, 0.01)
    R2_diag: Tuple[float, ...] = (0.05, 0.10,  0.10,  0.02)

    # ── Terminal cost ────────────────────────────────────────────────
    K1_scale: float = 1.0
    K2_scale: float = 1.0
    theta_values: Tuple[float, ...] = (-1.0, 1.0)
    target_z: Optional[Tuple[float, ...]] = None
    hexner_mod_type_state_weights: Optional[Tuple[Tuple[float, ...], ...]] = None
    interception_state_weights: Tuple[float, ...] = (
        1.0, 1.0, 1.0,
        0.25, 0.25, 0.25,
        0.10, 0.10, 0.10,
    )
    #   If None → default z concentrates on z-position:
    #   z = (0,0,1,  0,0,0,  0,0,0,  0,0,0)  so targets are at ±1 on the z-axis.

    # ── Control bounds ───────────────────────────────────────────────
    u_min: float = -10.0
    u_max: float = 10.0
    v_min: float = -10.0
    v_max: float = 10.0

    # ── Misc ─────────────────────────────────────────────────────────
    game_name: str = "hexner3d_quadrotor"

    @property
    def tau(self) -> float:
        return self.T / self.K

    @property
    def device_resolved(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def __post_init__(self) -> None:
        if self.control_cost_mode not in {"hover_relative", "absolute"}:
            raise ValueError(
                "control_cost_mode must be one of {'hover_relative', 'absolute'}, "
                f"got {self.control_cost_mode!r}"
            )
        if self.dynamics_model not in {"rigid_body", "interception"}:
            raise ValueError(
                "dynamics_model must be one of {'rigid_body', 'interception'}, "
                f"got {self.dynamics_model!r}"
            )
        if self.payoff_model not in {"hexner", "hexner_mod"}:
            raise ValueError(
                "payoff_model must be one of {'hexner', 'hexner_mod'}, "
                f"got {self.payoff_model!r}"
            )


# ── Convenience constructors ─────────────────────────────────────────────


def make_hexner3d_game_config(
    *,
    T: float = 2.0,
    K: int = 10,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> GameConfig:
    """Produce a default ``GameConfig`` for the 3-D Hexner quadrotor game."""
    return GameConfig(T=T, K=K, device=device, dtype=dtype)
