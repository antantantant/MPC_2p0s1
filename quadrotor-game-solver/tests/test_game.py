"""Tests for the Hexner3DQuadrotorGame class.

Verifies:
  1. Cost matrix shapes and symmetry
  2. Terminal cost quadratic structure
  3. Belief-averaged costs
  4. Default hover / initial state
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.utils.config import GameConfig
from src.game.quadrotor_game import Hexner3DQuadrotorGame


DTYPE = torch.float64
DEVICE = "cpu"


@pytest.fixture
def game() -> Hexner3DQuadrotorGame:
    cfg = GameConfig(T=2.0, K=10, I=2, dtype=DTYPE, device=DEVICE)
    return Hexner3DQuadrotorGame(cfg)


class TestCostMatrices:

    def test_R_shape(self, game: Hexner3DQuadrotorGame):
        assert game.R.shape == (game.I, game.du, game.du)

    def test_S_shape(self, game: Hexner3DQuadrotorGame):
        assert game.S.shape == (game.I, game.dv, game.dv)

    def test_Q_shape(self, game: Hexner3DQuadrotorGame):
        assert game.Q.shape == (game.I, game.dx, game.dx)

    def test_q_shape(self, game: Hexner3DQuadrotorGame):
        assert game.q.shape == (game.I, game.dx)

    def test_c_shape(self, game: Hexner3DQuadrotorGame):
        assert game.c.shape == (game.I,)

    def test_R_symmetric(self, game: Hexner3DQuadrotorGame):
        R = game.R
        torch.testing.assert_close(R, R.transpose(-1, -2))

    def test_S_symmetric(self, game: Hexner3DQuadrotorGame):
        S = game.S
        torch.testing.assert_close(S, S.transpose(-1, -2))

    def test_Q_symmetric(self, game: Hexner3DQuadrotorGame):
        Q = game.Q
        torch.testing.assert_close(Q, Q.transpose(-1, -2))

    def test_R_diagonal_values(self, game: Hexner3DQuadrotorGame):
        """R = 2 · diag(R1_diag) → diagonal entries = 2 × (0.05, 0.025, 0.025, 0.01)."""
        R0 = game.R[0]
        expected_diag = torch.tensor(
            [2 * 0.05, 2 * 0.025, 2 * 0.025, 2 * 0.01],
            dtype=DTYPE,
        )
        torch.testing.assert_close(R0.diag(), expected_diag)

    def test_R_type_independent(self, game: Hexner3DQuadrotorGame):
        """R is the same for all types (Hexner convention)."""
        torch.testing.assert_close(game.R[0], game.R[1])

    def test_Q_terminal_structure(self, game: Hexner3DQuadrotorGame):
        """Q_i = 2·diag(K1, -K2); K1 = K2 = I_3 in first 3 dims."""
        Q0 = game.Q[0]
        # P1 block: top-left 12×12 should have K1_scale in (0,0),(1,1),(2,2)
        assert float(Q0[0, 0]) == pytest.approx(2.0 * game.cfg.K1_scale)
        assert float(Q0[1, 1]) == pytest.approx(2.0 * game.cfg.K1_scale)
        assert float(Q0[2, 2]) == pytest.approx(2.0 * game.cfg.K1_scale)
        assert float(Q0[3, 3]) == pytest.approx(0.0)
        # P2 block: entries at (12,12) etc. should be -2 K2_scale
        assert float(Q0[12, 12]) == pytest.approx(-2.0 * game.cfg.K2_scale)
        assert float(Q0[13, 13]) == pytest.approx(-2.0 * game.cfg.K2_scale)
        assert float(Q0[14, 14]) == pytest.approx(-2.0 * game.cfg.K2_scale)
        assert float(Q0[15, 15]) == pytest.approx(0.0)


class TestTerminalCost:

    def test_terminal_evaluated_at_target(self, game: Hexner3DQuadrotorGame):
        """When both drones sit at the target z·θ_i, terminal cost should
        have a specific known value based on the quadratic form."""
        # For type 0 (θ=-1), target z-pos = -1
        # If both drones are at z=-1 with zero everything else:
        x = torch.zeros(game.dx, dtype=DTYPE)
        x[2] = -1.0   # P1 at z=-1 (which is z·θ_0 for θ_0=-1)
        x[14] = -1.0  # P2 at z=-1

        Q0, q0, c0 = game.Q[0], game.q[0], game.c[0]
        g = 0.5 * x @ Q0 @ x + q0 @ x + c0

        # g_0(x) = ||p1 - z·θ_0||²_K1 - ||p2 - z·θ_0||²_K2
        # Both at target → both norms are 0 → g = 0
        assert float(g.item()) == pytest.approx(0.0, abs=1e-10)


class TestBeliefAveraged:

    def test_running_cost_uniform_belief(self, game: Hexner3DQuadrotorGame):
        """With uniform belief, R̄ = R (since type-indep)."""
        belief = torch.tensor([0.5, 0.5], dtype=DTYPE)
        R_bar, S_bar = game.running_cost_mats(belief)
        torch.testing.assert_close(R_bar, game.R[0])
        torch.testing.assert_close(S_bar, game.S[0])

    def test_terminal_batch(self, game: Hexner3DQuadrotorGame):
        B = 4
        beliefs = torch.rand(B, game.I, dtype=DTYPE)
        beliefs = beliefs / beliefs.sum(dim=-1, keepdim=True)
        P, r, c = game.terminal_value_quad_batch(beliefs)
        assert P.shape == (B, game.dx, game.dx)
        assert r.shape == (B, game.dx)
        assert c.shape == (B,)


class TestDefaults:

    def test_default_initial_state_shape(self, game: Hexner3DQuadrotorGame):
        x0 = game.default_initial_state()
        assert x0.shape == (game.dx,)

    def test_default_hover_control_shape(self, game: Hexner3DQuadrotorGame):
        u0, v0 = game.default_hover_control()
        assert u0.shape == (game.du,)
        assert v0.shape == (game.dv,)

    def test_hover_relative_default_is_zero_delta(self, game: Hexner3DQuadrotorGame):
        u0, v0 = game.default_hover_control()
        torch.testing.assert_close(u0, torch.zeros_like(u0))
        torch.testing.assert_close(v0, torch.zeros_like(v0))

    def test_hover_physical_control_is_mg(self, game: Hexner3DQuadrotorGame):
        u_phys, v_phys = game.hover_physical_control()
        mg = game.cfg.quad_params.mass * game.cfg.quad_params.g
        assert float(u_phys[0].item()) == pytest.approx(mg)
        assert float(v_phys[0].item()) == pytest.approx(mg)

    def test_absolute_mode_default_hover_is_physical(self):
        game_abs = Hexner3DQuadrotorGame(
            GameConfig(
                T=2.0,
                K=10,
                I=2,
                dtype=DTYPE,
                device=DEVICE,
                control_cost_mode="absolute",
            )
        )
        u0, v0 = game_abs.default_hover_control()
        mg = game_abs.cfg.quad_params.mass * game_abs.cfg.quad_params.g
        assert float(u0[0].item()) == pytest.approx(mg)
        assert float(v0[0].item()) == pytest.approx(mg)

    def test_default_prior(self, game: Hexner3DQuadrotorGame):
        p0 = game.default_prior()
        assert p0.shape == (game.I,)
        assert float(p0.sum().item()) == pytest.approx(1.0)

    def test_type_target_positions(self, game: Hexner3DQuadrotorGame):
        tgt = game.type_target_positions()
        assert tgt.shape == (game.I, 3)
        # θ = (-1, 1), z-direction is z-axis → targets at z = ±1
        assert float(tgt[0, 2].item()) == pytest.approx(-1.0)
        assert float(tgt[1, 2].item()) == pytest.approx(1.0)


class TestDynamicsIntegration:

    def test_step_output_shape(self, game: Hexner3DQuadrotorGame):
        x = game.default_initial_state()
        u, v = game.default_hover_control()
        x_next = game.step(x, u, v)
        assert x_next.shape == (game.dx,)

    def test_linearize_output_shapes(self, game: Hexner3DQuadrotorGame):
        x = game.default_initial_state().unsqueeze(0)
        u, v = game.default_hover_control()
        u = u.unsqueeze(0)
        v = v.unsqueeze(0)
        A, B1, B2, d = game.linearize(x, u, v)
        assert A.shape == (1, game.dx, game.dx)
        assert B1.shape == (1, game.dx, game.du)
        assert B2.shape == (1, game.dx, game.dv)
        assert d.shape == (1, game.dx)


class TestHoverRelativeSemantics:

    def test_zero_delta_running_cost_is_zero(self, game: Hexner3DQuadrotorGame):
        belief = torch.tensor([0.5, 0.5], dtype=DTYPE)
        R_bar, S_bar = game.running_cost_mats(belief)
        u0, v0 = game.default_hover_control()
        stage = 0.5 * (u0 @ R_bar @ u0) - 0.5 * (v0 @ S_bar @ v0)
        assert float(stage.item()) == pytest.approx(0.0, abs=1e-12)

    def test_hover_rollout_stays_near_initial_altitude(self, game: Hexner3DQuadrotorGame):
        x = game.default_initial_state()
        u0, v0 = game.default_hover_control()
        z0_p1 = float(x[2].item())
        z0_p2 = float(x[14].item())

        for _ in range(5):
            x = game.step(x, u0, v0)

        assert abs(float(x[2].item()) - z0_p1) < 1e-2
        assert abs(float(x[14].item()) - z0_p2) < 1e-2
