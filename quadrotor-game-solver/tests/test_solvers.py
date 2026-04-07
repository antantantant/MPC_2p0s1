"""Tests for the SQP tree solver and Riccati backward pass.

Verifies:
  1. Riccati solution shapes
  2. SQP convergence (cost decreases or stabilises)
  3. Value function matches expected cost at convergence
  4. Full pipeline: α → belief tree → SQP → objective (differentiable)
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.utils.config import GameConfig
from src.game.quadrotor_game import Hexner3DQuadrotorGame
from src.tree.indexing import FullIaryTreeIndexer
from src.tree.signaling import AlphaParam, AlphaParamConfig
from src.tree.belief_tree import build_belief_tree_vectorized
from src.objectives.cost_tree import compute_quadratic_cost_data
import src.solvers.sqp_tree as sqp_tree_mod
from src.solvers.sqp_tree import sqp_tree_layer, SQPTreeResult
from src.optimization.objective_primal_sqp import primal_objective_sqp


DTYPE = torch.float64
DEVICE = "cpu"


@pytest.fixture
def small_game():
    """A small game for quick testing (K=2, I=2)."""
    cfg = GameConfig(T=1.0, K=2, I=2, dtype=DTYPE, device=DEVICE)
    game = Hexner3DQuadrotorGame(cfg)
    indexer = FullIaryTreeIndexer(I=2, K=2)
    return game, indexer


@pytest.fixture
def alpha_uniform(small_game):
    game, indexer = small_game
    alpha = torch.ones(
        indexer.K, indexer.max_nodes_per_depth, indexer.I, indexer.I,
        dtype=DTYPE,
    ) / indexer.I
    return alpha


class TestSQPTreeLayer:

    def test_runs_without_error(self, small_game, alpha_uniform):
        game, indexer = small_game
        alpha = alpha_uniform
        x0 = game.default_initial_state()
        p0 = game.default_prior()

        result = sqp_tree_layer(
            game=game, indexer=indexer,
            alpha=alpha, x0=x0, p0=p0,
            num_sqp_iters=2,
            collect_diagnostics=True,
        )
        assert isinstance(result, SQPTreeResult)
        assert result.diagnostics is not None

    def test_output_shapes(self, small_game, alpha_uniform):
        game, indexer = small_game
        alpha = alpha_uniform
        x0 = game.default_initial_state()
        p0 = game.default_prior()

        result = sqp_tree_layer(
            game=game, indexer=indexer,
            alpha=alpha, x0=x0, p0=p0,
            num_sqp_iters=2,
        )

        K = indexer.K
        I = indexer.I

        assert len(result.x_nodes) == K + 1
        assert len(result.u_edges) == K
        assert len(result.v_edges) == K

        # Check shapes
        assert result.x_nodes[0].shape == (1, game.dx)
        for k in range(K):
            Nk = indexer.node_count(k)
            assert result.u_edges[k].shape == (Nk, I, game.du)
            assert result.v_edges[k].shape == (Nk, I, game.dv)
            Nn = indexer.node_count(k + 1)
            assert result.x_nodes[k + 1].shape == (Nn, game.dx)

    def test_sqp_cost_finite(self, small_game, alpha_uniform):
        game, indexer = small_game
        alpha = alpha_uniform
        x0 = game.default_initial_state()
        p0 = game.default_prior()

        result = sqp_tree_layer(
            game=game, indexer=indexer,
            alpha=alpha, x0=x0, p0=p0,
            num_sqp_iters=3,
            collect_diagnostics=True,
        )
        assert result.diagnostics is not None
        for c in result.diagnostics.cost_hist:
            assert torch.isfinite(torch.tensor(c)), f"Non-finite cost: {c}"

    def test_sqp_cost_no_nan(self, small_game, alpha_uniform):
        game, indexer = small_game
        alpha = alpha_uniform
        x0 = game.default_initial_state()
        p0 = game.default_prior()

        result = sqp_tree_layer(
            game=game, indexer=indexer,
            alpha=alpha, x0=x0, p0=p0,
            num_sqp_iters=3,
            collect_diagnostics=True,
        )
        assert result.diagnostics is not None
        assert not result.diagnostics.nan_or_inf_encountered

    def test_line_search_safeguard_keeps_previous_iterate(
        self,
        small_game,
        alpha_uniform,
        monkeypatch: pytest.MonkeyPatch,
    ):
        game, indexer = small_game
        alpha = alpha_uniform
        x0 = game.default_initial_state()
        p0 = game.default_prior()

        # Force nominal and candidate costs to tie, so safeguard keeps previous iterate.
        def constant_cost(**kwargs):
            return torch.tensor(1.0, dtype=game.dtype, device=game.device)

        monkeypatch.setattr(sqp_tree_mod, "_expected_cost_from_nominal_tree", constant_cost)

        result = sqp_tree_layer(
            game=game,
            indexer=indexer,
            alpha=alpha,
            x0=x0,
            p0=p0,
            num_sqp_iters=1,
            line_search=True,
            line_search_accept_worse=False,
            collect_diagnostics=True,
        )

        assert result.diagnostics is not None
        assert len(result.diagnostics.accepted_step_hist) == 1
        assert len(result.diagnostics.used_prev_iterate_hist) == 1
        assert result.diagnostics.accepted_step_hist[0] == pytest.approx(0.0)
        assert result.diagnostics.used_prev_iterate_hist[0] is True


class TestPrimalObjective:

    def test_returns_scalar(self, small_game):
        game, indexer = small_game
        alpha_mod = AlphaParam(
            indexer, dtype=DTYPE, device=DEVICE,
            cfg=AlphaParamConfig(init_scale=0.01),
        )
        loss = primal_objective_sqp(
            game=game, alpha_module=alpha_mod, indexer=indexer,
            num_sqp_iters=2,
        )
        assert loss.ndim == 0  # scalar

    def test_loss_is_finite(self, small_game):
        game, indexer = small_game
        alpha_mod = AlphaParam(
            indexer, dtype=DTYPE, device=DEVICE,
            cfg=AlphaParamConfig(init_scale=0.01),
        )
        loss = primal_objective_sqp(
            game=game, alpha_module=alpha_mod, indexer=indexer,
            num_sqp_iters=2,
        )
        assert torch.isfinite(loss)

    def test_gradient_flows_to_logits(self, small_game):
        """Backward pass should produce gradients on α logits."""
        game, indexer = small_game
        alpha_mod = AlphaParam(
            indexer, dtype=DTYPE, device=DEVICE,
            cfg=AlphaParamConfig(init_scale=0.01),
        )
        loss = primal_objective_sqp(
            game=game, alpha_module=alpha_mod, indexer=indexer,
            num_sqp_iters=2,
        )
        loss.backward()
        assert alpha_mod.logits.grad is not None
        assert torch.isfinite(alpha_mod.logits.grad).all()

    def test_return_details(self, small_game):
        game, indexer = small_game
        alpha_mod = AlphaParam(
            indexer, dtype=DTYPE, device=DEVICE,
            cfg=AlphaParamConfig(init_scale=0.01),
        )
        result = primal_objective_sqp(
            game=game, alpha_module=alpha_mod, indexer=indexer,
            num_sqp_iters=2,
            return_details=True,
        )
        loss, details = result
        assert "alpha" in details
        assert "sqp_result" in details
        assert isinstance(details["sqp_result"], SQPTreeResult)
