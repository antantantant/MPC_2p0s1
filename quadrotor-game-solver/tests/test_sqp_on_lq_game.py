#!/usr/bin/env python3
"""Test SQP solver on the 2D LQ Hexner game from parent directory.

This verifies that the SQP solver works correctly on a known LQ game.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Import parent LQ game (use package name MPC_2p0s1)
from MPC_2p0s1.games.hexner_game import HexnerGame
from MPC_2p0s1.config.base_config import GameConfig as LQGameConfig

# Import quadrotor SQP solver
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.tree.indexing import FullIaryTreeIndexer
from src.tree.signaling import AlphaParam, AlphaParamConfig
from src.solvers.sqp_tree import sqp_tree_layer


class LQGameAdapter:
    """Adapter to make parent LQ game work with quadrotor SQP solver."""

    def __init__(self, lq_game: HexnerGame):
        self.lq_game = lq_game
        self.cfg = lq_game.cfg
        self.device = lq_game.device_resolved
        self.dtype = lq_game.dtype
        self.dx = lq_game.dx
        self.du = lq_game.du
        self.dv = lq_game.dv
        self.I = lq_game.I

    def step(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Linear dynamics: x_next = A·x + B1·u + B2·v"""
        return self.lq_game.step_dynamics(x, u, v)

    def linearize(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return constant A, B1, B2 and zero d (pure linear system)."""
        B = x.shape[0]
        A = self.lq_game.A.unsqueeze(0).expand(B, -1, -1)
        B1 = self.lq_game.B1.unsqueeze(0).expand(B, -1, -1)
        B2 = self.lq_game.B2.unsqueeze(0).expand(B, -1, -1)
        d = torch.zeros(B, self.dx, device=self.device, dtype=self.dtype)
        return A, B1, B2, d

    @property
    def R(self) -> torch.Tensor:
        return self.lq_game.R

    @property
    def S(self) -> torch.Tensor:
        return self.lq_game.S

    @property
    def Q(self) -> torch.Tensor:
        return self.lq_game.Q

    @property
    def q(self) -> torch.Tensor:
        return self.lq_game.q

    @property
    def c(self) -> torch.Tensor:
        return self.lq_game.c

    def running_cost_mats(self, belief: torch.Tensor):
        return self.lq_game.running_cost_mats(belief)

    def terminal_cost_quad(self, belief: torch.Tensor):
        """Belief-averaged terminal cost (Q̄, q̄, c̄)."""
        # LQ game computes: Q̄ = Σ_i p[i] Q_i, q̄ = Σ_i p[i] q_i, c̄ = Σ_i p[i] c_i
        Q_bar = torch.einsum("...i, iab -> ...ab", belief, self.lq_game.Q)
        q_bar = torch.einsum("...i, ia -> ...a", belief, self.lq_game.q)
        c_bar = torch.einsum("...i, i -> ...", belief, self.lq_game.c)
        return Q_bar, q_bar, c_bar

    def stage_cost_mats_batch(
        self, beliefs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorised running-cost data for a batch of beliefs."""
        B = beliefs.shape[0]
        # For Hexner game: Q=0, q=0, c=0 in running cost (control-only)
        Q = torch.zeros(self.dx, self.dx, device=self.device, dtype=self.dtype)
        q = torch.zeros(B, self.dx, device=self.device, dtype=self.dtype)
        c = torch.zeros(B, device=self.device, dtype=self.dtype)
        R_bar, S_bar = self.running_cost_mats(beliefs)
        return Q, q, c, R_bar, S_bar

    def terminal_value_quad_batch(
        self, beliefs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorised terminal cost for a batch of beliefs."""
        return self.terminal_cost_quad(beliefs)

    def default_initial_state(self) -> torch.Tensor:
        return self.lq_game.default_initial_state()

    def default_prior(self) -> torch.Tensor:
        return self.lq_game.default_prior()

    def default_hover_control(self) -> tuple[torch.Tensor, torch.Tensor]:
        """For LQ double-integrator, 'hover' is zero acceleration."""
        u0 = torch.zeros(self.du, device=self.device, dtype=self.dtype)
        v0 = torch.zeros(self.dv, device=self.device, dtype=self.dtype)
        return u0, v0


def main():
    print("=" * 80)
    print("Testing SQP Solver on 2D LQ Hexner Game (I=2)")
    print("=" * 80)

    # Create 2D LQ Hexner game (same as parent)
    lq_cfg = LQGameConfig(
        dx1=4,  # 2D position + velocity per player
        dx2=4,
        du=2,  # 2D acceleration
        dv=2,
        I=2,
        K=10,
        T=1.0,
        dtype=torch.float64,
        device="cpu",
    )

    lq_game = HexnerGame(lq_cfg)
    game = LQGameAdapter(lq_game)

    print(f"\nLQ Game created:")
    print(f"  dx={game.dx}, du={game.du}, dv={game.dv}, I={game.I}")
    print(f"  K={game.cfg.K}, T={game.cfg.T}, tau={game.cfg.tau}")

    # Create indexer and alpha param
    indexer = FullIaryTreeIndexer(K=lq_cfg.K, I=lq_cfg.I)
    alpha_param = AlphaParam(
        indexer=indexer,
        cfg=AlphaParamConfig(init_scale=0.01),
        device=game.device,
        dtype=game.dtype,
    )
    alpha = alpha_param()
    prior = game.default_prior()
    x0 = game.default_initial_state()

    print(f"\nInitial state x0:")
    print(f"  P1: {x0[:4].tolist()}")
    print(f"  P2: {x0[4:].tolist()}")
    print(f"Prior: {prior.tolist()}")

    # Run SQP
    print("\n" + "=" * 80)
    print("Running SQP on LQ game (should converge quickly)")
    print("=" * 80)

    sqp_result = sqp_tree_layer(
        game=game,
        indexer=indexer,
        alpha=alpha,
        x0=x0,
        p0=prior,
        num_sqp_iters=10,
        step_size=1.0,
        riccati_reg=1e-6,
        verbose=True,
        early_stop=True,
        line_search=False,
        collect_diagnostics=True,
    )

    # Print results
    print("\n" + "=" * 80)
    print("SQP Results")
    print("=" * 80)

    if sqp_result.diagnostics is not None:
        print(f"Converged: {sqp_result.diagnostics.converged}")
        print(f"Converged at iteration: {sqp_result.diagnostics.converged_iter}")
        print(f"Final cost: {sqp_result.diagnostics.cost_hist[-1]:.6f}")
        print(f"Cost history: {[f'{c:.6f}' for c in sqp_result.diagnostics.cost_hist]}")

    # Print trajectory
    x_root_0 = sqp_result.x_nodes[0][0]
    x_root_K = sqp_result.x_nodes[-1][0]
    print(f"\nRoot trajectory:")
    print(f"  x[0]: {x_root_0.tolist()}")
    print(f"  x[K]: {x_root_K.tolist()}")

    # Sanity checks
    print("\n" + "=" * 80)
    print("Sanity checks:")
    print("=" * 80)

    if sqp_result.diagnostics is not None:
        sqp_iters = (
            sqp_result.diagnostics.converged_iter
            if sqp_result.diagnostics.converged
            else 100
        )
        if sqp_iters <= 3:
            print(f"  ✓ SQP converged in {sqp_iters} iterations")
        else:
            print(f"  ✗ SQP took {sqp_iters} iterations (expected ≤3 for LQ)")

        final_cost = sqp_result.diagnostics.cost_hist[-1]
        if torch.isfinite(torch.tensor(final_cost)):
            print(f"  ✓ Final cost is finite: {final_cost:.6f}")
        else:
            print(f"  ✗ Final cost is not finite")

    all_finite = all(torch.isfinite(x).all() for x in sqp_result.x_nodes)
    if all_finite:
        print(f"  ✓ All trajectories are finite")
    else:
        print(f"  ✗ Some trajectories contain NaN/Inf")

    return 0


if __name__ == "__main__":
    sys.exit(main())
