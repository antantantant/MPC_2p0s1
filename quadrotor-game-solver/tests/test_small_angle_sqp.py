#!/usr/bin/env python3
"""Test nonlinear quadrotor SQP with small-angle approximation.

This uses polynomial approximations for trig functions:
  sin(x) ≈ x - x³/6
  cos(x) ≈ 1 - x²/2

The dynamics are still nonlinear but smoother than exact trig,
which should help SQP converge better.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from src.utils.config import GameConfig
from src.game.quadrotor_game import Hexner3DQuadrotorGame
from src.tree.indexing import FullIaryTreeIndexer
from src.tree.signaling import AlphaParam, AlphaParamConfig
from src.solvers.sqp_tree import sqp_tree_layer


def main():
    print("=" * 80)
    print("Testing Nonlinear Quadrotor SQP with Small-Angle Approximation")
    print("=" * 80)

    # Create nonlinear quadrotor game with small-angle approximation
    cfg = GameConfig(
        I=2,
        T=1.0,
        K=10,
        integrator="rk4",
        linearized_mode=False,       # Full nonlinear
        small_angle_approx=True,     # <-- Use polynomial approx for trig
        device="cpu",
        dtype=torch.float64,
    )

    game = Hexner3DQuadrotorGame(cfg)
    print(f"\nQuadrotor Game: dx={game.dx}, K={cfg.K}, I={cfg.I}")
    print(f"Small-angle approximation: {cfg.small_angle_approx}")

    # Create indexer and alpha param
    indexer = FullIaryTreeIndexer(K=cfg.K, I=cfg.I)
    alpha_param = AlphaParam(
        indexer=indexer,
        cfg=AlphaParamConfig(init_scale=0.01),
        device=cfg.device_resolved,
        dtype=cfg.dtype,
    )

    # Option 1: Test with uniform alpha
    alpha = alpha_param()
    prior = game.default_prior()
    x0 = game.default_initial_state()

    print(f"\nInitial state:")
    print(f"  P1: {x0[:3].tolist()}")
    print(f"  P2: {x0[12:15].tolist()}")

    # Run SQP
    print("\n" + "=" * 80)
    print("Running SQP (should converge better with smoother dynamics)")
    print("=" * 80)

    sqp_result = sqp_tree_layer(
        game=game,
        indexer=indexer,
        alpha=alpha,
        x0=x0,
        p0=prior,
        num_sqp_iters=20,
        step_size=0.5,
        riccati_reg=5e-2,
        verbose=True,
        early_stop=True,
        line_search=True,
        collect_diagnostics=True,
    )

    # Print results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)

    if sqp_result.diagnostics is not None:
        print(f"Converged: {sqp_result.diagnostics.converged}")
        print(f"Converged at iter: {sqp_result.diagnostics.converged_iter}")
        print(f"Final cost: {sqp_result.diagnostics.cost_hist[-1]:.6f}")
        print(
            f"Cost history (first 10): {[f'{c:.4f}' for c in sqp_result.diagnostics.cost_hist[:10]]}"
        )
        print(
            f"Final du/dv: du={sqp_result.diagnostics.du_max_hist[-1]:.4f}, dv={sqp_result.diagnostics.dv_max_hist[-1]:.4f}"
        )

    x_root_K = sqp_result.x_nodes[-1][0]
    print(f"\nFinal positions:")
    print(f"  P1: {x_root_K[:3].tolist()}")
    print(f"  P2: {x_root_K[12:15].tolist()}")

    all_finite = all(torch.isfinite(x).all() for x in sqp_result.x_nodes)
    print(f"  All finite: {all_finite}")

    # Sanity check
    if sqp_result.diagnostics is not None:
        du_final = sqp_result.diagnostics.du_max_hist[-1]
        dv_final = sqp_result.diagnostics.dv_max_hist[-1]

        if du_final < 0.01 and dv_final < 0.01:
            print("\n✓ SUCCESS: SQP converged with small-angle approximation!")
        else:
            print(f"\n✗ NOT CONVERGED: du={du_final:.4f}, dv={dv_final:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
