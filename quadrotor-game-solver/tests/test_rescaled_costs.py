#!/usr/bin/env python3
"""Test nonlinear SQP with rescaled terminal costs.

The issue: terminal cost is ~6x smaller than running cost, so the game
doesn't care about reaching the goal. We fix this by increasing K1_scale, K2_scale.
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


def test_config(K1_scale: float, K2_scale: float, label: str):
    """Test SQP with given terminal cost scales."""
    print("\n" + "=" * 80)
    print(f"{label}: K1={K1_scale}, K2={K2_scale}")
    print("=" * 80)

    cfg = GameConfig(
        I=2,
        T=1.0,
        K=10,
        integrator="rk4",
        linearized_mode=False,
        small_angle_approx=False,
        K1_scale=K1_scale,
        K2_scale=K2_scale,
        device="cpu",
        dtype=torch.float64,
    )

    game = Hexner3DQuadrotorGame(cfg)

    # Load LQ-trained alpha
    indexer = FullIaryTreeIndexer(K=cfg.K, I=cfg.I)
    alpha_param = AlphaParam(
        indexer=indexer,
        cfg=AlphaParamConfig(init_scale=0.01),
        device=cfg.device_resolved,
        dtype=cfg.dtype,
    )

    lq_ckpt_path = Path(__file__).parent.parent.parent / "runs" / "hexner_primal_test_stable" / "latest.pt"
    if lq_ckpt_path.exists():
        lq_ckpt = torch.load(lq_ckpt_path, map_location="cpu")
        alpha_param.load_state_dict(lq_ckpt["alpha_state_dict"])
        print("✓ Loaded LQ-trained alpha")
    else:
        print("⚠ Using uniform alpha (LQ checkpoint not found)")

    alpha = alpha_param()
    prior = game.default_prior()
    x0 = game.default_initial_state()

    # Run SQP
    sqp_result = sqp_tree_layer(
        game=game,
        indexer=indexer,
        alpha=alpha,
        x0=x0,
        p0=prior,
        num_sqp_iters=20,
        step_size=0.5,
        riccati_reg=5e-2,
        verbose=False,  # Less verbose for comparison
        early_stop=True,
        line_search=True,
        collect_diagnostics=True,
    )

    # Print summary
    if sqp_result.diagnostics is not None:
        converged = sqp_result.diagnostics.converged
        iters = sqp_result.diagnostics.converged_iter if converged else -1
        du_final = sqp_result.diagnostics.du_max_hist[-1]
        dv_final = sqp_result.diagnostics.dv_max_hist[-1]
        cost_final = sqp_result.diagnostics.cost_hist[-1]

        status = "✓ CONVERGED" if converged else "✗ NOT CONVERGED"
        print(f"\n{status}")
        print(f"  Iterations: {iters if converged else '20+'}")
        print(f"  Final cost: {cost_final:.4f}")
        print(f"  Final du/dv: {du_final:.4f} / {dv_final:.4f}")

        x_final = sqp_result.x_nodes[-1][0]
        print(f"  Final pos P1: [{x_final[0]:.2f}, {x_final[1]:.2f}, {x_final[2]:.2f}]")
        print(f"  Final pos P2: [{x_final[12]:.2f}, {x_final[13]:.2f}, {x_final[14]:.2f}]")

    return sqp_result


def main():
    print("=" * 80)
    print("Testing Nonlinear SQP with Rescaled Terminal Costs")
    print("=" * 80)

    # Test 1: Original (too small)
    print("\nOriginal cost scaling (terminal too weak):")
    test_config(K1_scale=1.0, K2_scale=1.0, label="TEST 1: Original (K=1)")

    # Test 2: Moderate increase (10×)
    print("\n\nModerate increase (10× stronger terminal cost):")
    test_config(K1_scale=10.0, K2_scale=10.0, label="TEST 2: Moderate (K=10)")

    # Test 3: Strong increase (50×)
    print("\n\nStrong increase (50× stronger terminal cost):")
    test_config(K1_scale=50.0, K2_scale=50.0, label="TEST 3: Strong (K=50)")

    # Test 4: Very strong (100×)
    print("\n\nVery strong (100× stronger terminal cost):")
    test_config(K1_scale=100.0, K2_scale=100.0, label="TEST 4: Very Strong (K=100)")

    print("\n" + "=" * 80)
    print("Summary: Check which cost scaling leads to SQP convergence")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
