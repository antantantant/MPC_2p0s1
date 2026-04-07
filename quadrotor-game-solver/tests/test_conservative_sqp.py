#!/usr/bin/env python3
"""Test nonlinear SQP with very conservative hyperparameters.

SQP SHOULD work on quadrotor problems - it's the standard method.
If it's not converging, we need to tune the hyperparameters properly.
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


def test_sqp_params(step_size: float, riccati_reg: float, label: str, num_iters: int = 30):
    """Test SQP with specific hyperparameters."""
    print("\n" + "=" * 80)
    print(f"{label}")
    print(f"  step_size={step_size}, riccati_reg={riccati_reg}, iters={num_iters}")
    print("=" * 80)

    cfg = GameConfig(
        I=2,
        T=1.0,
        K=10,
        integrator="rk4",
        linearized_mode=False,
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

    alpha = alpha_param()
    prior = game.default_prior()
    x0 = game.default_initial_state()

    # Run SQP with these parameters
    sqp_result = sqp_tree_layer(
        game=game,
        indexer=indexer,
        alpha=alpha,
        x0=x0,
        p0=prior,
        num_sqp_iters=num_iters,
        step_size=step_size,
        riccati_reg=riccati_reg,
        verbose=True,
        early_stop=True,
        line_search=True,
        collect_diagnostics=True,
    )

    # Print summary
    print("\n" + "-" * 80)
    if sqp_result.diagnostics is not None:
        converged = sqp_result.diagnostics.converged
        iters = sqp_result.diagnostics.converged_iter if converged else -1
        du_final = sqp_result.diagnostics.du_max_hist[-1]
        dv_final = sqp_result.diagnostics.dv_max_hist[-1]
        cost_final = sqp_result.diagnostics.cost_hist[-1]

        if converged:
            print(f"✓✓✓ CONVERGED in {iters} iterations! ✓✓✓")
        else:
            print(f"✗ Did not converge (du={du_final:.4f}, dv={dv_final:.4f})")

        print(f"  Final cost: {cost_final:.6f}")
        print(f"  Cost change: {sqp_result.diagnostics.cost_hist[0]:.6f} → {cost_final:.6f}")

    return sqp_result


def main():
    print("=" * 80)
    print("Testing Nonlinear Quadrotor SQP with Conservative Hyperparameters")
    print("=" * 80)
    print("\nSQP should work - let's find the right parameters!")

    # Test 1: Very small step, high regularization
    test_sqp_params(
        step_size=0.1,
        riccati_reg=0.5,
        label="TEST 1: Very Conservative (tiny steps, high reg)",
        num_iters=50
    )

    # Test 2: Tiny step, very high regularization
    test_sqp_params(
        step_size=0.05,
        riccati_reg=1.0,
        label="TEST 2: Ultra Conservative (even smaller steps)",
        num_iters=50
    )

    # Test 3: Medium step with adaptive regularization (through max_tries)
    print("\n" + "=" * 80)
    print("TEST 3: Adaptive Regularization")
    print("  Let Riccati solver increase reg automatically if needed")
    print("=" * 80)

    cfg = GameConfig(I=2, T=1.0, K=10, integrator="rk4", device="cpu", dtype=torch.float64)
    game = Hexner3DQuadrotorGame(cfg)
    indexer = FullIaryTreeIndexer(K=cfg.K, I=cfg.I)
    alpha_param = AlphaParam(indexer=indexer, cfg=AlphaParamConfig(init_scale=0.01),
                             device=cfg.device_resolved, dtype=cfg.dtype)

    lq_ckpt_path = Path(__file__).parent.parent.parent / "runs" / "hexner_primal_test_stable" / "latest.pt"
    if lq_ckpt_path.exists():
        lq_ckpt = torch.load(lq_ckpt_path, map_location="cpu")
        alpha_param.load_state_dict(lq_ckpt["alpha_state_dict"])

    sqp_result = sqp_tree_layer(
        game=game,
        indexer=indexer,
        alpha=alpha_param(),
        x0=game.default_initial_state(),
        p0=game.default_prior(),
        num_sqp_iters=50,
        step_size=0.2,
        riccati_reg=0.01,           # Start low
        max_riccati_reg_tries=10,   # But allow it to increase
        riccati_reg_factor=5.0,     # Increase aggressively if needed
        verbose=True,
        early_stop=True,
        line_search=True,
        collect_diagnostics=True,
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("If none converged, we may need to:")
    print("  1. Check for bugs in dynamics/linearization")
    print("  2. Use even smaller steps (< 0.05)")
    print("  3. Try different initialization (not hover)")
    print("  4. Reduce horizon (K < 10)")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
