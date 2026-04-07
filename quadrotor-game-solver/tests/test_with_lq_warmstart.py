#!/usr/bin/env python3
"""Test nonlinear quadrotor SQP with warm-start from trained LQ alpha."""

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
    print("Testing Nonlinear Quadrotor SQP with LQ Warm-Start")
    print("=" * 80)

    # Create nonlinear quadrotor game
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
    print(f"\nQuadrotor Game: dx={game.dx}, K={cfg.K}, I={cfg.I}")

    # Create indexer and alpha param
    indexer = FullIaryTreeIndexer(K=cfg.K, I=cfg.I)
    alpha_param = AlphaParam(
        indexer=indexer,
        cfg=AlphaParamConfig(init_scale=0.01),
        device=cfg.device_resolved,
        dtype=cfg.dtype,
    )

    # Load trained LQ alpha (from parent directory)
    lq_ckpt_path = Path(__file__).parent.parent.parent / "runs" / "hexner_primal_test_stable" / "latest.pt"
    if not lq_ckpt_path.exists():
        print(f"ERROR: LQ checkpoint not found at {lq_ckpt_path}")
        return 1

    print(f"\nLoading LQ trained alpha from: {lq_ckpt_path}")
    lq_ckpt = torch.load(lq_ckpt_path, map_location="cpu")
    lq_alpha_sd = lq_ckpt["alpha_state_dict"]

    print(f"  LQ alpha shape: {lq_alpha_sd['logits'].shape}")
    print(f"  Quadrotor alpha shape: {alpha_param.logits.shape}")

    # Load the trained logits
    alpha_param.load_state_dict(lq_alpha_sd)
    print("  ✓ Loaded trained LQ alpha into quadrotor alpha param")

    # Get alpha tensor
    alpha = alpha_param()
    prior = game.default_prior()
    x0 = game.default_initial_state()

    print(f"\nInitial state:")
    print(f"  P1: {x0[:3].tolist()}")
    print(f"  P2: {x0[12:15].tolist()}")

    # Run SQP with LQ-trained alpha
    print("\n" + "=" * 80)
    print("Running SQP with LQ-trained alpha")
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
