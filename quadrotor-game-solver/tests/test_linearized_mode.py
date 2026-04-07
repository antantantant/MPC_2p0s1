#!/usr/bin/env python3
"""Quick test to verify linearized mode works correctly.

This script:
1. Creates a game in linearized mode (frozen dynamics at hover)
2. Runs a single SQP solve
3. Verifies that SQP converges quickly (should be ~1-2 iters for LQ problem)
4. Prints trajectories and costs
"""

from __future__ import annotations

import sys

import torch

from src.utils.config import GameConfig
from src.game.quadrotor_game import Hexner3DQuadrotorGame
from src.tree.belief_tree import build_belief_tree_vectorized
from src.tree.indexing import FullIaryTreeIndexer
from src.objectives.cost_tree import compute_quadratic_cost_data
from src.solvers.sqp_tree import sqp_tree_layer
from src.tree.signaling import AlphaParam, AlphaParamConfig


def main():
    print("=" * 80)
    print("Testing Linearized Mode")
    print("=" * 80)

    # ── Create config with linearized mode ──────────────────────────────────
    cfg = GameConfig(
        I=2,
        T=1.0,
        K=10,
        integrator="rk4",
        linearized_mode=True,  # <-- KEY: freeze linearization at hover
        device="cpu",
        dtype=torch.float64,
    )

    print(f"\nConfig:")
    print(f"  I={cfg.I}, K={cfg.K}, T={cfg.T}, tau={cfg.tau:.3f}")
    print(f"  Linearized mode: {cfg.linearized_mode}")
    print(f"  Integrator: {cfg.integrator}")

    # ── Create game ─────────────────────────────────────────────────────────
    game = Hexner3DQuadrotorGame(cfg)
    print(f"\nGame created:")
    print(f"  dx={game.dx}, du={game.du}, dv={game.dv}")
    print(f"  Linearized mode active: {game.linearized_mode}")

    # ── Verify cached matrices exist ────────────────────────────────────────
    if game.linearized_mode:
        print(f"\nCached linearization matrices:")
        print(f"  A: {game._A_cached.shape}")
        print(f"  B1: {game._B1_cached.shape}")
        print(f"  B2: {game._B2_cached.shape}")
        print(f"  d: {game._d_cached.shape}")

    # ── Create alpha param (for uniform signaling) ──────────────────────────
    indexer = FullIaryTreeIndexer(K=cfg.K, I=cfg.I)
    alpha_param = AlphaParam(
        indexer=indexer,
        cfg=AlphaParamConfig(init_scale=0.01),
        device=cfg.device_resolved,
        dtype=cfg.dtype,
    )

    # Get alpha tensor (uniform signaling)
    alpha = alpha_param()  # (K, max_nodes, I, I)
    prior = game.default_prior()

    print(f"\nIndexer created:")
    print(f"  Depth K={indexer.K}, branching I={indexer.I}")
    print(f"  Alpha shape: {alpha.shape}")

    # ── Initial guess (hover) ───────────────────────────────────────────────
    x0 = game.default_initial_state()

    print(f"\nInitial state x0:")
    print(f"  P1 pos: {x0[:3].tolist()}")
    print(f"  P2 pos: {x0[12:15].tolist()}")

    # ── Run SQP ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Running SQP (should converge in 1-2 iters for LQ problem)")
    print("=" * 80)

    sqp_result = sqp_tree_layer(
        game=game,
        indexer=indexer,
        alpha=alpha,
        x0=x0,
        p0=prior,
        num_sqp_iters=10,
        step_size=1.0,
        riccati_reg=1e-3,
        verbose=True,
        early_stop=True,
        line_search=True,
        collect_diagnostics=True,
    )

    print("\n" + "=" * 80)
    print("SQP Results")
    print("=" * 80)
    if sqp_result.diagnostics is not None:
        print(f"Converged: {sqp_result.diagnostics.converged}")
        print(f"Converged at iteration: {sqp_result.diagnostics.converged_iter}")
        print(f"Final cost: {sqp_result.diagnostics.cost_hist[-1]:.6f}")
    else:
        print("No diagnostics available (collect_diagnostics=False)")

    # ── Print trajectory summary ────────────────────────────────────────────
    print(f"\nTrajectory shapes:")
    print(f"  x_nodes: {len(sqp_result.x_nodes)} states, root shape {sqp_result.x_nodes[0].shape}")
    print(f"  u_edges: {len(sqp_result.u_edges)} controls, root shape {sqp_result.u_edges[0].shape}")
    print(f"  v_edges: {len(sqp_result.v_edges)} controls, root shape {sqp_result.v_edges[0].shape}")

    # Print first and last state of root trajectory
    print(f"\nRoot trajectory (node 0):")
    x_root_0 = sqp_result.x_nodes[0][0]  # depth 0, node 0
    x_root_K = sqp_result.x_nodes[-1][0]  # depth K, node 0
    print(f"  x[0]: pos1={x_root_0[:3].tolist()}, pos2={x_root_0[12:15].tolist()}")
    print(f"  x[K]: pos1={x_root_K[:3].tolist()}, pos2={x_root_K[12:15].tolist()}")

    # Print control effort at root
    u_norm = (sqp_result.u_edges[0][0] ** 2).sum()
    v_norm = (sqp_result.v_edges[0][0] ** 2).sum()
    print(f"\nControl effort (root, first step):")
    print(f"  ||u[0]||^2 = {u_norm:.3f}")
    print(f"  ||v[0]||^2 = {v_norm:.3f}")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)

    # ── Sanity checks ───────────────────────────────────────────────────────
    print("\nSanity checks:")

    # Check 1: Should converge in few iterations
    if sqp_result.diagnostics is not None:
        sqp_iters = sqp_result.diagnostics.converged_iter if sqp_result.diagnostics.converged else 100
        if sqp_iters <= 3:
            print(f"  ✓ SQP converged in {sqp_iters} iterations (expected for LQ)")
        else:
            print(f"  ✗ SQP took {sqp_iters} iterations (expected ≤3 for LQ)")

        # Check 2: Cost should be finite
        final_cost = sqp_result.diagnostics.cost_hist[-1]
        if torch.isfinite(torch.tensor(final_cost)):
            print(f"  ✓ Final cost is finite: {final_cost:.6f}")
        else:
            print(f"  ✗ Final cost is not finite: {final_cost}")

    # Check 3: Trajectories should be finite
    all_finite = all(torch.isfinite(x).all() for x in sqp_result.x_nodes)
    if all_finite:
        print(f"  ✓ All trajectories are finite")
    else:
        print(f"  ✗ Some trajectories contain NaN/Inf")

    return 0


if __name__ == "__main__":
    sys.exit(main())
