#!/usr/bin/env python3
"""Test nonlinear SQP solver on 3D quadrotor dynamics.

This tests the actual nonlinear solver (not linearized mode).
"""

from __future__ import annotations

import sys

import torch

from src.utils.config import GameConfig
from src.game.quadrotor_game import Hexner3DQuadrotorGame
from src.tree.indexing import FullIaryTreeIndexer
from src.tree.signaling import AlphaParam, AlphaParamConfig
from src.solvers.sqp_tree import sqp_tree_layer


def main():
    print("=" * 80)
    print("Testing Nonlinear SQP on 3D Quadrotor (I=2)")
    print("=" * 80)

    # Create nonlinear quadrotor game (linearized_mode=False)
    cfg = GameConfig(
        I=2,
        T=1.0,
        K=10,
        integrator="rk4",
        linearized_mode=False,  # <-- NONLINEAR
        device="cpu",
        dtype=torch.float64,
    )

    game = Hexner3DQuadrotorGame(cfg)
    print(f"\nNonlinear Quadrotor Game:")
    print(f"  dx={game.dx}, du={game.du}, dv={game.dv}, I={game.I}")
    print(f"  K={cfg.K}, T={cfg.T}, tau={cfg.tau:.3f}")
    print(f"  Linearized mode: {game.linearized_mode}")

    # Create indexer and alpha
    indexer = FullIaryTreeIndexer(K=cfg.K, I=cfg.I)
    alpha_param = AlphaParam(
        indexer=indexer,
        cfg=AlphaParamConfig(init_scale=0.01),
        device=cfg.device_resolved,
        dtype=cfg.dtype,
    )
    alpha = alpha_param()
    prior = game.default_prior()
    x0 = game.default_initial_state()

    print(f"\nInitial state x0:")
    print(f"  P1 pos: {x0[:3].tolist()}")
    print(f"  P2 pos: {x0[12:15].tolist()}")
    print(f"Prior: {prior.tolist()}")

    # Test different hyperparameter configurations
    configs = [
        {
            "name": "Default",
            "num_sqp_iters": 10,
            "step_size": 0.5,
            "riccati_reg": 1e-1,
            "line_search": True,
        },
        {
            "name": "More iters + smaller reg",
            "num_sqp_iters": 20,
            "step_size": 0.5,
            "riccati_reg": 1e-2,
            "line_search": True,
        },
        {
            "name": "Smaller step + more reg",
            "num_sqp_iters": 15,
            "step_size": 0.3,
            "riccati_reg": 5e-2,
            "line_search": True,
        },
    ]

    for i, config in enumerate(configs):
        print("\n" + "=" * 80)
        print(f"Config {i+1}: {config['name']}")
        print("=" * 80)
        print(f"  SQP iters: {config['num_sqp_iters']}")
        print(f"  Step size: {config['step_size']}")
        print(f"  Riccati reg: {config['riccati_reg']}")
        print(f"  Line search: {config['line_search']}")
        print()

        try:
            sqp_result = sqp_tree_layer(
                game=game,
                indexer=indexer,
                alpha=alpha,
                x0=x0,
                p0=prior,
                num_sqp_iters=config["num_sqp_iters"],
                step_size=config["step_size"],
                riccati_reg=config["riccati_reg"],
                verbose=True,
                early_stop=True,
                line_search=config["line_search"],
                collect_diagnostics=True,
            )

            # Print results
            print("\nResults:")
            if sqp_result.diagnostics is not None:
                print(f"  Converged: {sqp_result.diagnostics.converged}")
                print(
                    f"  Converged at iter: {sqp_result.diagnostics.converged_iter}"
                )
                print(f"  Final cost: {sqp_result.diagnostics.cost_hist[-1]:.6f}")
                print(
                    f"  Cost history: {[f'{c:.6f}' for c in sqp_result.diagnostics.cost_hist[:5]]}..."
                )

            # Check trajectory
            x_root_K = sqp_result.x_nodes[-1][0]
            all_finite = all(torch.isfinite(x).all() for x in sqp_result.x_nodes)
            print(f"  Final P1 pos: {x_root_K[:3].tolist()}")
            print(f"  Final P2 pos: {x_root_K[12:15].tolist()}")
            print(f"  All finite: {all_finite}")

        except Exception as e:
            print(f"\n  ✗ ERROR: {str(e)}")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
