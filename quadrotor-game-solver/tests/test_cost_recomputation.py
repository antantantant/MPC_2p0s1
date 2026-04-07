#!/usr/bin/env python3
"""Test that cost recomputation gives same result for quadratic costs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import GameConfig
from src.game.quadrotor_game import Hexner3DQuadrotorGame
from src.tree.indexing import FullIaryTreeIndexer
from src.tree.belief_tree import build_belief_tree_vectorized
from src.tree.signaling import AlphaParam, AlphaParamConfig
from src.objectives.cost_tree import compute_quadratic_cost_data


def main():
    print("=" * 80)
    print("Testing Cost Recomputation for Quadratic Costs")
    print("=" * 80)

    cfg = GameConfig(
        I=2,
        T=1.0,
        K=10,
        integrator="rk4",
        device="cpu",
        dtype=torch.float64,
    )

    game = Hexner3DQuadrotorGame(cfg)
    indexer = FullIaryTreeIndexer(K=cfg.K, I=cfg.I)
    
    # Create alpha and belief tree
    alpha_param = AlphaParam(
        indexer=indexer,
        cfg=AlphaParamConfig(init_scale=0.01),
        device=cfg.device_resolved,
        dtype=cfg.dtype,
    )
    alpha = alpha_param()
    prior = game.default_prior()
    
    belief_tree = build_belief_tree_vectorized(
        alpha=alpha, p0=prior, indexer=indexer
    )

    # Compute cost data without trajectory (original behavior)
    print("\n1. Computing cost data without trajectory...")
    cost_data_1 = compute_quadratic_cost_data(
        game=game, belief_tree=belief_tree
    )

    # Create some arbitrary trajectory
    x0 = game.default_initial_state()
    u_hover, v_hover = game.default_hover_control()
    
    x_nodes = [None] * (cfg.K + 1)
    x_nodes[0] = x0.view(1, game.dx)
    
    u_edges = []
    v_edges = []
    
    for k in range(cfg.K):
        Nk = indexer.node_count(k)
        u_edges.append(
            u_hover.unsqueeze(0).unsqueeze(0).expand(Nk, cfg.I, game.du).clone()
        )
        v_edges.append(
            v_hover.unsqueeze(0).unsqueeze(0).expand(Nk, cfg.I, game.dv).clone()
        )
        
        # Forward step
        x_k = x_nodes[k]
        x_parent = x_k.unsqueeze(1).expand(Nk, cfg.I, game.dx).reshape(
            Nk * cfg.I, game.dx
        )
        u_flat = u_edges[k].reshape(Nk * cfg.I, game.du)
        v_flat = v_edges[k].reshape(Nk * cfg.I, game.dv)
        x_nodes[k + 1] = game.step(x_parent, u_flat, v_flat)

    # Compute cost data with trajectory (new behavior)
    print("2. Computing cost data WITH trajectory...")
    cost_data_2 = compute_quadratic_cost_data(
        game=game,
        belief_tree=belief_tree,
        x_nodes=x_nodes,
        u_edges=u_edges,
        v_edges=v_edges,
    )

    # Compare results
    print("\n3. Comparing results...")
    print("   For quadratic costs, both should give IDENTICAL results.\n")
    
    all_match = True
    for k in range(cfg.K):
        Q_match = torch.allclose(cost_data_1.Q[k], cost_data_2.Q[k])
        q_match = torch.allclose(cost_data_1.q[k], cost_data_2.q[k])
        c_match = torch.allclose(cost_data_1.c[k], cost_data_2.c[k])
        R_match = torch.allclose(cost_data_1.R[k], cost_data_2.R[k])
        S_match = torch.allclose(cost_data_1.S[k], cost_data_2.S[k])
        
        if not (Q_match and q_match and c_match and R_match and S_match):
            print(f"   ✗ Stage {k}: Mismatch!")
            all_match = False
    
    P_match = torch.allclose(cost_data_1.P_leaf, cost_data_2.P_leaf)
    r_match = torch.allclose(cost_data_1.r_leaf, cost_data_2.r_leaf)
    c_match = torch.allclose(cost_data_1.c_leaf, cost_data_2.c_leaf)
    
    if not (P_match and r_match and c_match):
        print("   ✗ Terminal: Mismatch!")
        all_match = False

    if all_match:
        print("   ✓ All cost coefficients match!")
        print("   ✓ Cost recomputation gives identical results for quadratic costs.")
        print("\n✓✓✓ TEST PASSED ✓✓✓")
        return 0
    else:
        print("\n✗✗✗ TEST FAILED ✗✗✗")
        print("Cost coefficients differ - this should not happen for quadratic costs!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
