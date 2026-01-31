"""
Verify the rollout fix addresses the off-by-one timing issue.

The bug: Control at k uses kappa_u computed from child's value (at k+1),
which incorporates the posterior belief. This means control at k "knows"
the revealed info one step early.

The fix: Aggregate kappa_u over actions using lambda_edge (edge probabilities),
so the control at k uses the expected feedforward term given the PRIOR belief.
"""
import torch
from MPC_2p0s1.config.base_config import GameConfig
from MPC_2p0s1.games.hexner_game import HexnerGame, HexnerParams
from MPC_2p0s1.tree.indexing import FullIaryTreeIndexer
from MPC_2p0s1.tree.belief_tree import build_belief_tree
from MPC_2p0s1.tree.averaged_costs import compute_averaged_costs
from MPC_2p0s1.tree.riccati_tree import riccati_backward
from MPC_2p0s1.tree.rollout import rollout_trajectory


def main():
    dtype = torch.float64
    device = torch.device('cpu')
    T, K = 1.0, 10
    tau = T / K
    
    cfg = GameConfig(I=2, T=T, K=K, dx1=4, dx2=4, du=2, dv=2, device='cpu', dtype=dtype)
    cfg.device_resolved = device
    cfg.dx = cfg.dx1 + cfg.dx2
    cfg.tau = tau
    
    game = HexnerGame(cfg=cfg, params=HexnerParams())
    indexer = FullIaryTreeIndexer(I=2, K=K)
    x0 = game.default_initial_state()
    p0 = torch.tensor([0.5, 0.5], dtype=dtype, device=device)
    
    # Reveal at k=5 (i.e., belief changes at k=6)
    reveal_k = 5
    alpha = torch.zeros(K, indexer.max_nodes_per_depth, 2, 2, dtype=dtype, device=device)
    for k in range(K):
        for node in range(indexer.node_count(k)):
            if k < reveal_k:
                alpha[k, node, 0, 0] = 1.0  # Pool
                alpha[k, node, 1, 0] = 1.0  # Pool
            else:
                alpha[k, node, 0, 0] = 1.0  # Separate
                alpha[k, node, 1, 1] = 1.0  # Separate
    
    belief_tree = build_belief_tree(alpha=alpha, p0=p0, indexer=indexer)
    avg_costs = compute_averaged_costs(game=game, belief_tree=belief_tree)
    riccati = riccati_backward(game=game, belief_tree=belief_tree, avg_costs=avg_costs, action_space=None)
    rollout = rollout_trajectory(
        game=game, indexer=indexer, belief_tree=belief_tree, riccati_sol=riccati,
        alpha=alpha, x0=x0, type_index=0, action_space=None, sample_actions=False
    )
    
    print("=" * 70)
    print("Verify Fix: Control at step k should use PRIOR belief, not POSTERIOR")
    print("=" * 70)
    print()
    print(f"Reveal at k={reveal_k} means:")
    print(f"  - Belief at k={reveal_k} is still [0.5, 0.5] (prior)")
    print(f"  - Belief at k={reveal_k+1} is [1, 0] (posterior, revealed)")
    print()
    print("Ground truth expectation (tr = 0.5, tau = 0.1):")
    print(f"  - Control at k <= 5 (t <= 0.5) should HEDGE (u_y = 0)")
    print(f"  - Control at k >= 6 (t > 0.5) should TARGET (u_y != 0)")
    print()
    
    print("Actual controls (y-component):")
    print("-" * 50)
    for k in range(K):
        u_y = rollout.u_traj[k, 1].item()
        belief = rollout.belief_traj[k].numpy()
        t_k = k * tau
        
        is_hedging = abs(u_y) < 0.01
        status = "HEDGE ✓" if is_hedging else f"TARGET (u_y={u_y:.4f})"
        
        # Check if correct
        should_hedge = t_k <= 0.5
        correct = is_hedging == should_hedge
        correctness = "✓" if correct else "✗ BUG!"
        
        print(f"  k={k:2d}, t={t_k:.1f}: belief={belief}, {status} {correctness}")
    
    print()
    print("If all ✓, the fix is working correctly!")


if __name__ == "__main__":
    main()
