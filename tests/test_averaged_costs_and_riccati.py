# test/test_averaged_costs_and_riccati.py
from __future__ import annotations

import torch

from MPC_2p0s1.tests.utils_small_hexner import make_small_hexner_setup
from MPC_2p0s1.tree.belief_tree import build_belief_tree
from MPC_2p0s1.tree.averaged_costs import compute_averaged_costs
from MPC_2p0s1.tree.riccati_tree import riccati_backward


def _is_symmetric(mat: torch.Tensor, atol: float = 1e-6) -> bool:
    return torch.allclose(mat, mat.transpose(-1, -2), atol=atol)


def test_averaged_costs_shapes_and_symmetry():
    setup = make_small_hexner_setup(I=2, K=2, T=1.0, device="cpu")
    game = setup.game
    indexer = setup.indexer

    alpha = setup.alpha_module()
    p0 = game.default_prior()

    belief_tree = build_belief_tree(alpha=alpha, p0=p0, indexer=indexer)
    avg_costs = compute_averaged_costs(game=game, belief_tree=belief_tree)

    dx = game.dx
    du = game.du
    dv = game.dv
    I = game.I
    K = indexer.K

    # Running costs on edges
    assert len(avg_costs.R_bar) == K
    assert len(avg_costs.S_bar) == K

    for k in range(K):
        num_nodes = indexer.node_count(k)
        R_k = avg_costs.R_bar[k]
        S_k = avg_costs.S_bar[k]

        assert R_k.shape == (num_nodes, I, du, du)
        assert S_k.shape == (num_nodes, I, dv, dv)

        # Symmetry of R̄ and S̄
        assert _is_symmetric(R_k)
        assert _is_symmetric(S_k)

    # Terminal value quads at leaves
    num_leaves = indexer.node_count(K)
    assert avg_costs.P_leaf.shape == (num_leaves, dx, dx)
    assert avg_costs.r_leaf.shape == (num_leaves, dx)
    assert avg_costs.c_leaf.shape == (num_leaves,)

    # P_leaf matrices should be symmetric.
    assert _is_symmetric(avg_costs.P_leaf)


def test_riccati_backward_basic_properties():
    setup = make_small_hexner_setup(I=2, K=2, T=1.0, device="cpu")
    game = setup.game
    indexer = setup.indexer

    alpha = setup.alpha_module()
    p0 = game.default_prior()

    belief_tree = build_belief_tree(alpha=alpha, p0=p0, indexer=indexer)
    avg_costs = compute_averaged_costs(game=game, belief_tree=belief_tree)
    riccati_sol = riccati_backward(
        game=game,
        belief_tree=belief_tree,
        avg_costs=avg_costs,
        action_space=None,
    )

    dx = game.dx
    du = game.du
    dv = game.dv
    I = game.I
    K = indexer.K

    # Node-wise value shapes
    assert len(riccati_sol.P_nodes) == K + 1
    assert len(riccati_sol.r_nodes) == K + 1
    assert len(riccati_sol.c_nodes) == K + 1

    for k in range(K + 1):
        num_nodes = indexer.node_count(k)
        P_k = riccati_sol.P_nodes[k]
        r_k = riccati_sol.r_nodes[k]
        c_k = riccati_sol.c_nodes[k]

        assert P_k.shape == (num_nodes, dx, dx)
        assert r_k.shape == (num_nodes, dx)
        assert c_k.shape == (num_nodes,)

        # P_k should be symmetric
        assert torch.allclose(P_k, P_k.transpose(-1, -2), atol=1e-6)

    # Feedback gains shapes on edges
    assert len(riccati_sol.K_u) == K
    assert len(riccati_sol.K_v) == K

    for k in range(K):
        num_nodes = indexer.node_count(k)
        Ku_k = riccati_sol.K_u[k]
        kvu_k = riccati_sol.K_v[k]
        kappa_u_k = riccati_sol.kappa_u[k]
        kappa_v_k = riccati_sol.kappa_v[k]

        assert Ku_k.shape == (num_nodes, I, du, dx)
        assert kvu_k.shape == (num_nodes, I, dv, dx)
        assert kappa_u_k.shape == (num_nodes, I, du)
        assert kappa_v_k.shape == (num_nodes, I, dv)

    # Root value at default x0 should be finite
    x0 = game.default_initial_state()
    value = riccati_sol.value_at_root(x0)
    assert value.ndim == 0
    assert torch.isfinite(value)