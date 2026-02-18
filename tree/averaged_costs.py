# tree/averaged_costs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from MPC_2p0s1.core.types import Tensor
from MPC_2p0s1.games.base_lq_game import BaseLQGame
from MPC_2p0s1.tree.belief_tree import BeliefTree
from MPC_2p0s1.tree.indexing import FullIaryTreeIndexer


@dataclass
class AveragedCostData:
    """
    Belief-averaged running and terminal cost data on the tree.

    Attributes
    ----------
    R_bar:
        List of length K. R_bar[k] has shape (num_nodes_k, I, du, du) and
        stores the belief-averaged P1 running-cost matrices R̄_{k,ω}^a on
        each edge (k, ω, a).
    S_bar:
        List of length K. S_bar[k] has shape (num_nodes_k, I, dv, dv) and
        stores the belief-averaged P2 running-cost matrices S̄_{k,ω}^a.
    P_leaf:
        Tensor of shape (num_nodes_K, dx, dx) with terminal quadratic
        matrices at leaves (no path-mass weighting).
    r_leaf:
        Tensor of shape (num_nodes_K, dx) with terminal linear terms.
    c_leaf:
        Tensor of shape (num_nodes_K,) with terminal constants.
    """

    R_bar: List[Tensor]
    S_bar: List[Tensor]
    P_leaf: Tensor
    r_leaf: Tensor
    c_leaf: Tensor


def compute_averaged_costs(
    game: BaseLQGame,
    belief_tree: BeliefTree,
) -> AveragedCostData:
    """
    Compute belief-averaged running-cost matrices and terminal value quads.

    Parameters
    ----------
    game:
        LQ game instance providing R_i, S_i and terminal cost data.
    belief_tree:
        BeliefTree containing beliefs p_{k,ω} and path masses λ_{k,ω}.

    Returns
    -------
    AveragedCostData
        Running-cost matrices R̄, S̄ on edges and terminal value data on leaves.
    """
    indexer: FullIaryTreeIndexer = belief_tree.indexer
    K = indexer.K
    I = game.I
    du = game.du
    dv = game.dv
    dx = game.dx

    device = game.device_resolved
    dtype = game.dtype

    R_bar: List[Tensor] = []
    S_bar: List[Tensor] = []

    # Running costs for edges k=0..K-1
    for k in range(K):
        num_nodes = indexer.node_count(k)
        num_nodes_next = indexer.node_count(k + 1)
        if num_nodes_next != num_nodes * I:
            raise RuntimeError(
                "compute_averaged_costs expects full I-ary indexing with "
                f"num_nodes_next={num_nodes_next} == num_nodes*I={num_nodes * I}"
            )

        # child_index(k, node, a) = node * I + a, so row-major reshape gives
        # beliefs per edge (node, action, type).
        p_next = belief_tree.beliefs[k + 1].reshape(num_nodes, I, I)  # (node, action, type)

        # Belief-averaged running costs on all edges at once:
        # (node, action, type) x (type, du, du) -> (node, action, du, du)
        R_k = torch.einsum("nai, ibc -> nabc", p_next, game.R)
        S_k = torch.einsum("nai, ibc -> nabc", p_next, game.S)

        R_bar.append(R_k)
        S_bar.append(S_k)

    # Terminal costs at leaves (depth K)
    num_leaves = indexer.node_count(K)
    belief_leaves = belief_tree.beliefs[K]  # (num_leaves, I)

    # We use conditional terminal values at each leaf (node mass = 1).
    # terminal_value_quad with node_mass=1 gives:
    # P = sum_i p[i] Q_i, r = sum_i p[i] q_i, c = sum_i p[i] c_i.
    P_leaf = torch.einsum("ni, iab -> nab", belief_leaves, game.Q)  # (num_leaves, dx, dx)
    r_leaf = torch.einsum("ni, ia -> na", belief_leaves, game.q)    # (num_leaves, dx)
    c_leaf = torch.einsum("ni, i -> n", belief_leaves, game.c)      # (num_leaves,)

    return AveragedCostData(
        R_bar=R_bar,
        S_bar=S_bar,
        P_leaf=P_leaf,
        r_leaf=r_leaf,
        c_leaf=c_leaf,
    )
