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
        R_k = torch.empty(
            num_nodes,
            I,
            du,
            du,
            device=device,
            dtype=dtype,
        )
        S_k = torch.empty(
            num_nodes,
            I,
            dv,
            dv,
            device=device,
            dtype=dtype,
        )

        for node_idx in range(num_nodes):
            for a in range(I):
                child_idx = indexer.child_index(k, node_idx, a)
                p_child = belief_tree.beliefs[k + 1][child_idx]  # (I,)
                R_edge, S_edge = game.running_cost_mats(p_child)
                R_k[node_idx, a] = R_edge
                S_k[node_idx, a] = S_edge

        R_bar.append(R_k)
        S_bar.append(S_k)

    # Terminal costs at leaves (depth K)
    num_leaves = indexer.node_count(K)
    P_leaf = torch.empty(num_leaves, dx, dx, device=device, dtype=dtype)
    r_leaf = torch.empty(num_leaves, dx, device=device, dtype=dtype)
    c_leaf = torch.empty(num_leaves, device=device, dtype=dtype)

    for node_idx in range(num_leaves):
        belief_leaf = belief_tree.beliefs[K][node_idx]        # (I,)
        # We treat terminal values as *conditional* on reaching the leaf;
        # overall expectations are built by the Riccati recursion and the
        # λ_{k,ω}^a aggregation, not by baking λ into P,r,c here.
        node_mass = torch.ones((), device=device, dtype=dtype)
        vq = game.terminal_value_quad(belief_leaf, node_mass=node_mass)
        P_leaf[node_idx] = vq.P
        r_leaf[node_idx] = vq.r
        c_leaf[node_idx] = vq.c

    return AveragedCostData(
        R_bar=R_bar,
        S_bar=S_bar,
        P_leaf=P_leaf,
        r_leaf=r_leaf,
        c_leaf=c_leaf,
    )