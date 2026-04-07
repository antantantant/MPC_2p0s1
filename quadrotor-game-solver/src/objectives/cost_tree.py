"""Compute belief-averaged quadratic cost data on the tree (vectorized).

Mirrors ``nl_sqp/cost_tree.py`` but adapted for the 3-D Hexner game
(no pursuit or goal-distance running-state cost — only control effort).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor

from ..tree.belief_tree import BeliefTree
from ..game.quadrotor_game import Hexner3DQuadrotorGame


@dataclass
class QuadraticCostData:
    """All quadratic cost coefficients needed by the Riccati / SQP solver.

    For each depth k = 0 … K-1, each edge (node, action) has:
        running cost:  (τ/2)[ x^T Q x + q^T x + c  +  u^T R u − v^T S v ]
    For the Hexner game Q = 0, q = 0, c = 0  (running = control only).
    """

    Q: List[Tensor]   # len K; Q[k]: (N_k, I, dx, dx)
    q: List[Tensor]   # len K; q[k]: (N_k, I, dx)
    c: List[Tensor]   # len K; c[k]: (N_k, I)

    R: List[Tensor]   # len K; R[k]: (N_k, I, du, du)
    S: List[Tensor]   # len K; S[k]: (N_k, I, dv, dv)

    # Terminal value at leaves (depth K)
    P_leaf: Tensor    # (N_K, dx, dx)
    r_leaf: Tensor    # (N_K, dx)
    c_leaf: Tensor    # (N_K,)


def compute_quadratic_cost_data(
    *,
    game: Hexner3DQuadrotorGame,
    belief_tree: BeliefTree,
    x_nodes: List[Tensor] | None = None,
    u_edges: List[Tensor] | None = None,
    v_edges: List[Tensor] | None = None,
) -> QuadraticCostData:
    """Compute belief-averaged quadratic coefficients on all edges and leaves.

    For faithful SQP, this should be called at each iteration with the current
    nominal trajectory (x_nodes, u_edges, v_edges) to quadratize costs around
    that trajectory. For quadratic costs (like Hexner), the trajectory is ignored
    and the result is trajectory-independent.

    For general nonlinear costs, this would compute Hessians and gradients at
    the nominal points. Currently implemented for quadratic costs only.

    Convention (same as nl_sqp and original LQ):
      - Running cost on edge (k, ω, a) uses the *child/posterior* belief
        at (k+1, ωa).
      - Terminal cost at leaf uses the leaf belief.

    Parameters
    ----------
    game : Hexner3DQuadrotorGame
    belief_tree : BeliefTree
    x_nodes : List[Tensor] | None
        Current nominal states (for nonlinear cost quadratization).
        Ignored for quadratic costs.
    u_edges : List[Tensor] | None
        Current nominal P1 controls. Ignored for quadratic costs.
    v_edges : List[Tensor] | None
        Current nominal P2 controls. Ignored for quadratic costs.
    """
    indexer = belief_tree.indexer
    K = indexer.K
    I = indexer.I

    Q_list: List[Tensor] = []
    q_list: List[Tensor] = []
    c_list: List[Tensor] = []
    R_list: List[Tensor] = []
    S_list: List[Tensor] = []

    for k in range(K):
        N = indexer.node_count(k)
        Nn = indexer.node_count(k + 1)          # == N * I

        beliefs_child = belief_tree.beliefs[k + 1]   # (Nn, I)

        # Vectorised running costs
        Qk, qk_flat, ck_flat, Rk_flat, Sk_flat = game.stage_cost_mats_batch(
            beliefs_child
        )

        # Reshape to (N, I, …)
        Q_edges = Qk.view(1, 1, game.dx, game.dx).expand(N, I, game.dx, game.dx)
        q_edges = qk_flat.view(N, I, game.dx)
        c_edges = ck_flat.view(N, I)
        R_edges = Rk_flat.view(N, I, game.du, game.du)
        S_edges = Sk_flat.view(N, I, game.dv, game.dv)

        Q_list.append(Q_edges)
        q_list.append(q_edges)
        c_list.append(c_edges)
        R_list.append(R_edges)
        S_list.append(S_edges)

    # Terminal (leaves at depth K)
    beliefs_leaf = belief_tree.beliefs[K]         # (N_K, I)
    P_term, r_term, c_term = game.terminal_value_quad_batch(beliefs_leaf)
    N_leaf = beliefs_leaf.shape[0]

    # P_term may be (N_K, dx, dx) or (dx, dx) depending on whether Q is
    # type-independent.  Make sure it's (N_K, dx, dx).
    if P_term.ndim == 2:
        P_leaf = P_term.unsqueeze(0).expand(N_leaf, game.dx, game.dx)
    else:
        P_leaf = P_term

    return QuadraticCostData(
        Q=Q_list,
        q=q_list,
        c=c_list,
        R=R_list,
        S=S_list,
        P_leaf=P_leaf,
        r_leaf=r_term,
        c_leaf=c_term,
    )
