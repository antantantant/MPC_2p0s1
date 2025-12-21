# tree/riccati_tree.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from MPC_2p0s1.core.types import Tensor, ValueQuad
from MPC_2p0s1.core.tensor_ops import symmetrize
from MPC_2p0s1.games.base_lq_game import BaseLQGame
from MPC_2p0s1.core.action_spaces import BoxActionSpace
from MPC_2p0s1.tree.averaged_costs import AveragedCostData
from MPC_2p0s1.tree.belief_tree import BeliefTree
from MPC_2p0s1.tree.indexing import FullIaryTreeIndexer


@dataclass
class RiccatiSolution:
    """
    Result of a tree-structured Riccati backward pass.

    Attributes
    ----------
    P_nodes, r_nodes, c_nodes:
        Lists of length K+1. P_nodes[k] has shape (num_nodes_k, dx, dx),
        r_nodes[k] has shape (num_nodes_k, dx), and c_nodes[k] has shape
        (num_nodes_k,). They encode the conditional value functions:

            V_{k,ω}(x) = 0.5 x^T P_{k,ω} x + r_{k,ω}^T x + c_{k,ω}.

    K_u, kappa_u:
        Lists of length K. K_u[k] has shape (num_nodes_k, I, du, dx) and
        kappa_u[k] has shape (num_nodes_k, I, du). For each edge (k,ω,a),
        the optimal P1 control is

            u_{k,ω}^a(x) = K_u[k,ω,a] x + kappa_u[k,ω,a].

    K_v, kappa_v:
        Analogous objects for P2’s best responses with shapes
        (num_nodes_k, I, dv, dx) and (num_nodes_k, I, dv).

    Notes
    -----
    These feedback prototypes correspond to the inner convex–concave LQ
    game for fixed α on the tree. They are not yet restricted by the
    box constraints; clipping should be applied downstream if desired.
    """

    P_nodes: List[Tensor]
    r_nodes: List[Tensor]
    c_nodes: List[Tensor]

    K_u: List[Tensor]
    kappa_u: List[Tensor]

    K_v: List[Tensor]
    kappa_v: List[Tensor]

    def root_value_quad(self) -> ValueQuad:
        """Return the quadratic value representation at the root (k=0, ω=∅)."""
        P0 = self.P_nodes[0][0]
        r0 = self.r_nodes[0][0]
        c0 = self.c_nodes[0][0]
        return ValueQuad(P=P0, r=r0, c=c0)

    def value_at_root(self, x0: Tensor) -> Tensor:
        """
        Evaluate the scalar conditional value at the root given x0.
        """
        vq = self.root_value_quad()
        return vq.evaluate(x0)


def _local_lq_saddle(
    A: Tensor,
    B1: Tensor,
    B2: Tensor,
    tau: float,
    R_bar: Tensor,
    S_bar: Tensor,
    P_plus: Tensor,
    r_plus: Tensor,
    c_plus: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Solve the one-step LQ min–max problem at a single edge.

    This implements the local game

        min_u max_v  τ ℓ̄(u, v) + V_+(A x + B1 u + B2 v),

    where
        ℓ̄(u, v) = 0.5 u^T R̄ u - 0.5 v^T S̄ v,
        V_+(x^+) = 0.5 x^{+T} P_+ x^+ + r_+^T x^+ + c_+.

    It returns:
        - P_loc, r_loc, c_loc for the optimized value
              V_loc(x) = 0.5 x^T P_loc x + r_loc^T x + c_loc,
        - K_u, kappa_u and K_v, kappa_v such that
              u*(x) = K_u x + kappa_u,
              v*(x) = K_v x + kappa_v.

    All tensors are assumed to live on the same device and have consistent
    dtypes; error checking is intentionally light here for efficiency.
    """
    dx = A.shape[0]
    du = B1.shape[1]
    dv = B2.shape[1]

    # Quadratic terms in u and v (Hessian blocks)
    H_uu = tau * R_bar + B1.T @ P_plus @ B1                 # (du, du)
    H_uv = B1.T @ P_plus @ B2                               # (du, dv)
    H_vu = H_uv.T                                           # (dv, du)
    H_vv = -tau * S_bar + B2.T @ P_plus @ B2               # (dv, dv)

    H = torch.empty(du + dv, du + dv, device=A.device, dtype=A.dtype)
    H[:du, :du] = H_uu
    H[:du, du:] = H_uv
    H[du:, :du] = H_vu
    H[du:, du:] = H_vv

    # Cross terms with x
    F_u = B1.T @ P_plus @ A  # (du, dx)
    F_v = B2.T @ P_plus @ A  # (dv, dx)
    F = torch.cat([F_u, F_v], dim=0)  # (du+dv, dx)

    # Linear terms in u and v from r_plus
    f_u = B1.T @ r_plus  # (du,)
    f_v = B2.T @ r_plus  # (dv,)
    f = torch.cat([f_u, f_v], dim=0)  # (du+dv,)

    # Pure x-quadratic and linear terms from next-step value
    Q = A.T @ P_plus @ A                # (dx, dx)
    q = A.T @ r_plus                    # (dx,)
    c = c_plus                          # scalar

    # Solve for H^{-1} F and H^{-1} f without forming explicit inverse.
    # H may be indefinite but is assumed nonsingular under Isaacs/LQ conditions.
    #   (du+dv, du+dv) @ (du+dv, dx) = (du+dv, dx)
    sol_F = torch.linalg.solve(H, F)
    sol_f = torch.linalg.solve(H, f.unsqueeze(-1))  # (du+dv, 1)

    # feedback gains: [u*; v*] = - H^{-1} F x - H^{-1} f
    K = -sol_F                        # (du+dv, dx)
    b = -sol_f.squeeze(-1)           # (du+dv,)

    K_u = K[:du, :]       # (du, dx)
    K_v = K[du:, :]       # (dv, dx)
    kappa_u = b[:du]      # (du,)
    kappa_v = b[du:]      # (dv,)

    # Optimized value: V_loc(x) = 0.5 x^T P_loc x + r_loc^T x + c_loc
    # where
    #   P_loc = Q - F^T H^{-1} F = Q - F^T sol_F,
    #   r_loc = q - F^T H^{-1} f = q - F^T sol_f,
    #   c_loc = c - 0.5 f^T H^{-1} f = c - 0.5 f^T sol_f.
    P_loc = Q - F.T @ sol_F
    P_loc = symmetrize(P_loc)  # numerical symmetrization

    r_loc = q - (F.T @ sol_f).squeeze(-1)  # (dx,)
    c_loc = c - 0.5 * torch.dot(f, sol_f.squeeze(-1))

    return P_loc, r_loc, c_loc, K_u, kappa_u, K_v, kappa_v


def riccati_backward(
    game: BaseLQGame,
    belief_tree: BeliefTree,
    avg_costs: AveragedCostData,
    action_space: BoxActionSpace | None = None,
) -> RiccatiSolution:
    """
    Run a tree-structured Riccati backward pass for fixed α.

    Parameters
    ----------
    game:
        LQ game instance providing dynamics and cost matrices.
    belief_tree:
        BeliefTree containing p_{k,ω} and λ_{k,ω}^a.
    avg_costs:
        Precomputed averaged running costs and terminal values.
    action_space:
        Optional BoxActionSpace. If provided, the computed feedback
        controls can be clipped downstream for rollouts. The Riccati
        update itself treats the problem as unconstrained.

    Returns
    -------
    RiccatiSolution
        Object containing node value matrices (P,r,c) and feedback
        prototypes (K_u, kappa_u, K_v, kappa_v) on each edge.
    """
    indexer: FullIaryTreeIndexer = belief_tree.indexer
    K = indexer.K
    I = indexer.I

    dx = game.dx
    du = game.du
    dv = game.dv
    tau = game.cfg.tau

    A = game.A
    B1 = game.B1
    B2 = game.B2

    device = game.device_resolved
    dtype = game.dtype

    # Node value storage: depth-major
    P_nodes: List[Tensor] = [None for _ in range(K + 1)]  # type: ignore
    r_nodes: List[Tensor] = [None for _ in range(K + 1)]  # type: ignore
    c_nodes: List[Tensor] = [None for _ in range(K + 1)]  # type: ignore

    # Initialize leaves (depth K) from avg_costs
    num_leaves = indexer.node_count(K)
    P_nodes[K] = avg_costs.P_leaf.clone()
    r_nodes[K] = avg_costs.r_leaf.clone()
    c_nodes[K] = avg_costs.c_leaf.clone()

    # Feedback storage for edges
    K_u: List[Tensor] = [None for _ in range(K)]           # type: ignore
    kappa_u: List[Tensor] = [None for _ in range(K)]       # type: ignore
    K_v: List[Tensor] = [None for _ in range(K)]           # type: ignore
    kappa_v: List[Tensor] = [None for _ in range(K)]       # type: ignore

    # Backward sweep over depths
    for k in reversed(range(K)):
        num_nodes = indexer.node_count(k)
        num_nodes_next = indexer.node_count(k + 1)

        P_k = torch.zeros(num_nodes, dx, dx, device=device, dtype=dtype)
        r_k = torch.zeros(num_nodes, dx, device=device, dtype=dtype)
        c_k = torch.zeros(num_nodes, device=device, dtype=dtype)

        K_u_k = torch.empty(num_nodes, I, du, dx, device=device, dtype=dtype)
        kappa_u_k = torch.empty(num_nodes, I, du, device=device, dtype=dtype)
        K_v_k = torch.empty(num_nodes, I, dv, dx, device=device, dtype=dtype)
        kappa_v_k = torch.empty(num_nodes, I, dv, device=device, dtype=dtype)

        P_next = P_nodes[k + 1]
        r_next = r_nodes[k + 1]
        c_next = c_nodes[k + 1]

        R_bar_k = avg_costs.R_bar[k]  # (num_nodes, I, du, du)
        S_bar_k = avg_costs.S_bar[k]  # (num_nodes, I, dv, dv)
        lam_edge_k = belief_tree.lambda_edge[k]  # (num_nodes, I)

        for node_idx in range(num_nodes):
            for a in range(I):
                child_idx = indexer.child_index(k, node_idx, a)
                if not (0 <= child_idx < num_nodes_next):
                    raise RuntimeError(
                        f"Inconsistent tree indexing at depth {k}: "
                        f"child_idx={child_idx}, num_nodes_next={num_nodes_next}"
                    )

                P_plus = P_next[child_idx]
                r_plus = r_next[child_idx]
                c_plus = c_next[child_idx]

                R_bar = R_bar_k[node_idx, a]
                S_bar = S_bar_k[node_idx, a]

                (
                    P_loc,
                    r_loc,
                    c_loc,
                    K_u_edge,
                    kappa_u_edge,
                    K_v_edge,
                    kappa_v_edge,
                ) = _local_lq_saddle(
                    A=A,
                    B1=B1,
                    B2=B2,
                    tau=tau,
                    R_bar=R_bar,
                    S_bar=S_bar,
                    P_plus=P_plus,
                    r_plus=r_plus,
                    c_plus=c_plus,
                )

                # Aggregate over child edges with weights λ_{k,ω}^a
                lam_a = lam_edge_k[node_idx, a]
                P_k[node_idx] += lam_a * P_loc
                r_k[node_idx] += lam_a * r_loc
                c_k[node_idx] += lam_a * c_loc

                K_u_k[node_idx, a] = K_u_edge
                kappa_u_k[node_idx, a] = kappa_u_edge
                K_v_k[node_idx, a] = K_v_edge
                kappa_v_k[node_idx, a] = kappa_v_edge

        P_nodes[k] = P_k
        r_nodes[k] = r_k
        c_nodes[k] = c_k

        K_u[k] = K_u_k
        kappa_u[k] = kappa_u_k
        K_v[k] = K_v_k
        kappa_v[k] = kappa_v_k

    return RiccatiSolution(
        P_nodes=P_nodes,
        r_nodes=r_nodes,
        c_nodes=c_nodes,
        K_u=K_u,
        kappa_u=kappa_u,
        K_v=K_v,
        kappa_v=kappa_v,
    )