# tree/rollout.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from MPC_2p0s1.core.types import Tensor
from MPC_2p0s1.core.action_spaces import BoxActionSpace
from MPC_2p0s1.games.base_lq_game import BaseLQGame
from MPC_2p0s1.tree.belief_tree import BeliefTree
from MPC_2p0s1.tree.indexing import FullIaryTreeIndexer
from MPC_2p0s1.tree.riccati_tree import RiccatiSolution


@dataclass
class RolloutResult:
    """
    Container for a single simulated trajectory on the tree.

    Attributes
    ----------
    x_traj:
        Tensor of shape (K+1, dx) with states x_0, ..., x_K.
    u_traj:
        Tensor of shape (K, du) with P1 controls.
    v_traj:
        Tensor of shape (K, dv) with P2 controls.
    belief_traj:
        Tensor of shape (K+1, I) with public beliefs p_0, ..., p_K.
    proto_indices:
        Long tensor of shape (K,) with the chosen prototype index a_k
        at each depth k.
    """

    x_traj: Tensor
    u_traj: Tensor
    v_traj: Tensor
    belief_traj: Tensor
    proto_indices: torch.LongTensor


def rollout_trajectory(
    game: BaseLQGame,
    indexer: FullIaryTreeIndexer,
    belief_tree: BeliefTree,
    riccati_sol: RiccatiSolution,
    alpha: Tensor,
    x0: Tensor,
    type_index: int,
    action_space: Optional[BoxActionSpace] = None,
    sample_actions: bool = False,
    generator: Optional[torch.Generator] = None,
) -> RolloutResult:
    """
    Roll out a single trajectory under the current α and feedback prototypes.

    Parameters
    ----------
    game:
        LQ game instance.
    indexer:
        Tree indexer (should match the one used for belief_tree / α).
    belief_tree:
        BeliefTree containing p_{k,ω}. The rollout path follows the
        same indexing, i.e., node (k, node_idx) and its children.
    riccati_sol:
        RiccatiSolution providing feedback gains K_u, kappa_u, K_v,
        kappa_v on each edge.
    alpha:
        Tensor of shape (K, max_nodes, I, I) with belief-splitting
        parameters α_{k,ω,i}^a (already softmax-normalized in the
        α-parameter module).
    x0:
        Initial state x0 ∈ R^{dx}.
    type_index:
        Realized payoff type i ∈ {0, ..., I-1} for P1.
    action_space:
        Optional BoxActionSpace; if provided, controls are clipped
        into the feasible box before applying the dynamics.
    sample_actions:
        If True, sample P1's prototype a_k at each node according to
        α_{k,ω,type}^·. If False, use argmax for a deterministic path.
    generator:
        Optional torch.Generator used for sampling.

    Returns
    -------
    RolloutResult
        State, control, belief, and prototype index trajectories.
    """
    K = indexer.K
    I = indexer.I

    dx = game.dx
    du = game.du
    dv = game.dv

    device = game.device_resolved
    dtype = game.dtype

    if x0.shape != (dx,):
        raise ValueError(f"rollout_trajectory: x0 must have shape ({dx},), got {tuple(x0.shape)}")
    if not (0 <= type_index < I):
        raise ValueError(f"rollout_trajectory: type_index={type_index} out of range [0, {I})")

    max_nodes = indexer.max_nodes_per_depth
    if alpha.shape != (K, max_nodes, I, I):
        raise ValueError(
            "rollout_trajectory: alpha shape mismatch, expected "
            f"({K}, {max_nodes}, {I}, {I}), got {tuple(alpha.shape)}"
        )

    if generator is None:
        generator = torch.Generator(device=device)

    x_traj = torch.empty(K + 1, dx, device=device, dtype=dtype)
    u_traj = torch.empty(K, du, device=device, dtype=dtype)
    v_traj = torch.empty(K, dv, device=device, dtype=dtype)
    belief_traj = torch.empty(K + 1, I, device=device, dtype=dtype)
    proto_indices = torch.empty(K, dtype=torch.long, device=device)

    # Initial conditions at root
    x = x0.to(device=device, dtype=dtype)
    node_idx = 0  # root at depth 0
    belief = belief_tree.beliefs[0][node_idx]

    x_traj[0] = x
    belief_traj[0] = belief

    for k in range(K):
        num_nodes_k = indexer.node_count(k)
        if node_idx >= num_nodes_k:
            raise RuntimeError(
                f"rollout_trajectory: node_idx={node_idx} out of range at depth {k} "
                f"(num_nodes_k={num_nodes_k})"
            )

        # P1 prototype selection for this type
        alpha_row = alpha[k, node_idx, type_index]  # (I,)
        if sample_actions:
            # Sample according to α_{k,ω,type}^a
            dist = torch.distributions.Categorical(probs=alpha_row)
            a = dist.sample()
        else:
            # Deterministic choice: argmax over prototypes
            a = torch.argmax(alpha_row)

        a_int = int(a.item())
        proto_indices[k] = a

        # ----------------------------------------------------------------
        # Control computation with correct timing for belief revelation
        # ----------------------------------------------------------------
        # The feedback gain K_u is the same for all actions (since P and R
        # are type-independent), so we can use the action-specific one.
        K_u_edge = riccati_sol.K_u[k][node_idx, a_int]        # (du, dx)
        K_v_edge = riccati_sol.K_v[k][node_idx, a_int]        # (dv, dx)
        
        kappa_u_edge = riccati_sol.kappa_u[k][node_idx, a_int]    # (du,)
        kappa_v_edge = riccati_sol.kappa_v[k][node_idx, a_int]    # (dv,)
        u = K_u_edge @ x + kappa_u_edge
        v = K_v_edge @ x + kappa_v_edge

        ## UPDATE: Aggregation is FUNDAMENTALLY WRONG!! CHECK GT WITH CONT. GAME
        # IMPORTANT: We must AGGREGATE the feedforward terms (kappa_u, kappa_v)
        # over actions using lambda_edge (edge probabilities from prior belief).
        #
        # PREVIOUS CODE IMPLEMENTS PRIMAL GAME
        #   kappa_u_edge = riccati_sol.kappa_u[k][node_idx, a_int]
        #   u = K_u_edge @ x + kappa_u_edge
        #
        # This used the action-specific kappa_u, which is computed from the
        # child's value function at depth k+1. The child's value incorporates
        # the POSTERIOR belief (after the action is observed), so the control
        # at step k was using revealed information one time step too early.
        #
        # For example, with reveal at k=5:
        #   - Belief at k=5 is [0.5, 0.5] (prior, not yet revealed)
        #   - Belief at k=6 is [1, 0] (posterior, revealed)
        #   - OLD: Control at k=5 used kappa_u from child with belief [1,0] → WRONG
        #   - NEW: Control at k=5 uses aggregated kappa_u from prior [0.5,0.5] → CORRECT
        #
        # The "fix": We can project this to make it non-anticipative: aggregate kappa_u over actions weighted by lambda_edge,
        # which are the edge probabilities computed from the PRIOR belief.
        # lam_edge = belief_tree.lambda_edge[k][node_idx]       # (I,)
        # kappa_u_all = riccati_sol.kappa_u[k][node_idx]        # (I, du)
        # kappa_v_all = riccati_sol.kappa_v[k][node_idx]        # (I, dv)
        # kappa_u_agg = torch.einsum('a, ad -> d', lam_edge, kappa_u_all)  # (du,)
        # kappa_v_agg = torch.einsum('a, ad -> d', lam_edge, kappa_v_all)  # (dv,)

        # u = K_u_edge @ x + kappa_u_agg
        # v = K_v_edge @ x + kappa_v_agg

        if action_space is not None:
            u = action_space.clip_u(u)
            v = action_space.clip_v(v)

        u_traj[k] = u
        v_traj[k] = v

        # State update
        x = game.step_dynamics(x, u, v)
        x_traj[k + 1] = x

        # Belief update follows the precomputed belief tree
        child_idx = indexer.child_index(k, node_idx, a_int)
        belief = belief_tree.beliefs[k + 1][child_idx]
        belief_traj[k + 1] = belief

        # Move to next node
        node_idx = child_idx

    return RolloutResult(
        x_traj=x_traj,
        u_traj=u_traj,
        v_traj=v_traj,
        belief_traj=belief_traj,
        proto_indices=proto_indices,
    )