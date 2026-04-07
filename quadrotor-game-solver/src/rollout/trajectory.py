"""Rollout a single trajectory through the public tree under α and feedback.

Mirrors ``nl_sqp/rollout_drone.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..game.quadrotor_game import Hexner3DQuadrotorGame
from ..tree.indexing import FullIaryTreeIndexer
from ..tree.belief_tree import BeliefTree
from ..solvers.riccati import RiccatiSolution
from ..solvers.action_spaces import BoxActionSpace


@dataclass
class RolloutResult:
    x_traj: Tensor              # (K+1, dx)
    u_traj: Tensor              # (K, du)
    v_traj: Tensor              # (K, dv)
    belief_traj: Tensor         # (K+1, I)
    proto_indices: torch.LongTensor   # (K,)


def rollout_trajectory(
    *,
    game: Hexner3DQuadrotorGame,
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
    """Roll out a single trajectory (one realised type) under current α."""

    K = indexer.K
    I = indexer.I
    dx = game.dx
    du = game.du
    dv = game.dv

    if x0.shape != (dx,):
        raise ValueError(f"x0 must have shape ({dx},), got {tuple(x0.shape)}")
    if not (0 <= type_index < I):
        raise ValueError(f"type_index out of range [0, {I})")

    max_nodes = indexer.max_nodes_per_depth
    if alpha.shape != (K, max_nodes, I, I):
        raise ValueError(
            f"alpha shape mismatch, expected {(K, max_nodes, I, I)}, "
            f"got {tuple(alpha.shape)}"
        )

    device = game.device
    dtype = game.dtype

    if generator is None:
        generator = torch.Generator(device=device)

    x_traj = torch.empty(K + 1, dx, device=device, dtype=dtype)
    u_traj = torch.empty(K, du, device=device, dtype=dtype)
    v_traj = torch.empty(K, dv, device=device, dtype=dtype)
    belief_traj = torch.empty(K + 1, I, device=device, dtype=dtype)
    proto_indices = torch.empty(K, dtype=torch.long, device=device)

    x = x0.to(device=device, dtype=dtype)
    node_idx = 0
    belief = belief_tree.beliefs[0][node_idx]

    x_traj[0] = x
    belief_traj[0] = belief

    for k in range(K):
        # Choose prototype a
        alpha_row = alpha[k, node_idx, type_index]   # (I,)
        if sample_actions:
            a = torch.multinomial(
                alpha_row, num_samples=1, replacement=True,
                generator=generator,
            ).squeeze(0)
        else:
            a = torch.argmax(alpha_row)
        a_int = int(a.item())
        proto_indices[k] = a

        # Feedback gains
        K_u_edge = riccati_sol.K_u[k][node_idx, a_int]
        K_v_edge = riccati_sol.K_v[k][node_idx, a_int]

        # Aggregate feedforward
        lam_edge = belief_tree.lambda_edge[k][node_idx]
        kappa_u_all = riccati_sol.kappa_u[k][node_idx]
        kappa_v_all = riccati_sol.kappa_v[k][node_idx]
        kappa_u_agg = torch.einsum("a, ad -> d", lam_edge, kappa_u_all)
        kappa_v_agg = torch.einsum("a, ad -> d", lam_edge, kappa_v_all)

        u = K_u_edge @ x + kappa_u_agg
        v = K_v_edge @ x + kappa_v_agg

        if action_space is not None:
            u = action_space.clip_u(u)
            v = action_space.clip_v(v)

        u_traj[k] = u
        v_traj[k] = v

        # Next state
        x = game.step(x, u, v)
        x_traj[k + 1] = x

        # Belief update
        child_idx = indexer.child_index(k, node_idx, a_int)
        belief = belief_tree.beliefs[k + 1][child_idx]
        belief_traj[k + 1] = belief
        node_idx = child_idx

    return RolloutResult(
        x_traj=x_traj,
        u_traj=u_traj,
        v_traj=v_traj,
        belief_traj=belief_traj,
        proto_indices=proto_indices,
    )
