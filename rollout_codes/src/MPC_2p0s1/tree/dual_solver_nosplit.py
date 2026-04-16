from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from MPC_2p0s1.core.action_spaces import BoxActionSpace
from MPC_2p0s1.core.types import Tensor
from MPC_2p0s1.games.base_lq_game import BaseLQGame
from MPC_2p0s1.outer_opt.dual_split_network import DualNoSplitP1Net, DualNoSplitP2Net
from MPC_2p0s1.outer_opt.dual_split_param import DualNoSplitP1PolicyParam, DualNoSplitP2PolicyParam
from MPC_2p0s1.tree.belief_tree import BeliefTree
from MPC_2p0s1.tree.indexing import FullIaryTreeIndexer
from MPC_2p0s1.tree.riccati_tree import RiccatiSolution


@dataclass
class DualNoSplitConfig:
    """
    Runtime options for simple no-split dual rollout.
    """

    terminal_smooth_temp: float = 0.0


@dataclass
class DualNoSplitRolloutResult:
    """
    Outputs of a single-path no-split dual rollout.
    """

    dual_value: Tensor            # ()
    terminal_terms: Tensor        # (I,)
    expected_type_cost: Tensor    # (I,)
    x_traj: Tensor                # (K+1, dx)
    phat_traj: Tensor             # (K+1, I)


@dataclass
class DualNoSplitPrimalP1RolloutResult:
    """
    Outputs for no-split dual rollout when P1 policy is fixed by primal solution.

    Here state/belief trajectories are tracked per realized type.
    """

    dual_value: Tensor               # ()
    terminal_terms: Tensor           # (I,)
    expected_type_cost: Tensor       # (I,)
    x_traj_by_type: Tensor           # (K+1, I, dx)
    belief_traj_by_type: Tensor      # (K+1, I, I)
    phat_traj: Tensor                # (K+1, I)


def _running_cost_per_type(game: BaseLQGame, u: Tensor, v: Tensor) -> Tensor:
    """
    Type-wise running costs l_i(u,v) for one step.

    Returns
    -------
    Tensor
        Shape (I,)
    """
    tau = float(game.cfg.tau)
    run_u = 0.5 * tau * torch.einsum("iab,a,b->i", game.R, u, u)
    run_v = 0.5 * tau * torch.einsum("iab,a,b->i", game.S, v, v)
    return run_u - run_v


def _terminal_cost_per_type(game: BaseLQGame, x: Tensor) -> Tensor:
    """
    Type-wise terminal costs g_i(x).

    Returns
    -------
    Tensor
        Shape (I,)
    """
    quad = 0.5 * torch.einsum("iab,a,b->i", game.Q, x, x)
    lin = torch.einsum("ia,a->i", game.q, x)
    return quad + lin + game.c


def rollout_dual_nosplit(
    game: BaseLQGame,
    p1_policy: DualNoSplitP1PolicyParam,
    p2_policy: DualNoSplitP2PolicyParam,
    *,
    x0: Tensor,
    phat0: Tensor,
    cfg: DualNoSplitConfig,
    action_space: Optional[BoxActionSpace] = None,
) -> DualNoSplitRolloutResult:
    """
    Roll out a single-path no-split dual game.

    Dynamics:
      x_{k+1} = A x_k + B1 u_k + B2 v_k
      phat_{k+1} = phat_k - l_k(u_k, v_k), elementwise over types.

    Objective:
      max_i [phat_K[i] - g_i(x_K)]  (or smooth-max if temperature > 0).
    """
    device = game.device_resolved
    dtype = game.dtype

    K = int(game.cfg.K)
    I = int(game.I)

    if p1_policy.K != K:
        raise ValueError(f"p1_policy.K={p1_policy.K} does not match game K={K}.")
    if p2_policy.K != K:
        raise ValueError(f"p2_policy.K={p2_policy.K} does not match game K={K}.")

    if x0.shape != (game.dx,):
        raise ValueError(f"x0 must have shape ({game.dx},), got {tuple(x0.shape)}")
    if phat0.shape != (I,):
        raise ValueError(f"phat0 must have shape ({I},), got {tuple(phat0.shape)}")

    x = x0.to(device=device, dtype=dtype)
    phat = phat0.to(device=device, dtype=dtype)

    x_traj = torch.empty((K + 1, game.dx), device=device, dtype=dtype)
    phat_traj = torch.empty((K + 1, I), device=device, dtype=dtype)

    x_traj[0] = x
    phat_traj[0] = phat

    type_cost = torch.zeros((I,), device=device, dtype=dtype)

    for k in range(K):
        u = p1_policy.action_at(k)
        v = p2_policy.action_at(k)

        if action_space is not None:
            u = action_space.clip_u(u)
            v = action_space.clip_v(v)

        l_type = _running_cost_per_type(game, u, v)
        type_cost = type_cost + l_type

        x = game.step_dynamics(x, u, v)
        phat = phat - l_type

        x_traj[k + 1] = x
        phat_traj[k + 1] = phat

    g_type = _terminal_cost_per_type(game, x)
    type_cost = type_cost + g_type

    terminal_terms = phat - g_type
    temp = float(cfg.terminal_smooth_temp)
    if temp > 0.0:
        dual_value = temp * torch.logsumexp(terminal_terms / temp, dim=0)
    else:
        dual_value = terminal_terms.max(dim=0).values

    return DualNoSplitRolloutResult(
        dual_value=dual_value,
        terminal_terms=terminal_terms,
        expected_type_cost=type_cost,
        x_traj=x_traj,
        phat_traj=phat_traj,
    )


def rollout_dual_nosplit_network(
    game: BaseLQGame,
    p1_model: DualNoSplitP1Net,
    p2_model: DualNoSplitP2Net,
    *,
    x0: Tensor,
    phat0: Tensor,
    cfg: DualNoSplitConfig,
    action_space: Optional[BoxActionSpace] = None,
) -> DualNoSplitRolloutResult:
    """
    Roll out a single-path no-split dual game with neural action policies.

    Policies are feedback maps:
      u_k = pi1(x_k, phat_k, t_k),
      v_k = pi2(x_k, phat_k, t_k).
    """
    device = game.device_resolved
    dtype = game.dtype

    K = int(game.cfg.K)
    I = int(game.I)

    if x0.shape != (game.dx,):
        raise ValueError(f"x0 must have shape ({game.dx},), got {tuple(x0.shape)}")
    if phat0.shape != (I,):
        raise ValueError(f"phat0 must have shape ({I},), got {tuple(phat0.shape)}")

    x = x0.to(device=device, dtype=dtype)
    phat = phat0.to(device=device, dtype=dtype)

    x_traj = torch.empty((K + 1, game.dx), device=device, dtype=dtype)
    phat_traj = torch.empty((K + 1, I), device=device, dtype=dtype)

    x_traj[0] = x
    phat_traj[0] = phat

    type_cost = torch.zeros((I,), device=device, dtype=dtype)

    for k in range(K):
        t_norm = float(k) / float(max(1, K))
        x_b = x.view(1, -1)
        phat_b = phat.view(1, -1)
        u = p1_model(x_b, phat_b, t_norm).squeeze(0)
        v = p2_model(x_b, phat_b, t_norm).squeeze(0)

        if action_space is not None:
            u = action_space.clip_u(u)
            v = action_space.clip_v(v)

        l_type = _running_cost_per_type(game, u, v)
        type_cost = type_cost + l_type

        x = game.step_dynamics(x, u, v)
        phat = phat - l_type

        x_traj[k + 1] = x
        phat_traj[k + 1] = phat

    g_type = _terminal_cost_per_type(game, x)
    type_cost = type_cost + g_type

    terminal_terms = phat - g_type
    temp = float(cfg.terminal_smooth_temp)
    if temp > 0.0:
        dual_value = temp * torch.logsumexp(terminal_terms / temp, dim=0)
    else:
        dual_value = terminal_terms.max(dim=0).values

    return DualNoSplitRolloutResult(
        dual_value=dual_value,
        terminal_terms=terminal_terms,
        expected_type_cost=type_cost,
        x_traj=x_traj,
        phat_traj=phat_traj,
    )


def rollout_dual_nosplit_primal_p1(
    game: BaseLQGame,
    indexer: FullIaryTreeIndexer,
    alpha: Tensor,
    belief_tree: BeliefTree,
    riccati_sol: RiccatiSolution,
    p2_policy: DualNoSplitP2PolicyParam,
    *,
    x0: Tensor,
    p0: Tensor,
    phat0: Tensor,
    cfg: DualNoSplitConfig,
    action_space: Optional[BoxActionSpace] = None,
) -> DualNoSplitPrimalP1RolloutResult:
    """
    No-split dual rollout with P1 controls from primal policy objects.

    For each type i:
      - P1 action index uses primal alpha at (k, node, i),
      - P1 control uses primal Riccati gains on that edge,
      - belief/node evolve by primal public signal tree.

    P2 remains no-split and type-independent in this simplified setting.
    """
    device = game.device_resolved
    dtype = game.dtype

    K = int(indexer.K)
    I = int(game.I)

    if p2_policy.K != K:
        raise ValueError(f"p2_policy.K={p2_policy.K} does not match indexer K={K}.")
    if alpha.shape[0] != K:
        raise ValueError(f"alpha first dim must equal K={K}, got {alpha.shape[0]}.")
    if x0.shape != (game.dx,):
        raise ValueError(f"x0 must have shape ({game.dx},), got {tuple(x0.shape)}")
    if p0.shape != (I,):
        raise ValueError(f"p0 must have shape ({I},), got {tuple(p0.shape)}")
    if phat0.shape != (I,):
        raise ValueError(f"phat0 must have shape ({I},), got {tuple(phat0.shape)}")

    x_by_type = x0.to(device=device, dtype=dtype).view(1, -1).repeat(I, 1)
    belief_by_type = p0.to(device=device, dtype=dtype).view(1, -1).repeat(I, 1)
    node_idx_by_type = torch.zeros((I,), dtype=torch.long, device=device)
    phat = phat0.to(device=device, dtype=dtype).clone()

    x_traj_by_type = torch.empty((K + 1, I, game.dx), device=device, dtype=dtype)
    belief_traj_by_type = torch.empty((K + 1, I, I), device=device, dtype=dtype)
    phat_traj = torch.empty((K + 1, I), device=device, dtype=dtype)
    type_cost = torch.zeros((I,), device=device, dtype=dtype)

    x_traj_by_type[0] = x_by_type
    belief_traj_by_type[0] = belief_by_type
    phat_traj[0] = phat

    for k in range(K):
        v = p2_policy.action_at(k)
        if action_space is not None:
            v = action_space.clip_v(v)

        next_x_by_type = torch.empty_like(x_by_type)
        next_belief_by_type = torch.empty_like(belief_by_type)
        next_node_idx_by_type = torch.empty_like(node_idx_by_type)

        for i in range(I):
            node_idx_i = int(node_idx_by_type[i].item())
            alpha_row = alpha[k, node_idx_i, i]
            a_i = int(torch.argmax(alpha_row).item())

            # Primal P1 control on selected public edge.
            K_u = riccati_sol.K_u[k][node_idx_i, a_i]          # (du, dx)
            lam_edge = belief_tree.lambda_edge[k][node_idx_i]  # (I,)
            kappa_u_all = riccati_sol.kappa_u[k][node_idx_i]   # (I, du)
            kappa_u = torch.einsum("a,ad->d", lam_edge, kappa_u_all)
            u_i = K_u @ x_by_type[i] + kappa_u
            if action_space is not None:
                u_i = action_space.clip_u(u_i)

            # Type-specific running cost component.
            tau = float(game.cfg.tau)
            run_u_i = 0.5 * tau * torch.einsum("a,ab,b->", u_i, game.R[i], u_i)
            run_v_i = 0.5 * tau * torch.einsum("a,ab,b->", v, game.S[i], v)
            ell_i = run_u_i - run_v_i
            type_cost[i] = type_cost[i] + ell_i
            phat[i] = phat[i] - ell_i

            x_next_i = game.step_dynamics(x_by_type[i], u_i, v)
            child_idx_i = indexer.child_index(k, node_idx_i, a_i)
            belief_next_i = belief_tree.beliefs[k + 1][child_idx_i]

            next_x_by_type[i] = x_next_i
            next_belief_by_type[i] = belief_next_i
            next_node_idx_by_type[i] = child_idx_i

        x_by_type = next_x_by_type
        belief_by_type = next_belief_by_type
        node_idx_by_type = next_node_idx_by_type

        x_traj_by_type[k + 1] = x_by_type
        belief_traj_by_type[k + 1] = belief_by_type
        phat_traj[k + 1] = phat

    # Terminal per-type term uses each type's own terminal state.
    g_type = torch.empty((I,), device=device, dtype=dtype)
    for i in range(I):
        g_i = game.terminal_cost_type(i, x_by_type[i])
        g_type[i] = g_i
        type_cost[i] = type_cost[i] + g_i

    terminal_terms = phat - g_type
    temp = float(cfg.terminal_smooth_temp)
    if temp > 0.0:
        dual_value = temp * torch.logsumexp(terminal_terms / temp, dim=0)
    else:
        dual_value = terminal_terms.max(dim=0).values

    return DualNoSplitPrimalP1RolloutResult(
        dual_value=dual_value,
        terminal_terms=terminal_terms,
        expected_type_cost=type_cost,
        x_traj_by_type=x_traj_by_type,
        belief_traj_by_type=belief_traj_by_type,
        phat_traj=phat_traj,
    )
