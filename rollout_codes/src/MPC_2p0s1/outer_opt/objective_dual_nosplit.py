from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional, Tuple

import torch

from MPC_2p0s1.core.action_spaces import BoxActionSpace
from MPC_2p0s1.core.types import Tensor
from MPC_2p0s1.games.base_lq_game import BaseLQGame
from MPC_2p0s1.outer_opt.dual_split_network import DualNoSplitP1Net, DualNoSplitP2Net
from MPC_2p0s1.outer_opt.dual_split_param import DualNoSplitP1PolicyParam, DualNoSplitP2PolicyParam
from MPC_2p0s1.tree.belief_tree import BeliefTree
from MPC_2p0s1.tree.dual_solver_nosplit import (
    DualNoSplitConfig,
    rollout_dual_nosplit,
    rollout_dual_nosplit_network,
    rollout_dual_nosplit_primal_p1,
)
from MPC_2p0s1.tree.indexing import FullIaryTreeIndexer
from MPC_2p0s1.tree.riccati_tree import RiccatiSolution


def dual_objective_nosplit(
    game: BaseLQGame,
    p1_policy: DualNoSplitP1PolicyParam,
    p2_policy: DualNoSplitP2PolicyParam,
    *,
    x0: Optional[Tensor] = None,
    phat0: Optional[Tensor] = None,
    action_space: Optional[BoxActionSpace] = None,
    solver_cfg: Optional[DualNoSplitConfig] = None,
    return_details: bool = False,
) -> Tensor | Tuple[Tensor, Dict[str, object]]:
    """
    Simple no-split dual objective.

    Computes
      V = max_i [phat_K[i] - g_i(x_K)]
    with dynamics
      phat_{k+1} = phat_k - l_k(u_k, v_k).
    """
    device = game.device_resolved
    dtype = game.dtype

    if solver_cfg is None:
        solver_cfg = DualNoSplitConfig()

    if x0 is None:
        x0 = game.default_initial_state()
    if phat0 is None:
        phat0 = torch.zeros(game.I, device=device, dtype=dtype)

    x0 = x0.to(device=device, dtype=dtype)
    phat0 = phat0.to(device=device, dtype=dtype)

    ro = rollout_dual_nosplit(
        game=game,
        p1_policy=p1_policy,
        p2_policy=p2_policy,
        x0=x0,
        phat0=phat0,
        cfg=solver_cfg,
        action_space=action_space,
    )

    loss = ro.dual_value
    if not return_details:
        return loss

    details: Dict[str, object] = {
        "terminal_terms": ro.terminal_terms,
        "expected_type_cost": ro.expected_type_cost,
        "x_traj": ro.x_traj,
        "phat_traj": ro.phat_traj,
        "solver_cfg": asdict(solver_cfg),
    }
    return loss, details


def dual_objective_nosplit_network(
    game: BaseLQGame,
    p1_model: DualNoSplitP1Net,
    p2_model: DualNoSplitP2Net,
    *,
    x0: Optional[Tensor] = None,
    phat0: Optional[Tensor] = None,
    action_space: Optional[BoxActionSpace] = None,
    solver_cfg: Optional[DualNoSplitConfig] = None,
    return_details: bool = False,
) -> Tensor | Tuple[Tensor, Dict[str, object]]:
    """
    No-split dual objective with neural feedback policies for both players.
    """
    device = game.device_resolved
    dtype = game.dtype

    if solver_cfg is None:
        solver_cfg = DualNoSplitConfig()

    if x0 is None:
        x0 = game.default_initial_state()
    if phat0 is None:
        phat0 = torch.zeros(game.I, device=device, dtype=dtype)

    x0 = x0.to(device=device, dtype=dtype)
    phat0 = phat0.to(device=device, dtype=dtype)

    ro = rollout_dual_nosplit_network(
        game=game,
        p1_model=p1_model,
        p2_model=p2_model,
        x0=x0,
        phat0=phat0,
        cfg=solver_cfg,
        action_space=action_space,
    )

    loss = ro.dual_value
    if not return_details:
        return loss

    details: Dict[str, object] = {
        "terminal_terms": ro.terminal_terms,
        "expected_type_cost": ro.expected_type_cost,
        "x_traj": ro.x_traj,
        "phat_traj": ro.phat_traj,
        "solver_cfg": asdict(solver_cfg),
    }
    return loss, details


def dual_objective_nosplit_primal_p1(
    game: BaseLQGame,
    indexer: FullIaryTreeIndexer,
    alpha: Tensor,
    belief_tree: BeliefTree,
    riccati_sol: RiccatiSolution,
    p2_policy: DualNoSplitP2PolicyParam,
    *,
    x0: Optional[Tensor] = None,
    p0: Optional[Tensor] = None,
    phat0: Optional[Tensor] = None,
    action_space: Optional[BoxActionSpace] = None,
    solver_cfg: Optional[DualNoSplitConfig] = None,
    return_details: bool = False,
) -> Tensor | Tuple[Tensor, Dict[str, object]]:
    """
    No-split dual objective with P1 fixed to primal-game policy.

    P1 controls are taken from primal alpha+belief-tree+Riccati objects,
    while P2 controls come from the no-split dual parameterization.
    """
    device = game.device_resolved
    dtype = game.dtype

    if solver_cfg is None:
        solver_cfg = DualNoSplitConfig()

    if x0 is None:
        x0 = game.default_initial_state()
    if p0 is None:
        p0 = game.default_prior()
    if phat0 is None:
        phat0 = torch.zeros(game.I, device=device, dtype=dtype)

    x0 = x0.to(device=device, dtype=dtype)
    p0 = p0.to(device=device, dtype=dtype)
    phat0 = phat0.to(device=device, dtype=dtype)

    ro = rollout_dual_nosplit_primal_p1(
        game=game,
        indexer=indexer,
        alpha=alpha,
        belief_tree=belief_tree,
        riccati_sol=riccati_sol,
        p2_policy=p2_policy,
        x0=x0,
        p0=p0,
        phat0=phat0,
        cfg=solver_cfg,
        action_space=action_space,
    )

    loss = ro.dual_value
    if not return_details:
        return loss

    details: Dict[str, object] = {
        "terminal_terms": ro.terminal_terms,
        "expected_type_cost": ro.expected_type_cost,
        "x_traj_by_type": ro.x_traj_by_type,
        "belief_traj_by_type": ro.belief_traj_by_type,
        "phat_traj": ro.phat_traj,
        "solver_cfg": asdict(solver_cfg),
    }
    return loss, details
