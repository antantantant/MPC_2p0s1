from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional, Tuple

import torch

from MPC_2p0s1.core.action_spaces import BoxActionSpace
from MPC_2p0s1.core.types import Tensor
from MPC_2p0s1.games.base_lq_game import BaseLQGame
from MPC_2p0s1.outer_opt.dual_split_network import DualSplitP1BRNet, DualSplitP2Net
from MPC_2p0s1.outer_opt.dual_split_param import DualNoSplitP1PolicyParam, DualSplitP2PolicyParam
from MPC_2p0s1.tree.dual_solver_split import (
    DualSplitConfig,
    DualSplitRolloutResult,
    rollout_dual_split_paths,
    rollout_dual_split_paths_network,
)


def dual_objective_split(
    game: BaseLQGame,
    p1_policy: DualNoSplitP1PolicyParam,
    p2_policy: DualSplitP2PolicyParam,
    *,
    paths: Tensor,
    x0: Optional[Tensor] = None,
    phat0: Optional[Tensor] = None,
    action_space: Optional[BoxActionSpace] = None,
    solver_cfg: Optional[DualSplitConfig] = None,
    edge_allow_bitmaps: Optional[list[Tensor | None]] = None,
    return_details: bool = False,
) -> Tensor | Tuple[Tensor, Dict[str, object]]:
    """
    Split dual objective over explicit path set.
    """
    device = game.device_resolved
    dtype = game.dtype

    if solver_cfg is None:
        solver_cfg = DualSplitConfig()

    if x0 is None:
        x0 = game.default_initial_state()
    if phat0 is None:
        phat0 = torch.zeros(game.I, device=device, dtype=dtype)

    x0 = x0.to(device=device, dtype=dtype)
    phat0 = phat0.to(device=device, dtype=dtype)
    paths = paths.to(device=device, dtype=torch.long)

    ro: DualSplitRolloutResult = rollout_dual_split_paths(
        game=game,
        p1_policy=p1_policy,
        p2_policy=p2_policy,
        paths=paths,
        x0=x0,
        phat0=phat0,
        cfg=solver_cfg,
        action_space=action_space,
        edge_allow_bitmaps=edge_allow_bitmaps,
    )

    loss = ro.dual_value
    if not return_details:
        return loss

    details: Dict[str, object] = {
        "path_value": ro.path_value,
        "prob_seq": ro.prob_seq,
        "terminal_terms": ro.terminal_terms,
        "expected_type_cost": ro.expected_type_cost,
        "type_cost_by_path": ro.type_cost_by_path,
        "x_final": ro.x_final,
        "phat_final": ro.phat_final,
        "paths": ro.paths,
        "solver_cfg": asdict(solver_cfg),
    }
    return loss, details


def dual_objective_split_network(
    game: BaseLQGame,
    p1_model: DualSplitP1BRNet,
    p2_model: DualSplitP2Net,
    *,
    paths: Tensor,
    x0: Optional[Tensor] = None,
    phat0: Optional[Tensor] = None,
    action_space: Optional[BoxActionSpace] = None,
    solver_cfg: Optional[DualSplitConfig] = None,
    edge_allow_bitmaps: Optional[list[Tensor | None]] = None,
    return_details: bool = False,
) -> Tensor | Tuple[Tensor, Dict[str, object]]:
    """
    Split dual objective over explicit path set with neural policies.
    """
    device = game.device_resolved
    dtype = game.dtype

    if solver_cfg is None:
        solver_cfg = DualSplitConfig()

    if x0 is None:
        x0 = game.default_initial_state()
    if phat0 is None:
        phat0 = torch.zeros(game.I, device=device, dtype=dtype)

    x0 = x0.to(device=device, dtype=dtype)
    phat0 = phat0.to(device=device, dtype=dtype)
    paths = paths.to(device=device, dtype=torch.long)

    ro: DualSplitRolloutResult = rollout_dual_split_paths_network(
        game=game,
        p1_model=p1_model,
        p2_model=p2_model,
        paths=paths,
        x0=x0,
        phat0=phat0,
        cfg=solver_cfg,
        action_space=action_space,
        edge_allow_bitmaps=edge_allow_bitmaps,
    )

    loss = ro.dual_value
    if not return_details:
        return loss

    details: Dict[str, object] = {
        "path_value": ro.path_value,
        "prob_seq": ro.prob_seq,
        "terminal_terms": ro.terminal_terms,
        "expected_type_cost": ro.expected_type_cost,
        "type_cost_by_path": ro.type_cost_by_path,
        "x_final": ro.x_final,
        "phat_final": ro.phat_final,
        "paths": ro.paths,
        "solver_cfg": asdict(solver_cfg),
    }
    return loss, details
