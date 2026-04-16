# outer_opt/objective_primal_sqp.py
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from MPC_2p0s1.core.types import Tensor
from MPC_2p0s1.core.action_spaces import BoxActionSpace
from MPC_2p0s1.tree.indexing import FullIaryTreeIndexer
from MPC_2p0s1.tree.belief_tree import build_belief_tree, BeliefTree
from MPC_2p0s1.tree.averaged_costs import compute_averaged_costs, AveragedCostData
from MPC_2p0s1.tree.sqp_tree import SQPConfig, sqp_solve_tree, SQPTreeSolution
from MPC_2p0s1.outer_opt.alpha_param import AlphaParam


def primal_objective_sqp(
    game,
    alpha_module: AlphaParam,
    indexer: FullIaryTreeIndexer,
    x0: Optional[Tensor] = None,
    p0: Optional[Tensor] = None,
    action_space: Optional[BoxActionSpace] = None,
    sqp_cfg: Optional[SQPConfig] = None,
    return_details: bool = False,
) -> Tensor | Tuple[Tensor, Dict[str, object]]:
    """
    Nonlinear (drone) version of the primal objective:
      α -> belief tree -> averaged costs -> SQP solve -> expected cost at root

    Returns a scalar 'loss' (P1's value) that you can minimize over α/logits.
    """
    device = game.device_resolved
    dtype = game.dtype

    if x0 is None:
        x0 = game.default_initial_state()
    if p0 is None:
        p0 = game.default_prior()

    x0 = x0.to(device=device, dtype=dtype)
    p0 = p0.to(device=device, dtype=dtype)

    alpha: Tensor = alpha_module()  # (K, max_nodes, I, I)

    belief_tree: BeliefTree = build_belief_tree(alpha=alpha, p0=p0, indexer=indexer)
    avg_costs: AveragedCostData = compute_averaged_costs(game=game, belief_tree=belief_tree)

    sol: SQPTreeSolution = sqp_solve_tree(
        game=game,
        belief_tree=belief_tree,
        avg_costs=avg_costs,
        x0=x0,
        sqp_cfg=sqp_cfg,
        action_space=action_space,
        return_debug=False,
    )

    loss: Tensor = sol.value()

    if not return_details:
        return loss

    details: Dict[str, object] = {
        "alpha": alpha,
        "belief_tree": belief_tree,
        "avg_costs": avg_costs,
        "sqp_solution": sol,
    }
    return loss, details

