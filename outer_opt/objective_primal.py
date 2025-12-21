# outer_opt/objective_primal.py
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ..core.types import Tensor
from ..core.action_spaces import BoxActionSpace
from ..games.base_lq_game import BaseLQGame
from ..tree.indexing import FullIaryTreeIndexer
from ..tree.belief_tree import build_belief_tree, BeliefTree
from ..tree.averaged_costs import compute_averaged_costs, AveragedCostData
from ..tree.riccati_tree import RiccatiSolution, riccati_backward
from .alpha_param import AlphaParam


def primal_objective(
    game: BaseLQGame,
    alpha_module: AlphaParam,
    indexer: FullIaryTreeIndexer,
    x0: Optional[Tensor] = None,
    p0: Optional[Tensor] = None,
    action_space: Optional[BoxActionSpace] = None,
    return_details: bool = False,
) -> Tensor | Tuple[Tensor, Dict[str, object]]:
    """
    Compute P1's value in the primal game for fixed α, with P2 best-responding.

    This function builds the entire computational graph corresponding to:
    - α → belief tree and path masses (Bayes updates),
    - belief tree → averaged running and terminal costs,
    - averaged costs → tree-structured Riccati recursion,
    - Riccati result → value at the initial state (root).

    All intermediate steps are differentiable, so PyTorch autograd can be
    used to obtain gradients with respect to the logits in `alpha_module`.

    Parameters
    ----------
    game:
        LQ game instance providing dynamics and cost matrices.
    alpha_module:
        AlphaParam module whose `forward()` returns α.
    indexer:
        Tree indexer specifying the I-ary structure.
    x0:
        Initial state x0 ∈ R^{dx}. If None, uses game.default_initial_state().
    p0:
        Initial prior belief p0 ∈ Δ(I). If None, uses game.default_prior().
    action_space:
        Optional BoxActionSpace. For the Riccati recursion, controls are
        treated as unconstrained; `action_space` is only needed if you
        intend to clamp controls downstream during rollouts.
    return_details:
        If True, return both the scalar value and a dictionary including
        intermediate objects that may be useful for analysis and plotting.

    Returns
    -------
    loss or (loss, details)
        - If return_details is False:
              loss: scalar tensor representing P1's value at (x0, p0).
        - If return_details is True:
              loss: scalar tensor,
              details: dict with keys:
                  "alpha": α tensor,
                  "belief_tree": BeliefTree,
                  "avg_costs": AveragedCostData,
                  "riccati_solution": RiccatiSolution.
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
    riccati_sol: RiccatiSolution = riccati_backward(
        game=game,
        belief_tree=belief_tree,
        avg_costs=avg_costs,
        action_space=action_space,
    )

    loss: Tensor = riccati_sol.value_at_root(x0)

    if not return_details:
        return loss

    details: Dict[str, object] = {
        "alpha": alpha,
        "belief_tree": belief_tree,
        "avg_costs": avg_costs,
        "riccati_solution": riccati_sol,
    }
    return loss, details