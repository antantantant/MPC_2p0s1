from __future__ import annotations

from typing import Optional

import torch

from MPC_2p0s1.core.action_spaces import BoxActionSpace
from MPC_2p0s1.core.types import Tensor
from MPC_2p0s1.games.base_lq_game import BaseLQGame
from MPC_2p0s1.outer_opt.alpha_param import AlphaParam
from MPC_2p0s1.outer_opt.objective_primal import primal_objective
from MPC_2p0s1.tree.indexing import FullIaryTreeIndexer
from MPC_2p0s1.tree.rollout import rollout_trajectory


def rollout_total_cost(
    game: BaseLQGame,
    x_traj: Tensor,
    u_traj: Tensor,
    v_traj: Tensor,
    type_index: int,
) -> Tensor:
    """
    Compute realized total cost for one rollout and one true type.
    """
    if not (0 <= int(type_index) < game.I):
        raise ValueError(f"type_index must be in [0, {game.I}), got {type_index}")

    R_i = game.R[int(type_index)]
    S_i = game.S[int(type_index)]
    tau = float(game.cfg.tau)

    run_u = 0.5 * tau * torch.einsum("kd,dd,kd->k", u_traj, R_i, u_traj)
    run_v = 0.5 * tau * torch.einsum("kd,dd,kd->k", v_traj, S_i, v_traj)
    running = (run_u - run_v).sum()
    terminal = game.terminal_cost_type(int(type_index), x_traj[-1])
    return running + terminal


@torch.no_grad()
def estimate_phat_from_primal_policy(
    game: BaseLQGame,
    alpha_module: AlphaParam,
    indexer: FullIaryTreeIndexer,
    x0: Optional[Tensor] = None,
    p0: Optional[Tensor] = None,
    action_space: Optional[BoxActionSpace] = None,
    sample_actions: bool = False,
    num_rollouts_per_type: int = 1,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """
    Derive a dual seed vector phat from current primal policy estimate.

    For each type i, this runs rollout(s) under the current primal policy and
    returns the average realized type-i cost.
    """
    device = game.device_resolved
    dtype = game.dtype

    if x0 is None:
        x0 = game.default_initial_state()
    if p0 is None:
        p0 = game.default_prior()

    x0 = x0.to(device=device, dtype=dtype)
    p0 = p0.to(device=device, dtype=dtype)
    if p0.shape != (game.I,):
        raise ValueError(f"p0 must have shape ({game.I},), got {tuple(p0.shape)}")

    _, details = primal_objective(
        game=game,
        alpha_module=alpha_module,
        indexer=indexer,
        x0=x0,
        p0=p0,
        action_space=action_space,
        return_details=True,
    )

    alpha = details["alpha"]
    belief_tree = details["belief_tree"]
    riccati_solution = details["riccati_solution"]

    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(game.cfg.seed))

    phat = torch.empty(game.I, device=device, dtype=dtype)
    n_roll = max(1, int(num_rollouts_per_type))

    for i in range(game.I):
        costs = []
        for _ in range(n_roll):
            ro = rollout_trajectory(
                game=game,
                indexer=indexer,
                belief_tree=belief_tree,
                riccati_sol=riccati_solution,
                alpha=alpha,
                x0=x0,
                type_index=i,
                action_space=action_space,
                sample_actions=bool(sample_actions),
                generator=generator,
            )
            costs.append(
                rollout_total_cost(
                    game=game,
                    x_traj=ro.x_traj,
                    u_traj=ro.u_traj,
                    v_traj=ro.v_traj,
                    type_index=i,
                )
            )
        phat[i] = torch.stack(costs, dim=0).mean()

    return phat
