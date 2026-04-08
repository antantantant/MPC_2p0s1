"""Differentiable receding-horizon MPC utilities for the signaling game.

This module supports two MPC uses:

1. ``mpc_rollout_for_type`` for greedy or sampled closed-loop execution.
2. Training objectives that repeatedly re-solve a local horizon problem.

For practical K=10 training, the sampled objective is the intended path. The
older exact expectation over public-action branches grows exponentially in K and
is only suitable for tiny horizons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from ..optimization.objective_primal_sqp import primal_objective_sqp
from ..rollout.trajectory import RolloutResult
from ..solvers.action_spaces import BoxActionSpace
from ..tree.indexing import FullIaryTreeIndexer


class _FixedAlpha(nn.Module):
    """Simple wrapper so ``primal_objective_sqp`` can consume a fixed alpha tensor."""

    def __init__(self, alpha: Tensor) -> None:
        super().__init__()
        self.register_buffer("_alpha", alpha)

    def forward(self) -> Tensor:
        return self._alpha


@dataclass
class MPCClosedLoopResult:
    rollout: RolloutResult
    total_cost: Tensor
    step_costs: List[Tensor]
    step_diagnostics: List[Dict[str, object]]
    policy_log_prob_sum: Optional[Tensor] = None


def extract_alpha_subtree(
    *,
    alpha_full: Tensor,
    full_indexer: FullIaryTreeIndexer,
    start_depth: int,
    start_node: int,
    horizon: Optional[int] = None,
) -> Tuple[Tensor, FullIaryTreeIndexer]:
    """Extract a differentiable alpha tensor for the subtree rooted at a public node."""
    if not (0 <= start_depth < full_indexer.K):
        raise ValueError(
            f"start_depth must be in [0, {full_indexer.K - 1}], got {start_depth}"
        )

    I = full_indexer.I
    remaining = full_indexer.K - start_depth
    local_horizon = remaining if horizon is None or horizon <= 0 else min(horizon, remaining)
    local_indexer = FullIaryTreeIndexer(I=I, K=local_horizon)
    max_nodes = local_indexer.max_nodes_per_depth

    levels: List[Tensor] = []
    for d in range(local_horizon):
        n_local = local_indexer.node_count(d)
        depth = start_depth + d
        scale = I ** d
        slices = [
            alpha_full[depth, start_node * scale + local_node]
            for local_node in range(n_local)
        ]
        level = torch.stack(slices, dim=0)
        if n_local < max_nodes:
            pad = alpha_full.new_full((max_nodes - n_local, I, I), 1.0 / float(I))
            level = torch.cat([level, pad], dim=0)
        levels.append(level)

    alpha_sub = torch.stack(levels, dim=0)
    return alpha_sub, local_indexer


def update_belief(
    *,
    belief: Tensor,
    alpha_node: Tensor,
    action_index: int,
    eps: float = 1e-10,
) -> Tensor:
    """Bayes update for the posterior belief after a realized public action."""
    lam = torch.einsum("i,ia->a", belief, alpha_node)
    denom = lam[action_index].clamp_min(eps)
    b_next = belief * alpha_node[:, action_index] / denom
    return b_next / b_next.sum().clamp_min(eps)


def _solve_local_tree(
    *,
    game,
    indexer: FullIaryTreeIndexer,
    alpha_local: Tensor,
    x_now: Tensor,
    p_now: Tensor,
    action_space: Optional[BoxActionSpace],
    num_sqp_iters: int,
    sqp_step_size: float,
    riccati_reg: float,
    sqp_verbose: bool,
    sqp_early_stop: bool,
    sqp_line_search: bool,
    ls_alpha_min: float,
    ls_backtrack: float,
    ls_max_steps: int,
    line_search_accept_worse: bool,
    track_grad: bool,
    collect_sqp_diagnostics: bool,
) -> Tuple[Tensor, Dict[str, object]]:
    alpha_mod = _FixedAlpha(alpha_local)
    if track_grad:
        return primal_objective_sqp(  # type: ignore[return-value]
            game=game,
            alpha_module=alpha_mod,
            indexer=indexer,
            x0=x_now,
            p0=p_now,
            action_space=action_space,
            num_sqp_iters=num_sqp_iters,
            sqp_step_size=sqp_step_size,
            riccati_reg=riccati_reg,
            sqp_verbose=sqp_verbose,
            sqp_early_stop=sqp_early_stop,
            sqp_line_search=sqp_line_search,
            ls_alpha_min=ls_alpha_min,
            ls_backtrack=ls_backtrack,
            ls_max_steps=ls_max_steps,
            line_search_accept_worse=line_search_accept_worse,
            u_init=None,
            v_init=None,
            collect_sqp_diagnostics=collect_sqp_diagnostics,
            return_details=True,
        )

    with torch.no_grad():
        return primal_objective_sqp(  # type: ignore[return-value]
            game=game,
            alpha_module=alpha_mod,
            indexer=indexer,
            x0=x_now,
            p0=p_now,
            action_space=action_space,
            num_sqp_iters=num_sqp_iters,
            sqp_step_size=sqp_step_size,
            riccati_reg=riccati_reg,
            sqp_verbose=sqp_verbose,
            sqp_early_stop=sqp_early_stop,
            sqp_line_search=sqp_line_search,
            ls_alpha_min=ls_alpha_min,
            ls_backtrack=ls_backtrack,
            ls_max_steps=ls_max_steps,
            line_search_accept_worse=line_search_accept_worse,
            u_init=None,
            v_init=None,
            collect_sqp_diagnostics=collect_sqp_diagnostics,
            return_details=True,
        )


def _type_belief(game, type_index: int) -> Tensor:
    belief = torch.zeros((1, game.cfg.I), dtype=game.dtype, device=game.device)
    belief[0, type_index] = 1.0
    return belief


def realized_stage_cost(
    *,
    game,
    x: Tensor,
    u: Tensor,
    v: Tensor,
    type_index: int,
) -> Tensor:
    """Executed stage cost for one realized payoff type."""
    belief = _type_belief(game, type_index)
    Q, q, c, R, S = game.stage_cost_mats_batch(belief)

    Q_use = Q if Q.ndim == 2 else Q[0]
    q_use = q[0]
    c_use = c[0]
    R_use = R[0]
    S_use = S[0]

    Qx = Q_use @ x
    Ru = R_use @ u
    Sv = S_use @ v
    stage = (
        0.5 * torch.dot(x, Qx)
        + torch.dot(q_use, x)
        + c_use
        + 0.5 * torch.dot(u, Ru)
        - 0.5 * torch.dot(v, Sv)
    )
    return game.cfg.tau * stage


def realized_terminal_cost(
    *,
    game,
    x: Tensor,
    type_index: int,
) -> Tensor:
    """Executed terminal cost for one realized payoff type."""
    belief = _type_belief(game, type_index)
    P, r, c = game.terminal_value_quad_batch(belief)

    P_use = P[0] if P.ndim == 3 else P
    r_use = r[0]
    c_use = c[0]

    Px = P_use @ x
    return 0.5 * torch.dot(x, Px) + torch.dot(r_use, x) + c_use


def _local_root_controls(
    *,
    sqp_res,
    x: Tensor,
    action: int,
    action_space: Optional[BoxActionSpace],
) -> Tuple[Tensor, Tensor]:
    Ku = sqp_res.riccati_sol.K_u[0][0, action]
    Kv = sqp_res.riccati_sol.K_v[0][0, action]
    kappa_u = sqp_res.riccati_sol.kappa_u[0][0, action]
    kappa_v = sqp_res.riccati_sol.kappa_v[0][0, action]

    u = Ku @ x + kappa_u
    v = Kv @ x + kappa_v
    if action_space is not None:
        u = action_space.clip_u(u)
        v = action_space.clip_v(v)
    return u, v


def mpc_rollout_for_type(
    *,
    game,
    full_indexer: FullIaryTreeIndexer,
    alpha_full: Tensor,
    x0: Tensor,
    p0: Tensor,
    type_index: int,
    rollout_steps: int,
    local_horizon: int,
    action_space: Optional[BoxActionSpace],
    num_sqp_iters: int,
    sqp_step_size: float,
    riccati_reg: float,
    sqp_verbose: bool = False,
    sqp_early_stop: bool = False,
    sqp_line_search: bool = True,
    ls_alpha_min: float = 0.01,
    ls_backtrack: float = 0.5,
    ls_max_steps: int = 8,
    line_search_accept_worse: bool = False,
    sample_actions: bool = False,
    generator: Optional[torch.Generator] = None,
    track_grad: bool = False,
    collect_step_diagnostics: bool = False,
) -> MPCClosedLoopResult:
    """Run a receding-horizon rollout for one realized type."""
    I = full_indexer.I
    K_eval = min(rollout_steps, full_indexer.K)

    if generator is None:
        generator = torch.Generator(device=game.device)

    x_traj = torch.empty(K_eval + 1, game.dx, dtype=game.dtype, device=game.device)
    u_traj = torch.empty(K_eval, game.du, dtype=game.dtype, device=game.device)
    v_traj = torch.empty(K_eval, game.dv, dtype=game.dtype, device=game.device)
    belief_traj = torch.empty(K_eval + 1, I, dtype=game.dtype, device=game.device)
    proto_indices = torch.empty(K_eval, dtype=torch.long, device=game.device)

    x = x0.to(device=game.device, dtype=game.dtype)
    belief = p0.to(device=game.device, dtype=game.dtype)
    node_idx = 0

    x_traj[0] = x
    belief_traj[0] = belief

    step_costs: List[Tensor] = []
    step_diags: List[Dict[str, object]] = []
    total_cost = torch.zeros((), dtype=game.dtype, device=game.device)
    log_prob_sum = torch.zeros((), dtype=game.dtype, device=game.device)

    for t in range(K_eval):
        node_before = node_idx
        alpha_local, local_indexer = extract_alpha_subtree(
            alpha_full=alpha_full,
            full_indexer=full_indexer,
            start_depth=t,
            start_node=node_idx,
            horizon=local_horizon,
        )
        local_loss, details = _solve_local_tree(
            game=game,
            indexer=local_indexer,
            alpha_local=alpha_local,
            x_now=x,
            p_now=belief,
            action_space=action_space,
            num_sqp_iters=num_sqp_iters,
            sqp_step_size=sqp_step_size,
            riccati_reg=riccati_reg,
            sqp_verbose=sqp_verbose,
            sqp_early_stop=sqp_early_stop,
            sqp_line_search=sqp_line_search,
            ls_alpha_min=ls_alpha_min,
            ls_backtrack=ls_backtrack,
            ls_max_steps=ls_max_steps,
            line_search_accept_worse=line_search_accept_worse,
            track_grad=track_grad,
            collect_sqp_diagnostics=collect_step_diagnostics,
        )

        sqp_res = details["sqp_result"]
        alpha_eval = details["alpha"]
        alpha_row = alpha_eval[0, 0, type_index]
        if sample_actions:
            action = int(torch.multinomial(alpha_row, 1, replacement=True, generator=generator).item())
            log_prob_sum = log_prob_sum + torch.log(alpha_row[action].clamp_min(1e-10))
        else:
            action = int(torch.argmax(alpha_row).item())
        proto_indices[t] = action

        u, v = _local_root_controls(
            sqp_res=sqp_res,
            x=x,
            action=action,
            action_space=action_space,
        )

        stage_cost = realized_stage_cost(game=game, x=x, u=u, v=v, type_index=type_index)
        total_cost = total_cost + stage_cost
        step_costs.append(stage_cost)

        u_traj[t] = u
        v_traj[t] = v

        x = game.step(x, u, v)
        x_traj[t + 1] = x

        alpha_node_full = alpha_full[t, node_idx]
        belief = update_belief(
            belief=belief,
            alpha_node=alpha_node_full,
            action_index=action,
        )
        belief_traj[t + 1] = belief
        node_idx = full_indexer.child_index(t, node_idx, action)

        if collect_step_diagnostics:
            sqp_diag = sqp_res.diagnostics
            if sqp_diag is not None:
                ric_retries = int(sum(1 for retry in sqp_diag.riccati_retry_hist if retry))
                keep_prev = int(sum(1 for keep in sqp_diag.used_prev_iterate_hist if keep))
                last_step = (
                    float(sqp_diag.accepted_step_hist[-1])
                    if sqp_diag.accepted_step_hist
                    else None
                )
                sqp_converged = bool(sqp_diag.converged)
                sqp_nan = bool(sqp_diag.nan_or_inf_encountered)
            else:
                ric_retries = 0
                keep_prev = 0
                last_step = None
                sqp_converged = False
                sqp_nan = False

            step_diags.append(
                {
                    "step": t,
                    "public_node_before": int(node_before),
                    "chosen_action": action,
                    "local_horizon": local_indexer.K,
                    "local_value": float(local_loss.detach().cpu().item()),
                    "executed_stage_cost": float(stage_cost.detach().cpu().item()),
                    "sqp_converged": sqp_converged,
                    "sqp_nan": sqp_nan,
                    "sqp_riccati_retries": ric_retries,
                    "sqp_used_prev_iterates": keep_prev,
                    "sqp_last_accepted_step": last_step,
                }
            )

    total_cost = total_cost + realized_terminal_cost(game=game, x=x, type_index=type_index)

    rollout = RolloutResult(
        x_traj=x_traj,
        u_traj=u_traj,
        v_traj=v_traj,
        belief_traj=belief_traj,
        proto_indices=proto_indices,
    )
    return MPCClosedLoopResult(
        rollout=rollout,
        total_cost=total_cost,
        step_costs=step_costs,
        step_diagnostics=step_diags,
        policy_log_prob_sum=log_prob_sum,
    )


def _expected_cost_from_node(
    *,
    game,
    full_indexer: FullIaryTreeIndexer,
    alpha_full: Tensor,
    x: Tensor,
    belief: Tensor,
    type_index: int,
    depth: int,
    node_idx: int,
    rollout_steps: int,
    local_horizon: int,
    action_space: Optional[BoxActionSpace],
    num_sqp_iters: int,
    sqp_step_size: float,
    riccati_reg: float,
    sqp_verbose: bool,
    sqp_early_stop: bool,
    sqp_line_search: bool,
    ls_alpha_min: float,
    ls_backtrack: float,
    ls_max_steps: int,
    line_search_accept_worse: bool,
) -> Tuple[Tensor, int]:
    if depth >= rollout_steps:
        return realized_terminal_cost(game=game, x=x, type_index=type_index), 0

    alpha_local, local_indexer = extract_alpha_subtree(
        alpha_full=alpha_full,
        full_indexer=full_indexer,
        start_depth=depth,
        start_node=node_idx,
        horizon=local_horizon,
    )
    _, details = _solve_local_tree(
        game=game,
        indexer=local_indexer,
        alpha_local=alpha_local,
        x_now=x,
        p_now=belief,
        action_space=action_space,
        num_sqp_iters=num_sqp_iters,
        sqp_step_size=sqp_step_size,
        riccati_reg=riccati_reg,
        sqp_verbose=sqp_verbose,
        sqp_early_stop=sqp_early_stop,
        sqp_line_search=sqp_line_search,
        ls_alpha_min=ls_alpha_min,
        ls_backtrack=ls_backtrack,
        ls_max_steps=ls_max_steps,
        line_search_accept_worse=line_search_accept_worse,
        track_grad=True,
        collect_sqp_diagnostics=False,
    )
    sqp_res = details["sqp_result"]

    alpha_node = alpha_full[depth, node_idx]
    alpha_row = alpha_node[type_index]

    total = torch.zeros((), dtype=game.dtype, device=game.device)
    solve_count = 1
    for action in range(full_indexer.I):
        u, v = _local_root_controls(
            sqp_res=sqp_res,
            x=x,
            action=action,
            action_space=action_space,
        )
        stage_cost = realized_stage_cost(game=game, x=x, u=u, v=v, type_index=type_index)
        x_next = game.step(x, u, v)
        belief_next = update_belief(
            belief=belief,
            alpha_node=alpha_node,
            action_index=action,
        )
        future_cost, child_solves = _expected_cost_from_node(
            game=game,
            full_indexer=full_indexer,
            alpha_full=alpha_full,
            x=x_next,
            belief=belief_next,
            type_index=type_index,
            depth=depth + 1,
            node_idx=full_indexer.child_index(depth, node_idx, action),
            rollout_steps=rollout_steps,
            local_horizon=local_horizon,
            action_space=action_space,
            num_sqp_iters=num_sqp_iters,
            sqp_step_size=sqp_step_size,
            riccati_reg=riccati_reg,
            sqp_verbose=sqp_verbose,
            sqp_early_stop=sqp_early_stop,
            sqp_line_search=sqp_line_search,
            ls_alpha_min=ls_alpha_min,
            ls_backtrack=ls_backtrack,
            ls_max_steps=ls_max_steps,
            line_search_accept_worse=line_search_accept_worse,
        )
        solve_count += child_solves
        total = total + alpha_row[action] * (stage_cost + future_cost)

    return total, solve_count


def mpc_closed_loop_expected_cost(
    *,
    game,
    indexer: FullIaryTreeIndexer,
    alpha_module: nn.Module,
    x0: Tensor,
    p0: Tensor,
    rollout_steps: int,
    local_horizon: int,
    action_space: Optional[BoxActionSpace],
    num_sqp_iters: int,
    sqp_step_size: float,
    riccati_reg: float,
    sqp_verbose: bool = False,
    sqp_early_stop: bool = False,
    sqp_line_search: bool = True,
    ls_alpha_min: float = 0.01,
    ls_backtrack: float = 0.5,
    ls_max_steps: int = 8,
    line_search_accept_worse: bool = False,
) -> Tuple[Tensor, Dict[str, object]]:
    """Exact expected MPC objective. Exponential in rollout_steps; use only for tiny K."""
    alpha_full = alpha_module()
    K_eval = min(rollout_steps, indexer.K)
    H_local = min(local_horizon, indexer.K)

    total = torch.zeros((), dtype=game.dtype, device=game.device)
    per_type_costs: List[Tensor] = []
    per_type_solve_counts: List[int] = []

    for type_idx in range(game.cfg.I):
        type_cost, solve_count = _expected_cost_from_node(
            game=game,
            full_indexer=indexer,
            alpha_full=alpha_full,
            x=x0,
            belief=p0,
            type_index=type_idx,
            depth=0,
            node_idx=0,
            rollout_steps=K_eval,
            local_horizon=H_local,
            action_space=action_space,
            num_sqp_iters=num_sqp_iters,
            sqp_step_size=sqp_step_size,
            riccati_reg=riccati_reg,
            sqp_verbose=sqp_verbose,
            sqp_early_stop=sqp_early_stop,
            sqp_line_search=sqp_line_search,
            ls_alpha_min=ls_alpha_min,
            ls_backtrack=ls_backtrack,
            ls_max_steps=ls_max_steps,
            line_search_accept_worse=line_search_accept_worse,
        )
        total = total + p0[type_idx] * type_cost
        per_type_costs.append(type_cost)
        per_type_solve_counts.append(solve_count)

    details: Dict[str, object] = {
        "alpha": alpha_full,
        "per_type_costs": per_type_costs,
        "per_type_solve_counts": per_type_solve_counts,
        "training_objective": "expected_closed_loop_cost",
    }
    return total, details


def mpc_sampled_closed_loop_objective(
    *,
    game,
    indexer: FullIaryTreeIndexer,
    alpha_module: nn.Module,
    x0: Tensor,
    p0: Tensor,
    rollout_steps: int,
    local_horizon: int,
    action_space: Optional[BoxActionSpace],
    num_sqp_iters: int,
    sqp_step_size: float,
    riccati_reg: float,
    sqp_verbose: bool = False,
    sqp_early_stop: bool = False,
    sqp_line_search: bool = True,
    ls_alpha_min: float = 0.01,
    ls_backtrack: float = 0.5,
    ls_max_steps: int = 8,
    line_search_accept_worse: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Dict[str, object]]:
    """Sampled on-policy MPC objective with a REINFORCE correction for public actions.

    This keeps K=10 practical by reducing the number of local solves per epoch
    from exponential in rollout length to linear in rollout length.
    """
    alpha_full = alpha_module()
    K_eval = min(rollout_steps, indexer.K)
    H_local = min(local_horizon, indexer.K)

    if generator is None:
        generator = torch.Generator(device=game.device)

    per_type_results: List[MPCClosedLoopResult] = []
    per_type_costs: List[Tensor] = []
    per_type_log_probs: List[Tensor] = []
    per_type_solve_counts: List[int] = []

    baseline = torch.zeros((), dtype=game.dtype, device=game.device)
    for type_idx in range(game.cfg.I):
        result = mpc_rollout_for_type(
            game=game,
            full_indexer=indexer,
            alpha_full=alpha_full,
            x0=x0,
            p0=p0,
            type_index=type_idx,
            rollout_steps=K_eval,
            local_horizon=H_local,
            action_space=action_space,
            num_sqp_iters=num_sqp_iters,
            sqp_step_size=sqp_step_size,
            riccati_reg=riccati_reg,
            sqp_verbose=sqp_verbose,
            sqp_early_stop=sqp_early_stop,
            sqp_line_search=sqp_line_search,
            ls_alpha_min=ls_alpha_min,
            ls_backtrack=ls_backtrack,
            ls_max_steps=ls_max_steps,
            line_search_accept_worse=line_search_accept_worse,
            sample_actions=True,
            generator=generator,
            track_grad=True,
            collect_step_diagnostics=False,
        )
        per_type_results.append(result)
        per_type_costs.append(result.total_cost)
        per_type_log_probs.append(
            result.policy_log_prob_sum
            if result.policy_log_prob_sum is not None
            else torch.zeros((), dtype=game.dtype, device=game.device)
        )
        per_type_solve_counts.append(len(result.step_costs))
        baseline = baseline + p0[type_idx] * result.total_cost.detach()

    total = torch.zeros((), dtype=game.dtype, device=game.device)
    for type_idx in range(game.cfg.I):
        pathwise = per_type_costs[type_idx]
        advantage = per_type_costs[type_idx].detach() - baseline
        reinforce = advantage * per_type_log_probs[type_idx]
        total = total + p0[type_idx] * (pathwise + reinforce)

    details: Dict[str, object] = {
        "alpha": alpha_full,
        "per_type_costs": per_type_costs,
        "per_type_log_probs": per_type_log_probs,
        "per_type_solve_counts": per_type_solve_counts,
        "training_objective": "sampled_closed_loop_cost",
        "baseline": baseline,
        "per_type_results": per_type_results,
    }
    return total, details
