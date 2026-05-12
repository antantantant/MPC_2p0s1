from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import time
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .solver_jax import drone_3d_diffmpc as base
from .solver_jax.public_tree import PublicTreeTopology, build_public_tree_topology, propagate_type_probabilities
from .solver_jax.tree import F32, MixedPrefixTreeSpec, build_alpha_from_logits, jax_softmax


@dataclass(frozen=True)
class Drone3DAffinePolicy:
    offense_feedback: jnp.ndarray
    offense_bias: jnp.ndarray
    defense_feedback: jnp.ndarray
    defense_bias: jnp.ndarray


@dataclass(frozen=True)
class Drone3DBeliefTables:
    node_type_probs: jnp.ndarray
    node_public_probs: jnp.ndarray
    node_beliefs: jnp.ndarray


def _tree_l2_norm(value: Any) -> float:
    leaves = jax.tree_util.tree_leaves(value)
    total = sum(float(jnp.sum(jnp.square(leaf))) for leaf in leaves)
    return float(np.sqrt(total))


def build_belief_tables(problem: base.Drone3DTreeProblem, alpha: jnp.ndarray) -> Drone3DBeliefTables:
    node_type_probs, node_public_probs, _, _ = propagate_type_probabilities(
        alpha,
        problem.topology,
        problem.prior,
    )
    node_beliefs = node_type_probs / jnp.clip(node_public_probs[:, None], 1e-8, None)
    return Drone3DBeliefTables(
        node_type_probs=node_type_probs,
        node_public_probs=node_public_probs,
        node_beliefs=node_beliefs,
    )


def node_index_from_prototypes(
    topology: PublicTreeTopology,
    prototypes_so_far: Sequence[int],
) -> int:
    node_idx = 0
    for step, prototype in enumerate(prototypes_so_far):
        node_idx = child_node_index(topology, step, node_idx, int(prototype))
    return int(node_idx)


def child_node_index(
    topology: PublicTreeTopology,
    time_step: int,
    node_idx: int,
    prototype_index: int,
) -> int:
    if not (0 <= int(time_step) < topology.total_depth):
        raise ValueError(f"time_step={time_step} out of range for total_depth={topology.total_depth}.")
    local_idx = int(topology.node_local_indices[int(node_idx)])
    if int(time_step) < topology.mixed_depth:
        if not (0 <= int(prototype_index) < topology.branch_factor):
            raise ValueError(
                f"prototype_index={prototype_index} out of range [0, {topology.branch_factor})."
            )
        edge_idx = topology.edge_offsets_py[int(time_step)] + local_idx * topology.branch_factor + int(
            prototype_index
        )
    else:
        if int(prototype_index) != 0:
            raise ValueError(
                f"Tail-step prototype_index must be 0, got {prototype_index} at step {time_step}."
            )
        edge_idx = topology.edge_offsets_py[int(time_step)] + local_idx
    return int(topology.edge_children[int(edge_idx)])


def rebase_alpha_logits_to_remaining(
    tree: MixedPrefixTreeSpec,
    alpha_logits: jnp.ndarray,
    prototypes_so_far: Sequence[int],
) -> jnp.ndarray:
    t = len(prototypes_so_far)
    if t >= tree.mixed_horizon_steps:
        return jnp.zeros((0, 1, tree.type_count, tree.type_count), dtype=alpha_logits.dtype)

    remaining_mixed_steps = tree.mixed_horizon_steps - t
    remaining_tree = MixedPrefixTreeSpec(
        type_count=tree.type_count,
        mixed_horizon_steps=remaining_mixed_steps,
        tail_horizon_steps=tree.tail_horizon_steps,
        force_identity_reveal=tree.force_identity_reveal,
    )
    rebased = jnp.zeros(
        (remaining_mixed_steps, remaining_tree.max_node_count, tree.type_count, tree.type_count),
        dtype=alpha_logits.dtype,
    )

    base_node = 0
    for prototype in prototypes_so_far[: tree.mixed_horizon_steps]:
        base_node = base_node * tree.type_count + int(prototype)

    for depth in range(remaining_mixed_steps):
        node_count = tree.type_count**depth
        old_depth = t + depth
        old_start = base_node * (tree.type_count**depth)
        old_stop = old_start + node_count
        rebased = rebased.at[depth, :node_count].set(alpha_logits[old_depth, old_start:old_stop])
    return rebased


def _deterministic_action0_logits(
    shape: tuple[int, ...],
    *,
    dtype: Any,
    logit: float,
) -> jnp.ndarray:
    if len(shape) < 1:
        raise ValueError("shape must include an action dimension.")
    action_count = shape[-1]
    logits = -float(logit) * jnp.ones(shape, dtype=dtype)
    logits = logits.at[..., 0].set(float(logit))
    if action_count == 1:
        logits = jnp.zeros(shape, dtype=dtype)
    return logits


def rebase_alpha_logits_to_fixed_padded(
    tree: MixedPrefixTreeSpec,
    alpha_logits: jnp.ndarray,
    prototypes_so_far: Sequence[int],
    *,
    active_mixed_steps: int,
    deterministic_logit: float = 20.0,
) -> jnp.ndarray:
    """Rebase a root alpha tensor into a fixed-shape padded online tensor.

    The first active_mixed_steps levels are copied from the root solution
    conditioned on prototypes_so_far. Remaining mixed levels are deterministic
    public-continuation padding, so they do not reveal additional type
    information before the active terminal depth.
    """
    active_mixed_steps = int(active_mixed_steps)
    if not (0 <= active_mixed_steps <= tree.mixed_horizon_steps):
        raise ValueError(
            f"active_mixed_steps={active_mixed_steps} must be in [0, {tree.mixed_horizon_steps}]."
        )
    if tree.mixed_horizon_steps == 0:
        return jnp.zeros((0, 1, tree.type_count, tree.type_count), dtype=jnp.asarray(alpha_logits).dtype)

    alpha_logits = jnp.asarray(alpha_logits, dtype=F32)
    full_shape = (tree.mixed_horizon_steps, tree.max_node_count, tree.type_count, tree.type_count)
    rebased = _deterministic_action0_logits(
        full_shape,
        dtype=alpha_logits.dtype,
        logit=deterministic_logit,
    )

    consumed_mixed = min(len(prototypes_so_far), tree.mixed_horizon_steps)
    base_node = 0
    for prototype in prototypes_so_far[:consumed_mixed]:
        base_node = base_node * tree.type_count + int(prototype)

    for depth in range(active_mixed_steps):
        old_depth = consumed_mixed + depth
        if old_depth >= tree.mixed_horizon_steps:
            break
        node_count = tree.type_count**depth
        old_start = base_node * (tree.type_count**depth)
        old_stop = old_start + node_count
        rebased = rebased.at[depth, :node_count].set(alpha_logits[old_depth, old_start:old_stop])

    if tree.force_identity_reveal and active_mixed_steps > 0:
        reveal_depth = active_mixed_steps - 1
        used_nodes = tree.type_count**reveal_depth
        identity = jnp.eye(tree.type_count, dtype=alpha_logits.dtype)
        identity_logits = jnp.where(identity > 0.0, deterministic_logit, -deterministic_logit)
        rebased = rebased.at[reveal_depth, :used_nodes].set(
            jnp.broadcast_to(identity_logits, (used_nodes, tree.type_count, tree.type_count))
        )
    return rebased


def _project_fixed_padded_logits(
    tree: MixedPrefixTreeSpec,
    theta: jnp.ndarray,
    *,
    active_mixed_steps: int,
    deterministic_logit: float,
) -> jnp.ndarray:
    """Keep padded mixed depths deterministic after optimizer updates."""
    active_mixed_steps = int(active_mixed_steps)
    theta = jnp.asarray(theta, dtype=F32)
    if tree.mixed_horizon_steps == 0:
        return theta

    deterministic = _deterministic_action0_logits(
        theta.shape,
        dtype=theta.dtype,
        logit=deterministic_logit,
    )
    depth_mask = (jnp.arange(tree.mixed_horizon_steps) < active_mixed_steps)[:, None, None, None]
    theta = jnp.where(depth_mask, theta, deterministic)

    if tree.force_identity_reveal and active_mixed_steps > 0:
        reveal_depth = active_mixed_steps - 1
        used_nodes = tree.type_count**reveal_depth
        identity = jnp.eye(tree.type_count, dtype=theta.dtype)
        identity_logits = jnp.where(identity > 0.0, deterministic_logit, -deterministic_logit)
        theta = theta.at[reveal_depth, :used_nodes].set(
            jnp.broadcast_to(identity_logits, (used_nodes, tree.type_count, tree.type_count))
        )
    return theta


def _padded_alpha_from_projected_logits(theta: jnp.ndarray) -> jnp.ndarray:
    # Projection encodes active reveals and deterministic padded continuation.
    return jax_softmax(theta, axis=-1).astype(F32)


@lru_cache(maxsize=None)
def _padded_lq_policy_value_kernel(
    node_offsets_py: tuple[int, ...],
    edge_offsets_py: tuple[int, ...],
    total_depth: int,
    node_count: int,
    edge_count: int,
    state_dim: int,
    control_dim: int,
    terminal_dim: int,
) -> Callable[..., tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    def _edge_end(depth: int) -> int:
        return edge_offsets_py[depth + 1] if depth + 1 < len(edge_offsets_py) else edge_count

    @jax.jit
    def kernel(
        lambda_edge: jnp.ndarray,
        node_terminal_p: jnp.ndarray,
        node_terminal_r: jnp.ndarray,
        node_terminal_c: jnp.ndarray,
        a_state: jnp.ndarray,
        control_matrix: jnp.ndarray,
        control_weight_diag: jnp.ndarray,
        terminal_selector: jnp.ndarray,
        x0: jnp.ndarray,
        active_total_steps: jnp.ndarray,
        edge_parents: jnp.ndarray,
        edge_children: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        dtype = x0.dtype
        p_nodes = jnp.zeros((node_count, state_dim, state_dim), dtype=dtype)
        r_nodes = jnp.zeros((node_count, state_dim), dtype=dtype)
        c_nodes = jnp.zeros((node_count,), dtype=dtype)
        edge_feedback = jnp.zeros((edge_count, control_dim, state_dim), dtype=dtype)
        edge_bias = jnp.zeros((edge_count, control_dim), dtype=dtype)

        control_weight = jnp.diag(control_weight_diag.astype(dtype))
        a_transpose = a_state.T
        b_transpose = control_matrix.T
        terminal_control = terminal_selector @ control_matrix
        terminal_state = terminal_selector @ a_state

        for depth in range(total_depth - 1, -1, -1):
            node_start = node_offsets_py[depth]
            node_end = node_offsets_py[depth + 1]
            edge_start = edge_offsets_py[depth]
            edge_end = _edge_end(depth)
            node_slice = slice(node_start, node_end)
            edge_slice = slice(edge_start, edge_end)
            local_parent_count = node_end - node_start
            local_parents = edge_parents[edge_slice] - node_start
            child_indices = edge_children[edge_slice]

            p_child = p_nodes[child_indices]
            r_child = r_nodes[child_indices]
            c_child = c_nodes[child_indices]
            child_is_active_terminal = jnp.asarray(depth + 1, dtype=jnp.int32) == active_total_steps
            p_child = jnp.where(child_is_active_terminal, node_terminal_p[child_indices], p_child)
            r_child = jnp.where(child_is_active_terminal, node_terminal_r[child_indices], r_child)
            c_child = jnp.where(child_is_active_terminal, node_terminal_c[child_indices], c_child)

            h_block = control_weight[None, :, :] + jnp.einsum(
                "ui,eij,jv->euv",
                b_transpose,
                p_child,
                control_matrix,
            )
            f_block = jnp.einsum("ui,eij,jv->euv", b_transpose, p_child, a_state)
            g_block = jnp.einsum("ui,ei->eu", b_transpose, r_child)

            def _terminal_edge_solve():
                upper = jnp.concatenate(
                    [
                        h_block,
                        jnp.broadcast_to(
                            terminal_control.T,
                            (child_indices.shape[0], control_dim, terminal_dim),
                        ),
                    ],
                    axis=2,
                )
                lower = jnp.concatenate(
                    [
                        jnp.broadcast_to(
                            terminal_control,
                            (child_indices.shape[0], terminal_dim, control_dim),
                        ),
                        jnp.zeros((child_indices.shape[0], terminal_dim, terminal_dim), dtype=dtype),
                    ],
                    axis=2,
                )
                block_matrix = jnp.concatenate([upper, lower], axis=1)
                rhs_matrix = -jnp.concatenate(
                    [
                        f_block,
                        jnp.broadcast_to(
                            terminal_state,
                            (child_indices.shape[0], terminal_dim, state_dim),
                        ),
                    ],
                    axis=1,
                )
                rhs_vector = -jnp.concatenate(
                    [
                        g_block,
                        jnp.zeros((child_indices.shape[0], terminal_dim), dtype=dtype),
                    ],
                    axis=1,
                )
                solution_matrix = jnp.linalg.solve(block_matrix, rhs_matrix)
                solution_vector = jnp.linalg.solve(block_matrix, rhs_vector[:, :, None]).squeeze(-1)
                local_feedback = solution_matrix[:, :control_dim, :]
                local_bias = solution_vector[:, :control_dim]
                a_closed = a_state[None, :, :] + jnp.einsum("ij,ejk->eik", control_matrix, local_feedback)
                b_closed = jnp.einsum("ij,ej->ei", control_matrix, local_bias)
                p_local = (
                    jnp.einsum("eui,uv,evj->eij", local_feedback, control_weight, local_feedback)
                    + jnp.einsum("eui,euv,evj->eij", a_closed, p_child, a_closed)
                )
                p_local = 0.5 * (p_local + jnp.swapaxes(p_local, -1, -2))
                child_affine = jnp.einsum("euv,ev->eu", p_child, b_closed) + r_child
                r_local = (
                    jnp.einsum("eui,uv,ev->ei", local_feedback, control_weight, local_bias)
                    + jnp.einsum("eui,eu->ei", a_closed, child_affine)
                )
                c_local = (
                    0.5 * jnp.einsum("eu,uv,ev->e", local_bias, control_weight, local_bias)
                    + 0.5 * jnp.einsum("eu,euv,ev->e", b_closed, p_child, b_closed)
                    + jnp.einsum("eu,eu->e", r_child, b_closed)
                    + c_child
                )
                return local_feedback, local_bias, p_local, r_local, c_local

            def _regular_edge_solve():
                rhs = jnp.concatenate([f_block, g_block[:, :, None]], axis=2)
                solve = jnp.linalg.solve(h_block, rhs)
                solve_f = solve[:, :, :state_dim]
                solve_g = solve[:, :, state_dim]
                local_feedback = -solve_f
                local_bias = -solve_g
                q_block = jnp.einsum("ui,eij,jv->euv", a_transpose, p_child, a_state)
                q_lin = jnp.einsum("ui,ei->eu", a_transpose, r_child)
                p_local = q_block - jnp.einsum("eui,euj->eij", f_block, solve_f)
                p_local = 0.5 * (p_local + jnp.swapaxes(p_local, -1, -2))
                r_local = q_lin - jnp.einsum("eui,eu->ei", f_block, solve_g)
                c_local = c_child - 0.5 * jnp.einsum("eu,eu->e", g_block, solve_g)
                return local_feedback, local_bias, p_local, r_local, c_local

            if terminal_dim > 0:
                local_feedback, local_bias, p_local, r_local, c_local = jax.lax.cond(
                    jnp.asarray(depth, dtype=jnp.int32) == active_total_steps - 1,
                    _terminal_edge_solve,
                    _regular_edge_solve,
                )
            else:
                local_feedback, local_bias, p_local, r_local, c_local = _regular_edge_solve()

            edge_feedback = edge_feedback.at[edge_slice].set(local_feedback)
            edge_bias = edge_bias.at[edge_slice].set(local_bias)

            edge_weights = lambda_edge[edge_slice]
            p_parent = jax.ops.segment_sum(
                edge_weights[:, None, None] * p_local,
                local_parents,
                num_segments=local_parent_count,
            )
            r_parent = jax.ops.segment_sum(
                edge_weights[:, None] * r_local,
                local_parents,
                num_segments=local_parent_count,
            )
            c_parent = jax.ops.segment_sum(
                edge_weights * c_local,
                local_parents,
                num_segments=local_parent_count,
            )
            active_node = jnp.asarray(depth, dtype=jnp.int32) == active_total_steps
            p_parent = jnp.where(active_node, p_parent + node_terminal_p[node_slice], p_parent)
            r_parent = jnp.where(active_node, r_parent + node_terminal_r[node_slice], r_parent)
            c_parent = jnp.where(active_node, c_parent + node_terminal_c[node_slice], c_parent)
            p_nodes = p_nodes.at[node_slice].set(p_parent)
            r_nodes = r_nodes.at[node_slice].set(r_parent)
            c_nodes = c_nodes.at[node_slice].set(c_parent)

        node_states = jnp.zeros((node_count, state_dim), dtype=dtype)
        edge_controls = jnp.zeros((edge_count, control_dim), dtype=dtype)
        node_states = node_states.at[0].set(x0)
        for depth in range(total_depth):
            edge_start = edge_offsets_py[depth]
            edge_end = _edge_end(depth)
            edge_slice = slice(edge_start, edge_end)
            parents = edge_parents[edge_slice]
            parent_states = node_states[parents]
            controls = (
                jnp.einsum("eij,ej->ei", edge_feedback[edge_slice], parent_states)
                + edge_bias[edge_slice]
            )
            child_states = parent_states @ a_state.T + controls @ control_matrix.T
            edge_controls = edge_controls.at[edge_slice].set(controls)
            node_states = node_states.at[edge_children[edge_slice]].set(child_states)

        root_value = (
            0.5 * jnp.einsum("i,ij,j->", x0, p_nodes[0], x0)
            + jnp.dot(r_nodes[0], x0)
            + c_nodes[0]
        )
        return node_states, edge_controls, edge_feedback, edge_bias, root_value

    return kernel


@lru_cache(maxsize=None)
def _padded_exact_lq_value_and_grad_kernel(
    branch_factor: int,
    mixed_depth: int,
    tail_depth: int,
    node_offsets_py: tuple[int, ...],
    edge_offsets_py: tuple[int, ...],
    total_depth: int,
    node_count: int,
    edge_count: int,
    state_dim: int,
    control_dim: int,
    terminal_dim: int,
) -> Callable[..., tuple[jnp.ndarray, jnp.ndarray]]:
    solve_kernel = _padded_lq_policy_value_kernel(
        node_offsets_py,
        edge_offsets_py,
        total_depth,
        node_count,
        edge_count,
        state_dim,
        control_dim,
        terminal_dim,
    )

    def _propagate(alpha: jnp.ndarray, prior: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        hidden_type_count = int(alpha.shape[-1])
        node_type_probs = jnp.zeros((node_count, hidden_type_count), dtype=F32)
        node_type_probs = node_type_probs.at[0].set(prior.astype(F32))
        edge_public_probs = jnp.zeros((edge_count,), dtype=F32)

        for depth in range(mixed_depth):
            node_offset = node_offsets_py[depth]
            next_offset = node_offsets_py[depth + 1]
            edge_offset = edge_offsets_py[depth]
            depth_node_count = branch_factor**depth
            child_count = branch_factor ** (depth + 1)
            depth_node_type_probs = node_type_probs[node_offset : node_offset + depth_node_count]
            alpha_depth = alpha[depth, :depth_node_count]
            child_type_probs = depth_node_type_probs[:, :, None] * alpha_depth
            child_type_probs = jnp.transpose(child_type_probs, (0, 2, 1)).reshape(
                (child_count, hidden_type_count)
            )
            node_type_probs = node_type_probs.at[next_offset : next_offset + child_count].set(child_type_probs)
            edge_public_probs = edge_public_probs.at[edge_offset : edge_offset + child_count].set(
                jnp.sum(child_type_probs, axis=1)
            )

        if tail_depth > 0:
            frontier_count = branch_factor**mixed_depth
            frontier_type_probs = node_type_probs[
                node_offsets_py[mixed_depth] : node_offsets_py[mixed_depth] + frontier_count
            ]
            for tail_step in range(tail_depth):
                depth = mixed_depth + tail_step
                node_offset = node_offsets_py[depth + 1]
                edge_offset = edge_offsets_py[depth]
                node_type_probs = node_type_probs.at[node_offset : node_offset + frontier_count].set(
                    frontier_type_probs
                )
                edge_public_probs = edge_public_probs.at[edge_offset : edge_offset + frontier_count].set(
                    jnp.sum(frontier_type_probs, axis=1)
                )

        node_public_probs = jnp.sum(node_type_probs, axis=1)
        return node_type_probs, node_public_probs, edge_public_probs

    @jax.jit
    def value_and_grad(
        theta: jnp.ndarray,
        prior: jnp.ndarray,
        offense_x0: jnp.ndarray,
        defense_x0: jnp.ndarray,
        offense_a_state: jnp.ndarray,
        defense_a_state: jnp.ndarray,
        offense_matrix: jnp.ndarray,
        defense_matrix: jnp.ndarray,
        offense_running_diag: jnp.ndarray,
        defense_running_diag: jnp.ndarray,
        terminal_pdiag: jnp.ndarray,
        terminal_r: jnp.ndarray,
        terminal_c: jnp.ndarray,
        offense_terminal_selector: jnp.ndarray,
        defense_terminal_selector: jnp.ndarray,
        active_total_steps: jnp.ndarray,
        edge_parents: jnp.ndarray,
        edge_children: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        def objective(theta_value: jnp.ndarray) -> jnp.ndarray:
            alpha = _padded_alpha_from_projected_logits(theta_value)
            node_type_probs, node_public_probs, edge_public_probs = _propagate(alpha, prior)
            parent_public_probs = node_public_probs[edge_parents]
            lambda_edge = edge_public_probs / jnp.clip(parent_public_probs, 1e-8, None)
            node_beliefs = node_type_probs / jnp.clip(node_public_probs[:, None], 1e-8, None)
            terminal_p = jnp.einsum("ni,ijk->njk", node_beliefs, jax.vmap(jnp.diag)(terminal_pdiag))
            terminal_r_node = node_beliefs @ terminal_r
            terminal_c_node = node_beliefs @ terminal_c
            _, _, _, _, offense_value = solve_kernel(
                lambda_edge.astype(F32),
                terminal_p.astype(F32),
                terminal_r_node.astype(F32),
                terminal_c_node.astype(F32),
                offense_a_state.astype(F32),
                offense_matrix.astype(F32),
                offense_running_diag.astype(F32),
                offense_terminal_selector.astype(F32),
                offense_x0.astype(F32),
                active_total_steps.astype(jnp.int32),
                edge_parents,
                edge_children,
            )
            _, _, _, _, defense_value = solve_kernel(
                lambda_edge.astype(F32),
                terminal_p.astype(F32),
                terminal_r_node.astype(F32),
                terminal_c_node.astype(F32),
                defense_a_state.astype(F32),
                defense_matrix.astype(F32),
                defense_running_diag.astype(F32),
                defense_terminal_selector.astype(F32),
                defense_x0.astype(F32),
                active_total_steps.astype(jnp.int32),
                edge_parents,
                edge_children,
            )
            return offense_value - defense_value

        return jax.value_and_grad(objective)(theta)

    return value_and_grad


def _padded_probability_tables(
    problem: base.Drone3DTreeProblem,
    theta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    alpha = _padded_alpha_from_projected_logits(theta)
    node_type_probs, node_public_probs, _, edge_public_probs = propagate_type_probabilities(
        alpha,
        problem.topology,
        problem.prior,
    )
    return alpha, node_type_probs, node_public_probs, edge_public_probs


def _padded_player_terminal_tables(
    problem: base.Drone3DTreeProblem,
    node_type_probs: jnp.ndarray,
    node_public_probs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    node_beliefs = node_type_probs / jnp.clip(node_public_probs[:, None], 1e-8, None)
    terminal_p = jnp.einsum("ni,ijk->njk", node_beliefs, jax.vmap(jnp.diag)(problem.terminal_pdiag))
    terminal_r = node_beliefs @ problem.terminal_r
    terminal_c = node_beliefs @ problem.terminal_c
    return terminal_p, terminal_r, terminal_c


def _player_terminal_selector(problem: base.Drone3DTreeProblem, player: str) -> jnp.ndarray:
    if player not in ("offense", "defense"):
        raise ValueError("player must be 'offense' or 'defense'.")
    if player not in problem.terminal_velocity_constraint_players:
        return jnp.zeros((0, 6), dtype=F32)
    return jnp.asarray(
        (
            (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        dtype=F32,
    )


def _extract_unconstrained_player_policy_fast(
    problem: base.Drone3DTreeProblem,
    theta: jnp.ndarray,
    *,
    player: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if player == "offense":
        state_slice = slice(0, 6)
        control_matrix = problem.offense_matrix[state_slice, :]
        running_weights = jnp.asarray(problem.cfg.running_r, dtype=F32)
    elif player == "defense":
        state_slice = slice(6, 12)
        control_matrix = problem.defense_matrix[state_slice, :]
        running_weights = jnp.asarray(problem.cfg.running_s, dtype=F32)
    else:
        raise ValueError("player must be 'offense' or 'defense'.")

    alpha = build_alpha_from_logits(theta, problem.tree)
    node_type_probs, node_public_probs, _, edge_public_probs = propagate_type_probabilities(
        alpha,
        problem.topology,
        problem.prior,
    )
    leaf_type_probs = node_type_probs[problem.topology.leaf_nodes]
    leaf_public_probs = node_public_probs[problem.topology.leaf_nodes]
    parent_public_probs = node_public_probs[problem.topology.edge_parents]
    lambda_edge = edge_public_probs / jnp.clip(parent_public_probs, 1e-8, None)
    leaf_beliefs = leaf_type_probs / jnp.clip(leaf_public_probs[:, None], 1e-8, None)
    terminal_p = jnp.einsum("li,ijk->ljk", leaf_beliefs, jax.vmap(jnp.diag)(problem.terminal_pdiag))
    terminal_r = leaf_beliefs @ problem.terminal_r
    terminal_selector = _player_terminal_selector(problem, player)
    kernel = _lq_policy_kernel(
        problem.topology.node_offsets_py,
        problem.topology.edge_offsets_py,
        problem.topology.total_depth,
        problem.topology.node_count,
        problem.topology.edge_count,
        6,
        3,
        terminal_selector.shape[0],
    )
    return kernel(
        lambda_edge.astype(F32),
        terminal_p.astype(F32),
        terminal_r.astype(F32),
        problem.a_state[state_slice, state_slice].astype(F32),
        control_matrix.astype(F32),
        (problem.cfg.dt * running_weights).astype(F32),
        terminal_selector.astype(F32),
        problem.topology.edge_parents,
        problem.topology.edge_children,
        problem.topology.leaf_nodes,
    )


@lru_cache(maxsize=None)
def _full_horizon_exact_lq_value_and_grad_kernel(
    branch_factor: int,
    mixed_depth: int,
    tail_depth: int,
    force_identity_reveal: bool,
    node_offsets_py: tuple[int, ...],
    edge_offsets_py: tuple[int, ...],
    total_depth: int,
    node_count: int,
    edge_count: int,
    leaf_count: int,
    state_dim: int,
    control_dim: int,
    terminal_dim: int,
) -> Callable[..., tuple[jnp.ndarray, jnp.ndarray]]:
    solve_kernel = base._drone_single_player_lq_tree_solve_kernel(
        node_offsets_py,
        edge_offsets_py,
        total_depth,
        node_count,
        edge_count,
        leaf_count,
        state_dim,
        control_dim,
        terminal_dim,
    )
    tree = MixedPrefixTreeSpec(
        type_count=branch_factor,
        mixed_horizon_steps=mixed_depth,
        tail_horizon_steps=tail_depth,
        force_identity_reveal=force_identity_reveal,
    )
    topology = build_public_tree_topology(tree)

    @jax.jit
    def value_and_grad(
        theta: jnp.ndarray,
        prior: jnp.ndarray,
        x0_full: jnp.ndarray,
        offense_a_state: jnp.ndarray,
        defense_a_state: jnp.ndarray,
        offense_matrix: jnp.ndarray,
        defense_matrix: jnp.ndarray,
        offense_running_diag: jnp.ndarray,
        defense_running_diag: jnp.ndarray,
        offense_terminal_selector: jnp.ndarray,
        defense_terminal_selector: jnp.ndarray,
        terminal_pdiag: jnp.ndarray,
        terminal_r: jnp.ndarray,
        terminal_c: jnp.ndarray,
        edge_parents: jnp.ndarray,
        edge_children: jnp.ndarray,
        leaf_nodes: jnp.ndarray,
        leaf_parent_edges: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        def _player_value(
            theta_value: jnp.ndarray,
            player_x0: jnp.ndarray,
            a_state: jnp.ndarray,
            control_matrix: jnp.ndarray,
            running_diag: jnp.ndarray,
            terminal_selector: jnp.ndarray,
        ) -> jnp.ndarray:
            alpha = build_alpha_from_logits(theta_value, tree)
            node_type_probs, node_public_probs, _, edge_public_probs = propagate_type_probabilities(
                alpha,
                topology,
                prior,
            )
            leaf_type_probs = node_type_probs[leaf_nodes]
            leaf_public_probs = node_public_probs[leaf_nodes]
            parent_public_probs = node_public_probs[edge_parents]
            lambda_edge = edge_public_probs / jnp.clip(parent_public_probs, 1e-8, None)
            leaf_beliefs = leaf_type_probs / jnp.clip(leaf_public_probs[:, None], 1e-8, None)
            terminal_p = jnp.einsum("li,ijk->ljk", leaf_beliefs, jax.vmap(jnp.diag)(terminal_pdiag))
            terminal_r_leaf = leaf_beliefs @ terminal_r
            terminal_c_leaf = leaf_beliefs @ terminal_c
            _, _, _, _, value = solve_kernel(
                lambda_edge.astype(F32),
                terminal_p.astype(F32),
                terminal_r_leaf.astype(F32),
                terminal_c_leaf.astype(F32),
                leaf_public_probs.astype(F32),
                a_state.astype(F32),
                control_matrix.astype(F32),
                running_diag.astype(F32),
                terminal_selector.astype(F32),
                player_x0.astype(F32),
                edge_parents,
                edge_children,
                leaf_nodes,
                leaf_parent_edges,
            )
            return value

        def objective(theta_value: jnp.ndarray) -> jnp.ndarray:
            offense_value = _player_value(
                theta_value,
                x0_full[:state_dim],
                offense_a_state,
                offense_matrix,
                offense_running_diag,
                offense_terminal_selector,
            )
            defense_value = _player_value(
                theta_value,
                x0_full[state_dim : 2 * state_dim],
                defense_a_state,
                defense_matrix,
                defense_running_diag,
                defense_terminal_selector,
            )
            return offense_value - defense_value

        return jax.value_and_grad(objective)(theta)

    return value_and_grad


def solve_full_horizon_unconstrained_bilevel_with_policy(
    problem: base.Drone3DTreeProblem,
    cfg: base.Drone3DBilevelConfig,
    *,
    alpha_logits: jnp.ndarray | None = None,
    progress_callback=None,
    control_tolerance: float | None = None,
) -> tuple[dict[str, Any], Drone3DAffinePolicy, Drone3DBeliefTables]:
    function_start = time.perf_counter()
    if problem.has_box_inequality_constraints or problem.cfg.squash_controls:
        raise ValueError("Fast full-horizon online solve currently supports only unconstrained exact-LQ problems.")
    state_dim = 6
    control_dim = 3
    offense_a_state = problem.a_state[:6, :6]
    defense_a_state = problem.a_state[6:12, 6:12]
    offense_matrix = problem.offense_matrix[:6, :]
    defense_matrix = problem.defense_matrix[6:12, :]
    offense_running = jnp.asarray(problem.cfg.running_r, dtype=F32)
    defense_running = jnp.asarray(problem.cfg.running_s, dtype=F32)
    offense_terminal_selector = _player_terminal_selector(problem, "offense")
    defense_terminal_selector = _player_terminal_selector(problem, "defense")
    if offense_terminal_selector.shape[0] != defense_terminal_selector.shape[0]:
        raise ValueError("Fast full-horizon online solve currently expects matching terminal constraint dimensions.")

    theta = (
        problem.init_alpha_logits(
            seed=cfg.outer.seed,
            init_scale=cfg.outer.alpha_init_scale,
            mode=cfg.outer.alpha_init_mode,
        )
        if alpha_logits is None
        else jnp.asarray(alpha_logits, dtype=F32)
    )
    value_and_grad = _full_horizon_exact_lq_value_and_grad_kernel(
        problem.topology.branch_factor,
        problem.topology.mixed_depth,
        problem.topology.tail_depth,
        problem.tree.force_identity_reveal,
        problem.topology.node_offsets_py,
        problem.topology.edge_offsets_py,
        problem.topology.total_depth,
        problem.topology.node_count,
        problem.topology.edge_count,
        problem.topology.leaf_count,
        state_dim,
        control_dim,
        offense_terminal_selector.shape[0],
    )
    leaf_parent_edges = (
        problem.topology.leaf_edge_paths[:, -1]
        if problem.topology.total_depth > 0
        else jnp.zeros((problem.topology.leaf_count,), dtype=problem.topology.leaf_nodes.dtype)
    )
    prep_sec = time.perf_counter() - function_start
    history: list[dict[str, Any]] = []
    previous_loss: float | None = None
    outer_status = "max_steps"
    grad_theta = jnp.zeros_like(theta)
    loss_value = jnp.asarray(0.0, dtype=F32)

    for outer_step in range(int(cfg.outer.steps)):
        step_start = time.perf_counter()
        loss_value, grad_theta = value_and_grad(
            theta,
            problem.prior.astype(F32),
            problem.x0.astype(F32),
            offense_a_state.astype(F32),
            defense_a_state.astype(F32),
            offense_matrix.astype(F32),
            defense_matrix.astype(F32),
            (problem.cfg.dt * offense_running).astype(F32),
            (problem.cfg.dt * defense_running).astype(F32),
            offense_terminal_selector.astype(F32),
            defense_terminal_selector.astype(F32),
            problem.terminal_pdiag.astype(F32),
            problem.terminal_r.astype(F32),
            problem.terminal_c.astype(F32),
            problem.topology.edge_parents,
            problem.topology.edge_children,
            problem.topology.leaf_nodes,
            leaf_parent_edges,
        )
        theta = theta - float(cfg.outer.lr_alpha) * grad_theta
        if cfg.outer.alpha_logit_clip is not None:
            theta = jnp.clip(theta, -cfg.outer.alpha_logit_clip, cfg.outer.alpha_logit_clip)

        grad_norm = _tree_l2_norm(grad_theta)
        loss_float = float(loss_value)
        elapsed = time.perf_counter() - step_start
        history_entry = {
            "outer_step": outer_step + 1,
            "loss": loss_float,
            "loss_change": None if previous_loss is None else loss_float - previous_loss,
            "alpha_grad_norm": grad_norm,
            "step_elapsed_sec": elapsed,
            "solver_variant": "player_separable_lq",
            "variant2_constraint_mode": "exact_lq_full_horizon_receding",
            "box_constraints_active": False,
            "forward_solve_sec": elapsed,
            "forward_num_iterations": 2,
            "forward_initial_residual_norm": 0.0,
            "forward_residual_norm": 0.0,
            "forward_status": int(base.SolverStatus.SUCCESS),
            "forward_linear_mode": "player_separable[full_horizon_lq_riccati,full_horizon_lq_riccati]",
            "full_horizon_receding": True,
        }
        history.append(history_entry)
        if progress_callback is not None:
            progress_callback(history_entry)
        if outer_step + 1 >= max(1, int(cfg.outer.min_steps)):
            stop_reasons: list[str] = []
            if cfg.outer.grad_tolerance is not None and grad_norm <= float(cfg.outer.grad_tolerance):
                stop_reasons.append("grad_tolerance")
            loss_change = history_entry["loss_change"]
            if (
                cfg.outer.loss_change_tolerance is not None
                and loss_change is not None
                and abs(float(loss_change)) <= float(cfg.outer.loss_change_tolerance)
                and (cfg.outer.grad_tolerance is None or grad_norm <= float(cfg.outer.grad_tolerance))
            ):
                stop_reasons.append("loss_change_tolerance")
            if stop_reasons:
                outer_status = "+".join(stop_reasons)
                break
        previous_loss = loss_float

    final_eval_start = time.perf_counter()
    final_loss, final_grad = value_and_grad(
        theta,
        problem.prior.astype(F32),
        problem.x0.astype(F32),
        offense_a_state.astype(F32),
        defense_a_state.astype(F32),
        offense_matrix.astype(F32),
        defense_matrix.astype(F32),
        (problem.cfg.dt * offense_running).astype(F32),
        (problem.cfg.dt * defense_running).astype(F32),
        offense_terminal_selector.astype(F32),
        defense_terminal_selector.astype(F32),
        problem.terminal_pdiag.astype(F32),
        problem.terminal_r.astype(F32),
        problem.terminal_c.astype(F32),
        problem.topology.edge_parents,
        problem.topology.edge_children,
        problem.topology.leaf_nodes,
        leaf_parent_edges,
    )
    jax.block_until_ready(final_loss)
    final_loss_float = float(final_loss)
    final_grad_norm = _tree_l2_norm(final_grad)
    final_eval_sec = time.perf_counter() - final_eval_start
    theta = jax.lax.stop_gradient(theta)
    belief_start = time.perf_counter()
    alpha = build_alpha_from_logits(theta, problem.tree)
    beliefs = build_belief_tables(problem, alpha)
    jax.block_until_ready(beliefs.node_beliefs)
    belief_sec = time.perf_counter() - belief_start
    policy_start = time.perf_counter()
    offense_feedback, offense_bias = _extract_unconstrained_player_policy_fast(problem, theta, player="offense")
    defense_feedback, defense_bias = _extract_unconstrained_player_policy_fast(problem, theta, player="defense")
    jax.block_until_ready(offense_feedback)
    jax.block_until_ready(defense_feedback)
    policy_extract_sec = time.perf_counter() - policy_start
    policy = Drone3DAffinePolicy(
        offense_feedback=offense_feedback,
        offense_bias=offense_bias,
        defense_feedback=defense_feedback,
        defense_bias=defense_bias,
    )
    result = {
        "objective": final_loss_float,
        "objective_lq": final_loss_float,
        "alpha": alpha,
        "alpha_logits": theta,
        "solver_variant": "player_separable_lq",
        "variant2_constraint_mode": "exact_lq_full_horizon_receding",
        "box_constraints_active": False,
        "residual_norm": 0.0,
        "num_iterations": 1,
        "num_iterations_total": 2,
        "num_iterations_offense": 1,
        "num_iterations_defense": 1,
        "status": int(base.SolverStatus.SUCCESS),
        "status_offense": int(base.SolverStatus.SUCCESS),
        "status_defense": int(base.SolverStatus.SUCCESS),
        "status_reason_offense": "full_horizon_receding_exact_lq",
        "status_reason_defense": "full_horizon_receding_exact_lq",
        "forward_linear_mode": "player_separable[full_horizon_lq_riccati,full_horizon_lq_riccati]",
        "forward_linear_mode_offense": "full_horizon_lq_riccati",
        "forward_linear_mode_defense": "full_horizon_lq_riccati",
        "forward_status": int(base.SolverStatus.SUCCESS),
        "forward_residual_norm": 0.0,
        "forward_initial_residual_norm": 0.0,
        "forward_num_iterations": 2,
        "history": history,
        "outer_status": outer_status,
        "outer_steps_used": len(history),
        "alpha_grad_norm": final_grad_norm,
        "full_horizon_receding": True,
        "final_eval_sec": float(final_eval_sec),
        "belief_build_sec": float(belief_sec),
        "policy_extract_sec": float(policy_extract_sec),
        "function_internal_sec": float(time.perf_counter() - function_start),
        "prep_sec": float(prep_sec),
    }
    return result, policy, beliefs


def _padded_exact_lq_solve_once(
    problem: base.Drone3DTreeProblem,
    theta: jnp.ndarray,
    *,
    active_total_steps: int,
) -> tuple[dict[str, Any], Drone3DAffinePolicy, Drone3DBeliefTables]:
    offense_problem = base._make_drone3d_player_problem(problem, "offense")
    defense_problem = base._make_drone3d_player_problem(problem, "defense")
    if offense_problem.terminal_selector.shape[0] != defense_problem.terminal_selector.shape[0]:
        raise ValueError("Padded exact-LQ online solve currently expects matching terminal constraint dimensions.")

    alpha, node_type_probs, node_public_probs, edge_public_probs = _padded_probability_tables(problem, theta)
    parent_public_probs = node_public_probs[problem.topology.edge_parents]
    lambda_edge = edge_public_probs / jnp.clip(parent_public_probs, 1e-8, None)
    terminal_p, terminal_r, terminal_c = _padded_player_terminal_tables(
        problem,
        node_type_probs,
        node_public_probs,
    )

    kernel = _padded_lq_policy_value_kernel(
        problem.topology.node_offsets_py,
        problem.topology.edge_offsets_py,
        problem.topology.total_depth,
        problem.topology.node_count,
        problem.topology.edge_count,
        offense_problem.state_dim,
        offense_problem.control_dim,
        offense_problem.terminal_selector.shape[0],
    )
    active_steps = jnp.asarray(int(active_total_steps), dtype=jnp.int32)
    offense_states, offense_controls, offense_feedback, offense_bias, offense_value = kernel(
        lambda_edge.astype(F32),
        terminal_p.astype(F32),
        terminal_r.astype(F32),
        terminal_c.astype(F32),
        offense_problem.a_state.astype(F32),
        offense_problem.control_matrix.astype(F32),
        (problem.cfg.dt * offense_problem.running_weights).astype(F32),
        offense_problem.terminal_selector.astype(F32),
        offense_problem.x0.astype(F32),
        active_steps,
        problem.topology.edge_parents,
        problem.topology.edge_children,
    )
    defense_states, defense_controls, defense_feedback, defense_bias, defense_value = kernel(
        lambda_edge.astype(F32),
        terminal_p.astype(F32),
        terminal_r.astype(F32),
        terminal_c.astype(F32),
        defense_problem.a_state.astype(F32),
        defense_problem.control_matrix.astype(F32),
        (problem.cfg.dt * defense_problem.running_weights).astype(F32),
        defense_problem.terminal_selector.astype(F32),
        defense_problem.x0.astype(F32),
        active_steps,
        problem.topology.edge_parents,
        problem.topology.edge_children,
    )

    empty_controls = jnp.zeros((problem.topology.edge_count, 0), dtype=offense_controls.dtype)
    offense_point = base.EqualityGamePoint(
        primal=base.EqualityGamePrimal(
            node_states=offense_states,
            offense_controls=offense_controls,
            defense_controls=empty_controls,
        ),
        dual=base.EqualityGameDual(
            node_multipliers=jnp.zeros_like(offense_states),
            terminal_multipliers=jnp.zeros(
                (problem.topology.leaf_count, offense_problem.terminal_selector.shape[0]),
                dtype=offense_states.dtype,
            ),
        ),
    )
    defense_point = base.EqualityGamePoint(
        primal=base.EqualityGamePrimal(
            node_states=defense_states,
            offense_controls=defense_controls,
            defense_controls=empty_controls,
        ),
        dual=base.EqualityGameDual(
            node_multipliers=jnp.zeros_like(defense_states),
            terminal_multipliers=jnp.zeros(
                (problem.topology.leaf_count, defense_problem.terminal_selector.shape[0]),
                dtype=defense_states.dtype,
            ),
        ),
    )
    full_point = base._combine_drone3d_player_points(offense_point, defense_point)
    objective = offense_value - defense_value
    beliefs = Drone3DBeliefTables(
        node_type_probs=node_type_probs,
        node_public_probs=node_public_probs,
        node_beliefs=node_type_probs / jnp.clip(node_public_probs[:, None], 1e-8, None),
    )
    policy = Drone3DAffinePolicy(
        offense_feedback=offense_feedback,
        offense_bias=offense_bias,
        defense_feedback=defense_feedback,
        defense_bias=defense_bias,
    )
    result = {
        "objective": float(objective),
        "objective_lq": float(objective),
        "objective_offense_lq": float(offense_value),
        "objective_defense_lq": float(defense_value),
        "alpha": alpha,
        "alpha_logits": theta,
        "point": full_point,
        "solver_variant": "player_separable_lq",
        "variant2_constraint_mode": "exact_lq_padded_active_horizon",
        "box_constraints_active": False,
        "residual_norm": 0.0,
        "num_iterations": 1,
        "num_iterations_total": 2,
        "num_iterations_offense": 1,
        "num_iterations_defense": 1,
        "status": int(base.SolverStatus.SUCCESS),
        "status_offense": int(base.SolverStatus.SUCCESS),
        "status_defense": int(base.SolverStatus.SUCCESS),
        "status_reason_offense": "padded_exact_lq",
        "status_reason_defense": "padded_exact_lq",
        "forward_linear_mode": "player_separable[padded_lq_riccati,padded_lq_riccati]",
        "forward_linear_mode_offense": "padded_lq_riccati",
        "forward_linear_mode_defense": "padded_lq_riccati",
        "forward_status": int(base.SolverStatus.SUCCESS),
        "forward_residual_norm": 0.0,
        "forward_initial_residual_norm": 0.0,
        "forward_num_iterations": 2,
        "fixed_shape_padded": True,
        "active_total_horizon_steps": int(active_total_steps),
    }
    return result, policy, beliefs


def solve_padded_unconstrained_bilevel_with_policy(
    problem: base.Drone3DTreeProblem,
    cfg: base.Drone3DBilevelConfig,
    *,
    active_mixed_steps: int,
    active_total_steps: int,
    alpha_logits: jnp.ndarray | None = None,
    deterministic_logit: float = 20.0,
    progress_callback=None,
) -> tuple[dict[str, Any], Drone3DAffinePolicy, Drone3DBeliefTables]:
    if problem.has_box_inequality_constraints or problem.cfg.squash_controls:
        raise ValueError("Padded fixed-shape online solve currently supports only unconstrained exact-LQ problems.")
    if not (1 <= int(active_total_steps) <= problem.topology.total_depth):
        raise ValueError(
            f"active_total_steps={active_total_steps} must be in [1, {problem.topology.total_depth}]."
        )
    if not (0 <= int(active_mixed_steps) <= problem.tree.mixed_horizon_steps):
        raise ValueError(
            f"active_mixed_steps={active_mixed_steps} must be in [0, {problem.tree.mixed_horizon_steps}]."
        )

    offense_problem = base._make_drone3d_player_problem(problem, "offense")
    defense_problem = base._make_drone3d_player_problem(problem, "defense")
    if not offense_problem.supports_exact_lq_solver() or not defense_problem.supports_exact_lq_solver():
        raise ValueError("Padded fixed-shape online solve requires exact unconstrained LQ player problems.")
    if offense_problem.terminal_selector.shape[0] != defense_problem.terminal_selector.shape[0]:
        raise ValueError("Padded fixed-shape online solve currently expects matching terminal constraint dimensions.")

    clip = cfg.outer.alpha_logit_clip
    if clip is not None:
        deterministic_logit = min(float(deterministic_logit), float(clip))
    theta = (
        problem.init_alpha_logits(
            seed=cfg.outer.seed,
            init_scale=cfg.outer.alpha_init_scale,
            mode=cfg.outer.alpha_init_mode,
        )
        if alpha_logits is None
        else jnp.asarray(alpha_logits, dtype=F32)
    )
    theta = _project_fixed_padded_logits(
        problem.tree,
        theta,
        active_mixed_steps=int(active_mixed_steps),
        deterministic_logit=float(deterministic_logit),
    )

    value_and_grad = _padded_exact_lq_value_and_grad_kernel(
        problem.topology.branch_factor,
        problem.topology.mixed_depth,
        problem.topology.tail_depth,
        problem.topology.node_offsets_py,
        problem.topology.edge_offsets_py,
        problem.topology.total_depth,
        problem.topology.node_count,
        problem.topology.edge_count,
        offense_problem.state_dim,
        offense_problem.control_dim,
        offense_problem.terminal_selector.shape[0],
    )
    optimizer = base._make_optimizer(cfg.outer.optimizer, cfg.outer.lr_alpha)
    opt_state = optimizer.init(theta)
    history: list[dict[str, Any]] = []
    previous_loss: float | None = None
    outer_status = "max_steps"

    for outer_step in range(int(cfg.outer.steps)):
        step_start = time.perf_counter()
        loss_value, grad_theta = value_and_grad(
            theta,
            problem.prior.astype(F32),
            offense_problem.x0.astype(F32),
            defense_problem.x0.astype(F32),
            offense_problem.a_state.astype(F32),
            defense_problem.a_state.astype(F32),
            offense_problem.control_matrix.astype(F32),
            defense_problem.control_matrix.astype(F32),
            (problem.cfg.dt * offense_problem.running_weights).astype(F32),
            (problem.cfg.dt * defense_problem.running_weights).astype(F32),
            problem.terminal_pdiag.astype(F32),
            problem.terminal_r.astype(F32),
            problem.terminal_c.astype(F32),
            offense_problem.terminal_selector.astype(F32),
            defense_problem.terminal_selector.astype(F32),
            jnp.asarray(int(active_total_steps), dtype=jnp.int32),
            problem.topology.edge_parents,
            problem.topology.edge_children,
        )
        updates, opt_state = optimizer.update(grad_theta, opt_state, theta)
        theta = base.optax.apply_updates(theta, updates)
        if cfg.outer.alpha_logit_clip is not None:
            theta = jnp.clip(theta, -cfg.outer.alpha_logit_clip, cfg.outer.alpha_logit_clip)
        theta = _project_fixed_padded_logits(
            problem.tree,
            theta,
            active_mixed_steps=int(active_mixed_steps),
            deterministic_logit=float(deterministic_logit),
        )
        alpha = _padded_alpha_from_projected_logits(theta)
        _, node_public_probs, _, _ = propagate_type_probabilities(alpha, problem.topology, problem.prior)
        grad_norm = _tree_l2_norm(grad_theta)
        loss_float = float(loss_value)
        history_entry = {
            "outer_step": outer_step + 1,
            "loss": loss_float,
            "loss_change": None if previous_loss is None else loss_float - previous_loss,
            "alpha_grad_norm": grad_norm,
            "step_elapsed_sec": time.perf_counter() - step_start,
            "solver_variant": "player_separable_lq",
            "variant2_constraint_mode": "exact_lq_padded_active_horizon",
            "box_constraints_active": False,
            "forward_solve_sec": time.perf_counter() - step_start,
            "forward_num_iterations": 2,
            "forward_initial_residual_norm": 0.0,
            "forward_residual_norm": 0.0,
            "forward_status": int(base.SolverStatus.SUCCESS),
            "forward_linear_mode": "player_separable[padded_lq_riccati,padded_lq_riccati]",
            "leaf_belief_min": 0.0,
            "leaf_belief_max": 1.0,
            "path_probability_min": float(jnp.min(node_public_probs)),
            "path_probability_max": float(jnp.max(node_public_probs)),
            "active_mixed_horizon_steps": int(active_mixed_steps),
            "active_total_horizon_steps": int(active_total_steps),
            "fixed_shape_padded": True,
        }
        history.append(history_entry)
        if progress_callback is not None:
            progress_callback(history_entry)
        if outer_step + 1 >= max(1, int(cfg.outer.min_steps)):
            stop_reasons: list[str] = []
            if cfg.outer.grad_tolerance is not None and grad_norm <= float(cfg.outer.grad_tolerance):
                stop_reasons.append("grad_tolerance")
            loss_change = history_entry["loss_change"]
            if (
                cfg.outer.loss_change_tolerance is not None
                and loss_change is not None
                and abs(float(loss_change)) <= float(cfg.outer.loss_change_tolerance)
                and (cfg.outer.grad_tolerance is None or grad_norm <= float(cfg.outer.grad_tolerance))
            ):
                stop_reasons.append("loss_change_tolerance")
            if stop_reasons:
                outer_status = "+".join(stop_reasons)
                break
        previous_loss = loss_float

    result, policy, beliefs = _padded_exact_lq_solve_once(
        problem,
        theta,
        active_total_steps=int(active_total_steps),
    )
    result.update(
        {
            "history": history,
            "outer_status": outer_status,
            "outer_steps_used": len(history),
            "active_mixed_horizon_steps": int(active_mixed_steps),
            "active_total_horizon_steps": int(active_total_steps),
            "fixed_shape_padded": True,
        }
    )
    return result, policy, beliefs


def make_context_problem(
    cfg_template: base.Drone3DProblemConfig,
    *,
    x0: jnp.ndarray,
    prior: jnp.ndarray,
    mixed_horizon_steps: int,
    tail_horizon_steps: int,
    force_identity_reveal: bool,
) -> base.Drone3DTreeProblem:
    cfg = replace(
        cfg_template,
        horizon_seconds=cfg_template.dt * float(mixed_horizon_steps + tail_horizon_steps),
        prior=tuple(float(value) for value in jnp.asarray(prior, dtype=F32).tolist()),
        initial_state=tuple(float(value) for value in jnp.asarray(x0, dtype=F32).tolist()),
    )
    tree = MixedPrefixTreeSpec(
        type_count=2,
        mixed_horizon_steps=int(mixed_horizon_steps),
        tail_horizon_steps=int(tail_horizon_steps),
        force_identity_reveal=bool(force_identity_reveal),
    )
    return base.Drone3DTreeProblem(tree, cfg)


@lru_cache(maxsize=None)
def _lq_policy_kernel(
    node_offsets_py: tuple[int, ...],
    edge_offsets_py: tuple[int, ...],
    total_depth: int,
    node_count: int,
    edge_count: int,
    state_dim: int,
    control_dim: int,
    terminal_dim: int,
):
    def _edge_end(depth: int) -> int:
        return edge_offsets_py[depth + 1] if depth + 1 < len(edge_offsets_py) else edge_count

    @jax.jit
    def kernel(
        lambda_edge: jnp.ndarray,
        leaf_p: jnp.ndarray,
        leaf_r: jnp.ndarray,
        a_state: jnp.ndarray,
        control_matrix: jnp.ndarray,
        control_weight_diag: jnp.ndarray,
        terminal_selector: jnp.ndarray,
        edge_parents: jnp.ndarray,
        edge_children: jnp.ndarray,
        leaf_nodes: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        dtype = leaf_r.dtype
        p_nodes = jnp.zeros((node_count, state_dim, state_dim), dtype=dtype)
        r_nodes = jnp.zeros((node_count, state_dim), dtype=dtype)
        edge_feedback = jnp.zeros((edge_count, control_dim, state_dim), dtype=dtype)
        edge_bias = jnp.zeros((edge_count, control_dim), dtype=dtype)

        p_nodes = p_nodes.at[leaf_nodes].set(leaf_p)
        r_nodes = r_nodes.at[leaf_nodes].set(leaf_r)

        control_weight = jnp.diag(control_weight_diag.astype(dtype))
        a_transpose = a_state.T
        b_transpose = control_matrix.T
        terminal_control = terminal_selector @ control_matrix
        terminal_state = terminal_selector @ a_state

        for depth in range(total_depth - 1, -1, -1):
            node_start = node_offsets_py[depth]
            node_end = node_offsets_py[depth + 1]
            edge_start = edge_offsets_py[depth]
            edge_end = _edge_end(depth)
            node_slice = slice(node_start, node_end)
            edge_slice = slice(edge_start, edge_end)
            local_parent_count = node_end - node_start
            local_parents = edge_parents[edge_slice] - node_start
            child_indices = edge_children[edge_slice]

            p_child = p_nodes[child_indices]
            r_child = r_nodes[child_indices]

            h_block = control_weight[None, :, :] + jnp.einsum(
                "ui,eij,jv->euv",
                b_transpose,
                p_child,
                control_matrix,
            )
            f_block = jnp.einsum("ui,eij,jv->euv", b_transpose, p_child, a_state)
            g_block = jnp.einsum("ui,ei->eu", b_transpose, r_child)

            if terminal_dim > 0 and depth == total_depth - 1:
                upper = jnp.concatenate(
                    [
                        h_block,
                        jnp.broadcast_to(
                            terminal_control.T,
                            (child_indices.shape[0], control_dim, terminal_dim),
                        ),
                    ],
                    axis=2,
                )
                lower = jnp.concatenate(
                    [
                        jnp.broadcast_to(
                            terminal_control,
                            (child_indices.shape[0], terminal_dim, control_dim),
                        ),
                        jnp.zeros((child_indices.shape[0], terminal_dim, terminal_dim), dtype=dtype),
                    ],
                    axis=2,
                )
                block_matrix = jnp.concatenate([upper, lower], axis=1)
                rhs_matrix = -jnp.concatenate(
                    [
                        f_block,
                        jnp.broadcast_to(
                            terminal_state,
                            (child_indices.shape[0], terminal_dim, state_dim),
                        ),
                    ],
                    axis=1,
                )
                rhs_vector = -jnp.concatenate(
                    [
                        g_block,
                        jnp.zeros((child_indices.shape[0], terminal_dim), dtype=dtype),
                    ],
                    axis=1,
                )
                solution_matrix = jnp.linalg.solve(block_matrix, rhs_matrix)
                solution_vector = jnp.linalg.solve(block_matrix, rhs_vector[:, :, None]).squeeze(-1)
                local_feedback = solution_matrix[:, :control_dim, :]
                local_bias = solution_vector[:, :control_dim]
            else:
                rhs = jnp.concatenate([f_block, g_block[:, :, None]], axis=2)
                solve = jnp.linalg.solve(h_block, rhs)
                local_feedback = -solve[:, :, :state_dim]
                local_bias = -solve[:, :, state_dim]

            edge_feedback = edge_feedback.at[edge_slice].set(local_feedback)
            edge_bias = edge_bias.at[edge_slice].set(local_bias)

            a_closed = a_state[None, :, :] + jnp.einsum("ij,ejk->eik", control_matrix, local_feedback)
            b_closed = jnp.einsum("ij,ej->ei", control_matrix, local_bias)
            p_local = (
                jnp.einsum("eui,uv,evj->eij", local_feedback, control_weight, local_feedback)
                + jnp.einsum("eui,euv,evj->eij", a_closed, p_child, a_closed)
            )
            p_local = 0.5 * (p_local + jnp.swapaxes(p_local, -1, -2))
            child_affine = jnp.einsum("euv,ev->eu", p_child, b_closed) + r_child
            r_local = (
                jnp.einsum("eui,uv,ev->ei", local_feedback, control_weight, local_bias)
                + jnp.einsum("eui,eu->ei", a_closed, child_affine)
            )

            edge_weights = lambda_edge[edge_slice]
            p_parent = jax.ops.segment_sum(
                edge_weights[:, None, None] * p_local,
                local_parents,
                num_segments=local_parent_count,
            )
            r_parent = jax.ops.segment_sum(
                edge_weights[:, None] * r_local,
                local_parents,
                num_segments=local_parent_count,
            )
            p_nodes = p_nodes.at[node_slice].set(p_parent)
            r_nodes = r_nodes.at[node_slice].set(r_parent)

        return edge_feedback, edge_bias

    return kernel


@lru_cache(maxsize=None)
def _control_box_policy_kernel(
    node_offsets_py: tuple[int, ...],
    edge_offsets_py: tuple[int, ...],
    total_depth: int,
    node_count: int,
    edge_count: int,
    state_dim: int,
    control_dim: int,
    terminal_dim: int,
):
    def _edge_end(depth: int) -> int:
        return edge_offsets_py[depth + 1] if depth + 1 < len(edge_offsets_py) else edge_count

    @jax.jit
    def kernel(
        lambda_edge: jnp.ndarray,
        leaf_p: jnp.ndarray,
        leaf_r: jnp.ndarray,
        a_state: jnp.ndarray,
        control_matrix: jnp.ndarray,
        control_weight_diag: jnp.ndarray,
        terminal_selector: jnp.ndarray,
        edge_parents: jnp.ndarray,
        edge_children: jnp.ndarray,
        leaf_nodes: jnp.ndarray,
        free_control_mask: jnp.ndarray,
        fixed_controls: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        dtype = leaf_r.dtype
        p_nodes = jnp.zeros((node_count, state_dim, state_dim), dtype=dtype)
        r_nodes = jnp.zeros((node_count, state_dim), dtype=dtype)
        edge_feedback = jnp.zeros((edge_count, control_dim, state_dim), dtype=dtype)
        edge_bias = jnp.zeros((edge_count, control_dim), dtype=dtype)

        p_nodes = p_nodes.at[leaf_nodes].set(leaf_p)
        r_nodes = r_nodes.at[leaf_nodes].set(leaf_r)

        control_weight = jnp.diag(control_weight_diag.astype(dtype))
        control_identity = jnp.eye(control_dim, dtype=dtype)
        b_transpose = control_matrix.T
        terminal_control = terminal_selector @ control_matrix
        terminal_state = terminal_selector @ a_state

        for depth in range(total_depth - 1, -1, -1):
            node_start = node_offsets_py[depth]
            node_end = node_offsets_py[depth + 1]
            edge_start = edge_offsets_py[depth]
            edge_end = _edge_end(depth)
            node_slice = slice(node_start, node_end)
            edge_slice = slice(edge_start, edge_end)
            local_parent_count = node_end - node_start
            local_parents = edge_parents[edge_slice] - node_start
            child_indices = edge_children[edge_slice]

            p_child = p_nodes[child_indices]
            r_child = r_nodes[child_indices]
            free = free_control_mask[edge_slice].astype(dtype)
            active = 1.0 - free
            fixed = fixed_controls[edge_slice]

            h_block = control_weight[None, :, :] + jnp.einsum(
                "ui,eij,jv->euv",
                b_transpose,
                p_child,
                control_matrix,
            )
            f_block = jnp.einsum("ui,eij,jv->euv", b_transpose, p_child, a_state)
            g_block = jnp.einsum("ui,ei->eu", b_transpose, r_child)
            masked_h = h_block * free[:, :, None] * free[:, None, :]
            masked_h = masked_h + control_identity[None, :, :] * active[:, None, :]
            fixed_grad = jnp.einsum("euv,ev->eu", h_block, fixed) + g_block

            if terminal_dim > 0 and depth == total_depth - 1:
                top_right = terminal_control.T[None, :, :] * free[:, :, None]
                bottom_left = terminal_control[None, :, :] * free[:, None, :]
                upper = jnp.concatenate([masked_h, top_right], axis=2)
                lower = jnp.concatenate(
                    [
                        bottom_left,
                        jnp.zeros((child_indices.shape[0], terminal_dim, terminal_dim), dtype=dtype),
                    ],
                    axis=2,
                )
                block_matrix = jnp.concatenate([upper, lower], axis=1)
                rhs_matrix = -jnp.concatenate(
                    [
                        f_block * free[:, :, None],
                        jnp.broadcast_to(
                            terminal_state,
                            (child_indices.shape[0], terminal_dim, state_dim),
                        ),
                    ],
                    axis=1,
                )
                rhs_vector = -jnp.concatenate(
                    [
                        fixed_grad * free,
                        jnp.einsum("ij,ej->ei", terminal_control, fixed),
                    ],
                    axis=1,
                )
                solution_matrix = jnp.linalg.solve(block_matrix, rhs_matrix)
                solution_vector = jnp.linalg.solve(block_matrix, rhs_vector[:, :, None]).squeeze(-1)
                local_feedback_var = solution_matrix[:, :control_dim, :]
                local_bias_var = solution_vector[:, :control_dim]
            else:
                rhs = jnp.concatenate(
                    [
                        -(f_block * free[:, :, None]),
                        -(fixed_grad * free)[:, :, None],
                    ],
                    axis=2,
                )
                solve = jnp.linalg.solve(masked_h, rhs)
                local_feedback_var = solve[:, :, :state_dim]
                local_bias_var = solve[:, :, state_dim]

            local_feedback = free[:, :, None] * local_feedback_var
            local_bias = fixed + free * local_bias_var
            edge_feedback = edge_feedback.at[edge_slice].set(local_feedback)
            edge_bias = edge_bias.at[edge_slice].set(local_bias)

            a_closed = a_state[None, :, :] + jnp.einsum("ij,ejk->eik", control_matrix, local_feedback)
            b_closed = jnp.einsum("ij,ej->ei", control_matrix, local_bias)
            p_local = (
                jnp.einsum("eui,uv,evj->eij", local_feedback, control_weight, local_feedback)
                + jnp.einsum("eui,euv,evj->eij", a_closed, p_child, a_closed)
            )
            p_local = 0.5 * (p_local + jnp.swapaxes(p_local, -1, -2))
            child_affine = jnp.einsum("euv,ev->eu", p_child, b_closed) + r_child
            r_local = (
                jnp.einsum("eui,uv,ev->ei", local_feedback, control_weight, local_bias)
                + jnp.einsum("eui,eu->ei", a_closed, child_affine)
            )

            edge_weights = lambda_edge[edge_slice]
            p_parent = jax.ops.segment_sum(
                edge_weights[:, None, None] * p_local,
                local_parents,
                num_segments=local_parent_count,
            )
            r_parent = jax.ops.segment_sum(
                edge_weights[:, None] * r_local,
                local_parents,
                num_segments=local_parent_count,
            )
            p_nodes = p_nodes.at[node_slice].set(p_parent)
            r_nodes = r_nodes.at[node_slice].set(r_parent)

        return edge_feedback, edge_bias

    return kernel


@lru_cache(maxsize=None)
def _affine_control_policy_kernel(
    node_offsets_py: tuple[int, ...],
    edge_offsets_py: tuple[int, ...],
    total_depth: int,
    node_count: int,
    edge_count: int,
    state_dim: int,
    control_dim: int,
):
    def _edge_end(depth: int) -> int:
        return edge_offsets_py[depth + 1] if depth + 1 < len(edge_offsets_py) else edge_count

    @jax.jit
    def kernel(
        lambda_edge: jnp.ndarray,
        leaf_p: jnp.ndarray,
        leaf_r: jnp.ndarray,
        a_state: jnp.ndarray,
        control_matrix: jnp.ndarray,
        control_weight_diag: jnp.ndarray,
        edge_parents: jnp.ndarray,
        edge_children: jnp.ndarray,
        leaf_nodes: jnp.ndarray,
        free_control_mask: jnp.ndarray,
        fixed_control_feedback: jnp.ndarray,
        fixed_control_bias: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        dtype = leaf_r.dtype
        p_nodes = jnp.zeros((node_count, state_dim, state_dim), dtype=dtype)
        r_nodes = jnp.zeros((node_count, state_dim), dtype=dtype)
        edge_feedback = jnp.zeros((edge_count, control_dim, state_dim), dtype=dtype)
        edge_bias = jnp.zeros((edge_count, control_dim), dtype=dtype)

        p_nodes = p_nodes.at[leaf_nodes].set(leaf_p)
        r_nodes = r_nodes.at[leaf_nodes].set(leaf_r)

        control_weight = jnp.diag(control_weight_diag.astype(dtype))
        control_identity = jnp.eye(control_dim, dtype=dtype)
        b_transpose = control_matrix.T

        for depth in range(total_depth - 1, -1, -1):
            node_start = node_offsets_py[depth]
            node_end = node_offsets_py[depth + 1]
            edge_start = edge_offsets_py[depth]
            edge_end = _edge_end(depth)
            node_slice = slice(node_start, node_end)
            edge_slice = slice(edge_start, edge_end)
            local_parent_count = node_end - node_start
            local_parents = edge_parents[edge_slice] - node_start
            child_indices = edge_children[edge_slice]

            p_child = p_nodes[child_indices]
            r_child = r_nodes[child_indices]
            free = free_control_mask[edge_slice].astype(dtype)
            active = 1.0 - free
            base_feedback = fixed_control_feedback[edge_slice]
            base_bias = fixed_control_bias[edge_slice]

            a_base = a_state[None, :, :] + jnp.einsum("ij,ejk->eik", control_matrix, base_feedback)
            b_base = jnp.einsum("ij,ej->ei", control_matrix, base_bias)

            h_block = control_weight[None, :, :] + jnp.einsum(
                "ui,eij,jv->euv",
                b_transpose,
                p_child,
                control_matrix,
            )
            f_block = (
                jnp.einsum("ui,eij,ejk->euk", b_transpose, p_child, a_base)
                + jnp.einsum("uv,evk->euk", control_weight, base_feedback)
            )
            g_block = (
                jnp.einsum(
                    "ui,ei->eu",
                    b_transpose,
                    jnp.einsum("eij,ej->ei", p_child, b_base) + r_child,
                )
                + jnp.einsum("uv,ev->eu", control_weight, base_bias)
            )
            masked_h = h_block * free[:, :, None] * free[:, None, :]
            masked_h = masked_h + control_identity[None, :, :] * active[:, None, :]
            rhs = jnp.concatenate(
                [
                    -(f_block * free[:, :, None]),
                    -(g_block * free)[:, :, None],
                ],
                axis=2,
            )
            solve = jnp.linalg.solve(masked_h, rhs)
            variable_feedback = solve[:, :, :state_dim]
            variable_bias = solve[:, :, state_dim]

            local_feedback = base_feedback + free[:, :, None] * variable_feedback
            local_bias = base_bias + free * variable_bias
            edge_feedback = edge_feedback.at[edge_slice].set(local_feedback)
            edge_bias = edge_bias.at[edge_slice].set(local_bias)

            a_closed = a_state[None, :, :] + jnp.einsum("ij,ejk->eik", control_matrix, local_feedback)
            b_closed = jnp.einsum("ij,ej->ei", control_matrix, local_bias)
            p_local = (
                jnp.einsum("eui,uv,evj->eij", local_feedback, control_weight, local_feedback)
                + jnp.einsum("eui,euv,evj->eij", a_closed, p_child, a_closed)
            )
            p_local = 0.5 * (p_local + jnp.swapaxes(p_local, -1, -2))
            child_affine = jnp.einsum("euv,ev->eu", p_child, b_closed) + r_child
            r_local = (
                jnp.einsum("eui,uv,ev->ei", local_feedback, control_weight, local_bias)
                + jnp.einsum("eui,eu->ei", a_closed, child_affine)
            )

            edge_weights = lambda_edge[edge_slice]
            p_parent = jax.ops.segment_sum(
                edge_weights[:, None, None] * p_local,
                local_parents,
                num_segments=local_parent_count,
            )
            r_parent = jax.ops.segment_sum(
                edge_weights[:, None] * r_local,
                local_parents,
                num_segments=local_parent_count,
            )
            p_nodes = p_nodes.at[node_slice].set(p_parent)
            r_nodes = r_nodes.at[node_slice].set(r_parent)

        return edge_feedback, edge_bias

    return kernel


def _infer_active_masks(
    player_problem: base.Drone3DSinglePlayerTreeProblem,
    point: base.EqualityGamePoint,
    control_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    controls = np.asarray(point.primal.offense_controls, dtype=np.float32)
    lower = np.asarray(player_problem.control_lower_bounds, dtype=np.float32)
    upper = np.asarray(player_problem.control_upper_bounds, dtype=np.float32)
    lower_finite = np.isfinite(lower)[None, :]
    upper_finite = np.isfinite(upper)[None, :]
    active_tolerance = max(10.0 * float(control_tolerance), 1e-6)
    lower_active = lower_finite & (controls <= lower[None, :] + active_tolerance)
    upper_active = upper_finite & (controls >= upper[None, :] - active_tolerance)
    both_active = lower_active & upper_active
    if np.any(both_active):
        midpoint = 0.5 * (lower[None, :] + upper[None, :])
        choose_upper = controls >= midpoint
        lower_active = lower_active & ~(both_active & choose_upper)
        upper_active = upper_active & ~(both_active & ~choose_upper)
    return controls, lower, upper, lower_active, upper_active


def _extract_unconstrained_player_policy(
    player_problem: base.Drone3DSinglePlayerTreeProblem,
    theta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    lambda_edge, _, _, terminal_p, terminal_r, _ = player_problem._exact_lq_tree_inputs(theta)
    kernel = _lq_policy_kernel(
        player_problem.topology.node_offsets_py,
        player_problem.topology.edge_offsets_py,
        player_problem.topology.total_depth,
        player_problem.topology.node_count,
        player_problem.topology.edge_count,
        player_problem.state_dim,
        player_problem.control_dim,
        player_problem.terminal_selector.shape[0],
    )
    return kernel(
        lambda_edge.astype(F32),
        terminal_p.astype(F32),
        terminal_r.astype(F32),
        player_problem.a_state.astype(F32),
        player_problem.control_matrix.astype(F32),
        (player_problem.parent.cfg.dt * player_problem.running_weights).astype(F32),
        player_problem.terminal_selector.astype(F32),
        player_problem.topology.edge_parents,
        player_problem.topology.edge_children,
        player_problem.topology.leaf_nodes,
    )


def _extract_boxed_player_policy(
    player_problem: base.Drone3DSinglePlayerTreeProblem,
    theta: jnp.ndarray,
    point: base.EqualityGamePoint,
    control_tolerance: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    _, lower, upper, lower_active, upper_active = _infer_active_masks(
        player_problem,
        point,
        control_tolerance,
    )
    free_mask = ~(lower_active | upper_active)
    lambda_edge, _, _, terminal_p, terminal_r, _ = player_problem._exact_lq_tree_inputs(theta)

    if player_problem.terminal_selector.shape[0] > 0:
        effective_free_mask, fixed_feedback, fixed_bias = player_problem._terminal_velocity_preeliminated_control_affines(
            free_mask,
            lower_active,
            upper_active,
            lower,
            upper,
            tolerance=float(control_tolerance),
        )
        kernel = _affine_control_policy_kernel(
            player_problem.topology.node_offsets_py,
            player_problem.topology.edge_offsets_py,
            player_problem.topology.total_depth,
            player_problem.topology.node_count,
            player_problem.topology.edge_count,
            player_problem.state_dim,
            player_problem.control_dim,
        )
        return kernel(
            lambda_edge.astype(F32),
            terminal_p.astype(F32),
            terminal_r.astype(F32),
            player_problem.a_state.astype(F32),
            player_problem.control_matrix.astype(F32),
            (player_problem.parent.cfg.dt * player_problem.running_weights).astype(F32),
            player_problem.topology.edge_parents,
            player_problem.topology.edge_children,
            player_problem.topology.leaf_nodes,
            jnp.asarray(effective_free_mask),
            jnp.asarray(fixed_feedback, dtype=F32),
            jnp.asarray(fixed_bias, dtype=F32),
        )

    fixed_controls = np.zeros_like(np.asarray(point.primal.offense_controls, dtype=np.float32))
    fixed_controls = np.where(lower_active, lower[None, :], fixed_controls)
    fixed_controls = np.where(upper_active, upper[None, :], fixed_controls)
    kernel = _control_box_policy_kernel(
        player_problem.topology.node_offsets_py,
        player_problem.topology.edge_offsets_py,
        player_problem.topology.total_depth,
        player_problem.topology.node_count,
        player_problem.topology.edge_count,
        player_problem.state_dim,
        player_problem.control_dim,
        player_problem.terminal_selector.shape[0],
    )
    return kernel(
        lambda_edge.astype(F32),
        terminal_p.astype(F32),
        terminal_r.astype(F32),
        player_problem.a_state.astype(F32),
        player_problem.control_matrix.astype(F32),
        (player_problem.parent.cfg.dt * player_problem.running_weights).astype(F32),
        player_problem.terminal_selector.astype(F32),
        player_problem.topology.edge_parents,
        player_problem.topology.edge_children,
        player_problem.topology.leaf_nodes,
        jnp.asarray(free_mask),
        jnp.asarray(fixed_controls, dtype=F32),
    )


def extract_affine_policy(
    problem: base.Drone3DTreeProblem,
    theta: jnp.ndarray,
    point: base.EqualityGamePoint,
    *,
    control_tolerance: float = 1e-6,
) -> Drone3DAffinePolicy:
    offense_problem = base._make_drone3d_player_problem(problem, "offense")
    defense_problem = base._make_drone3d_player_problem(problem, "defense")
    offense_point, defense_point = base._split_drone3d_full_point_by_player(problem, point)

    if problem.has_box_inequality_constraints and not problem.cfg.squash_controls:
        offense_feedback, offense_bias = _extract_boxed_player_policy(
            offense_problem,
            theta,
            offense_point,
            control_tolerance,
        )
        defense_feedback, defense_bias = _extract_boxed_player_policy(
            defense_problem,
            theta,
            defense_point,
            control_tolerance,
        )
    else:
        offense_feedback, offense_bias = _extract_unconstrained_player_policy(offense_problem, theta)
        defense_feedback, defense_bias = _extract_unconstrained_player_policy(defense_problem, theta)

    return Drone3DAffinePolicy(
        offense_feedback=offense_feedback,
        offense_bias=offense_bias,
        defense_feedback=defense_feedback,
        defense_bias=defense_bias,
    )


def solve_bilevel_with_policy(
    problem: base.Drone3DTreeProblem,
    cfg: base.Drone3DBilevelConfig,
    *,
    alpha_logits: jnp.ndarray | None = None,
    warm_point: base.EqualityGamePoint | None = None,
    use_box_gpu_wrapper: bool = True,
    progress_callback=None,
    control_tolerance: float | None = None,
) -> tuple[dict, Drone3DAffinePolicy, Drone3DBeliefTables]:
    if use_box_gpu_wrapper:
        result = base_box_gpu.solve_drone3d_bilevel_box_gpu(
            problem,
            cfg,
            alpha_logits=alpha_logits,
            warm_point=warm_point,
            progress_callback=progress_callback,
        )
    else:
        result = base.solve_drone3d_bilevel(
            problem,
            cfg,
            alpha_logits=alpha_logits,
            warm_point=warm_point,
            progress_callback=progress_callback,
        )
    theta = jnp.asarray(result["alpha_logits"], dtype=F32)
    alpha = build_alpha_from_logits(theta, problem.tree)
    beliefs = build_belief_tables(problem, alpha)
    policy = extract_affine_policy(
        problem,
        theta,
        result["point"],
        control_tolerance=float(control_tolerance or cfg.inner.residual_tolerance),
    )
    return result, policy, beliefs


def solve_fixed_alpha_with_policy(
    problem: base.Drone3DTreeProblem,
    inner_cfg: base.TreeDiffMPCConfig,
    *,
    alpha_logits: jnp.ndarray | None = None,
    initial_point: base.EqualityGamePoint | None = None,
    use_box_gpu_wrapper: bool = True,
    control_tolerance: float | None = None,
) -> tuple[dict, Drone3DAffinePolicy, Drone3DBeliefTables]:
    if use_box_gpu_wrapper:
        result = base_box_gpu.solve_drone3d_fixed_alpha_box_gpu(
            problem,
            inner_cfg,
            alpha_logits=alpha_logits,
            initial_point=initial_point,
        )
    else:
        result = base.solve_drone3d_fixed_alpha(
            problem,
            inner_cfg,
            alpha_logits=alpha_logits,
            initial_point=initial_point,
        )
    theta = jnp.asarray(result["alpha_logits"], dtype=F32)
    alpha = build_alpha_from_logits(theta, problem.tree)
    beliefs = build_belief_tables(problem, alpha)
    policy = extract_affine_policy(
        problem,
        theta,
        result["point"],
        control_tolerance=float(control_tolerance or inner_cfg.residual_tolerance),
    )
    return result, policy, beliefs


# Imported late to keep the public helper names near the top of the file.
from .solver_jax import drone_3d_box_gpu as base_box_gpu  # noqa: E402
