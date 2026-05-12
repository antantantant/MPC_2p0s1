from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import time
from typing import Any, Callable, NamedTuple
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.optimize as spo
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .public_tree import PublicTreeTopology, build_public_tree_topology, propagate_type_probabilities
from .tree import F32, MixedPrefixTreeSpec, build_alpha_from_logits
from .tree_diffmpc import (
  EqualityGameDual,
  EqualityGamePoint,
  EqualityGamePrimal,
  SolverStatus,
  TreeDiffMPCConfig,
  TreeDiffMPCSolver,
  TreeSchurBlockData,
  TreeSchurPreconditioner,
  apply_tree_schur_preconditioner,
  build_tree_schur_preconditioner,
)
from .hexner_diffmpc import (
  BoxBarrierTerms,
  _aggregate_line_search_alphas,
  _aggregate_player_diagnostics,
  _aggregate_status_codes,
  _box_barrier_terms,
  _make_optimizer,
  _normalize_box_bounds,
  _normalize_terminal_velocity_constraint_players,
  _tree_l2_norm,
  _tree_max_abs,
  _validate_box_bounds,
)


class Drone3DSinglePlayerLinearization(NamedTuple):
  node_public_probs: jnp.ndarray
  edge_public_probs: jnp.ndarray
  leaf_type_probs: jnp.ndarray
  leaf_public_probs: jnp.ndarray
  controls: jnp.ndarray
  control_jac: jnp.ndarray
  control_second: jnp.ndarray
  dynamics_jac: jnp.ndarray
  control_grad: jnp.ndarray
  control_hdiag: jnp.ndarray
  node_grad: jnp.ndarray
  node_hdiag: jnp.ndarray
  node_constraints: jnp.ndarray
  terminal_constraints: jnp.ndarray


class Drone3DExactLQPlayerSolve(NamedTuple):
  point: EqualityGamePoint
  objective: jnp.ndarray
  mode: str
  status: int
  iterations: int
  active_lower_count: int
  active_upper_count: int
  warm_start_active_lower_count: int = 0
  warm_start_active_upper_count: int = 0
  status_reason: str = "not_recorded"
  primal_bound_violation: float = 0.0
  primal_terminal_violation: float = 0.0
  primal_dynamics_violation: float = 0.0
  active_stationarity_violation: float = 0.0
  repeated_active_set: bool = False
  active_set_iteration_log: tuple[dict[str, Any], ...] = ()
  kkt_solve_log: tuple[dict[str, Any], ...] = ()


class Drone3DBoxWarmStart(NamedTuple):
  point: EqualityGamePoint | None
  mode: str
  prepass_sec: float
  projected_control_delta_max: float
  used_projection: bool
  feasible: bool
  fallback_reason: str | None
  prepass_offense_control_min: jnp.ndarray
  prepass_offense_control_max: jnp.ndarray
  prepass_defense_control_min: jnp.ndarray
  prepass_defense_control_max: jnp.ndarray
  prepass_offense_control_max_abs: float
  prepass_defense_control_max_abs: float
  prepass_offense_control_max_violation: float
  prepass_defense_control_max_violation: float
  prepass_offense_controls_within_box: bool
  prepass_defense_controls_within_box: bool


@lru_cache(maxsize=None)
def _drone_single_player_lq_tree_solve_kernel(
  node_offsets_py: tuple[int, ...],
  edge_offsets_py: tuple[int, ...],
  total_depth: int,
  node_count: int,
  edge_count: int,
  leaf_count: int,
  state_dim: int,
  control_dim: int,
  terminal_dim: int,
) -> Callable[..., tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
  def _edge_end(depth: int) -> int:
    return edge_offsets_py[depth + 1] if depth + 1 < len(edge_offsets_py) else edge_count

  @jax.jit
  def kernel(
    lambda_edge: jnp.ndarray,
    leaf_p: jnp.ndarray,
    leaf_r: jnp.ndarray,
    leaf_c: jnp.ndarray,
    leaf_public_probs: jnp.ndarray,
    a_state: jnp.ndarray,
    control_matrix: jnp.ndarray,
    control_weight_diag: jnp.ndarray,
    terminal_selector: jnp.ndarray,
    x0: jnp.ndarray,
    edge_parents: jnp.ndarray,
    edge_children: jnp.ndarray,
    leaf_nodes: jnp.ndarray,
    leaf_parent_edges: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dtype = x0.dtype
    p_nodes = jnp.zeros((node_count, state_dim, state_dim), dtype=dtype)
    r_nodes = jnp.zeros((node_count, state_dim), dtype=dtype)
    c_nodes = jnp.zeros((node_count,), dtype=dtype)
    edge_feedback = jnp.zeros((edge_count, control_dim, state_dim), dtype=dtype)
    edge_bias = jnp.zeros((edge_count, control_dim), dtype=dtype)
    edge_terminal_feedback = jnp.zeros((edge_count, terminal_dim, state_dim), dtype=dtype)
    edge_terminal_bias = jnp.zeros((edge_count, terminal_dim), dtype=dtype)

    p_nodes = p_nodes.at[leaf_nodes].set(leaf_p)
    r_nodes = r_nodes.at[leaf_nodes].set(leaf_r)
    c_nodes = c_nodes.at[leaf_nodes].set(leaf_c)

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
            jnp.broadcast_to(terminal_control.T, (child_indices.shape[0], control_dim, terminal_dim)),
          ],
          axis=2,
        )
        lower = jnp.concatenate(
          [
            jnp.broadcast_to(terminal_control, (child_indices.shape[0], terminal_dim, control_dim)),
            jnp.zeros((child_indices.shape[0], terminal_dim, terminal_dim), dtype=dtype),
          ],
          axis=2,
        )
        block_matrix = jnp.concatenate([upper, lower], axis=1)
        rhs_matrix = -jnp.concatenate(
          [
            f_block,
            jnp.broadcast_to(terminal_state, (child_indices.shape[0], terminal_dim, state_dim)),
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
        local_terminal_feedback = solution_matrix[:, control_dim:, :]
        local_terminal_bias = solution_vector[:, control_dim:]
        edge_terminal_feedback = edge_terminal_feedback.at[edge_slice].set(local_terminal_feedback)
        edge_terminal_bias = edge_terminal_bias.at[edge_slice].set(local_terminal_bias)

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
      else:
        rhs = jnp.concatenate([f_block, g_block[:, :, None]], axis=2)
        solve = jnp.linalg.solve(h_block, rhs)
        solve_f = solve[:, :, :state_dim]
        solve_g = solve[:, :, state_dim]

        local_feedback = -solve_f
        local_bias = -solve_g
        q_block = jnp.einsum("ui,eij,jv->euv", a_transpose, p_child, a_state)
        q_lin = jnp.einsum("ui,ei->eu", a_transpose, r_child)
        q_const = c_child

        p_local = q_block - jnp.einsum("eui,euj->eij", f_block, solve_f)
        p_local = 0.5 * (p_local + jnp.swapaxes(p_local, -1, -2))
        r_local = q_lin - jnp.einsum("eui,eu->ei", f_block, solve_g)
        c_local = q_const - 0.5 * jnp.einsum("eu,eu->e", g_block, solve_g)

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
      p_nodes = p_nodes.at[node_slice].set(p_parent)
      r_nodes = r_nodes.at[node_slice].set(r_parent)
      c_nodes = c_nodes.at[node_slice].set(c_parent)

    node_states = jnp.zeros((node_count, state_dim), dtype=dtype)
    edge_controls = jnp.zeros((edge_count, control_dim), dtype=dtype)
    node_duals = jnp.zeros((node_count, state_dim), dtype=dtype)
    terminal_duals = jnp.zeros((leaf_count, terminal_dim), dtype=dtype)
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

    if leaf_count > 0:
      leaf_states = node_states[leaf_nodes]
      leaf_grad = jnp.einsum("nij,nj->ni", leaf_p, leaf_states) + leaf_r
      leaf_grad = leaf_public_probs[:, None] * leaf_grad
      if terminal_dim > 0:
        leaf_parent_states = node_states[edge_parents[leaf_parent_edges]]
        terminal_duals_cond = (
          jnp.einsum("eij,ej->ei", edge_terminal_feedback[leaf_parent_edges], leaf_parent_states)
          + edge_terminal_bias[leaf_parent_edges]
        )
        terminal_duals = leaf_public_probs[:, None] * terminal_duals_cond
        leaf_duals = -leaf_grad - terminal_duals @ terminal_selector
      else:
        leaf_duals = -leaf_grad
      node_duals = node_duals.at[leaf_nodes].set(leaf_duals)

    for depth in range(total_depth - 1, -1, -1):
      node_start = node_offsets_py[depth]
      node_end = node_offsets_py[depth + 1]
      edge_start = edge_offsets_py[depth]
      edge_end = _edge_end(depth)
      edge_slice = slice(edge_start, edge_end)
      local_parent_count = node_end - node_start
      local_parents = edge_parents[edge_slice] - node_start
      depth_parent_duals = jax.ops.segment_sum(
        node_duals[edge_children[edge_slice]] @ a_state,
        local_parents,
        num_segments=local_parent_count,
      )
      node_duals = node_duals.at[node_start:node_end].set(depth_parent_duals)

    root_value = (
      0.5 * jnp.einsum("i,ij,j->", x0, p_nodes[0], x0)
      + jnp.dot(r_nodes[0], x0)
      + c_nodes[0]
    )
    return node_states, edge_controls, node_duals, terminal_duals, root_value

  return kernel


@lru_cache(maxsize=None)
def _drone_single_player_lq_newton_tree_step_kernel(
  node_offsets_py: tuple[int, ...],
  edge_offsets_py: tuple[int, ...],
  total_depth: int,
  node_count: int,
  edge_count: int,
  leaf_count: int,
  state_dim: int,
  control_dim: int,
) -> Callable[..., tuple[jnp.ndarray, jnp.ndarray]]:
  def _edge_end(depth: int) -> int:
    return edge_offsets_py[depth + 1] if depth + 1 < len(edge_offsets_py) else edge_count

  @jax.jit
  def kernel(
    lambda_edge: jnp.ndarray,
    leaf_p: jnp.ndarray,
    leaf_r: jnp.ndarray,
    edge_hdiag: jnp.ndarray,
    edge_grad: jnp.ndarray,
    a_state: jnp.ndarray,
    control_matrix: jnp.ndarray,
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
    a_transpose = a_state.T
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
      control_weight = jnp.eye(control_dim, dtype=dtype)[None, :, :] * edge_hdiag[edge_slice, :, None]
      h_block = control_weight + jnp.einsum("ui,eij,jv->euv", b_transpose, p_child, control_matrix)
      f_block = jnp.einsum("ui,eij,jv->euv", b_transpose, p_child, a_state)
      g_block = edge_grad[edge_slice] + jnp.einsum("ui,ei->eu", b_transpose, r_child)
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
      p_nodes = p_nodes.at[node_slice].set(p_parent)
      r_nodes = r_nodes.at[node_slice].set(r_parent)

    node_step = jnp.zeros((node_count, state_dim), dtype=dtype)
    control_step = jnp.zeros((edge_count, control_dim), dtype=dtype)
    for depth in range(total_depth):
      edge_start = edge_offsets_py[depth]
      edge_end = _edge_end(depth)
      edge_slice = slice(edge_start, edge_end)
      parents = edge_parents[edge_slice]
      parent_steps = node_step[parents]
      local_controls = (
        jnp.einsum("eij,ej->ei", edge_feedback[edge_slice], parent_steps)
        + edge_bias[edge_slice]
      )
      child_steps = parent_steps @ a_state.T + local_controls @ control_matrix.T
      control_step = control_step.at[edge_slice].set(local_controls)
      node_step = node_step.at[edge_children[edge_slice]].set(child_steps)
    return node_step, control_step

  return kernel


@lru_cache(maxsize=None)
def _drone_single_player_lq_control_box_tree_solve_kernel(
  node_offsets_py: tuple[int, ...],
  edge_offsets_py: tuple[int, ...],
  total_depth: int,
  node_count: int,
  edge_count: int,
  leaf_count: int,
  state_dim: int,
  control_dim: int,
  terminal_dim: int,
) -> Callable[..., tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
  def _edge_end(depth: int) -> int:
    return edge_offsets_py[depth + 1] if depth + 1 < len(edge_offsets_py) else edge_count

  @jax.jit
  def kernel(
    lambda_edge: jnp.ndarray,
    leaf_p: jnp.ndarray,
    leaf_r: jnp.ndarray,
    leaf_c: jnp.ndarray,
    leaf_public_probs: jnp.ndarray,
    a_state: jnp.ndarray,
    control_matrix: jnp.ndarray,
    control_weight_diag: jnp.ndarray,
    terminal_selector: jnp.ndarray,
    x0: jnp.ndarray,
    edge_parents: jnp.ndarray,
    edge_children: jnp.ndarray,
    leaf_nodes: jnp.ndarray,
    leaf_parent_edges: jnp.ndarray,
    free_control_mask: jnp.ndarray,
    fixed_controls: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dtype = x0.dtype
    p_nodes = jnp.zeros((node_count, state_dim, state_dim), dtype=dtype)
    r_nodes = jnp.zeros((node_count, state_dim), dtype=dtype)
    c_nodes = jnp.zeros((node_count,), dtype=dtype)
    edge_feedback = jnp.zeros((edge_count, control_dim, state_dim), dtype=dtype)
    edge_bias = jnp.zeros((edge_count, control_dim), dtype=dtype)
    edge_terminal_feedback = jnp.zeros((edge_count, terminal_dim, state_dim), dtype=dtype)
    edge_terminal_bias = jnp.zeros((edge_count, terminal_dim), dtype=dtype)

    p_nodes = p_nodes.at[leaf_nodes].set(leaf_p)
    r_nodes = r_nodes.at[leaf_nodes].set(leaf_r)
    c_nodes = c_nodes.at[leaf_nodes].set(leaf_c)

    control_weight = jnp.diag(control_weight_diag.astype(dtype))
    control_identity = jnp.eye(control_dim, dtype=dtype)
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
            jnp.broadcast_to(terminal_state, (child_indices.shape[0], terminal_dim, state_dim)),
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
        local_terminal_feedback = solution_matrix[:, control_dim:, :]
        local_terminal_bias = solution_vector[:, control_dim:]
        edge_terminal_feedback = edge_terminal_feedback.at[edge_slice].set(local_terminal_feedback)
        edge_terminal_bias = edge_terminal_bias.at[edge_slice].set(local_terminal_bias)
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
      c_local = (
        0.5 * jnp.einsum("eu,uv,ev->e", local_bias, control_weight, local_bias)
        + 0.5 * jnp.einsum("eu,euv,ev->e", b_closed, p_child, b_closed)
        + jnp.einsum("eu,eu->e", r_child, b_closed)
        + c_child
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
      c_parent = jax.ops.segment_sum(
        edge_weights * c_local,
        local_parents,
        num_segments=local_parent_count,
      )
      p_nodes = p_nodes.at[node_slice].set(p_parent)
      r_nodes = r_nodes.at[node_slice].set(r_parent)
      c_nodes = c_nodes.at[node_slice].set(c_parent)

    node_states = jnp.zeros((node_count, state_dim), dtype=dtype)
    edge_controls = jnp.zeros((edge_count, control_dim), dtype=dtype)
    node_duals = jnp.zeros((node_count, state_dim), dtype=dtype)
    terminal_duals = jnp.zeros((leaf_count, terminal_dim), dtype=dtype)
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

    if leaf_count > 0:
      leaf_states = node_states[leaf_nodes]
      leaf_grad = jnp.einsum("nij,nj->ni", leaf_p, leaf_states) + leaf_r
      leaf_grad = leaf_public_probs[:, None] * leaf_grad
      if terminal_dim > 0:
        leaf_parent_states = node_states[edge_parents[leaf_parent_edges]]
        terminal_duals_cond = (
          jnp.einsum("eij,ej->ei", edge_terminal_feedback[leaf_parent_edges], leaf_parent_states)
          + edge_terminal_bias[leaf_parent_edges]
        )
        terminal_duals = leaf_public_probs[:, None] * terminal_duals_cond
        leaf_duals = -leaf_grad - terminal_duals @ terminal_selector
      else:
        leaf_duals = -leaf_grad
      node_duals = node_duals.at[leaf_nodes].set(leaf_duals)

    for depth in range(total_depth - 1, -1, -1):
      node_start = node_offsets_py[depth]
      node_end = node_offsets_py[depth + 1]
      edge_start = edge_offsets_py[depth]
      edge_end = _edge_end(depth)
      edge_slice = slice(edge_start, edge_end)
      local_parent_count = node_end - node_start
      local_parents = edge_parents[edge_slice] - node_start
      depth_parent_duals = jax.ops.segment_sum(
        node_duals[edge_children[edge_slice]] @ a_state,
        local_parents,
        num_segments=local_parent_count,
      )
      node_duals = node_duals.at[node_start:node_end].set(depth_parent_duals)

    root_value = (
      0.5 * jnp.einsum("i,ij,j->", x0, p_nodes[0], x0)
      + jnp.dot(r_nodes[0], x0)
      + c_nodes[0]
    )
    return node_states, edge_controls, node_duals, terminal_duals, root_value

  return kernel


@lru_cache(maxsize=None)
def _drone_single_player_lq_affine_control_tree_solve_kernel(
  node_offsets_py: tuple[int, ...],
  edge_offsets_py: tuple[int, ...],
  total_depth: int,
  node_count: int,
  edge_count: int,
  leaf_count: int,
  state_dim: int,
  control_dim: int,
  terminal_dim: int,
) -> Callable[..., tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
  def _edge_end(depth: int) -> int:
    return edge_offsets_py[depth + 1] if depth + 1 < len(edge_offsets_py) else edge_count

  @jax.jit
  def kernel(
    lambda_edge: jnp.ndarray,
    leaf_p: jnp.ndarray,
    leaf_r: jnp.ndarray,
    leaf_c: jnp.ndarray,
    edge_public_probs: jnp.ndarray,
    leaf_public_probs: jnp.ndarray,
    a_state: jnp.ndarray,
    control_matrix: jnp.ndarray,
    control_weight_diag: jnp.ndarray,
    terminal_selector: jnp.ndarray,
    x0: jnp.ndarray,
    edge_parents: jnp.ndarray,
    edge_children: jnp.ndarray,
    leaf_nodes: jnp.ndarray,
    leaf_parent_edges: jnp.ndarray,
    free_control_mask: jnp.ndarray,
    fixed_control_feedback: jnp.ndarray,
    fixed_control_bias: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dtype = x0.dtype
    p_nodes = jnp.zeros((node_count, state_dim, state_dim), dtype=dtype)
    r_nodes = jnp.zeros((node_count, state_dim), dtype=dtype)
    c_nodes = jnp.zeros((node_count,), dtype=dtype)
    edge_feedback = jnp.zeros((edge_count, control_dim, state_dim), dtype=dtype)
    edge_bias = jnp.zeros((edge_count, control_dim), dtype=dtype)

    p_nodes = p_nodes.at[leaf_nodes].set(leaf_p)
    r_nodes = r_nodes.at[leaf_nodes].set(leaf_r)
    c_nodes = c_nodes.at[leaf_nodes].set(leaf_c)

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
      c_child = c_nodes[child_indices]
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
        jnp.einsum("ui,ei->eu", b_transpose, jnp.einsum("eij,ej->ei", p_child, b_base) + r_child)
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
      c_local = (
        0.5 * jnp.einsum("eu,uv,ev->e", local_bias, control_weight, local_bias)
        + 0.5 * jnp.einsum("eu,euv,ev->e", b_closed, p_child, b_closed)
        + jnp.einsum("eu,eu->e", r_child, b_closed)
        + c_child
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
      c_parent = jax.ops.segment_sum(
        edge_weights * c_local,
        local_parents,
        num_segments=local_parent_count,
      )
      p_nodes = p_nodes.at[node_slice].set(p_parent)
      r_nodes = r_nodes.at[node_slice].set(r_parent)
      c_nodes = c_nodes.at[node_slice].set(c_parent)

    node_states = jnp.zeros((node_count, state_dim), dtype=dtype)
    edge_controls = jnp.zeros((edge_count, control_dim), dtype=dtype)
    node_duals = jnp.zeros((node_count, state_dim), dtype=dtype)
    terminal_duals = jnp.zeros((leaf_count, terminal_dim), dtype=dtype)
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

    if leaf_count > 0:
      leaf_states = node_states[leaf_nodes]
      leaf_grad = jnp.einsum("nij,nj->ni", leaf_p, leaf_states) + leaf_r
      base_leaf_duals = -leaf_public_probs[:, None] * leaf_grad
      if terminal_dim > 0:
        leaf_controls = edge_controls[leaf_parent_edges]
        leaf_edge_probs = edge_public_probs[leaf_parent_edges]
        terminal_control_jac = terminal_selector @ control_matrix
        stationarity_without_terminal = (
          leaf_edge_probs[:, None] * control_weight_diag[None, :] * leaf_controls
          - jnp.einsum("li,ij->lj", base_leaf_duals, control_matrix)
        )
        terminal_duals = jax.vmap(
          lambda residual: jnp.linalg.solve(terminal_control_jac.T, -residual)
        )(stationarity_without_terminal)
        leaf_duals = base_leaf_duals - terminal_duals @ terminal_selector
      else:
        leaf_duals = base_leaf_duals
      node_duals = node_duals.at[leaf_nodes].set(leaf_duals)

    for depth in range(total_depth - 1, -1, -1):
      node_start = node_offsets_py[depth]
      node_end = node_offsets_py[depth + 1]
      edge_start = edge_offsets_py[depth]
      edge_end = _edge_end(depth)
      edge_slice = slice(edge_start, edge_end)
      local_parent_count = node_end - node_start
      local_parents = edge_parents[edge_slice] - node_start
      depth_parent_duals = jax.ops.segment_sum(
        node_duals[edge_children[edge_slice]] @ a_state,
        local_parents,
        num_segments=local_parent_count,
      )
      node_duals = node_duals.at[node_start:node_end].set(depth_parent_duals)

    root_value = (
      0.5 * jnp.einsum("i,ij,j->", x0, p_nodes[0], x0)
      + jnp.dot(r_nodes[0], x0)
      + c_nodes[0]
    )
    return node_states, edge_controls, node_duals, terminal_duals, root_value

  return kernel


@lru_cache(maxsize=None)
def _drone3d_single_player_reduced_matvec_kernel(
  node_count: int,
  edge_count: int,
  leaf_count: int,
  state_dim: int,
  control_dim: int,
  terminal_dim: int,
) -> Callable[..., jnp.ndarray]:
  node_state_size = node_count * state_dim
  flat_control_size = edge_count * control_dim
  node_dual_size = node_count * state_dim
  terminal_dual_size = leaf_count * terminal_dim

  def pack_reduced(
    node_states: jnp.ndarray,
    flat_controls: jnp.ndarray,
    node_duals: jnp.ndarray,
    terminal_duals: jnp.ndarray,
  ) -> jnp.ndarray:
    pieces = [
      jnp.reshape(node_states, (node_state_size,)),
      jnp.reshape(flat_controls, (flat_control_size,)),
      jnp.reshape(node_duals, (node_dual_size,)),
    ]
    if terminal_dim > 0:
      pieces.append(jnp.reshape(terminal_duals, (terminal_dual_size,)))
    return jnp.concatenate(pieces, axis=0)

  def unpack_reduced(flat_value: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    offset = 0
    node_states = jnp.reshape(flat_value[offset : offset + node_state_size], (node_count, state_dim))
    offset += node_state_size
    flat_controls = jnp.reshape(flat_value[offset : offset + flat_control_size], (edge_count, control_dim))
    offset += flat_control_size
    node_duals = jnp.reshape(flat_value[offset : offset + node_dual_size], (node_count, state_dim))
    offset += node_dual_size
    if terminal_dim > 0:
      terminal_duals = jnp.reshape(
        flat_value[offset : offset + terminal_dual_size],
        (leaf_count, terminal_dim),
      )
    else:
      terminal_duals = jnp.zeros((leaf_count, 0), dtype=flat_value.dtype)
    return node_states, flat_controls, node_duals, terminal_duals

  @jax.jit
  def kernel(
    flat_value: jnp.ndarray,
    active_node_scale: jnp.ndarray,
    active_edge_scale: jnp.ndarray,
    active_leaf_scale: jnp.ndarray,
    flat_control_scale: jnp.ndarray,
    curved_control_mask: jnp.ndarray,
    effective_control_diag: jnp.ndarray,
    dynamics_jac: jnp.ndarray,
    node_hdiag: jnp.ndarray,
    a_state: jnp.ndarray,
    terminal_selector: jnp.ndarray,
    edge_parents: jnp.ndarray,
    edge_children: jnp.ndarray,
    leaf_nodes: jnp.ndarray,
    reg: jnp.ndarray,
  ) -> jnp.ndarray:
    node_states, flat_controls_raw, node_duals, terminal_duals = unpack_reduced(flat_value)
    flat_controls = flat_control_scale * flat_controls_raw
    child_duals = node_duals[edge_children]
    dual_term = jnp.einsum("ei,eij->ej", child_duals, dynamics_jac)
    safe_denominator = jnp.where(curved_control_mask, effective_control_diag, jnp.ones_like(effective_control_diag))
    curved_controls = jnp.where(
      curved_control_mask,
      dual_term / safe_denominator,
      jnp.zeros_like(flat_controls),
    )
    total_controls = flat_controls + curved_controls

    parent_dual_pullback = jax.ops.segment_sum(
      child_duals @ a_state,
      edge_parents,
      num_segments=node_count,
    )
    state_active = node_duals - parent_dual_pullback + node_hdiag * node_states + reg * node_states
    if terminal_dim > 0:
      terminal_state_term = jax.ops.segment_sum(
        terminal_duals @ terminal_selector,
        leaf_nodes,
        num_segments=node_count,
      )
      state_active = state_active + terminal_state_term
    state_matvec = active_node_scale * state_active + (1.0 - active_node_scale) * node_states

    flat_control_active = effective_control_diag * flat_controls - dual_term
    flat_control_matvec = (
      flat_control_scale * flat_control_active + (1.0 - flat_control_scale) * flat_controls_raw
    )

    predicted_states = (
      node_states[edge_parents] @ a_state.T
      + jnp.einsum("eij,ej->ei", dynamics_jac, total_controls)
    )
    edge_constraint = active_edge_scale * (node_states[edge_children] - predicted_states)
    node_constraint_active = jnp.zeros_like(node_duals)
    node_constraint_active = node_constraint_active.at[0].set(node_states[0])
    node_constraint_active = node_constraint_active.at[edge_children].add(edge_constraint)
    node_constraint_active = node_constraint_active + reg * node_duals
    node_constraint_matvec = (
      active_node_scale * node_constraint_active + (1.0 - active_node_scale) * node_duals
    )
    if terminal_dim > 0:
      terminal_constraint_active = active_leaf_scale * (
        node_states[leaf_nodes] @ terminal_selector.T + reg * terminal_duals
      )
      terminal_constraint_matvec = (
        terminal_constraint_active + (1.0 - active_leaf_scale) * terminal_duals
      )
    else:
      terminal_constraint_matvec = terminal_duals

    return pack_reduced(
      state_matvec,
      flat_control_matvec,
      node_constraint_matvec,
      terminal_constraint_matvec,
    )

  return kernel


@lru_cache(maxsize=None)
def _drone3d_single_player_preconditioner_kernel(
  node_offsets_py: tuple[int, ...],
  total_depth: int,
  node_count: int,
  edge_count: int,
  leaf_count: int,
  state_dim: int,
  control_dim: int,
  terminal_dim: int,
) -> Callable[..., jnp.ndarray]:
  node_state_size = node_count * state_dim
  flat_control_size = edge_count * control_dim
  node_dual_size = node_count * state_dim
  terminal_dual_size = leaf_count * terminal_dim

  def pack_reduced(
    node_states: jnp.ndarray,
    flat_controls: jnp.ndarray,
    node_duals: jnp.ndarray,
    terminal_duals: jnp.ndarray,
  ) -> jnp.ndarray:
    pieces = [
      jnp.reshape(node_states, (node_state_size,)),
      jnp.reshape(flat_controls, (flat_control_size,)),
      jnp.reshape(node_duals, (node_dual_size,)),
    ]
    if terminal_dim > 0:
      pieces.append(jnp.reshape(terminal_duals, (terminal_dual_size,)))
    return jnp.concatenate(pieces, axis=0)

  def unpack_reduced(flat_value: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    offset = 0
    node_states = jnp.reshape(flat_value[offset : offset + node_state_size], (node_count, state_dim))
    offset += node_state_size
    flat_controls = jnp.reshape(flat_value[offset : offset + flat_control_size], (edge_count, control_dim))
    offset += flat_control_size
    node_duals = jnp.reshape(flat_value[offset : offset + node_dual_size], (node_count, state_dim))
    offset += node_dual_size
    if terminal_dim > 0:
      terminal_duals = jnp.reshape(
        flat_value[offset : offset + terminal_dual_size],
        (leaf_count, terminal_dim),
      )
    else:
      terminal_duals = jnp.zeros((leaf_count, 0), dtype=flat_value.dtype)
    return node_states, flat_controls, node_duals, terminal_duals

  def apply_tree_schur(
    rhs: jnp.ndarray,
    schur_inv: jnp.ndarray,
    parent_coupling: jnp.ndarray,
    node_parents: jnp.ndarray,
  ) -> jnp.ndarray:
    reduced_rhs = rhs
    for depth in range(total_depth, 0, -1):
      start = node_offsets_py[depth]
      end = node_offsets_py[depth + 1] if depth < total_depth else node_count
      local_coupling = parent_coupling[start:end]
      parents = node_parents[start:end]
      child_response = jnp.einsum("nij,nj->ni", schur_inv[start:end], reduced_rhs[start:end])
      parent_update = -jnp.einsum("nij,nj->ni", local_coupling, child_response)
      reduced_rhs = reduced_rhs.at[parents].add(parent_update)

    solution = jnp.zeros_like(rhs)
    solution = solution.at[0:1].set(jnp.einsum("nij,nj->ni", schur_inv[0:1], reduced_rhs[0:1]))
    for depth in range(1, total_depth + 1):
      start = node_offsets_py[depth]
      end = node_offsets_py[depth + 1] if depth < total_depth else node_count
      local_coupling = parent_coupling[start:end]
      parent_solution = solution[node_parents[start:end]]
      local_rhs = reduced_rhs[start:end] - jnp.einsum("nji,nj->ni", local_coupling, parent_solution)
      local_solution = jnp.einsum("nij,nj->ni", schur_inv[start:end], local_rhs)
      solution = solution.at[start:end].set(local_solution)
    return solution

  @jax.jit
  def kernel(
    flat_value: jnp.ndarray,
    active_node_scale: jnp.ndarray,
    active_edge_scale: jnp.ndarray,
    active_leaf_scale: jnp.ndarray,
    flat_control_scale: jnp.ndarray,
    flat_control_mask: jnp.ndarray,
    dynamics_jac: jnp.ndarray,
    a_state: jnp.ndarray,
    terminal_selector: jnp.ndarray,
    edge_parents: jnp.ndarray,
    edge_children: jnp.ndarray,
    leaf_nodes: jnp.ndarray,
    node_parents: jnp.ndarray,
    state_inv: jnp.ndarray,
    control_safe: jnp.ndarray,
    variable_mask: jnp.ndarray,
    parent_coupling: jnp.ndarray,
    schur_inv: jnp.ndarray,
  ) -> jnp.ndarray:
    state_rhs, flat_control_rhs_raw, dual_rhs, terminal_rhs = unpack_reduced(flat_value)
    state_rhs = active_node_scale * state_rhs
    flat_control_rhs = flat_control_scale * flat_control_rhs_raw
    dual_rhs = active_node_scale * dual_rhs
    terminal_rhs = active_leaf_scale * terminal_rhs

    state_trial = state_inv * state_rhs
    flat_control_trial = jnp.where(
      flat_control_mask,
      flat_control_rhs / control_safe,
      jnp.zeros_like(flat_control_rhs),
    )
    predicted_trial = (
      state_trial[edge_parents] @ a_state.T
      + jnp.einsum("eij,ej->ei", dynamics_jac, flat_control_trial)
    )
    edge_constraint_trial = active_edge_scale * (
      state_trial[edge_children] - predicted_trial
    )
    node_constraint_trial = jnp.zeros_like(dual_rhs)
    node_constraint_trial = node_constraint_trial.at[0].set(state_trial[0])
    node_constraint_trial = node_constraint_trial.at[edge_children].add(edge_constraint_trial)
    if terminal_dim > 0:
      terminal_constraint_trial = active_leaf_scale * (
        state_trial[leaf_nodes] @ terminal_selector.T
      )
    else:
      terminal_constraint_trial = terminal_rhs

    packed_rhs = dual_rhs - node_constraint_trial
    if terminal_dim > 0:
      packed_rhs = jnp.concatenate([packed_rhs, terminal_rhs - terminal_constraint_trial], axis=1)
    packed_rhs = packed_rhs * variable_mask
    packed_solution = apply_tree_schur(packed_rhs, schur_inv, parent_coupling, node_parents)
    packed_solution = packed_solution * variable_mask
    node_duals = packed_solution[:, :state_dim]
    if terminal_dim > 0:
      terminal_duals = packed_solution[leaf_nodes, state_dim : state_dim + terminal_dim]
    else:
      terminal_duals = terminal_rhs

    child_duals = node_duals[edge_children]
    parent_dual_pullback = jax.ops.segment_sum(
      child_duals @ a_state,
      edge_parents,
      num_segments=node_count,
    )
    terminal_state_term = jnp.zeros_like(state_rhs)
    if terminal_dim > 0:
      terminal_state_term = jax.ops.segment_sum(
        terminal_duals @ terminal_selector,
        leaf_nodes,
        num_segments=node_count,
      )
    state_solution = active_node_scale * (
      state_inv * (state_rhs - node_duals + parent_dual_pullback - terminal_state_term)
    )
    flat_control_solution = jnp.where(
      flat_control_mask,
      (flat_control_rhs + jnp.einsum("ei,eij->ej", child_duals, dynamics_jac)) / control_safe,
      jnp.zeros_like(flat_control_rhs),
    )
    return pack_reduced(
      state_solution,
      flat_control_scale * flat_control_solution,
      active_node_scale * node_duals,
      active_leaf_scale * terminal_duals,
    )

  return kernel


@lru_cache(maxsize=None)
def _drone3d_single_player_tree_sweep_kernel(
  node_offsets_py: tuple[int, ...],
  edge_offsets_py: tuple[int, ...],
  total_depth: int,
  node_count: int,
  edge_count: int,
  leaf_count: int,
  state_dim: int,
  control_dim: int,
  terminal_dim: int,
) -> Callable[..., tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
  def _edge_end(depth: int) -> int:
    return edge_offsets_py[depth + 1] if depth + 1 < len(edge_offsets_py) else edge_count

  @jax.jit
  def kernel(
    state_rhs: jnp.ndarray,
    dual_rhs: jnp.ndarray,
    terminal_rhs: jnp.ndarray,
    node_hdiag: jnp.ndarray,
    dynamics_jac: jnp.ndarray,
    effective_control_diag: jnp.ndarray,
    curved_control_rhs: jnp.ndarray,
    a_state: jnp.ndarray,
    terminal_selector: jnp.ndarray,
    node_parents: jnp.ndarray,
    node_parent_edges: jnp.ndarray,
    edge_parents: jnp.ndarray,
    edge_children: jnp.ndarray,
    leaf_nodes: jnp.ndarray,
    reg: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dtype = state_rhs.dtype
    eye_state = jnp.eye(state_dim, dtype=dtype)
    eye_terminal = jnp.eye(terminal_dim, dtype=dtype)

    control_inv = 1.0 / effective_control_diag
    control_schur = jnp.einsum(
      "eik,ek,ejk->eij",
      dynamics_jac,
      control_inv,
      dynamics_jac,
    )
    edge_affine = control_schur - reg * eye_state[None, :, :]

    p_matrix = jnp.zeros((node_count, state_dim, state_dim), dtype=dtype)
    q_vector = jnp.zeros((node_count, state_dim), dtype=dtype)
    mu_matrix = jnp.zeros((node_count, terminal_dim, state_dim), dtype=dtype)
    mu_vector = jnp.zeros((node_count, terminal_dim), dtype=dtype)

    a_transpose = a_state.T
    for depth in range(total_depth, 0, -1):
      start = node_offsets_py[depth]
      end = node_offsets_py[depth + 1] if depth < total_depth else node_count
      node_slice = slice(start, end)
      node_count_depth = end - start
      h_reg_diag = node_hdiag[node_slice] + reg
      h_reg_matrix = eye_state[None, :, :] * h_reg_diag[:, None, :]
      b_block = state_rhs[node_slice]
      r_block = dual_rhs[node_slice]
      incoming_edges = node_parent_edges[node_slice]
      m_block_edge = edge_affine[incoming_edges]

      if depth == total_depth:
        if terminal_dim > 0:
          k_block = eye_state[None, :, :] + jnp.einsum("nij,njk->nik", h_reg_matrix, m_block_edge)
          upper = jnp.concatenate(
            [k_block, jnp.broadcast_to(terminal_selector.T, (node_count_depth, state_dim, terminal_dim))],
            axis=2,
          )
          lower = jnp.concatenate(
            [
              jnp.einsum("ij,njk->nik", terminal_selector, m_block_edge),
              jnp.broadcast_to(reg * eye_terminal, (node_count_depth, terminal_dim, terminal_dim)),
            ],
            axis=2,
          )
          block_matrix = jnp.concatenate([upper, lower], axis=1)
          rhs_matrix = jnp.concatenate(
            [
              -jnp.einsum("nij,jk->nik", h_reg_matrix, a_state),
              -jnp.broadcast_to(terminal_selector @ a_state, (node_count_depth, terminal_dim, state_dim)),
            ],
            axis=1,
          )
          rhs_vector = jnp.concatenate(
            [
              b_block - jnp.einsum("nij,nj->ni", h_reg_matrix, r_block),
              terminal_rhs - r_block @ terminal_selector.T,
            ],
            axis=1,
          )
          solution_matrix = jnp.linalg.solve(block_matrix, rhs_matrix)
          solution_vector = jnp.linalg.solve(block_matrix, rhs_vector[:, :, None]).squeeze(-1)
          p_matrix = p_matrix.at[node_slice].set(solution_matrix[:, :state_dim, :])
          q_vector = q_vector.at[node_slice].set(solution_vector[:, :state_dim])
          mu_matrix = mu_matrix.at[node_slice].set(solution_matrix[:, state_dim:, :])
          mu_vector = mu_vector.at[node_slice].set(solution_vector[:, state_dim:])
        else:
          k_block = eye_state[None, :, :] + jnp.einsum("nij,njk->nik", h_reg_matrix, m_block_edge)
          rhs_matrix = -jnp.einsum("nij,jk->nik", h_reg_matrix, a_state)
          rhs_vector = b_block - jnp.einsum("nij,nj->ni", h_reg_matrix, r_block)
          solution_matrix = jnp.linalg.solve(k_block, rhs_matrix)
          solution_vector = jnp.linalg.solve(k_block, rhs_vector[:, :, None]).squeeze(-1)
          p_matrix = p_matrix.at[node_slice].set(solution_matrix)
          q_vector = q_vector.at[node_slice].set(solution_vector)
      else:
        edge_start = edge_offsets_py[depth]
        edge_end = _edge_end(depth)
        depth_edges = slice(edge_start, edge_end)
        local_parents = edge_parents[depth_edges] - start
        child_p = p_matrix[edge_children[depth_edges]]
        child_q = q_vector[edge_children[depth_edges]]
        c_block = jax.ops.segment_sum(
          jnp.einsum("ij,ejk->eik", a_transpose, child_p),
          local_parents,
          num_segments=node_count_depth,
        )
        d_block = jax.ops.segment_sum(
          child_q @ a_state,
          local_parents,
          num_segments=node_count_depth,
        )
        h_minus_c = h_reg_matrix - c_block
        k_block = eye_state[None, :, :] + jnp.einsum("nij,njk->nik", h_minus_c, m_block_edge)
        rhs_matrix = -jnp.einsum("nij,jk->nik", h_minus_c, a_state)
        rhs_vector = b_block + d_block - jnp.einsum("nij,nj->ni", h_minus_c, r_block)
        solution_matrix = jnp.linalg.solve(k_block, rhs_matrix)
        solution_vector = jnp.linalg.solve(k_block, rhs_vector[:, :, None]).squeeze(-1)
        p_matrix = p_matrix.at[node_slice].set(solution_matrix)
        q_vector = q_vector.at[node_slice].set(solution_vector)

    x_solution = jnp.zeros((node_count, state_dim), dtype=dtype)
    lambda_solution = jnp.zeros((node_count, state_dim), dtype=dtype)
    mu_solution = jnp.zeros((leaf_count, terminal_dim), dtype=dtype)

    root_slice = slice(0, 1)
    root_constraint_rhs = dual_rhs[root_slice][0]
    if total_depth > 0:
      edge_start = edge_offsets_py[0]
      edge_end = _edge_end(0)
      local_parents = edge_parents[edge_start:edge_end]
      child_p = p_matrix[edge_children[edge_start:edge_end]]
      child_q = q_vector[edge_children[edge_start:edge_end]]
      c_root = jax.ops.segment_sum(
        jnp.einsum("ij,ejk->eik", a_transpose, child_p),
        local_parents,
        num_segments=1,
      )[0]
      d_root = jax.ops.segment_sum(
        child_q @ a_state,
        local_parents,
        num_segments=1,
      )[0]
    else:
      c_root = jnp.zeros((state_dim, state_dim), dtype=dtype)
      d_root = jnp.zeros((state_dim,), dtype=dtype)
    h_root_matrix = eye_state * (node_hdiag[root_slice][0] + reg)[None, :]
    h_minus_c_root = h_root_matrix - c_root
    root_matrix = eye_state - reg * h_minus_c_root
    root_rhs = state_rhs[root_slice][0] + d_root - jnp.einsum(
      "ij,j->i",
      h_minus_c_root,
      root_constraint_rhs,
    )
    lambda_root = jnp.linalg.solve(root_matrix, root_rhs)
    x_root = root_constraint_rhs - reg * lambda_root
    x_solution = x_solution.at[root_slice].set(x_root[None, :])
    lambda_solution = lambda_solution.at[root_slice].set(lambda_root[None, :])

    for depth in range(1, total_depth + 1):
      start = node_offsets_py[depth]
      end = node_offsets_py[depth + 1] if depth < total_depth else node_count
      node_slice = slice(start, end)
      parents = node_parents[node_slice]
      x_parent = x_solution[parents]
      lambda_block = jnp.einsum("nij,nj->ni", p_matrix[node_slice], x_parent) + q_vector[node_slice]
      x_block = (
        x_parent @ a_state.T
        + jnp.einsum("nij,nj->ni", edge_affine[node_parent_edges[node_slice]], lambda_block)
        + dual_rhs[node_slice]
      )
      x_solution = x_solution.at[node_slice].set(x_block)
      lambda_solution = lambda_solution.at[node_slice].set(lambda_block)
      if terminal_dim > 0 and depth == total_depth:
        mu_block = jnp.einsum("nij,nj->ni", mu_matrix[node_slice], x_parent) + mu_vector[node_slice]
        mu_solution = mu_solution.at[:].set(mu_block)

    child_duals = lambda_solution[edge_children]
    dual_term = jnp.einsum("ei,eij->ej", child_duals, dynamics_jac)
    controls = (curved_control_rhs + dual_term) / effective_control_diag
    return x_solution, controls, lambda_solution, mu_solution

  return kernel


@dataclass(frozen=True)
class Drone3DProblemConfig:
  horizon_seconds: float = 4.8
  dt: float = 0.3
  mass_kg: float = 1.3
  max_accel: float = 8.0
  squash_controls: bool = False
  prior: tuple[float, float] = (0.5, 0.5)
  running_r: tuple[float, float, float] = (0.025, 0.0125, 0.025)
  running_s: tuple[float, float, float] = (0.010, 0.020, 0.010)
  type_values: tuple[float, float] = (-1.0, 1.0)
  target_vector: tuple[float, float, float, float, float, float] = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
  target_positions: tuple[tuple[float, float, float], ...] | None = None
  terminal_type_diags: tuple[tuple[float, ...], ...] | None = None
  initial_state: tuple[float, ...] = (
    -1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
     1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
  )
  enforce_terminal_velocity_constraints: bool = True
  terminal_velocity_constraint_players: tuple[str, ...] | None = None
  offense_state_lower_bounds: tuple[float, ...] | None = None
  offense_state_upper_bounds: tuple[float, ...] | None = None
  defense_state_lower_bounds: tuple[float, ...] | None = None
  defense_state_upper_bounds: tuple[float, ...] | None = None
  offense_control_lower_bounds: tuple[float, ...] | None = None
  offense_control_upper_bounds: tuple[float, ...] | None = None
  defense_control_lower_bounds: tuple[float, ...] | None = None
  defense_control_upper_bounds: tuple[float, ...] | None = None
  inequality_barrier_weight: float = 1e-4

  @property
  def total_steps(self) -> int:
    return int(round(self.horizon_seconds / self.dt))


@dataclass(frozen=True)
class Drone3DOuterConfig:
  steps: int = 32
  min_steps: int = 4
  lr_alpha: float = 0.03
  optimizer: str = "adam"
  alpha_init_scale: float = 0.5
  alpha_init_mode: str = "symmetry_breaking"
  alpha_logit_clip: float | None = 8.0
  seed: int = 0
  grad_tolerance: float | None = 1e-4
  loss_change_tolerance: float | None = 1e-5


@dataclass(frozen=True)
class Drone3DBilevelConfig:
  inner: TreeDiffMPCConfig = TreeDiffMPCConfig()
  outer: Drone3DOuterConfig = Drone3DOuterConfig()


@dataclass(frozen=True)
class Drone3DAlphaModelConfig:
  hidden_sizes: tuple[int, ...] = (128, 128)
  learning_rate: float = 1e-3
  steps: int = 300
  batch_size: int = 32
  seed: int = 0


def build_drone3d_tree_spec(
  *,
  total_steps: int,
  reveal_step: int,
  force_identity_reveal: bool = True,
) -> MixedPrefixTreeSpec:
  if total_steps < 0:
    raise ValueError("total_steps must be nonnegative.")
  if not (0 <= reveal_step <= total_steps):
    raise ValueError("reveal_step must lie in [0, total_steps].")
  return MixedPrefixTreeSpec(
    type_count=2,
    mixed_horizon_steps=reveal_step,
    tail_horizon_steps=total_steps - reveal_step,
    force_identity_reveal=force_identity_reveal,
  )


def build_drone3d_reveal_schedule_alpha_logits(
  tree: MixedPrefixTreeSpec,
  *,
  reveal_depth: int | None,
  nonreveal_probs: tuple[float, float] = (0.5, 0.5),
  logit_scale: float = 12.0,
) -> jnp.ndarray:
  if tree.mixed_horizon_steps == 0:
    return jnp.zeros((0, 1, tree.type_count, tree.type_count), dtype=F32)
  if tree.type_count != 2:
    raise ValueError("Drone 3D reveal schedule helper currently assumes two types.")
  if reveal_depth is not None and not (0 <= reveal_depth < tree.mixed_horizon_steps):
    raise ValueError(f"reveal_depth must be in [0, {tree.mixed_horizon_steps}) or None.")

  logits = jnp.zeros(
    (tree.mixed_horizon_steps, tree.max_node_count, tree.type_count, tree.type_count),
    dtype=F32,
  )
  nonreveal = jnp.array(nonreveal_probs, dtype=F32)
  nonreveal = nonreveal / jnp.sum(nonreveal)
  nonreveal_log = jnp.log(jnp.clip(nonreveal, 1e-8, None))
  identity_probs = jnp.eye(tree.type_count, dtype=F32)
  identity_log = jnp.log(jnp.clip(identity_probs, 1e-8, None))

  for depth in range(tree.mixed_horizon_steps):
    node_count = tree.node_count(depth)
    block = jnp.broadcast_to(nonreveal_log, (node_count, tree.type_count, tree.type_count))
    if reveal_depth is not None and depth >= reveal_depth:
      block = jnp.broadcast_to(identity_log, (node_count, tree.type_count, tree.type_count))
    logits = logits.at[depth, :node_count].set(logit_scale * block)
  return logits


@dataclass(frozen=True)
class Drone3DTreeProblem:
  tree: MixedPrefixTreeSpec
  cfg: Drone3DProblemConfig = Drone3DProblemConfig()

  def __post_init__(self) -> None:
    topology = build_public_tree_topology(self.tree)
    prior = jnp.array(self.cfg.prior, dtype=F32)
    if prior.shape != (self.tree.type_count,):
      raise ValueError(
        f"Drone3DTreeProblem prior must have shape ({self.tree.type_count},), got {prior.shape}.",
      )
    if bool(jnp.any(prior < 0.0)):
      raise ValueError("Drone3DTreeProblem prior must be nonnegative.")
    prior_sum = jnp.sum(prior)
    if float(prior_sum) <= 0.0:
      raise ValueError("Drone3DTreeProblem prior must have positive mass.")
    prior = prior / prior_sum

    theta_values = jnp.asarray(self.cfg.type_values, dtype=F32)
    if theta_values.shape != (self.tree.type_count,):
      raise ValueError(
        f"type_values must have shape ({self.tree.type_count},), got {theta_values.shape}.",
      )
    target_vector = jnp.asarray(self.cfg.target_vector, dtype=F32)
    if target_vector.shape != (6,):
      raise ValueError("target_vector must have shape (6,).")
    if self.cfg.target_positions is None:
      targets = theta_values[:, None] * target_vector[None, :]
    else:
      target_positions = jnp.asarray(self.cfg.target_positions, dtype=F32)
      if target_positions.shape != (self.tree.type_count, 3):
        raise ValueError(
          f"target_positions must have shape ({self.tree.type_count}, 3), "
          f"got {target_positions.shape}.",
        )
      target_velocities = jnp.zeros((self.tree.type_count, 3), dtype=F32)
      targets = jnp.concatenate([target_positions, target_velocities], axis=1)

    if self.cfg.terminal_type_diags is None:
      if self.tree.type_count != 2:
        raise ValueError("Default Drone3D terminal diagonals assume exactly two types.")
      terminal_type_diags = jnp.asarray(
        (
          (1.0, 20.0, 20.0, 20.0, 20.0, 20.0),
          (20.0, 1.0, 20.0, 20.0, 20.0, 20.0),
        ),
        dtype=F32,
      )
    else:
      terminal_type_diags = jnp.asarray(self.cfg.terminal_type_diags, dtype=F32)
      if terminal_type_diags.shape != (self.tree.type_count, 6):
        raise ValueError(
          f"terminal_type_diags must have shape ({self.tree.type_count}, 6), "
          f"got {terminal_type_diags.shape}.",
        )
    terminal_type_matrices = jax.vmap(jnp.diag)(terminal_type_diags)
    terminal_pdiag = 2.0 * terminal_type_diags
    terminal_r = -2.0 * terminal_type_diags * targets
    terminal_c = jnp.sum(terminal_type_diags * jnp.square(targets), axis=1)

    x0 = jnp.asarray(self.cfg.initial_state, dtype=F32)
    if x0.shape != (12,):
      raise ValueError("initial_state must have shape (12,).")

    dt = float(self.cfg.dt)
    eye3 = jnp.eye(3, dtype=F32)
    zero3 = jnp.zeros((3, 3), dtype=F32)
    a_player = jnp.block(
      [
        [eye3, dt * eye3],
        [zero3, eye3],
      ],
    )
    b_player = jnp.concatenate([0.5 * (dt**2) * eye3, dt * eye3], axis=0)
    a_state = jnp.block(
      [
        [a_player, jnp.zeros((6, 6), dtype=F32)],
        [jnp.zeros((6, 6), dtype=F32), a_player],
      ],
    )
    offense_matrix = jnp.zeros((12, 3), dtype=F32).at[:6, :].set(b_player)
    defense_matrix = jnp.zeros((12, 3), dtype=F32).at[6:, :].set(b_player)

    terminal_velocity_constraint_players = _normalize_terminal_velocity_constraint_players(
      self.cfg.enforce_terminal_velocity_constraints,
      self.cfg.terminal_velocity_constraint_players,
    )
    terminal_rows: list[jnp.ndarray] = []
    if "offense" in terminal_velocity_constraint_players:
      terminal_rows.extend(
        [
          jnp.array((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=F32),
          jnp.array((0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=F32),
          jnp.array((0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=F32),
        ],
      )
    if "defense" in terminal_velocity_constraint_players:
      terminal_rows.extend(
        [
          jnp.array((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0), dtype=F32),
          jnp.array((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0), dtype=F32),
          jnp.array((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), dtype=F32),
        ],
      )
    terminal_selector = (
      jnp.stack(terminal_rows, axis=0) if terminal_rows else jnp.zeros((0, 12), dtype=F32)
    )

    if float(self.cfg.inequality_barrier_weight) < 0.0:
      raise ValueError("inequality_barrier_weight must be nonnegative.")
    offense_state_lower_bounds = _normalize_box_bounds(
      self.cfg.offense_state_lower_bounds,
      dim=6,
      name="offense_state_lower_bounds",
      default=-jnp.inf,
    )
    offense_state_upper_bounds = _normalize_box_bounds(
      self.cfg.offense_state_upper_bounds,
      dim=6,
      name="offense_state_upper_bounds",
      default=jnp.inf,
    )
    defense_state_lower_bounds = _normalize_box_bounds(
      self.cfg.defense_state_lower_bounds,
      dim=6,
      name="defense_state_lower_bounds",
      default=-jnp.inf,
    )
    defense_state_upper_bounds = _normalize_box_bounds(
      self.cfg.defense_state_upper_bounds,
      dim=6,
      name="defense_state_upper_bounds",
      default=jnp.inf,
    )
    offense_control_lower_bounds = _normalize_box_bounds(
      self.cfg.offense_control_lower_bounds,
      dim=3,
      name="offense_control_lower_bounds",
      default=-jnp.inf,
    )
    offense_control_upper_bounds = _normalize_box_bounds(
      self.cfg.offense_control_upper_bounds,
      dim=3,
      name="offense_control_upper_bounds",
      default=jnp.inf,
    )
    defense_control_lower_bounds = _normalize_box_bounds(
      self.cfg.defense_control_lower_bounds,
      dim=3,
      name="defense_control_lower_bounds",
      default=-jnp.inf,
    )
    defense_control_upper_bounds = _normalize_box_bounds(
      self.cfg.defense_control_upper_bounds,
      dim=3,
      name="defense_control_upper_bounds",
      default=jnp.inf,
    )
    _validate_box_bounds(offense_state_lower_bounds, offense_state_upper_bounds, name="offense_state_bounds")
    _validate_box_bounds(defense_state_lower_bounds, defense_state_upper_bounds, name="defense_state_bounds")
    _validate_box_bounds(offense_control_lower_bounds, offense_control_upper_bounds, name="offense_control_bounds")
    _validate_box_bounds(defense_control_lower_bounds, defense_control_upper_bounds, name="defense_control_bounds")
    has_box_inequality_constraints = bool(
      jnp.any(jnp.isfinite(offense_state_lower_bounds))
      or jnp.any(jnp.isfinite(offense_state_upper_bounds))
      or jnp.any(jnp.isfinite(defense_state_lower_bounds))
      or jnp.any(jnp.isfinite(defense_state_upper_bounds))
      or jnp.any(jnp.isfinite(offense_control_lower_bounds))
      or jnp.any(jnp.isfinite(offense_control_upper_bounds))
      or jnp.any(jnp.isfinite(defense_control_lower_bounds))
      or jnp.any(jnp.isfinite(defense_control_upper_bounds))
    )

    object.__setattr__(self, "topology", topology)
    object.__setattr__(self, "prior", prior)
    object.__setattr__(self, "theta_values", theta_values)
    object.__setattr__(self, "targets", targets)
    object.__setattr__(self, "target_vector", target_vector)
    object.__setattr__(self, "terminal_type_diags", terminal_type_diags)
    object.__setattr__(self, "terminal_type_matrices", terminal_type_matrices)
    object.__setattr__(self, "terminal_pdiag", terminal_pdiag)
    object.__setattr__(self, "terminal_r", terminal_r)
    object.__setattr__(self, "terminal_c", terminal_c)
    object.__setattr__(self, "x0", x0)
    object.__setattr__(self, "a_state", a_state)
    object.__setattr__(self, "offense_matrix", offense_matrix)
    object.__setattr__(self, "defense_matrix", defense_matrix)
    object.__setattr__(self, "terminal_velocity_constraint_players", terminal_velocity_constraint_players)
    object.__setattr__(self, "terminal_selector", terminal_selector)
    object.__setattr__(self, "offense_state_lower_bounds", offense_state_lower_bounds)
    object.__setattr__(self, "offense_state_upper_bounds", offense_state_upper_bounds)
    object.__setattr__(self, "defense_state_lower_bounds", defense_state_lower_bounds)
    object.__setattr__(self, "defense_state_upper_bounds", defense_state_upper_bounds)
    object.__setattr__(self, "offense_control_lower_bounds", offense_control_lower_bounds)
    object.__setattr__(self, "offense_control_upper_bounds", offense_control_upper_bounds)
    object.__setattr__(self, "defense_control_lower_bounds", defense_control_lower_bounds)
    object.__setattr__(self, "defense_control_upper_bounds", defense_control_upper_bounds)
    object.__setattr__(self, "inequality_barrier_weight", float(self.cfg.inequality_barrier_weight))
    object.__setattr__(self, "has_box_inequality_constraints", has_box_inequality_constraints)
    object.__setattr__(self, "state_dim", 12)
    object.__setattr__(self, "control_dim", 3)

  def empty_alpha_logits(self) -> jnp.ndarray:
    if self.tree.mixed_horizon_steps == 0:
      return jnp.zeros((0, 1, self.tree.type_count, self.tree.type_count), dtype=F32)
    return jnp.zeros(
      (self.tree.mixed_horizon_steps, self.tree.max_node_count, self.tree.type_count, self.tree.type_count),
      dtype=F32,
    )

  def init_alpha_logits(
    self,
    *,
    seed: int,
    init_scale: float,
    mode: str = "symmetry_breaking",
  ) -> jnp.ndarray:
    if self.tree.mixed_horizon_steps == 0:
      return self.empty_alpha_logits()
    normalized = mode.lower()
    if normalized == "zero":
      return self.empty_alpha_logits()
    if normalized == "random":
      key = jax.random.PRNGKey(seed)
      return init_scale * jax.random.normal(
        key,
        shape=self.empty_alpha_logits().shape,
        dtype=F32,
      )
    if normalized != "symmetry_breaking":
      raise ValueError(f"Unsupported Drone3D alpha init mode: {mode}")
    key = jax.random.PRNGKey(seed)
    jitter = 0.05 * init_scale * jax.random.normal(
      key,
      shape=self.empty_alpha_logits().shape,
      dtype=F32,
    )
    logits = jitter
    type_action_pattern = jnp.array(((1.0, -1.0), (-1.0, 1.0)), dtype=F32)
    horizon = max(1, self.tree.mixed_horizon_steps)
    for depth in range(self.tree.mixed_horizon_steps):
      node_count = self.tree.node_count(depth)
      depth_fraction = float(depth + 1) / float(horizon)
      reveal_weight = 0.1 + 0.9 * (depth_fraction**2)
      node_block = (init_scale * reveal_weight) * type_action_pattern
      logits = logits.at[depth, :node_count].add(node_block[None, :, :])
    return logits

  def _control_linearization(
    self,
    raw_control: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if self.cfg.squash_controls:
      accel = self.cfg.max_accel * jnp.tanh(raw_control)
      jac = self.cfg.max_accel * (1.0 - jnp.square(jnp.tanh(raw_control)))
      second = -2.0 * jnp.tanh(raw_control) * jac
    else:
      accel = raw_control
      jac = jnp.ones_like(raw_control)
      second = jnp.zeros_like(raw_control)
    return accel, jac, second

  def edge_dynamics(
    self,
    parent_state: jnp.ndarray,
    offense_control: jnp.ndarray,
    defense_control: jnp.ndarray,
  ) -> jnp.ndarray:
    offense_accel, _, _ = self._control_linearization(offense_control)
    defense_accel, _, _ = self._control_linearization(defense_control)
    return (
      self.a_state @ parent_state
      + self.offense_matrix @ offense_accel
      + self.defense_matrix @ defense_accel
    )

  def _state_barrier_terms(self, node_states: jnp.ndarray) -> tuple[BoxBarrierTerms, BoxBarrierTerms]:
    offense_terms = _box_barrier_terms(
      node_states[:, 0:6],
      self.offense_state_lower_bounds,
      self.offense_state_upper_bounds,
      weight=self.inequality_barrier_weight,
    )
    defense_terms = _box_barrier_terms(
      node_states[:, 6:12],
      self.defense_state_lower_bounds,
      self.defense_state_upper_bounds,
      weight=self.inequality_barrier_weight,
    )
    return offense_terms, defense_terms

  def _control_barrier_terms(
    self,
    offense_controls: jnp.ndarray,
    defense_controls: jnp.ndarray,
  ) -> tuple[BoxBarrierTerms, BoxBarrierTerms]:
    offense_terms = _box_barrier_terms(
      offense_controls,
      self.offense_control_lower_bounds,
      self.offense_control_upper_bounds,
      weight=self.inequality_barrier_weight,
    )
    defense_terms = _box_barrier_terms(
      defense_controls,
      self.defense_control_lower_bounds,
      self.defense_control_upper_bounds,
      weight=self.inequality_barrier_weight,
    )
    return offense_terms, defense_terms

  def objective(
    self,
    primal: EqualityGamePrimal,
    theta: Any,
  ) -> jnp.ndarray:
    alpha = build_alpha_from_logits(theta, self.tree)
    node_type_probs, _, _, edge_public_probs = propagate_type_probabilities(alpha, self.topology, self.prior)
    leaf_type_probs = node_type_probs[self.topology.leaf_nodes]
    offense_controls, _, _ = self._control_linearization(primal.offense_controls)
    defense_controls, _, _ = self._control_linearization(primal.defense_controls)

    running_r = jnp.asarray(self.cfg.running_r, dtype=F32)
    running_s = jnp.asarray(self.cfg.running_s, dtype=F32)
    running_loss = 0.5 * (
      jnp.sum(running_r[None, :] * jnp.square(offense_controls), axis=1)
      - jnp.sum(running_s[None, :] * jnp.square(defense_controls), axis=1)
    )
    running_loss = self.cfg.dt * jnp.sum(edge_public_probs * running_loss)

    leaf_states = primal.node_states[self.topology.leaf_nodes]
    offense_states = leaf_states[:, 0:6]
    defense_states = leaf_states[:, 6:12]
    offense_delta = offense_states[:, None, :] - self.targets[None, :, :]
    defense_delta = defense_states[:, None, :] - self.targets[None, :, :]
    offense_terminal = jnp.sum(jnp.square(offense_delta) * self.terminal_type_diags[None, :, :], axis=-1)
    defense_terminal = jnp.sum(jnp.square(defense_delta) * self.terminal_type_diags[None, :, :], axis=-1)
    terminal_loss = jnp.sum(leaf_type_probs * (offense_terminal - defense_terminal))

    offense_state_terms, defense_state_terms = self._state_barrier_terms(primal.node_states)
    offense_control_terms, defense_control_terms = self._control_barrier_terms(
      primal.offense_controls,
      primal.defense_controls,
    )
    feasible = (
      offense_state_terms.feasible
      & defense_state_terms.feasible
      & offense_control_terms.feasible
      & defense_control_terms.feasible
    )
    objective = (
      running_loss
      + terminal_loss
      + offense_state_terms.objective
      - defense_state_terms.objective
      + offense_control_terms.objective
      - defense_control_terms.objective
    )
    return jnp.where(feasible, objective, jnp.asarray(jnp.inf, dtype=objective.dtype))

  def lq_objective_no_barrier(
    self,
    primal: EqualityGamePrimal,
    theta: Any,
  ) -> jnp.ndarray:
    alpha = build_alpha_from_logits(theta, self.tree)
    node_type_probs, _, _, edge_public_probs = propagate_type_probabilities(alpha, self.topology, self.prior)
    leaf_type_probs = node_type_probs[self.topology.leaf_nodes]
    offense_controls, _, _ = self._control_linearization(primal.offense_controls)
    defense_controls, _, _ = self._control_linearization(primal.defense_controls)

    running_r = jnp.asarray(self.cfg.running_r, dtype=F32)
    running_s = jnp.asarray(self.cfg.running_s, dtype=F32)
    running_loss = 0.5 * (
      jnp.sum(running_r[None, :] * jnp.square(offense_controls), axis=1)
      - jnp.sum(running_s[None, :] * jnp.square(defense_controls), axis=1)
    )
    running_loss = self.cfg.dt * jnp.sum(edge_public_probs * running_loss)

    leaf_states = primal.node_states[self.topology.leaf_nodes]
    offense_states = leaf_states[:, 0:6]
    defense_states = leaf_states[:, 6:12]
    offense_delta = offense_states[:, None, :] - self.targets[None, :, :]
    defense_delta = defense_states[:, None, :] - self.targets[None, :, :]
    offense_terminal = jnp.sum(jnp.square(offense_delta) * self.terminal_type_diags[None, :, :], axis=-1)
    defense_terminal = jnp.sum(jnp.square(defense_delta) * self.terminal_type_diags[None, :, :], axis=-1)
    terminal_loss = jnp.sum(leaf_type_probs * (offense_terminal - defense_terminal))
    return running_loss + terminal_loss

  def evaluate(
    self,
    point: EqualityGamePoint,
    theta: Any,
  ) -> dict[str, Any]:
    alpha = build_alpha_from_logits(theta, self.tree)
    node_type_probs, node_public_probs, _, edge_public_probs = propagate_type_probabilities(alpha, self.topology, self.prior)
    leaf_type_probs = node_type_probs[self.topology.leaf_nodes]
    leaf_public_probs = node_public_probs[self.topology.leaf_nodes]
    prior_safe = jnp.clip(self.prior, 1e-8, None)
    path_type_probabilities = leaf_type_probs / prior_safe[None, :]

    leaf_states = point.primal.node_states[self.topology.leaf_nodes]
    offense_states = leaf_states[:, 0:6]
    defense_states = leaf_states[:, 6:12]
    offense_delta = offense_states[:, None, :] - self.targets[None, :, :]
    defense_delta = defense_states[:, None, :] - self.targets[None, :, :]
    offense_terminal = jnp.sum(jnp.square(offense_delta) * self.terminal_type_diags[None, :, :], axis=-1)
    defense_terminal = jnp.sum(jnp.square(defense_delta) * self.terminal_type_diags[None, :, :], axis=-1)

    leaf_state_paths = point.primal.node_states[self.topology.leaf_node_paths]
    edge_u_paths = (
      point.primal.offense_controls[self.topology.leaf_edge_paths]
      if self.topology.total_depth > 0
      else jnp.zeros((self.topology.leaf_count, 0, self.control_dim), dtype=point.primal.node_states.dtype)
    )
    edge_v_paths = (
      point.primal.defense_controls[self.topology.leaf_edge_paths]
      if self.topology.total_depth > 0
      else jnp.zeros((self.topology.leaf_count, 0, self.control_dim), dtype=point.primal.node_states.dtype)
    )
    edge_u = self.cfg.max_accel * jnp.tanh(edge_u_paths) if self.cfg.squash_controls else edge_u_paths
    edge_v = self.cfg.max_accel * jnp.tanh(edge_v_paths) if self.cfg.squash_controls else edge_v_paths

    return {
      "objective": float(self.objective(point.primal, theta)),
      "alpha": alpha,
      "alpha_logits": theta,
      "node_public_probabilities": node_public_probs,
      "edge_public_probabilities": edge_public_probs,
      "path_probabilities": leaf_public_probs,
      "path_type_probabilities": path_type_probabilities,
      "leaf_beliefs": leaf_type_probs / jnp.clip(leaf_public_probs[:, None], 1e-8, None),
      "frontier_beliefs": leaf_type_probs / jnp.clip(leaf_public_probs[:, None], 1e-8, None),
      "states": leaf_state_paths,
      "u_seq": edge_u,
      "v_seq": edge_v,
      "terminal_velocities_offense": leaf_states[:, 3:6],
      "terminal_velocities_defense": leaf_states[:, 9:12],
      "terminal_costs_offense": offense_terminal,
      "terminal_costs_defense": defense_terminal,
      "targets": self.targets,
      "prior": self.prior,
      "box_constraints_active": self.has_box_inequality_constraints,
      "type_count": self.tree.type_count,
      "mixed_horizon_steps": self.tree.mixed_horizon_steps,
      "tail_horizon_steps": self.tree.tail_horizon_steps,
      "topology_leaf_nodes": self.topology.leaf_nodes,
    }


@dataclass(frozen=True)
class Drone3DSinglePlayerTreeProblem:
  parent: Drone3DTreeProblem
  player: str

  def __post_init__(self) -> None:
    if self.player not in ("offense", "defense"):
      raise ValueError("player must be 'offense' or 'defense'.")
    state_slice = slice(0, 6) if self.player == "offense" else slice(6, 12)
    control_matrix = (
      self.parent.offense_matrix[state_slice, :]
      if self.player == "offense"
      else self.parent.defense_matrix[state_slice, :]
    )
    running_weights = (
      jnp.asarray(self.parent.cfg.running_r, dtype=F32)
      if self.player == "offense"
      else jnp.asarray(self.parent.cfg.running_s, dtype=F32)
    )
    terminal_selector = (
      jnp.asarray(
        (
          (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
          (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
          (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        dtype=F32,
      )
      if self.player in self.parent.terminal_velocity_constraint_players
      else jnp.zeros((0, 6), dtype=F32)
    )
    if self.player == "offense":
      state_lower_bounds = self.parent.offense_state_lower_bounds
      state_upper_bounds = self.parent.offense_state_upper_bounds
      control_lower_bounds = self.parent.offense_control_lower_bounds
      control_upper_bounds = self.parent.offense_control_upper_bounds
    else:
      state_lower_bounds = self.parent.defense_state_lower_bounds
      state_upper_bounds = self.parent.defense_state_upper_bounds
      control_lower_bounds = self.parent.defense_control_lower_bounds
      control_upper_bounds = self.parent.defense_control_upper_bounds
    has_box_inequality_constraints = bool(
      jnp.any(jnp.isfinite(state_lower_bounds))
      or jnp.any(jnp.isfinite(state_upper_bounds))
      or jnp.any(jnp.isfinite(control_lower_bounds))
      or jnp.any(jnp.isfinite(control_upper_bounds))
    )
    object.__setattr__(self, "tree", self.parent.tree)
    object.__setattr__(self, "topology", self.parent.topology)
    object.__setattr__(self, "prior", self.parent.prior)
    object.__setattr__(self, "targets", self.parent.targets)
    object.__setattr__(self, "terminal_pdiag", self.parent.terminal_pdiag)
    object.__setattr__(self, "terminal_r", self.parent.terminal_r)
    object.__setattr__(self, "terminal_c", self.parent.terminal_c)
    object.__setattr__(self, "state_dim", 6)
    object.__setattr__(self, "control_dim", 3)
    object.__setattr__(self, "a_state", self.parent.a_state[state_slice, state_slice])
    object.__setattr__(self, "control_matrix", control_matrix)
    object.__setattr__(self, "x0", self.parent.x0[state_slice])
    object.__setattr__(self, "running_weights", running_weights)
    object.__setattr__(self, "terminal_selector", terminal_selector)
    object.__setattr__(self, "state_lower_bounds", state_lower_bounds)
    object.__setattr__(self, "state_upper_bounds", state_upper_bounds)
    object.__setattr__(self, "control_lower_bounds", control_lower_bounds)
    object.__setattr__(self, "control_upper_bounds", control_upper_bounds)
    object.__setattr__(self, "inequality_barrier_weight", self.parent.inequality_barrier_weight)
    object.__setattr__(self, "has_box_inequality_constraints", has_box_inequality_constraints)
    leaf_index_by_node = -jnp.ones((self.topology.node_count,), dtype=jnp.int32)
    leaf_node_mask = jnp.zeros((self.topology.node_count,), dtype=F32)
    for leaf_idx, node_idx in enumerate(self.topology.leaf_nodes.tolist()):
      leaf_index_by_node = leaf_index_by_node.at[int(node_idx)].set(int(leaf_idx))
      leaf_node_mask = leaf_node_mask.at[int(node_idx)].set(1.0)
    object.__setattr__(self, "leaf_index_by_node", leaf_index_by_node)
    object.__setattr__(self, "leaf_node_mask", leaf_node_mask)

  def kkt_operator_is_symmetric(self) -> bool:
    return True

  def reduced_dual_operator_is_spd(self, regularization: float) -> bool:
    del regularization
    return False

  def reduced_dual_operator_cg_sign(self, regularization: float) -> float | None:
    if self.parent.cfg.squash_controls or float(regularization) <= 0.0:
      return None
    return -1.0

  def supports_reduced_dual_adjoint(self, regularization: float) -> bool:
    return float(regularization) > 0.0

  def singularity_aware_linearized_system_uses_regularization(self) -> bool:
    return False

  def supports_exact_lq_solver(self) -> bool:
    return not self.parent.cfg.squash_controls and not self.has_box_inequality_constraints

  def supports_exact_control_box_lq_solver(self) -> bool:
    if self.parent.cfg.squash_controls:
      return False
    if self.terminal_selector.shape[0] > 0:
      return False
    if not self.has_box_inequality_constraints:
      return False
    state_boxes_active = bool(
      jnp.any(jnp.isfinite(self.state_lower_bounds))
      or jnp.any(jnp.isfinite(self.state_upper_bounds))
    )
    control_boxes_active = bool(
      jnp.any(jnp.isfinite(self.control_lower_bounds))
      or jnp.any(jnp.isfinite(self.control_upper_bounds))
    )
    return control_boxes_active and not state_boxes_active

  def supports_exact_control_box_terminal_lq_solver(self) -> bool:
    if self.parent.cfg.squash_controls:
      return False
    if self.terminal_selector.shape[0] == 0:
      return False
    state_boxes_active = bool(
      jnp.any(jnp.isfinite(self.state_lower_bounds))
      or jnp.any(jnp.isfinite(self.state_upper_bounds))
    )
    control_boxes_active = bool(
      jnp.any(jnp.isfinite(self.control_lower_bounds))
      or jnp.any(jnp.isfinite(self.control_upper_bounds))
    )
    return control_boxes_active and not state_boxes_active

  def _exact_lq_tree_inputs(
    self,
    theta: Any,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    alpha = build_alpha_from_logits(theta, self.tree)
    node_type_probs, node_public_probs, _, edge_public_probs = propagate_type_probabilities(
      alpha,
      self.topology,
      self.prior,
    )
    leaf_type_probs = node_type_probs[self.topology.leaf_nodes]
    leaf_public_probs = node_public_probs[self.topology.leaf_nodes]
    parent_public_probs = node_public_probs[self.topology.edge_parents]
    lambda_edge = edge_public_probs / jnp.clip(parent_public_probs, 1e-8, None)
    leaf_beliefs = leaf_type_probs / jnp.clip(leaf_public_probs[:, None], 1e-8, None)
    terminal_p = jnp.einsum("li,ijk->ljk", leaf_beliefs, jax.vmap(jnp.diag)(self.terminal_pdiag))
    terminal_r = leaf_beliefs @ self.terminal_r
    terminal_c = leaf_beliefs @ self.terminal_c
    return lambda_edge, edge_public_probs, leaf_public_probs, terminal_p, terminal_r, terminal_c

  def _exact_control_stationarity_residual(
    self,
    edge_public_probs: jnp.ndarray,
    controls: jnp.ndarray,
    node_duals: jnp.ndarray,
  ) -> jnp.ndarray:
    child_multipliers = node_duals[self.topology.edge_children]
    return (
      self.parent.cfg.dt
      * edge_public_probs[:, None]
      * self.running_weights[None, :]
      * controls
      - jnp.einsum("ei,ij->ej", child_multipliers, self.control_matrix)
    )

  def solve_exact_lq(self, theta: Any) -> Drone3DExactLQPlayerSolve:
    if not self.supports_exact_lq_solver():
      raise ValueError(
        "Exact Drone3D Variant 2 LQ solve requires squash_controls=False and no active box constraints.",
      )

    lambda_edge, _, leaf_public_probs, terminal_p, terminal_r, terminal_c = self._exact_lq_tree_inputs(theta)

    kernel = _drone_single_player_lq_tree_solve_kernel(
      self.topology.node_offsets_py,
      self.topology.edge_offsets_py,
      self.topology.total_depth,
      self.topology.node_count,
      self.topology.edge_count,
      self.topology.leaf_count,
      self.state_dim,
      self.control_dim,
      self.terminal_selector.shape[0],
    )
    control_weight_diag = self.parent.cfg.dt * self.running_weights
    leaf_parent_edges = (
      self.topology.leaf_edge_paths[:, -1]
      if self.topology.total_depth > 0
      else jnp.zeros((self.topology.leaf_count,), dtype=self.topology.leaf_nodes.dtype)
    )
    node_states, controls, node_duals, terminal_duals, value = kernel(
      lambda_edge.astype(F32),
      terminal_p.astype(F32),
      terminal_r.astype(F32),
      terminal_c.astype(F32),
      leaf_public_probs.astype(F32),
      self.a_state.astype(F32),
      self.control_matrix.astype(F32),
      control_weight_diag.astype(F32),
      self.terminal_selector.astype(F32),
      self.x0.astype(F32),
      self.topology.edge_parents,
      self.topology.edge_children,
      self.topology.leaf_nodes,
      leaf_parent_edges,
    )
    point = EqualityGamePoint(
      primal=EqualityGamePrimal(
        node_states=node_states,
        offense_controls=controls,
        defense_controls=self._zero_aux_controls(node_states.dtype),
      ),
      dual=EqualityGameDual(
        node_multipliers=node_duals,
        terminal_multipliers=terminal_duals,
      ),
    )
    return Drone3DExactLQPlayerSolve(
      point=point,
      objective=value,
      mode="player_separable_lq_riccati",
      status=int(SolverStatus.SUCCESS),
      iterations=1,
      active_lower_count=0,
      active_upper_count=0,
    )

  def solve_exact_control_box_lq(
    self,
    theta: Any,
    *,
    max_active_set_iterations: int = 24,
    control_tolerance: float = 1e-6,
    initial_point: EqualityGamePoint | None = None,
  ) -> Drone3DExactLQPlayerSolve:
    if not self.supports_exact_control_box_lq_solver():
      raise ValueError(
        "Exact Drone3D control-box Variant 2 solve currently requires squash_controls=False, "
        "active control boxes, no terminal velocity equality, and no active state boxes.",
      )

    lambda_edge, edge_public_probs, leaf_public_probs, terminal_p, terminal_r, terminal_c = self._exact_lq_tree_inputs(theta)
    kernel = _drone_single_player_lq_control_box_tree_solve_kernel(
      self.topology.node_offsets_py,
      self.topology.edge_offsets_py,
      self.topology.total_depth,
      self.topology.node_count,
      self.topology.edge_count,
      self.topology.leaf_count,
      self.state_dim,
      self.control_dim,
      self.terminal_selector.shape[0],
    )
    control_weight_diag = self.parent.cfg.dt * self.running_weights
    leaf_parent_edges = (
      self.topology.leaf_edge_paths[:, -1]
      if self.topology.total_depth > 0
      else jnp.zeros((self.topology.leaf_count,), dtype=self.topology.leaf_nodes.dtype)
    )

    free_mask = np.ones((self.topology.edge_count, self.control_dim), dtype=bool)
    lower_active = np.zeros((self.topology.edge_count, self.control_dim), dtype=bool)
    upper_active = np.zeros((self.topology.edge_count, self.control_dim), dtype=bool)
    fixed_controls = np.zeros((self.topology.edge_count, self.control_dim), dtype=np.float32)
    lower = np.asarray(self.control_lower_bounds, dtype=np.float32)
    upper = np.asarray(self.control_upper_bounds, dtype=np.float32)
    lower_finite = np.isfinite(lower)[None, :]
    upper_finite = np.isfinite(upper)[None, :]
    if initial_point is not None:
      initial_controls = np.asarray(initial_point.primal.offense_controls, dtype=np.float32)
      if initial_controls.shape == (self.topology.edge_count, self.control_dim):
        active_tolerance = max(10.0 * control_tolerance, 1e-6)
        lower_active = lower_finite & (initial_controls <= lower[None, :] + active_tolerance)
        upper_active = upper_finite & (initial_controls >= upper[None, :] - active_tolerance)
        both_active = lower_active & upper_active
        if np.any(both_active):
          midpoint = 0.5 * (lower[None, :] + upper[None, :])
          choose_upper = initial_controls >= midpoint
          lower_active = lower_active & ~(both_active & choose_upper)
          upper_active = upper_active & ~(both_active & ~choose_upper)
        free_mask = ~(lower_active | upper_active)
        fixed_controls = np.where(lower_active, lower[None, :], fixed_controls)
        fixed_controls = np.where(upper_active, upper[None, :], fixed_controls)
    warm_start_active_lower_count = int(np.sum(lower_active))
    warm_start_active_upper_count = int(np.sum(upper_active))
    last_point: EqualityGamePoint | None = None
    last_value = jnp.asarray(0.0, dtype=F32)
    status = int(SolverStatus.MAX_ITERATIONS)
    status_reason = "max_active_set_iterations"
    iterations_used = 0
    seen_active_sets: set[bytes] = set()
    repeated_active_set_final = False
    active_set_iteration_log: list[dict[str, Any]] = []
    best_point: EqualityGamePoint | None = None
    best_value = jnp.asarray(0.0, dtype=F32)
    best_lower_active: np.ndarray | None = None
    best_upper_active: np.ndarray | None = None
    best_stationarity_inf = np.inf
    max_pivots = self._active_set_max_pivots()
    max_release_pivots = max(16, max_pivots // 8)

    for iteration in range(max_active_set_iterations):
      iterations_used = iteration + 1
      active_signature = np.packbits(
        np.concatenate([lower_active.reshape(-1), upper_active.reshape(-1)]).astype(np.uint8),
      ).tobytes()
      repeated_active_set = active_signature in seen_active_sets
      seen_active_sets.add(active_signature)
      node_states, controls, node_duals, terminal_duals, value = kernel(
        lambda_edge.astype(F32),
        terminal_p.astype(F32),
        terminal_r.astype(F32),
        terminal_c.astype(F32),
        leaf_public_probs.astype(F32),
        self.a_state.astype(F32),
        self.control_matrix.astype(F32),
        control_weight_diag.astype(F32),
        self.terminal_selector.astype(F32),
        self.x0.astype(F32),
        self.topology.edge_parents,
        self.topology.edge_children,
        self.topology.leaf_nodes,
        leaf_parent_edges,
        jnp.asarray(free_mask),
        jnp.asarray(fixed_controls, dtype=F32),
      )
      last_value = value
      last_point = EqualityGamePoint(
        primal=EqualityGamePrimal(
          node_states=node_states,
          offense_controls=controls,
          defense_controls=self._zero_aux_controls(node_states.dtype),
        ),
        dual=EqualityGameDual(
          node_multipliers=node_duals,
          terminal_multipliers=terminal_duals,
        ),
      )
      controls_np = np.asarray(controls)
      stationarity_np = np.asarray(
        self._exact_control_stationarity_residual(edge_public_probs, controls, node_duals)
      )

      lower_violation = (
        free_mask
        & lower_finite
        & (controls_np < lower[None, :] - control_tolerance)
      )
      upper_violation = (
        free_mask
        & upper_finite
        & (controls_np > upper[None, :] + control_tolerance)
      )
      lower_violation_amount = np.maximum(0.0, lower[None, :] - controls_np)
      upper_violation_amount = np.maximum(0.0, controls_np - upper[None, :])
      lower_finite_full = np.broadcast_to(np.isfinite(lower)[None, :], controls_np.shape)
      upper_finite_full = np.broadcast_to(np.isfinite(upper)[None, :], controls_np.shape)
      release_lower = lower_active & (stationarity_np < -control_tolerance)
      release_upper = upper_active & (stationarity_np > control_tolerance)
      stationarity_violation = np.zeros_like(stationarity_np)
      stationarity_violation[lower_active] = np.maximum(0.0, -stationarity_np[lower_active])
      stationarity_violation[upper_active] = np.maximum(0.0, stationarity_np[upper_active])
      stationarity_inf = float(np.max(stationarity_violation)) if stationarity_violation.size else 0.0
      iteration_record: dict[str, Any] = {
        "iteration": int(iteration + 1),
        "active_lower_count": int(np.sum(lower_active)),
        "active_upper_count": int(np.sum(upper_active)),
        "lower_violation_count": int(np.sum(lower_violation)),
        "upper_violation_count": int(np.sum(upper_violation)),
        "lower_violation_max": float(np.max(lower_violation_amount[lower_finite_full])) if np.any(lower_finite_full) else 0.0,
        "upper_violation_max": float(np.max(upper_violation_amount[upper_finite_full])) if np.any(upper_finite_full) else 0.0,
        "stationarity_inf": None if np.any(lower_violation) or np.any(upper_violation) else stationarity_inf,
        "release_lower_count": int(np.sum(release_lower)),
        "release_upper_count": int(np.sum(release_upper)),
        "repeated_active_set": bool(repeated_active_set),
      }
      if not np.any(lower_violation) and not np.any(upper_violation) and stationarity_inf < best_stationarity_inf:
        best_stationarity_inf = stationarity_inf
        best_point = last_point
        best_value = last_value
        best_lower_active = lower_active.copy()
        best_upper_active = upper_active.copy()
      if repeated_active_set:
        repeated_active_set_final = True
        if not np.any(lower_violation) and not np.any(upper_violation) and stationarity_inf <= max(10.0 * control_tolerance, 1e-5):
          status = int(SolverStatus.SUCCESS)
          status_reason = "repeated_active_set_with_stationarity_tolerance"
          iteration_record["action"] = "accept_repeated_active_set"
        else:
          status_reason = "repeated_active_set"
          iteration_record["action"] = "stop_repeated_active_set"
        active_set_iteration_log.append(iteration_record)
        break

      selected_lower_violation = self._active_set_pivot_mask(lower_violation, lower_violation_amount, max_pivots)
      selected_upper_violation = self._active_set_pivot_mask(upper_violation, upper_violation_amount, max_pivots)
      if np.any(selected_lower_violation) or np.any(selected_upper_violation):
        next_lower_active = lower_active | selected_lower_violation
        next_upper_active = upper_active | selected_upper_violation
        iteration_record["action"] = "add_bound_violations"
        iteration_record["selected_lower_violation_count"] = int(np.sum(selected_lower_violation))
        iteration_record["selected_upper_violation_count"] = int(np.sum(selected_upper_violation))
        active_set_iteration_log.append(iteration_record)
        status_reason = "adding_bound_violations"
      else:
        release_scores = np.zeros_like(stationarity_np)
        release_scores[release_lower] = -stationarity_np[release_lower]
        release_scores[release_upper] = stationarity_np[release_upper]
        selected_release_lower = self._active_set_pivot_mask(release_lower, release_scores, max_release_pivots)
        selected_release_upper = self._active_set_pivot_mask(release_upper, release_scores, max_release_pivots)
        if np.any(selected_release_lower) or np.any(selected_release_upper):
          next_lower_active = lower_active & ~selected_release_lower
          next_upper_active = upper_active & ~selected_release_upper
          iteration_record["action"] = "release_active_bounds"
          iteration_record["selected_release_lower_count"] = int(np.sum(selected_release_lower))
          iteration_record["selected_release_upper_count"] = int(np.sum(selected_release_upper))
          active_set_iteration_log.append(iteration_record)
          status_reason = "releasing_active_bounds"
        else:
          status = int(SolverStatus.SUCCESS)
          status_reason = "success"
          next_lower_active = lower_active
          next_upper_active = upper_active
          iteration_record["action"] = "success"
          active_set_iteration_log.append(iteration_record)
      both_active = next_lower_active & next_upper_active
      if np.any(both_active):
        choose_upper = controls_np >= 0.5 * (lower[None, :] + upper[None, :])
        next_lower_active = np.where(both_active & choose_upper, False, next_lower_active)
        next_upper_active = np.where(both_active & ~choose_upper, False, next_upper_active)
      next_free_mask = ~(next_lower_active | next_upper_active)
      next_fixed_controls = np.zeros_like(fixed_controls)
      next_fixed_controls = np.where(next_lower_active, lower[None, :], next_fixed_controls)
      next_fixed_controls = np.where(next_upper_active, upper[None, :], next_fixed_controls)

      if (
        np.array_equal(next_free_mask, free_mask)
        and np.array_equal(next_lower_active, lower_active)
        and np.array_equal(next_upper_active, upper_active)
      ):
        status = int(SolverStatus.SUCCESS)
        free_mask = next_free_mask
        lower_active = next_lower_active
        upper_active = next_upper_active
        fixed_controls = next_fixed_controls
        break

      free_mask = next_free_mask
      lower_active = next_lower_active
      upper_active = next_upper_active
      fixed_controls = next_fixed_controls
      if status == int(SolverStatus.SUCCESS):
        break

    if last_point is None:
      raise RuntimeError("Exact Drone3D control-box solve did not produce a point.")
    if status != int(SolverStatus.SUCCESS) and best_point is not None:
      last_point = best_point
      last_value = best_value
      lower_active = best_lower_active
      upper_active = best_upper_active
      status_reason = f"{status_reason}_best_feasible_active_set"
    node_states_np = np.asarray(last_point.primal.node_states, dtype=np.float64)
    controls_np = np.asarray(last_point.primal.offense_controls, dtype=np.float64)
    bound_violation, terminal_violation, dynamics_violation = self._control_primal_residuals_np(
      node_states_np,
      controls_np,
      lower=np.asarray(self.control_lower_bounds, dtype=np.float64),
      upper=np.asarray(self.control_upper_bounds, dtype=np.float64),
    )
    active_stationarity_violation = (
      0.0 if status == int(SolverStatus.SUCCESS) and np.isinf(best_stationarity_inf)
      else float(best_stationarity_inf)
    )

    return Drone3DExactLQPlayerSolve(
      point=last_point,
      objective=last_value,
      mode="player_separable_lq_box_active_set",
      status=status,
      iterations=iterations_used,
      active_lower_count=int(np.sum(lower_active)),
      active_upper_count=int(np.sum(upper_active)),
      warm_start_active_lower_count=warm_start_active_lower_count,
      warm_start_active_upper_count=warm_start_active_upper_count,
      status_reason=status_reason,
      primal_bound_violation=bound_violation,
      primal_terminal_violation=terminal_violation,
      primal_dynamics_violation=dynamics_violation,
      active_stationarity_violation=active_stationarity_violation,
      repeated_active_set=repeated_active_set_final,
      active_set_iteration_log=tuple(active_set_iteration_log),
    )

  def solve_exact_control_box_lq_ipm(
    self,
    theta: Any,
    *,
    max_iterations: int = 300,
    tolerance: float = 1e-6,
    initial_point: EqualityGamePoint | None = None,
  ) -> Drone3DExactLQPlayerSolve:
    if not self.supports_exact_control_box_lq_solver():
      raise ValueError(
        "Interior-point Drone3D control-box LQ solve requires squash_controls=False, "
        "active control boxes, no terminal velocity equality, and no active state boxes.",
      )

    lambda_edge, edge_public_probs, leaf_public_probs, terminal_p, terminal_r, terminal_c = self._exact_lq_tree_inputs(theta)
    edge_count = self.topology.edge_count
    node_count = self.topology.node_count
    control_dim = self.control_dim
    state_dim = self.state_dim
    terminal_dim = self.terminal_selector.shape[0]
    state_size = node_count * state_dim
    control_size = edge_count * control_dim
    variable_size = state_size + control_size

    a_state = np.asarray(self.a_state, dtype=np.float64)
    control_matrix = np.asarray(self.control_matrix, dtype=np.float64)
    x0 = np.asarray(self.x0, dtype=np.float64)
    edge_parents = np.asarray(self.topology.edge_parents)
    edge_children = np.asarray(self.topology.edge_children)
    leaf_nodes = np.asarray(self.topology.leaf_nodes)
    lower = np.broadcast_to(np.asarray(self.control_lower_bounds, dtype=np.float64), (edge_count, control_dim)).reshape(-1)
    upper = np.broadcast_to(np.asarray(self.control_upper_bounds, dtype=np.float64), (edge_count, control_dim)).reshape(-1)
    edge_weights = np.asarray(edge_public_probs, dtype=np.float64)
    leaf_public = np.asarray(leaf_public_probs, dtype=np.float64)
    terminal_p_np = np.asarray(terminal_p, dtype=np.float64)
    terminal_r_np = np.asarray(terminal_r, dtype=np.float64)
    terminal_c_np = np.asarray(terminal_c, dtype=np.float64)
    running_weights = np.asarray(self.running_weights, dtype=np.float64)

    h_rows: list[np.ndarray] = []
    h_cols: list[np.ndarray] = []
    h_data: list[np.ndarray] = []
    gradient = np.zeros((variable_size,), dtype=np.float64)
    constant = 0.0
    for edge_idx in range(edge_count):
      control_offset = state_size + edge_idx * control_dim
      diag = float(self.parent.cfg.dt) * edge_weights[edge_idx] * running_weights
      indices = control_offset + np.arange(control_dim)
      h_rows.append(indices)
      h_cols.append(indices)
      h_data.append(diag)
    for leaf_idx, node_idx_raw in enumerate(leaf_nodes):
      node_idx = int(node_idx_raw)
      state_offset = node_idx * state_dim
      state_indices = state_offset + np.arange(state_dim)
      p_leaf = leaf_public[leaf_idx] * terminal_p_np[leaf_idx]
      r_leaf = leaf_public[leaf_idx] * terminal_r_np[leaf_idx]
      h_rows.append(np.repeat(state_indices, state_dim))
      h_cols.append(np.tile(state_indices, state_dim))
      h_data.append(p_leaf.reshape(-1))
      gradient[state_indices] += r_leaf
      constant += float(leaf_public[leaf_idx] * terminal_c_np[leaf_idx])
    hessian_base = sp.coo_matrix(
      (
        np.concatenate(h_data) if h_data else np.zeros((0,), dtype=np.float64),
        (
          np.concatenate(h_rows) if h_rows else np.zeros((0,), dtype=np.int64),
          np.concatenate(h_cols) if h_cols else np.zeros((0,), dtype=np.int64),
        ),
      ),
      shape=(variable_size, variable_size),
    ).tocsc()

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs: list[float] = []
    row = 0
    for dim in range(state_dim):
      rows.append(row); cols.append(dim); data.append(1.0); rhs.append(float(x0[dim])); row += 1
    for edge_idx in range(edge_count):
      parent = int(edge_parents[edge_idx])
      child = int(edge_children[edge_idx])
      parent_offset = parent * state_dim
      child_offset = child * state_dim
      control_offset = state_size + edge_idx * control_dim
      for dim in range(state_dim):
        rows.append(row); cols.append(child_offset + dim); data.append(1.0)
        for parent_dim in range(state_dim):
          coeff = -float(a_state[dim, parent_dim])
          if coeff != 0.0:
            rows.append(row); cols.append(parent_offset + parent_dim); data.append(coeff)
        for control_dim_idx in range(control_dim):
          coeff = -float(control_matrix[dim, control_dim_idx])
          if coeff != 0.0:
            rows.append(row); cols.append(control_offset + control_dim_idx); data.append(coeff)
        rhs.append(0.0); row += 1
    constraint_matrix = sp.coo_matrix((data, (rows, cols)), shape=(row, variable_size)).tocsc()
    constraint_rhs = np.asarray(rhs, dtype=np.float64)
    zero_dual = sp.csc_matrix((constraint_matrix.shape[0], constraint_matrix.shape[0]), dtype=np.float64)
    newton_kernel = _drone_single_player_lq_newton_tree_step_kernel(
      self.topology.node_offsets_py,
      self.topology.edge_offsets_py,
      self.topology.total_depth,
      node_count,
      edge_count,
      self.topology.leaf_count,
      state_dim,
      control_dim,
    )

    def rollout(control_flat: np.ndarray) -> np.ndarray:
      controls_2d = control_flat.reshape((edge_count, control_dim))
      node_states_value = np.zeros((node_count, state_dim), dtype=np.float64)
      node_states_value[0] = x0
      for edge_idx in range(edge_count):
        parent = int(edge_parents[edge_idx])
        child = int(edge_children[edge_idx])
        node_states_value[child] = node_states_value[parent] @ a_state.T + controls_2d[edge_idx] @ control_matrix.T
      return node_states_value

    margin = 1e-3
    if initial_point is not None:
      init_controls = np.asarray(initial_point.primal.offense_controls, dtype=np.float64).reshape(-1)
      if init_controls.shape != (control_size,):
        init_controls = np.zeros((control_size,), dtype=np.float64)
    else:
      init_controls = np.zeros((control_size,), dtype=np.float64)
    span = upper - lower
    interior_lower = lower + np.minimum(margin, 0.25 * span)
    interior_upper = upper - np.minimum(margin, 0.25 * span)
    control_flat = np.minimum(np.maximum(init_controls, interior_lower), interior_upper)
    node_states = rollout(control_flat)
    variables = np.concatenate([node_states.reshape(-1), control_flat])

    mu = 1.0
    status = int(SolverStatus.MAX_ITERATIONS)
    status_reason = "max_ipm_iterations"
    iteration_log: list[dict[str, Any]] = []
    best_variables = variables.copy()
    best_kkt_inf = np.inf
    best_objective = 0.5 * float(variables @ (hessian_base @ variables)) + float(gradient @ variables) + constant
    max_iterations = max(1, int(max_iterations))
    tol = max(float(tolerance), 1e-7)

    def objective_no_barrier(var: np.ndarray) -> float:
      return 0.5 * float(var @ (hessian_base @ var)) + float(gradient @ var) + constant

    def barrier_terms(control_value: np.ndarray, mu_value: float) -> tuple[np.ndarray, np.ndarray, float, float]:
      slack_l = control_value - lower
      slack_u = upper - control_value
      if np.any(slack_l <= 0.0) or np.any(slack_u <= 0.0):
        return np.zeros_like(control_value), np.zeros_like(control_value), np.inf, np.inf
      barrier_grad = mu_value * (-1.0 / slack_l + 1.0 / slack_u)
      barrier_hdiag = mu_value * (1.0 / np.square(slack_l) + 1.0 / np.square(slack_u))
      barrier_obj = -mu_value * float(np.sum(np.log(slack_l) + np.log(slack_u)))
      complementarity = float(np.max(np.maximum(mu_value / slack_l, mu_value / slack_u)))
      return barrier_grad, barrier_hdiag, barrier_obj, complementarity

    for iteration in range(max_iterations):
      control_value = variables[state_size:]
      barrier_grad, barrier_hdiag, barrier_obj, complementarity = barrier_terms(control_value, mu)
      if not np.isfinite(barrier_obj):
        status_reason = "nonpositive_slack"
        break
      grad = np.asarray(hessian_base @ variables + gradient, dtype=np.float64)
      grad[state_size:] += barrier_grad
      primal_residual = constraint_matrix @ variables - constraint_rhs
      kkt_rhs_inf = float(max(np.max(np.abs(grad)) if grad.size else 0.0, np.max(np.abs(primal_residual)) if primal_residual.size else 0.0))
      solve_method = "tree_riccati"
      if float(np.max(np.abs(primal_residual))) <= 1e-7:
        state_current = variables[:state_size].reshape((node_count, state_dim))
        control_current = control_value.reshape((edge_count, control_dim))
        leaf_r_local = np.einsum("lij,lj->li", terminal_p_np, state_current[leaf_nodes]) + terminal_r_np
        edge_safe = np.maximum(edge_weights, 1e-12)
        local_edge_hdiag = (
          float(self.parent.cfg.dt) * running_weights[None, :]
          + barrier_hdiag.reshape((edge_count, control_dim)) / edge_safe[:, None]
        )
        local_edge_grad = (
          float(self.parent.cfg.dt) * running_weights[None, :] * control_current
          + barrier_grad.reshape((edge_count, control_dim)) / edge_safe[:, None]
        )
        try:
          node_delta_jnp, control_delta_jnp = newton_kernel(
            lambda_edge.astype(F32),
            terminal_p.astype(F32),
            jnp.asarray(leaf_r_local, dtype=F32),
            jnp.asarray(local_edge_hdiag, dtype=F32),
            jnp.asarray(local_edge_grad, dtype=F32),
            self.a_state.astype(F32),
            self.control_matrix.astype(F32),
            self.topology.edge_parents,
            self.topology.edge_children,
            self.topology.leaf_nodes,
          )
          step = np.concatenate([
            np.asarray(node_delta_jnp, dtype=np.float64).reshape(-1),
            np.asarray(control_delta_jnp, dtype=np.float64).reshape(-1),
          ])
        except Exception:
          step = np.full((variable_size,), np.nan, dtype=np.float64)
      else:
        step = np.full((variable_size,), np.nan, dtype=np.float64)
      if not np.all(np.isfinite(step)):
        solve_method = "spsolve"
        hessian = hessian_base + sp.diags(
          np.concatenate([np.zeros((state_size,), dtype=np.float64), barrier_hdiag]),
          format="csc",
        )
        kkt_matrix = sp.bmat([[hessian, constraint_matrix.T], [constraint_matrix, zero_dual]], format="csc")
        kkt_rhs = -np.concatenate([grad, primal_residual])
        try:
          with warnings.catch_warnings():
            warnings.simplefilter("error", spla.MatrixRankWarning)
            solution = spla.spsolve(kkt_matrix, kkt_rhs)
        except (spla.MatrixRankWarning, RuntimeError, ValueError):
          lsqr_result = spla.lsqr(kkt_matrix, kkt_rhs, atol=1e-10, btol=1e-10, iter_lim=4000)
          solution = lsqr_result[0]
          solve_method = "lsqr"
        if not np.all(np.isfinite(solution)):
          status_reason = "nonfinite_newton_step"
          break
        step = solution[:variable_size]
      projected_grad = np.asarray(hessian_base @ variables + gradient, dtype=np.float64)[state_size:]
      lower_active = control_value <= lower + 1e-5
      upper_active = control_value >= upper - 1e-5
      free = ~(lower_active | upper_active)
      kkt_violation = np.zeros_like(projected_grad)
      kkt_violation[free] = np.abs(projected_grad[free])
      kkt_violation[lower_active] = np.maximum(0.0, -projected_grad[lower_active])
      kkt_violation[upper_active] = np.maximum(0.0, projected_grad[upper_active])
      projected_kkt_inf = float(np.max(kkt_violation)) if kkt_violation.size else 0.0
      base_objective = objective_no_barrier(variables)
      if projected_kkt_inf < best_kkt_inf:
        best_kkt_inf = projected_kkt_inf
        best_variables = variables.copy()
        best_objective = base_objective
      iteration_log.append({
        "iteration": int(iteration + 1),
        "mu": float(mu),
        "objective": base_objective,
        "barrier_objective": float(base_objective + barrier_obj),
        "projected_kkt_inf": projected_kkt_inf,
        "newton_residual_inf": kkt_rhs_inf,
        "complementarity_proxy": complementarity,
        "solve_method": solve_method,
        "active_lower_count": int(np.sum(lower_active)),
        "active_upper_count": int(np.sum(upper_active)),
        "action": "ipm_newton_step",
      })
      if projected_kkt_inf <= max(10.0 * tol, 1e-5) and mu <= 1e-6:
        status = int(SolverStatus.SUCCESS)
        status_reason = "success"
        break
      alpha = 1.0
      step_control = step[state_size:]
      decreasing = step_control < 0.0
      increasing = step_control > 0.0
      if np.any(decreasing):
        alpha = min(alpha, 0.995 * float(np.min((control_value[decreasing] - lower[decreasing]) / (-step_control[decreasing]))))
      if np.any(increasing):
        alpha = min(alpha, 0.995 * float(np.min((upper[increasing] - control_value[increasing]) / step_control[increasing])))
      current_merit = base_objective + barrier_obj + 100.0 * float(np.linalg.norm(primal_residual, ord=1))
      accepted = False
      for _ in range(30):
        trial = variables + alpha * step
        trial_control = trial[state_size:]
        _, _, trial_barrier, _ = barrier_terms(trial_control, mu)
        if np.isfinite(trial_barrier):
          trial_primal = constraint_matrix @ trial - constraint_rhs
          trial_merit = objective_no_barrier(trial) + trial_barrier + 100.0 * float(np.linalg.norm(trial_primal, ord=1))
          if trial_merit <= current_merit or alpha < 1e-8:
            accepted = True
            break
        alpha *= 0.5
      if not accepted:
        status_reason = "line_search_failed"
        break
      variables = trial
      if (iteration + 1) % 8 == 0:
        mu = max(mu * 0.2, 1e-8)

    if status != int(SolverStatus.SUCCESS):
      variables = best_variables
      status_reason = f"{status_reason}_best_ipm_iterate"
    controls = np.minimum(np.maximum(variables[state_size:], lower), upper).reshape((edge_count, control_dim))
    node_states = rollout(controls.reshape(-1))
    bound_violation, terminal_violation, dynamics_violation = self._control_primal_residuals_np(
      node_states,
      controls,
      lower=np.asarray(self.control_lower_bounds, dtype=np.float64),
      upper=np.asarray(self.control_upper_bounds, dtype=np.float64),
    )
    active_tolerance = max(10.0 * float(tolerance), 1e-6)
    active_lower_count = int(np.sum(controls.reshape(-1) <= lower + active_tolerance))
    active_upper_count = int(np.sum(controls.reshape(-1) >= upper - active_tolerance))
    point = EqualityGamePoint(
      primal=EqualityGamePrimal(
        node_states=jnp.asarray(node_states, dtype=F32),
        offense_controls=jnp.asarray(controls, dtype=F32),
        defense_controls=self._zero_aux_controls(F32),
      ),
      dual=EqualityGameDual(
        node_multipliers=jnp.zeros((node_count, state_dim), dtype=F32),
        terminal_multipliers=jnp.zeros((self.topology.leaf_count, terminal_dim), dtype=F32),
      ),
    )
    return Drone3DExactLQPlayerSolve(
      point=point,
      objective=jnp.asarray(best_objective, dtype=F32),
      mode="player_separable_lq_box_ipm_newton",
      status=status,
      iterations=len(iteration_log),
      active_lower_count=active_lower_count,
      active_upper_count=active_upper_count,
      status_reason=status_reason,
      primal_bound_violation=bound_violation,
      primal_terminal_violation=terminal_violation,
      primal_dynamics_violation=dynamics_violation,
      active_stationarity_violation=float(best_kkt_inf),
      active_set_iteration_log=tuple(iteration_log),
    )

  def solve_exact_control_box_terminal_lq_dense(
    self,
    theta: Any,
    *,
    max_active_set_iterations: int = 48,
    control_tolerance: float = 1e-6,
    kkt_rcond: float = 1e-10,
  ) -> Drone3DExactLQPlayerSolve:
    if not self.supports_exact_control_box_terminal_lq_solver():
      raise ValueError(
        "Dense exact Drone3D control-box terminal LQ solve requires squash_controls=False, "
        "active control boxes, terminal velocity equalities, and no active state boxes.",
      )

    _, edge_public_probs, leaf_public_probs, terminal_p, terminal_r, terminal_c = self._exact_lq_tree_inputs(theta)
    edge_count = self.topology.edge_count
    node_count = self.topology.node_count
    control_dim = self.control_dim
    state_dim = self.state_dim
    terminal_dim = self.terminal_selector.shape[0]
    control_size = edge_count * control_dim

    a_state = np.asarray(self.a_state, dtype=np.float64)
    control_matrix = np.asarray(self.control_matrix, dtype=np.float64)
    x0 = np.asarray(self.x0, dtype=np.float64)
    edge_parents = np.asarray(self.topology.edge_parents)
    edge_children = np.asarray(self.topology.edge_children)
    leaf_nodes = np.asarray(self.topology.leaf_nodes)

    state_map = np.zeros((node_count, state_dim, control_size), dtype=np.float64)
    state_affine = np.zeros((node_count, state_dim), dtype=np.float64)
    state_affine[0] = x0
    for edge_idx in range(edge_count):
      parent_idx = int(edge_parents[edge_idx])
      child_idx = int(edge_children[edge_idx])
      control_slice = slice(edge_idx * control_dim, (edge_idx + 1) * control_dim)
      state_map[child_idx] = a_state @ state_map[parent_idx]
      state_map[child_idx, :, control_slice] += control_matrix
      state_affine[child_idx] = a_state @ state_affine[parent_idx]

    hessian = np.zeros((control_size, control_size), dtype=np.float64)
    gradient = np.zeros((control_size,), dtype=np.float64)
    constant = 0.0
    edge_weights = np.asarray(edge_public_probs, dtype=np.float64)
    running_weights = np.asarray(self.running_weights, dtype=np.float64)
    for edge_idx in range(edge_count):
      control_slice = slice(edge_idx * control_dim, (edge_idx + 1) * control_dim)
      hessian[control_slice, control_slice] += np.diag(
        float(self.parent.cfg.dt) * edge_weights[edge_idx] * running_weights,
      )

    leaf_public = np.asarray(leaf_public_probs, dtype=np.float64)
    terminal_p_np = np.asarray(terminal_p, dtype=np.float64)
    terminal_r_np = np.asarray(terminal_r, dtype=np.float64)
    terminal_c_np = np.asarray(terminal_c, dtype=np.float64)
    for leaf_idx, node_idx in enumerate(leaf_nodes):
      node_map = state_map[int(node_idx)]
      node_affine = state_affine[int(node_idx)]
      p_leaf = leaf_public[leaf_idx] * terminal_p_np[leaf_idx]
      r_leaf = leaf_public[leaf_idx] * terminal_r_np[leaf_idx]
      c_leaf = leaf_public[leaf_idx] * terminal_c_np[leaf_idx]
      hessian += node_map.T @ p_leaf @ node_map
      gradient += node_map.T @ (p_leaf @ node_affine + r_leaf)
      constant += 0.5 * float(node_affine @ p_leaf @ node_affine) + float(r_leaf @ node_affine) + float(c_leaf)

    terminal_selector = np.asarray(self.terminal_selector, dtype=np.float64)
    if terminal_dim > 0 and leaf_nodes.size > 0:
      equality_matrix = np.concatenate(
        [terminal_selector @ state_map[int(node_idx)] for node_idx in leaf_nodes],
        axis=0,
      )
      equality_rhs = np.concatenate(
        [-terminal_selector @ state_affine[int(node_idx)] for node_idx in leaf_nodes],
        axis=0,
      )
    else:
      equality_matrix = np.zeros((0, control_size), dtype=np.float64)
      equality_rhs = np.zeros((0,), dtype=np.float64)

    lower = np.broadcast_to(
      np.asarray(self.control_lower_bounds, dtype=np.float64),
      (edge_count, control_dim),
    ).reshape(control_size)
    upper = np.broadcast_to(
      np.asarray(self.control_upper_bounds, dtype=np.float64),
      (edge_count, control_dim),
    ).reshape(control_size)
    lower_finite = np.isfinite(lower)
    upper_finite = np.isfinite(upper)
    lower_active = np.zeros((control_size,), dtype=bool)
    upper_active = np.zeros((control_size,), dtype=bool)
    controls_flat = np.zeros((control_size,), dtype=np.float64)
    equality_multipliers = np.zeros((equality_matrix.shape[0],), dtype=np.float64)
    status = int(SolverStatus.MAX_ITERATIONS)
    iterations_used = 0

    def _solve_with_active_set() -> tuple[np.ndarray, np.ndarray]:
      active_indices = np.nonzero(lower_active | upper_active)[0]
      active_rhs = np.where(lower_active[active_indices], lower[active_indices], upper[active_indices])
      if active_indices.size > 0:
        active_matrix = np.zeros((active_indices.size, control_size), dtype=np.float64)
        active_matrix[np.arange(active_indices.size), active_indices] = 1.0
        constraint_matrix = np.concatenate([equality_matrix, active_matrix], axis=0)
        constraint_rhs = np.concatenate([equality_rhs, active_rhs], axis=0)
      else:
        constraint_matrix = equality_matrix
        constraint_rhs = equality_rhs

      if constraint_matrix.shape[0] > 0:
        upper_block = np.concatenate([hessian, constraint_matrix.T], axis=1)
        lower_block = np.concatenate(
          [
            constraint_matrix,
            np.zeros((constraint_matrix.shape[0], constraint_matrix.shape[0]), dtype=np.float64),
          ],
          axis=1,
        )
        kkt_matrix = np.concatenate([upper_block, lower_block], axis=0)
        kkt_rhs = np.concatenate([-gradient, constraint_rhs], axis=0)
      else:
        kkt_matrix = hessian
        kkt_rhs = -gradient
      solution = np.linalg.lstsq(kkt_matrix, kkt_rhs, rcond=kkt_rcond)[0]
      return solution[:control_size], solution[control_size : control_size + equality_matrix.shape[0]]

    for iteration in range(max_active_set_iterations):
      iterations_used = iteration + 1
      controls_flat, equality_multipliers = _solve_with_active_set()
      lower_violation = lower_finite & ~lower_active & ~upper_active & (controls_flat < lower - control_tolerance)
      upper_violation = upper_finite & ~lower_active & ~upper_active & (controls_flat > upper + control_tolerance)
      if np.any(lower_violation) or np.any(upper_violation):
        lower_active |= lower_violation
        upper_active |= upper_violation
        continue

      equality_gradient = (
        equality_matrix.T @ equality_multipliers
        if equality_matrix.shape[0] > 0
        else np.zeros_like(controls_flat)
      )
      reduced_gradient = hessian @ controls_flat + gradient + equality_gradient
      release_lower = lower_active & (reduced_gradient < -control_tolerance)
      release_upper = upper_active & (reduced_gradient > control_tolerance)
      if np.any(release_lower) or np.any(release_upper):
        lower_active &= ~release_lower
        upper_active &= ~release_upper
        continue

      status = int(SolverStatus.SUCCESS)
      break

    controls_flat = np.minimum(np.maximum(controls_flat, lower), upper)
    controls = controls_flat.reshape((edge_count, control_dim))
    node_states = np.einsum("nif,f->ni", state_map, controls_flat) + state_affine
    objective = 0.5 * float(controls_flat @ hessian @ controls_flat) + float(gradient @ controls_flat) + constant

    point = EqualityGamePoint(
      primal=EqualityGamePrimal(
        node_states=jnp.asarray(node_states, dtype=F32),
        offense_controls=jnp.asarray(controls, dtype=F32),
        defense_controls=self._zero_aux_controls(F32),
      ),
      dual=EqualityGameDual(
        node_multipliers=jnp.zeros((node_count, state_dim), dtype=F32),
        terminal_multipliers=jnp.zeros((self.topology.leaf_count, terminal_dim), dtype=F32),
      ),
    )
    return Drone3DExactLQPlayerSolve(
      point=point,
      objective=jnp.asarray(objective, dtype=F32),
      mode="player_separable_lq_box_terminal_dense_active_set",
      status=status,
      iterations=iterations_used,
      active_lower_count=int(np.sum(lower_active)),
      active_upper_count=int(np.sum(upper_active)),
    )

  def solve_exact_control_box_terminal_lq_sparse(
    self,
    theta: Any,
    *,
    max_active_set_iterations: int = 48,
    control_tolerance: float = 1e-6,
    prefer_riccati_active_set: bool = True,
    initial_point: EqualityGamePoint | None = None,
  ) -> Drone3DExactLQPlayerSolve:
    if not self.supports_exact_control_box_terminal_lq_solver():
      raise ValueError(
        "Sparse exact Drone3D control-box terminal LQ solve requires squash_controls=False, "
        "active control boxes, terminal velocity equalities, and no active state boxes.",
      )
    if prefer_riccati_active_set:
      try:
        riccati_result = self._solve_exact_control_box_terminal_lq_riccati_active_set(
          theta,
          max_active_set_iterations=max_active_set_iterations,
          control_tolerance=control_tolerance,
        )
        if (
          int(riccati_result.status) == int(SolverStatus.SUCCESS)
          and riccati_result.active_lower_count == 0
          and riccati_result.active_upper_count == 0
        ):
          return riccati_result
      except (np.linalg.LinAlgError, RuntimeError, ValueError):
        pass
    return self._solve_exact_control_box_terminal_lq_sparse_global(
      theta,
      max_active_set_iterations=max_active_set_iterations,
      control_tolerance=control_tolerance,
      initial_point=initial_point,
    )

  def solve_exact_control_box_terminal_lq_active_tree(
    self,
    theta: Any,
    *,
    max_active_set_iterations: int = 48,
    control_tolerance: float = 1e-6,
    initial_point: EqualityGamePoint | None = None,
  ) -> Drone3DExactLQPlayerSolve:
    if not self.supports_exact_control_box_terminal_lq_solver():
      raise ValueError(
        "Active-tree Drone3D control-box terminal LQ solve requires squash_controls=False, "
        "active control boxes, terminal velocity equalities, and no active state boxes.",
      )
    if initial_point is not None:
      try:
        return self._solve_exact_control_box_terminal_lq_riccati_active_set(
          theta,
          max_active_set_iterations=1,
          control_tolerance=control_tolerance,
          initial_point=initial_point,
          mode="player_separable_lq_box_terminal_active_tree",
          freeze_initial_active_set=True,
        )
      except (np.linalg.LinAlgError, RuntimeError, ValueError):
        pass
    try:
      return self._solve_exact_control_box_terminal_lq_riccati_active_set(
        theta,
        max_active_set_iterations=max_active_set_iterations,
        control_tolerance=control_tolerance,
        mode="player_separable_lq_box_terminal_active_tree",
      )
    except (np.linalg.LinAlgError, RuntimeError, ValueError):
      return self._solve_exact_control_box_terminal_lq_sparse_global(
        theta,
        max_active_set_iterations=max_active_set_iterations,
        control_tolerance=control_tolerance,
        initial_point=initial_point,
      )

  @staticmethod
  def _active_set_pivot_mask(candidates: np.ndarray, scores: np.ndarray, max_pivots: int) -> np.ndarray:
    candidate_shape = candidates.shape
    candidates_flat = np.asarray(candidates, dtype=bool).reshape(-1)
    scores_flat = np.asarray(scores, dtype=np.float64).reshape(-1)
    selected_flat = np.zeros_like(candidates_flat, dtype=bool)
    candidate_indices = np.nonzero(candidates_flat)[0]
    if candidate_indices.size == 0:
      return selected_flat.reshape(candidate_shape)
    pivot_count = min(int(max_pivots), int(candidate_indices.size))
    if pivot_count <= 0:
      return selected_flat.reshape(candidate_shape)
    candidate_scores = scores_flat[candidate_indices]
    if pivot_count < candidate_indices.size:
      local = np.argpartition(-candidate_scores, pivot_count - 1)[:pivot_count]
      candidate_indices = candidate_indices[local]
    selected_flat[candidate_indices] = True
    return selected_flat.reshape(candidate_shape)

  def _active_set_max_pivots(self) -> int:
    control_size = max(1, int(self.topology.edge_count) * int(self.control_dim))
    return max(16, min(2048, int(np.ceil(2.0 * np.sqrt(control_size)))))

  def _control_primal_residuals_np(
    self,
    node_states: np.ndarray,
    controls: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
  ) -> tuple[float, float, float]:
    lower_2d = np.broadcast_to(lower, controls.shape)
    upper_2d = np.broadcast_to(upper, controls.shape)
    lower_finite = np.isfinite(lower_2d)
    upper_finite = np.isfinite(upper_2d)
    lower_residual = np.maximum(0.0, lower_2d - controls)
    upper_residual = np.maximum(0.0, controls - upper_2d)
    bound_violation = max(
      float(np.max(lower_residual[lower_finite])) if np.any(lower_finite) else 0.0,
      float(np.max(upper_residual[upper_finite])) if np.any(upper_finite) else 0.0,
    )
    a_state = np.asarray(self.a_state, dtype=np.float64)
    control_matrix = np.asarray(self.control_matrix, dtype=np.float64)
    x0 = np.asarray(self.x0, dtype=np.float64)
    edge_parents = np.asarray(self.topology.edge_parents)
    edge_children = np.asarray(self.topology.edge_children)
    dynamics_violation = float(np.max(np.abs(node_states[0] - x0))) if node_states.size else 0.0
    if self.topology.edge_count > 0:
      predicted_children = node_states[edge_parents] @ a_state.T + controls @ control_matrix.T
      dynamics_violation = max(
        dynamics_violation,
        float(np.max(np.abs(node_states[edge_children] - predicted_children))),
      )
    terminal_selector = np.asarray(self.terminal_selector, dtype=np.float64)
    leaf_nodes = np.asarray(self.topology.leaf_nodes)
    if terminal_selector.shape[0] > 0 and leaf_nodes.size > 0:
      terminal_violation = float(np.max(np.abs(node_states[leaf_nodes] @ terminal_selector.T)))
    else:
      terminal_violation = 0.0
    return bound_violation, terminal_violation, dynamics_violation

  def _rerollout_states_np(self, controls: np.ndarray) -> np.ndarray:
    node_states = np.zeros((self.topology.node_count, self.state_dim), dtype=np.float64)
    node_states[0] = np.asarray(self.x0, dtype=np.float64)
    a_state = np.asarray(self.a_state, dtype=np.float64)
    control_matrix = np.asarray(self.control_matrix, dtype=np.float64)
    for edge_idx in range(self.topology.edge_count):
      parent_idx = int(self.topology.edge_parents[edge_idx])
      child_idx = int(self.topology.edge_children[edge_idx])
      node_states[child_idx] = node_states[parent_idx] @ a_state.T + controls[edge_idx] @ control_matrix.T
    return node_states

  def _box_projected_rollout_warm_point(
    self,
    theta: Any,
    *,
    control_tolerance: float,
  ) -> Drone3DExactLQPlayerSolve:
    base = self.solve_exact_lq(theta)
    controls = np.asarray(base.point.primal.offense_controls, dtype=np.float64)
    lower = np.asarray(self.control_lower_bounds, dtype=np.float64)
    upper = np.asarray(self.control_upper_bounds, dtype=np.float64)
    clipped_controls = np.minimum(np.maximum(controls, lower[None, :]), upper[None, :])
    node_states = self._rerollout_states_np(clipped_controls)
    bound_violation, terminal_violation, dynamics_violation = self._control_primal_residuals_np(
      node_states,
      clipped_controls,
      lower=lower,
      upper=upper,
    )
    status = (
      int(SolverStatus.SUCCESS)
      if terminal_violation <= max(10.0 * control_tolerance, 1e-6)
      else int(SolverStatus.MAX_ITERATIONS)
    )
    point = EqualityGamePoint(
      primal=EqualityGamePrimal(
        node_states=jnp.asarray(node_states, dtype=F32),
        offense_controls=jnp.asarray(clipped_controls, dtype=F32),
        defense_controls=self._zero_aux_controls(F32),
      ),
      dual=EqualityGameDual(
        node_multipliers=jnp.zeros((self.topology.node_count, self.state_dim), dtype=F32),
        terminal_multipliers=jnp.zeros((self.topology.leaf_count, self.terminal_selector.shape[0]), dtype=F32),
      ),
    )
    return Drone3DExactLQPlayerSolve(
      point=point,
      objective=base.objective,
      mode="player_separable_lq_box_projected_rollout_warm_point",
      status=status,
      iterations=1,
      active_lower_count=int(np.sum(clipped_controls <= lower[None, :] + control_tolerance)),
      active_upper_count=int(np.sum(clipped_controls >= upper[None, :] - control_tolerance)),
      status_reason="box_projected_rollout_terminal_feasible" if status == int(SolverStatus.SUCCESS) else "box_projected_rollout_terminal_violation",
      primal_bound_violation=bound_violation,
      primal_terminal_violation=terminal_violation,
      primal_dynamics_violation=dynamics_violation,
    )

  def _solve_exact_control_box_terminal_lq_riccati_active_set(
    self,
    theta: Any,
    *,
    max_active_set_iterations: int,
    control_tolerance: float,
    initial_point: EqualityGamePoint | None = None,
    mode: str = "player_separable_lq_box_terminal_riccati_active_set",
    freeze_initial_active_set: bool = False,
  ) -> Drone3DExactLQPlayerSolve:
    lambda_edge, edge_public_probs, leaf_public_probs, terminal_p, terminal_r, terminal_c = self._exact_lq_tree_inputs(theta)
    if self.terminal_selector.shape != (self.control_dim, self.state_dim):
      raise ValueError("Riccati terminal active-set pre-elimination expects one terminal velocity row per control.")
    terminal_velocity_block = np.asarray(self.terminal_selector[:, self.control_dim : 2 * self.control_dim])
    if not np.allclose(terminal_velocity_block, np.eye(self.control_dim), atol=1e-6):
      raise ValueError("Riccati terminal active-set pre-elimination expects terminal velocity selector rows.")

    kernel = _drone_single_player_lq_affine_control_tree_solve_kernel(
      self.topology.node_offsets_py,
      self.topology.edge_offsets_py,
      self.topology.total_depth,
      self.topology.node_count,
      self.topology.edge_count,
      self.topology.leaf_count,
      self.state_dim,
      self.control_dim,
      self.terminal_selector.shape[0],
    )
    control_weight_diag = self.parent.cfg.dt * self.running_weights
    leaf_parent_edges = (
      self.topology.leaf_edge_paths[:, -1]
      if self.topology.total_depth > 0
      else jnp.zeros((self.topology.leaf_count,), dtype=self.topology.leaf_nodes.dtype)
    )

    free_mask = np.ones((self.topology.edge_count, self.control_dim), dtype=bool)
    lower_active = np.zeros((self.topology.edge_count, self.control_dim), dtype=bool)
    upper_active = np.zeros((self.topology.edge_count, self.control_dim), dtype=bool)
    lower = np.asarray(self.control_lower_bounds, dtype=np.float32)
    upper = np.asarray(self.control_upper_bounds, dtype=np.float32)
    lower_finite = np.isfinite(lower)[None, :]
    upper_finite = np.isfinite(upper)[None, :]
    if initial_point is not None:
      initial_controls = np.asarray(initial_point.primal.offense_controls, dtype=np.float32)
      if initial_controls.shape == (self.topology.edge_count, self.control_dim):
        active_tolerance = max(10.0 * control_tolerance, 1e-6)
        lower_active = lower_finite & (initial_controls <= lower[None, :] + active_tolerance)
        upper_active = upper_finite & (initial_controls >= upper[None, :] - active_tolerance)
        both_active = lower_active & upper_active
        if np.any(both_active):
          midpoint = 0.5 * (lower[None, :] + upper[None, :])
          choose_upper = initial_controls >= midpoint
          lower_active = lower_active & ~(both_active & choose_upper)
          upper_active = upper_active & ~(both_active & ~choose_upper)
        free_mask = ~(lower_active | upper_active)
    warm_start_active_lower_count = int(np.sum(lower_active))
    warm_start_active_upper_count = int(np.sum(upper_active))
    last_point: EqualityGamePoint | None = None
    last_value = jnp.asarray(0.0, dtype=F32)
    status = int(SolverStatus.MAX_ITERATIONS)
    status_reason = "max_active_set_iterations"
    iterations_used = 0
    repeated_active_set_final = False
    seen_active_sets: set[bytes] = set()
    active_set_iteration_log: list[dict[str, Any]] = []
    best_point: EqualityGamePoint | None = None
    best_value = jnp.asarray(0.0, dtype=F32)
    best_lower_active: np.ndarray | None = None
    best_upper_active: np.ndarray | None = None
    best_stationarity_inf = np.inf
    max_pivots = self._active_set_max_pivots()

    def _terminal_active_set_feasible(candidate_lower: np.ndarray, candidate_upper: np.ndarray) -> bool:
      try:
        self._terminal_velocity_preeliminated_control_affines(
          ~(candidate_lower | candidate_upper),
          candidate_lower,
          candidate_upper,
          lower,
          upper,
          tolerance=control_tolerance,
        )
      except RuntimeError:
        return False
      return True

    def _terminal_feasible_bound_additions(
      candidate_lower: np.ndarray,
      candidate_upper: np.ndarray,
      lower_scores: np.ndarray,
      upper_scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
      candidates: list[tuple[float, str, int]] = []
      for flat_idx in np.nonzero(candidate_lower.reshape(-1))[0]:
        candidates.append((float(lower_scores.reshape(-1)[flat_idx]), "lower", int(flat_idx)))
      for flat_idx in np.nonzero(candidate_upper.reshape(-1))[0]:
        candidates.append((float(upper_scores.reshape(-1)[flat_idx]), "upper", int(flat_idx)))
      candidates.sort(key=lambda item: item[0], reverse=True)
      candidates = candidates[:max_pivots]
      trial_count = len(candidates)
      while trial_count > 0:
        trial_lower = lower_active.copy()
        trial_upper = upper_active.copy()
        for _, side, flat_idx in candidates[:trial_count]:
          if side == "lower":
            trial_lower.reshape(-1)[flat_idx] = True
            trial_upper.reshape(-1)[flat_idx] = False
          else:
            trial_upper.reshape(-1)[flat_idx] = True
            trial_lower.reshape(-1)[flat_idx] = False
        if _terminal_active_set_feasible(trial_lower, trial_upper):
          return trial_lower, trial_upper, trial_count
        if trial_count == 1:
          break
        trial_count = max(1, trial_count // 2)
      for _, side, flat_idx in candidates[: min(len(candidates), 8)]:
        trial_lower = lower_active.copy()
        trial_upper = upper_active.copy()
        if side == "lower":
          trial_lower.reshape(-1)[flat_idx] = True
          trial_upper.reshape(-1)[flat_idx] = False
        else:
          trial_upper.reshape(-1)[flat_idx] = True
          trial_lower.reshape(-1)[flat_idx] = False
        if _terminal_active_set_feasible(trial_lower, trial_upper):
          return trial_lower, trial_upper, 1
      return lower_active.copy(), upper_active.copy(), 0

    for iteration in range(max_active_set_iterations):
      iterations_used = iteration + 1
      active_signature = np.packbits(
        np.concatenate([lower_active.reshape(-1), upper_active.reshape(-1)]).astype(np.uint8),
      ).tobytes()
      repeated_active_set = active_signature in seen_active_sets
      seen_active_sets.add(active_signature)
      try:
        (
          effective_free_mask,
          fixed_feedback,
          fixed_bias,
        ) = self._terminal_velocity_preeliminated_control_affines(
          free_mask,
          lower_active,
          upper_active,
          lower,
          upper,
          tolerance=control_tolerance,
        )
      except RuntimeError as exc:
        active_mask = lower_active | upper_active
        if not np.any(active_mask):
          raise
        depth_scores = np.broadcast_to(
          np.asarray(self.topology.edge_depths, dtype=np.float64)[:, None] + 1.0,
          active_mask.shape,
        )
        selected_release = self._active_set_pivot_mask(active_mask, depth_scores, max_pivots)
        lower_active &= ~selected_release
        upper_active &= ~selected_release
        free_mask = ~(lower_active | upper_active)
        active_set_iteration_log.append(
          {
            "iteration": int(iteration + 1),
            "active_lower_count": int(np.sum(lower_active)),
            "active_upper_count": int(np.sum(upper_active)),
            "lower_violation_count": None,
            "upper_violation_count": None,
            "stationarity_inf": None,
            "release_lower_count": None,
            "release_upper_count": None,
            "selected_release_count": int(np.sum(selected_release)),
            "repeated_active_set": bool(repeated_active_set),
            "action": "release_terminal_infeasible_active_bounds",
            "terminal_preelimination_error": str(exc),
          },
        )
        status_reason = "terminal_preelimination_infeasible_releasing_active_bounds"
        continue
      effective_free_mask_jnp = jnp.asarray(effective_free_mask)
      fixed_feedback_jnp = jnp.asarray(fixed_feedback, dtype=F32)

      def objective_with_aux(fixed_bias_value: jnp.ndarray):
        node_states_value, controls_value, node_duals_value, terminal_duals_value, value_value = kernel(
          lambda_edge.astype(F32),
          terminal_p.astype(F32),
          terminal_r.astype(F32),
          terminal_c.astype(F32),
          edge_public_probs.astype(F32),
          leaf_public_probs.astype(F32),
          self.a_state.astype(F32),
          self.control_matrix.astype(F32),
          control_weight_diag.astype(F32),
          self.terminal_selector.astype(F32),
          self.x0.astype(F32),
          self.topology.edge_parents,
          self.topology.edge_children,
          self.topology.leaf_nodes,
          leaf_parent_edges,
          effective_free_mask_jnp,
          fixed_feedback_jnp,
          fixed_bias_value,
        )
        return value_value, (node_states_value, controls_value, node_duals_value, terminal_duals_value)

      (value, (node_states, controls, node_duals, terminal_duals)), fixed_bias_grad = jax.value_and_grad(
        objective_with_aux,
        has_aux=True,
      )(jnp.asarray(fixed_bias, dtype=F32))
      controls_np = np.asarray(controls)
      if not (
        np.all(np.isfinite(np.asarray(node_states)))
        and np.all(np.isfinite(controls_np))
        and np.all(np.isfinite(np.asarray(node_duals)))
        and np.all(np.isfinite(np.asarray(terminal_duals)))
        and np.isfinite(float(value))
      ):
        raise RuntimeError("Riccati active-set equality solve produced nonfinite values.")

      last_value = value
      last_point = EqualityGamePoint(
        primal=EqualityGamePrimal(
          node_states=node_states,
          offense_controls=controls,
          defense_controls=self._zero_aux_controls(node_states.dtype),
        ),
        dual=EqualityGameDual(
          node_multipliers=node_duals,
          terminal_multipliers=terminal_duals,
        ),
      )
      fixed_bias_grad_np = np.asarray(fixed_bias_grad)

      lower_violation = (
        free_mask
        & lower_finite
        & (controls_np < lower[None, :] - control_tolerance)
      )
      upper_violation = (
        free_mask
        & upper_finite
        & (controls_np > upper[None, :] + control_tolerance)
      )
      lower_violation_amount = np.maximum(0.0, lower[None, :] - controls_np)
      upper_violation_amount = np.maximum(0.0, controls_np - upper[None, :])
      lower_finite_full = np.broadcast_to(np.isfinite(lower)[None, :], controls_np.shape)
      upper_finite_full = np.broadcast_to(np.isfinite(upper)[None, :], controls_np.shape)
      iteration_record: dict[str, Any] = {
        "iteration": int(iteration + 1),
        "active_lower_count": int(np.sum(lower_active)),
        "active_upper_count": int(np.sum(upper_active)),
        "lower_violation_count": int(np.sum(lower_violation)),
        "upper_violation_count": int(np.sum(upper_violation)),
        "lower_violation_max": float(np.max(lower_violation_amount[lower_finite_full])) if np.any(lower_finite_full) else 0.0,
        "upper_violation_max": float(np.max(upper_violation_amount[upper_finite_full])) if np.any(upper_finite_full) else 0.0,
        "repeated_active_set": bool(repeated_active_set),
      }
      if freeze_initial_active_set:
        if np.any(lower_violation) or np.any(upper_violation):
          raise RuntimeError("Fixed active-tree warm set produced control bound violations.")
        status = int(SolverStatus.SUCCESS)
        status_reason = "fixed_active_set_success"
        iteration_record.update(
          {
            "stationarity_inf": 0.0,
            "release_lower_count": 0,
            "release_upper_count": 0,
            "action": "fixed_active_set_success",
          },
        )
        active_set_iteration_log.append(iteration_record)
        break
      release_lower = lower_active & (fixed_bias_grad_np < -control_tolerance)
      release_upper = upper_active & (fixed_bias_grad_np > control_tolerance)
      stationarity_violation = np.zeros_like(fixed_bias_grad_np)
      stationarity_violation[lower_active] = np.maximum(0.0, -fixed_bias_grad_np[lower_active])
      stationarity_violation[upper_active] = np.maximum(0.0, fixed_bias_grad_np[upper_active])
      stationarity_inf = float(np.max(stationarity_violation)) if stationarity_violation.size else 0.0
      iteration_record.update(
        {
          "stationarity_inf": None if np.any(lower_violation) or np.any(upper_violation) else stationarity_inf,
          "release_lower_count": int(np.sum(release_lower)),
          "release_upper_count": int(np.sum(release_upper)),
        },
      )
      if not np.any(lower_violation) and not np.any(upper_violation) and stationarity_inf < best_stationarity_inf:
        best_stationarity_inf = stationarity_inf
        best_point = last_point
        best_value = last_value
        best_lower_active = lower_active.copy()
        best_upper_active = upper_active.copy()
      if repeated_active_set:
        repeated_active_set_final = True
        if not np.any(lower_violation) and not np.any(upper_violation) and stationarity_inf <= max(10.0 * control_tolerance, 1e-5):
          status = int(SolverStatus.SUCCESS)
          status_reason = "repeated_active_set_with_stationarity_tolerance"
          iteration_record["action"] = "accept_repeated_active_set"
        else:
          status_reason = "repeated_active_set"
          iteration_record["action"] = "stop_repeated_active_set"
        active_set_iteration_log.append(iteration_record)
        break

      selected_lower_violation = self._active_set_pivot_mask(lower_violation, lower_violation_amount, max_pivots)
      selected_upper_violation = self._active_set_pivot_mask(upper_violation, upper_violation_amount, max_pivots)
      if np.any(selected_lower_violation) or np.any(selected_upper_violation):
        next_lower_active, next_upper_active, accepted_add_count = _terminal_feasible_bound_additions(
          selected_lower_violation,
          selected_upper_violation,
          lower_violation_amount,
          upper_violation_amount,
        )
        iteration_record["action"] = "add_bound_violations"
        iteration_record["selected_lower_violation_count"] = int(np.sum(selected_lower_violation))
        iteration_record["selected_upper_violation_count"] = int(np.sum(selected_upper_violation))
        iteration_record["accepted_bound_add_count"] = int(accepted_add_count)
        active_set_iteration_log.append(iteration_record)
        status_reason = "adding_bound_violations"
        if accepted_add_count == 0:
          status_reason = "no_terminal_feasible_bound_addition"
          break
      else:
        release_scores = np.zeros_like(fixed_bias_grad_np)
        release_scores[release_lower] = -fixed_bias_grad_np[release_lower]
        release_scores[release_upper] = fixed_bias_grad_np[release_upper]
        selected_release_lower = self._active_set_pivot_mask(release_lower, release_scores, max_pivots)
        selected_release_upper = self._active_set_pivot_mask(release_upper, release_scores, max_pivots)
        if np.any(selected_release_lower) or np.any(selected_release_upper):
          next_lower_active = lower_active & ~selected_release_lower
          next_upper_active = upper_active & ~selected_release_upper
          iteration_record["action"] = "release_active_bounds"
          iteration_record["selected_release_lower_count"] = int(np.sum(selected_release_lower))
          iteration_record["selected_release_upper_count"] = int(np.sum(selected_release_upper))
          active_set_iteration_log.append(iteration_record)
          status_reason = "releasing_active_bounds"
        else:
          status = int(SolverStatus.SUCCESS)
          status_reason = "success"
          next_lower_active = lower_active
          next_upper_active = upper_active
          iteration_record["action"] = "success"
          active_set_iteration_log.append(iteration_record)

      both_active = next_lower_active & next_upper_active
      if np.any(both_active):
        midpoint = 0.5 * (lower[None, :] + upper[None, :])
        choose_upper = controls_np >= midpoint
        next_lower_active = np.where(both_active & choose_upper, False, next_lower_active)
        next_upper_active = np.where(both_active & ~choose_upper, False, next_upper_active)
      next_free_mask = ~(next_lower_active | next_upper_active)

      if (
        np.array_equal(next_free_mask, free_mask)
        and np.array_equal(next_lower_active, lower_active)
        and np.array_equal(next_upper_active, upper_active)
      ):
        status = int(SolverStatus.SUCCESS)
        free_mask = next_free_mask
        lower_active = next_lower_active
        upper_active = next_upper_active
        break

      free_mask = next_free_mask
      lower_active = next_lower_active
      upper_active = next_upper_active

      if status == int(SolverStatus.SUCCESS):
        break

    if last_point is None:
      raise RuntimeError("Riccati active-set solve did not produce a point.")
    if status != int(SolverStatus.SUCCESS) and best_point is not None:
      last_point = best_point
      last_value = best_value
      lower_active = best_lower_active
      upper_active = best_upper_active
      status_reason = f"{status_reason}_best_feasible_active_set"
    node_states_np = np.asarray(last_point.primal.node_states, dtype=np.float64)
    controls_np = np.asarray(last_point.primal.offense_controls, dtype=np.float64)
    bound_violation, terminal_violation, dynamics_violation = self._control_primal_residuals_np(
      node_states_np,
      controls_np,
      lower=np.asarray(self.control_lower_bounds, dtype=np.float64),
      upper=np.asarray(self.control_upper_bounds, dtype=np.float64),
    )
    active_stationarity_violation = (
      0.0 if status == int(SolverStatus.SUCCESS) and np.isinf(best_stationarity_inf)
      else float(best_stationarity_inf)
    )

    return Drone3DExactLQPlayerSolve(
      point=last_point,
      objective=last_value,
      mode=mode,
      status=status,
      iterations=iterations_used,
      active_lower_count=int(np.sum(lower_active)),
      active_upper_count=int(np.sum(upper_active)),
      warm_start_active_lower_count=warm_start_active_lower_count,
      warm_start_active_upper_count=warm_start_active_upper_count,
      status_reason=status_reason,
      primal_bound_violation=bound_violation,
      primal_terminal_violation=terminal_violation,
      primal_dynamics_violation=dynamics_violation,
      active_stationarity_violation=active_stationarity_violation,
      repeated_active_set=repeated_active_set_final,
      active_set_iteration_log=tuple(active_set_iteration_log),
    )

  def _terminal_velocity_preeliminated_control_affines(
    self,
    free_mask: np.ndarray,
    lower_active: np.ndarray,
    upper_active: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    tolerance: float,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_count = self.topology.edge_count
    control_dim = self.control_dim
    state_dim = self.state_dim
    dt_by_dim = np.asarray(
      [self.control_matrix[control_dim + dim, dim] for dim in range(control_dim)],
      dtype=np.float64,
    )
    if not np.all(np.abs(dt_by_dim) > 0.0):
      raise ValueError("Terminal velocity pre-elimination requires nonzero velocity-control gains.")

    effective_free = np.asarray(free_mask, dtype=bool).copy()
    fixed_feedback = np.zeros((edge_count, control_dim, state_dim), dtype=np.float32)
    fixed_bias = np.zeros((edge_count, control_dim), dtype=np.float32)
    fixed_bias = np.where(lower_active, lower[None, :], fixed_bias)
    fixed_bias = np.where(upper_active, upper[None, :], fixed_bias).astype(np.float32, copy=False)

    node_has_velocity = np.zeros((self.topology.node_count, control_dim), dtype=bool)
    node_velocity = np.zeros((self.topology.node_count, control_dim), dtype=np.float64)
    node_has_velocity[np.asarray(self.topology.leaf_nodes), :] = True

    edge_offsets = self.topology.edge_offsets_py
    for depth in range(self.topology.total_depth - 1, -1, -1):
      edge_start = edge_offsets[depth]
      edge_end = edge_offsets[depth + 1] if depth + 1 < len(edge_offsets) else self.topology.edge_count
      for edge_idx in range(edge_start, edge_end):
        parent = int(self.topology.edge_parents[edge_idx])
        child = int(self.topology.edge_children[edge_idx])
        for dim in range(control_dim):
          if not node_has_velocity[child, dim]:
            continue
          desired_child_velocity = node_velocity[child, dim]
          dt_dim = dt_by_dim[dim]
          if free_mask[edge_idx, dim]:
            effective_free[edge_idx, dim] = False
            fixed_feedback[edge_idx, dim, control_dim + dim] = np.float32(-1.0 / dt_dim)
            fixed_bias[edge_idx, dim] = np.float32(desired_child_velocity / dt_dim)
          else:
            desired_parent_velocity = desired_child_velocity - dt_dim * float(fixed_bias[edge_idx, dim])
            if node_has_velocity[parent, dim]:
              if abs(float(node_velocity[parent, dim]) - desired_parent_velocity) > max(10.0 * tolerance, 1e-7):
                raise RuntimeError("Inconsistent propagated terminal velocity constraints in active set.")
            node_has_velocity[parent, dim] = True
            node_velocity[parent, dim] = desired_parent_velocity

    root_velocity = np.asarray(self.x0[self.control_dim : 2 * self.control_dim], dtype=np.float64)
    root_constrained = node_has_velocity[0]
    if np.any(np.abs(root_velocity[root_constrained] - node_velocity[0, root_constrained]) > max(10.0 * tolerance, 1e-7)):
      raise RuntimeError("Active set makes terminal velocity constraints infeasible from the root state.")
    return effective_free, fixed_feedback, fixed_bias

  def _solve_exact_control_box_terminal_lq_sparse_global(
    self,
    theta: Any,
    *,
    max_active_set_iterations: int,
    control_tolerance: float,
    initial_point: EqualityGamePoint | None = None,
  ) -> Drone3DExactLQPlayerSolve:

    _, edge_public_probs, leaf_public_probs, terminal_p, terminal_r, terminal_c = self._exact_lq_tree_inputs(theta)
    edge_count = self.topology.edge_count
    node_count = self.topology.node_count
    control_dim = self.control_dim
    state_dim = self.state_dim
    terminal_dim = self.terminal_selector.shape[0]
    state_size = node_count * state_dim
    control_size = edge_count * control_dim
    variable_size = state_size + control_size

    a_state = np.asarray(self.a_state, dtype=np.float64)
    control_matrix = np.asarray(self.control_matrix, dtype=np.float64)
    x0 = np.asarray(self.x0, dtype=np.float64)
    edge_parents = np.asarray(self.topology.edge_parents)
    edge_children = np.asarray(self.topology.edge_children)
    leaf_nodes = np.asarray(self.topology.leaf_nodes)
    terminal_selector = np.asarray(self.terminal_selector, dtype=np.float64)

    h_diag = np.zeros((variable_size,), dtype=np.float64)
    gradient = np.zeros((variable_size,), dtype=np.float64)
    constant = 0.0
    h_rows: list[np.ndarray] = []
    h_cols: list[np.ndarray] = []
    h_data: list[np.ndarray] = []

    edge_weights = np.asarray(edge_public_probs, dtype=np.float64)
    running_weights = np.asarray(self.running_weights, dtype=np.float64)
    for edge_idx in range(edge_count):
      control_offset = state_size + edge_idx * control_dim
      diag = float(self.parent.cfg.dt) * edge_weights[edge_idx] * running_weights
      indices = control_offset + np.arange(control_dim)
      h_rows.append(indices)
      h_cols.append(indices)
      h_data.append(diag)
      h_diag[indices] += diag

    leaf_public = np.asarray(leaf_public_probs, dtype=np.float64)
    terminal_p_np = np.asarray(terminal_p, dtype=np.float64)
    terminal_r_np = np.asarray(terminal_r, dtype=np.float64)
    terminal_c_np = np.asarray(terminal_c, dtype=np.float64)
    for leaf_idx, node_idx_raw in enumerate(leaf_nodes):
      node_idx = int(node_idx_raw)
      state_offset = node_idx * state_dim
      state_indices = state_offset + np.arange(state_dim)
      p_leaf = leaf_public[leaf_idx] * terminal_p_np[leaf_idx]
      r_leaf = leaf_public[leaf_idx] * terminal_r_np[leaf_idx]
      h_rows.append(np.repeat(state_indices, state_dim))
      h_cols.append(np.tile(state_indices, state_dim))
      h_data.append(p_leaf.reshape(-1))
      gradient[state_indices] += r_leaf
      constant += float(leaf_public[leaf_idx] * terminal_c_np[leaf_idx])

    hessian = sp.coo_matrix(
      (
        np.concatenate(h_data) if h_data else np.zeros((0,), dtype=np.float64),
        (
          np.concatenate(h_rows) if h_rows else np.zeros((0,), dtype=np.int64),
          np.concatenate(h_cols) if h_cols else np.zeros((0,), dtype=np.int64),
        ),
      ),
      shape=(variable_size, variable_size),
    ).tocsc()

    lower = np.broadcast_to(
      np.asarray(self.control_lower_bounds, dtype=np.float64),
      (edge_count, control_dim),
    ).reshape(control_size)
    upper = np.broadcast_to(
      np.asarray(self.control_upper_bounds, dtype=np.float64),
      (edge_count, control_dim),
    ).reshape(control_size)
    lower_finite = np.isfinite(lower)
    upper_finite = np.isfinite(upper)
    lower_active = np.zeros((control_size,), dtype=bool)
    upper_active = np.zeros((control_size,), dtype=bool)
    if initial_point is not None:
      initial_controls = np.asarray(initial_point.primal.offense_controls, dtype=np.float64).reshape(-1)
      if initial_controls.shape == (control_size,):
        active_tolerance = max(10.0 * control_tolerance, 1e-6)
        lower_active = lower_finite & (initial_controls <= lower + active_tolerance)
        upper_active = upper_finite & (initial_controls >= upper - active_tolerance)
        both_active = lower_active & upper_active
        if np.any(both_active):
          midpoint = 0.5 * (lower + upper)
          choose_upper = initial_controls >= midpoint
          lower_active = lower_active & ~(both_active & choose_upper)
          upper_active = upper_active & ~(both_active & ~choose_upper)
    warm_start_active_lower_count = int(np.sum(lower_active))
    warm_start_active_upper_count = int(np.sum(upper_active))
    variables = np.zeros((variable_size,), dtype=np.float64)
    status = int(SolverStatus.MAX_ITERATIONS)
    status_reason = "max_active_set_iterations"
    iterations_used = 0
    seen_active_sets: set[bytes] = set()
    repeated_active_set_final = False
    active_set_iteration_log: list[dict[str, Any]] = []
    kkt_solve_log: list[dict[str, Any]] = []
    best_feasible_variables: np.ndarray | None = None
    best_feasible_lower_active: np.ndarray | None = None
    best_feasible_upper_active: np.ndarray | None = None
    best_feasible_stationarity_inf = np.inf
    base_constraint_count = state_dim + edge_count * state_dim + node_count * 0 + leaf_nodes.size * terminal_dim

    def _base_constraints() -> tuple[sp.csc_matrix, np.ndarray]:
      rows: list[int] = []
      cols: list[int] = []
      data: list[float] = []
      rhs: list[float] = []
      row = 0
      for dim in range(state_dim):
        rows.append(row)
        cols.append(dim)
        data.append(1.0)
        rhs.append(float(x0[dim]))
        row += 1
      for edge_idx in range(edge_count):
        parent = int(edge_parents[edge_idx])
        child = int(edge_children[edge_idx])
        parent_offset = parent * state_dim
        child_offset = child * state_dim
        control_offset = state_size + edge_idx * control_dim
        for dim in range(state_dim):
          rows.append(row)
          cols.append(child_offset + dim)
          data.append(1.0)
          for parent_dim in range(state_dim):
            coeff = -float(a_state[dim, parent_dim])
            if coeff != 0.0:
              rows.append(row)
              cols.append(parent_offset + parent_dim)
              data.append(coeff)
          for control_dim_idx in range(control_dim):
            coeff = -float(control_matrix[dim, control_dim_idx])
            if coeff != 0.0:
              rows.append(row)
              cols.append(control_offset + control_dim_idx)
              data.append(coeff)
          rhs.append(0.0)
          row += 1
      for node_idx_raw in leaf_nodes:
        node_idx = int(node_idx_raw)
        state_offset = node_idx * state_dim
        for terminal_row in range(terminal_dim):
          for dim in range(state_dim):
            coeff = float(terminal_selector[terminal_row, dim])
            if coeff != 0.0:
              rows.append(row)
              cols.append(state_offset + dim)
              data.append(coeff)
          rhs.append(0.0)
          row += 1
      matrix = sp.coo_matrix((data, (rows, cols)), shape=(row, variable_size)).tocsc()
      return matrix, np.asarray(rhs, dtype=np.float64)

    base_matrix, base_rhs = _base_constraints()
    del base_constraint_count

    max_pivots = self._active_set_max_pivots()

    def _solve_with_active_set() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
      active_indices = np.nonzero(lower_active | upper_active)[0]
      if active_indices.size > 0:
        active_rows = np.arange(active_indices.size)
        active_cols = state_size + active_indices
        active_matrix = sp.coo_matrix(
          (np.ones((active_indices.size,), dtype=np.float64), (active_rows, active_cols)),
          shape=(active_indices.size, variable_size),
        ).tocsc()
        active_rhs = np.where(lower_active[active_indices], lower[active_indices], upper[active_indices])
        constraint_matrix = sp.vstack([base_matrix, active_matrix], format="csc")
        constraint_rhs = np.concatenate([base_rhs, active_rhs])
      else:
        constraint_matrix = base_matrix
        constraint_rhs = base_rhs
      zero_block = sp.csc_matrix((constraint_matrix.shape[0], constraint_matrix.shape[0]), dtype=np.float64)
      kkt_matrix = sp.bmat(
        [[hessian, constraint_matrix.T], [constraint_matrix, zero_block]],
        format="csc",
      )
      kkt_rhs = np.concatenate([-gradient, constraint_rhs])
      method = "spsolve"
      lsqr_iterations: int | None = None
      lsqr_residual_norm: float | None = None
      try:
        with warnings.catch_warnings():
          warnings.simplefilter("error", spla.MatrixRankWarning)
          solution = spla.spsolve(kkt_matrix, kkt_rhs)
      except (spla.MatrixRankWarning, RuntimeError, ValueError):
        method = "lsqr_rank_or_runtime_fallback"
        lsqr_result = spla.lsqr(kkt_matrix, kkt_rhs, atol=1e-9, btol=1e-9, iter_lim=2000)
        solution = lsqr_result[0]
        lsqr_iterations = int(lsqr_result[2])
        lsqr_residual_norm = float(lsqr_result[3])
      if not np.all(np.isfinite(solution)):
        method = "lsqr_nonfinite_retry"
        lsqr_result = spla.lsqr(kkt_matrix, kkt_rhs, atol=1e-9, btol=1e-9, iter_lim=4000)
        solution = lsqr_result[0]
        lsqr_iterations = int(lsqr_result[2])
        lsqr_residual_norm = float(lsqr_result[3])
      kkt_residual_norm = (
        float(np.linalg.norm(kkt_matrix @ solution - kkt_rhs))
        if np.all(np.isfinite(solution))
        else float("inf")
      )
      solve_record = {
        "method": method,
        "active_constraint_count": int(active_indices.size),
        "constraint_count": int(constraint_matrix.shape[0]),
        "variable_count": int(variable_size),
        "kkt_size": int(kkt_matrix.shape[0]),
        "kkt_residual_norm": kkt_residual_norm,
        "lsqr_iterations": lsqr_iterations,
        "lsqr_residual_norm": lsqr_residual_norm,
      }
      return solution[:variable_size], solution[variable_size : variable_size + base_matrix.shape[0]], solve_record

    for iteration in range(max_active_set_iterations):
      iterations_used = iteration + 1
      active_signature = np.packbits(
        np.concatenate([lower_active, upper_active]).astype(np.uint8),
      ).tobytes()
      repeated_active_set = active_signature in seen_active_sets
      seen_active_sets.add(active_signature)
      variables, base_multipliers, solve_record = _solve_with_active_set()
      solve_record["iteration"] = int(iteration + 1)
      kkt_solve_log.append(solve_record)
      controls_flat = variables[state_size:]
      lower_violation = lower_finite & ~lower_active & ~upper_active & (controls_flat < lower - control_tolerance)
      upper_violation = upper_finite & ~lower_active & ~upper_active & (controls_flat > upper + control_tolerance)
      lower_violation_amount = np.maximum(0.0, lower - controls_flat)
      upper_violation_amount = np.maximum(0.0, controls_flat - upper)
      lower_violation_count = int(np.sum(lower_violation))
      upper_violation_count = int(np.sum(upper_violation))
      iteration_record: dict[str, Any] = {
        "iteration": int(iteration + 1),
        "active_lower_count": int(np.sum(lower_active)),
        "active_upper_count": int(np.sum(upper_active)),
        "lower_violation_count": lower_violation_count,
        "upper_violation_count": upper_violation_count,
        "lower_violation_max": float(np.max(lower_violation_amount[lower_finite])) if np.any(lower_finite) else 0.0,
        "upper_violation_max": float(np.max(upper_violation_amount[upper_finite])) if np.any(upper_finite) else 0.0,
        "repeated_active_set": bool(repeated_active_set),
        "kkt_method": solve_record["method"],
        "kkt_residual_norm": solve_record["kkt_residual_norm"],
      }
      if np.any(lower_violation) or np.any(upper_violation):
        selected_lower_violation = self._active_set_pivot_mask(lower_violation, lower_violation_amount, max_pivots)
        selected_upper_violation = self._active_set_pivot_mask(upper_violation, upper_violation_amount, max_pivots)
        iteration_record.update(
          {
            "stationarity_inf": None,
            "release_lower_count": 0,
            "release_upper_count": 0,
            "action": "add_bound_violations",
            "selected_lower_violation_count": int(np.sum(selected_lower_violation)),
            "selected_upper_violation_count": int(np.sum(selected_upper_violation)),
          },
        )
        active_set_iteration_log.append(iteration_record)
        status_reason = "adding_bound_violations"
        lower_active |= selected_lower_violation
        upper_active |= selected_upper_violation
        continue

      reduced_gradient = hessian @ variables + gradient + base_matrix.T @ base_multipliers
      control_gradient = np.asarray(reduced_gradient[state_size:]).reshape(-1)
      stationarity_violation = np.zeros_like(control_gradient)
      stationarity_violation[lower_active] = np.maximum(0.0, -control_gradient[lower_active])
      stationarity_violation[upper_active] = np.maximum(0.0, control_gradient[upper_active])
      stationarity_inf = float(np.max(stationarity_violation)) if stationarity_violation.size else 0.0
      release_lower = lower_active & (control_gradient < -control_tolerance)
      release_upper = upper_active & (control_gradient > control_tolerance)
      iteration_record.update(
        {
          "stationarity_inf": stationarity_inf,
          "release_lower_count": int(np.sum(release_lower)),
          "release_upper_count": int(np.sum(release_upper)),
        },
      )
      if stationarity_inf < best_feasible_stationarity_inf:
        best_feasible_stationarity_inf = stationarity_inf
        best_feasible_variables = variables.copy()
        best_feasible_lower_active = lower_active.copy()
        best_feasible_upper_active = upper_active.copy()
      if repeated_active_set:
        repeated_active_set_final = True
        if stationarity_inf <= max(10.0 * control_tolerance, 1e-5):
          status = int(SolverStatus.SUCCESS)
          status_reason = "repeated_active_set_with_stationarity_tolerance"
          iteration_record["action"] = "accept_repeated_active_set"
        else:
          status_reason = "repeated_active_set_stationarity_violation"
          iteration_record["action"] = "stop_repeated_active_set"
        active_set_iteration_log.append(iteration_record)
        break
      if np.any(release_lower) or np.any(release_upper):
        release_scores = np.zeros_like(control_gradient)
        release_scores[release_lower] = -control_gradient[release_lower]
        release_scores[release_upper] = control_gradient[release_upper]
        selected_release_lower = self._active_set_pivot_mask(release_lower, release_scores, max_pivots)
        selected_release_upper = self._active_set_pivot_mask(release_upper, release_scores, max_pivots)
        iteration_record["action"] = "release_active_bounds"
        iteration_record["selected_release_lower_count"] = int(np.sum(selected_release_lower))
        iteration_record["selected_release_upper_count"] = int(np.sum(selected_release_upper))
        active_set_iteration_log.append(iteration_record)
        status_reason = "releasing_active_bounds"
        lower_active &= ~selected_release_lower
        upper_active &= ~selected_release_upper
        continue

      status = int(SolverStatus.SUCCESS)
      status_reason = "success"
      iteration_record["action"] = "success"
      active_set_iteration_log.append(iteration_record)
      break

    if status != int(SolverStatus.SUCCESS) and best_feasible_variables is not None:
      variables = best_feasible_variables
      lower_active = best_feasible_lower_active
      upper_active = best_feasible_upper_active
      status_reason = f"{status_reason}_best_feasible_active_set"
    controls_flat = variables[state_size:]
    node_states = variables[:state_size].reshape((node_count, state_dim))
    controls = controls_flat.reshape((edge_count, control_dim))
    objective = 0.5 * float(variables @ (hessian @ variables)) + float(gradient @ variables) + constant
    lower_residual = np.maximum(0.0, lower - controls_flat)
    upper_residual = np.maximum(0.0, controls_flat - upper)
    primal_bound_violation = max(
      float(np.max(lower_residual[lower_finite])) if np.any(lower_finite) else 0.0,
      float(np.max(upper_residual[upper_finite])) if np.any(upper_finite) else 0.0,
    )
    root_dynamics_violation = float(np.max(np.abs(node_states[0] - x0))) if node_states.size else 0.0
    dynamics_violation = root_dynamics_violation
    if edge_count > 0:
      predicted_children = node_states[edge_parents] @ a_state.T + controls @ control_matrix.T
      dynamics_violation = max(
        dynamics_violation,
        float(np.max(np.abs(node_states[edge_children] - predicted_children))),
      )
    if terminal_dim > 0 and leaf_nodes.size > 0:
      terminal_violation = float(np.max(np.abs(node_states[leaf_nodes] @ terminal_selector.T)))
    else:
      terminal_violation = 0.0
    active_stationarity_violation = (
      0.0 if status == int(SolverStatus.SUCCESS) and np.isinf(best_feasible_stationarity_inf)
      else float(best_feasible_stationarity_inf)
    )

    point = EqualityGamePoint(
      primal=EqualityGamePrimal(
        node_states=jnp.asarray(node_states, dtype=F32),
        offense_controls=jnp.asarray(controls, dtype=F32),
        defense_controls=self._zero_aux_controls(F32),
      ),
      dual=EqualityGameDual(
        node_multipliers=jnp.zeros((node_count, state_dim), dtype=F32),
        terminal_multipliers=jnp.zeros((self.topology.leaf_count, terminal_dim), dtype=F32),
      ),
    )
    return Drone3DExactLQPlayerSolve(
      point=point,
      objective=jnp.asarray(objective, dtype=F32),
      mode="player_separable_lq_box_terminal_sparse_active_set",
      status=status,
      iterations=iterations_used,
      active_lower_count=int(np.sum(lower_active)),
      active_upper_count=int(np.sum(upper_active)),
      warm_start_active_lower_count=warm_start_active_lower_count,
      warm_start_active_upper_count=warm_start_active_upper_count,
      status_reason=status_reason,
      primal_bound_violation=primal_bound_violation,
      primal_terminal_violation=terminal_violation,
      primal_dynamics_violation=dynamics_violation,
      active_stationarity_violation=active_stationarity_violation,
      repeated_active_set=repeated_active_set_final,
      active_set_iteration_log=tuple(active_set_iteration_log),
      kkt_solve_log=tuple(kkt_solve_log),
    )

  def _branch_activity_masks(
    self,
    linearization: Drone3DSinglePlayerLinearization,
    prune_threshold: float,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    threshold = max(0.0, float(prune_threshold))
    if threshold <= 0.0:
      active_node_mask = jnp.ones((self.topology.node_count,), dtype=bool)
      active_edge_mask = jnp.ones((self.topology.edge_count,), dtype=bool)
    else:
      active_node_mask = linearization.node_public_probs > threshold
      active_node_mask = active_node_mask.at[0].set(True)
      active_edge_mask = jnp.logical_and(
        linearization.edge_public_probs > threshold,
        active_node_mask[self.topology.edge_children],
      )
    active_leaf_mask = active_node_mask[self.topology.leaf_nodes]
    return active_node_mask, active_edge_mask, active_leaf_mask

  def solve_singularity_aware_linearized_system(
    self,
    *,
    residual: EqualityGamePoint,
    point_current: EqualityGamePoint,
    theta: Any,
    linearization: Drone3DSinglePlayerLinearization,
    regularization: float,
    cfg: TreeDiffMPCConfig,
  ) -> tuple[EqualityGamePoint, float, dict[str, Any]]:
    del point_current
    dtype = linearization.node_constraints.dtype
    reg = jnp.asarray(regularization, dtype=dtype)
    curvature_threshold = max(0.0, float(cfg.partial_elimination_curvature_threshold))
    prune_threshold = max(0.0, float(cfg.near_zero_branch_prune_threshold))
    active_node_mask, active_edge_mask, active_leaf_mask = self._branch_activity_masks(
      linearization,
      prune_threshold,
    )
    active_node_scale = active_node_mask.astype(dtype)[:, None]
    active_edge_scale = active_edge_mask.astype(dtype)[:, None]
    active_leaf_scale = active_leaf_mask.astype(dtype)[:, None]

    effective_control_diag = linearization.control_hdiag + reg
    curved_control_mask = jnp.logical_and(
      active_edge_mask[:, None],
      jnp.abs(effective_control_diag) > curvature_threshold,
    )
    flat_control_mask = jnp.logical_and(
      active_edge_mask[:, None],
      jnp.logical_not(curved_control_mask),
    )
    flat_control_scale = flat_control_mask.astype(dtype)

    zero_node_states = jnp.zeros((self.topology.node_count, self.state_dim), dtype=dtype)
    zero_controls = jnp.zeros((self.topology.edge_count, self.control_dim), dtype=dtype)
    zero_aux_controls = self._zero_aux_controls(dtype)
    zero_node_duals = jnp.zeros((self.topology.node_count, self.state_dim), dtype=dtype)
    zero_terminal_duals = jnp.zeros(
      (self.topology.leaf_count, self.terminal_selector.shape[0]),
      dtype=dtype,
    )
    node_state_size = self.topology.node_count * self.state_dim
    flat_control_size = self.topology.edge_count * self.control_dim
    node_dual_size = self.topology.node_count * self.state_dim
    terminal_dual_size = self.topology.leaf_count * self.terminal_selector.shape[0]

    def pack_reduced(point_value: EqualityGamePoint) -> jnp.ndarray:
      pieces = []
      if node_state_size > 0:
        pieces.append(jnp.reshape(point_value.primal.node_states, (-1,)))
      if flat_control_size > 0:
        pieces.append(jnp.reshape(point_value.primal.offense_controls, (-1,)))
      if node_dual_size > 0:
        pieces.append(jnp.reshape(point_value.dual.node_multipliers, (-1,)))
      if terminal_dual_size > 0:
        pieces.append(jnp.reshape(point_value.dual.terminal_multipliers, (-1,)))
      if not pieces:
        return jnp.zeros((0,), dtype=dtype)
      return jnp.concatenate(pieces, axis=0)

    def unpack_reduced(flat_value: jnp.ndarray) -> EqualityGamePoint:
      offset = 0
      node_states = zero_node_states
      if node_state_size > 0:
        node_slice = flat_value[offset : offset + node_state_size]
        offset += node_state_size
        node_states = jnp.reshape(node_slice, (self.topology.node_count, self.state_dim))
      controls = zero_controls
      if flat_control_size > 0:
        control_slice = flat_value[offset : offset + flat_control_size]
        offset += flat_control_size
        controls = jnp.reshape(control_slice, (self.topology.edge_count, self.control_dim))
      node_duals = zero_node_duals
      if node_dual_size > 0:
        dual_slice = flat_value[offset : offset + node_dual_size]
        offset += node_dual_size
        node_duals = jnp.reshape(dual_slice, (self.topology.node_count, self.state_dim))
      terminal_duals = zero_terminal_duals
      if terminal_dual_size > 0:
        terminal_slice = flat_value[offset : offset + terminal_dual_size]
        terminal_duals = jnp.reshape(
          terminal_slice,
          (self.topology.leaf_count, self.terminal_selector.shape[0]),
        )
      return EqualityGamePoint(
        primal=EqualityGamePrimal(
          node_states=node_states,
          offense_controls=controls,
          defense_controls=zero_aux_controls,
        ),
        dual=EqualityGameDual(
          node_multipliers=node_duals,
          terminal_multipliers=terminal_duals,
        ),
      )

    def curved_control_response(
      node_duals: jnp.ndarray,
      control_rhs: jnp.ndarray,
    ) -> jnp.ndarray:
      child_duals = node_duals[self.topology.edge_children]
      dual_term = jnp.einsum("ei,eij->ej", child_duals, linearization.dynamics_jac)
      safe_denominator = jnp.where(curved_control_mask, effective_control_diag, jnp.ones_like(effective_control_diag))
      return jnp.where(
        curved_control_mask,
        (control_rhs + dual_term) / safe_denominator,
        jnp.zeros_like(control_rhs),
      )

    zero_curved_control_rhs = jnp.zeros_like(linearization.control_hdiag)

    def reduced_matvec(flat_value: jnp.ndarray) -> jnp.ndarray:
      reduced_point = unpack_reduced(flat_value)
      node_states = reduced_point.primal.node_states
      flat_controls = flat_control_scale * reduced_point.primal.offense_controls
      node_duals = reduced_point.dual.node_multipliers
      terminal_duals = reduced_point.dual.terminal_multipliers
      curved_controls = curved_control_response(node_duals, zero_curved_control_rhs)
      total_controls = flat_controls + curved_controls
      child_duals = node_duals[self.topology.edge_children]

      parent_dual_pullback = jax.ops.segment_sum(
        child_duals @ self.a_state,
        self.topology.edge_parents,
        num_segments=self.topology.node_count,
      )
      state_active = node_duals - parent_dual_pullback + linearization.node_hdiag * node_states + reg * node_states
      if self.terminal_selector.shape[0] > 0:
        terminal_state_term = jax.ops.segment_sum(
          terminal_duals @ self.terminal_selector,
          self.topology.leaf_nodes,
          num_segments=self.topology.node_count,
        )
        state_active = state_active + terminal_state_term
      state_matvec = active_node_scale * state_active + (1.0 - active_node_scale) * node_states

      flat_control_active = (
        effective_control_diag * flat_controls
        - jnp.einsum("ei,eij->ej", child_duals, linearization.dynamics_jac)
      )
      flat_control_matvec = (
        flat_control_scale * flat_control_active
        + (1.0 - flat_control_scale) * reduced_point.primal.offense_controls
      )

      predicted_states = (
        node_states[self.topology.edge_parents] @ self.a_state.T
        + jnp.einsum("eij,ej->ei", linearization.dynamics_jac, total_controls)
      )
      edge_constraint = active_edge_scale * (node_states[self.topology.edge_children] - predicted_states)
      node_constraint_active = jnp.zeros_like(node_duals)
      node_constraint_active = node_constraint_active.at[0].set(node_states[0])
      node_constraint_active = node_constraint_active.at[self.topology.edge_children].add(edge_constraint)
      node_constraint_active = node_constraint_active + reg * node_duals
      node_constraint_matvec = (
        active_node_scale * node_constraint_active
        + (1.0 - active_node_scale) * node_duals
      )
      terminal_constraint_active = (
        active_leaf_scale
        * (node_states[self.topology.leaf_nodes] @ self.terminal_selector.T + reg * terminal_duals)
      )
      terminal_constraint_matvec = (
        terminal_constraint_active
        + (1.0 - active_leaf_scale) * terminal_duals
      )

      return pack_reduced(
        EqualityGamePoint(
          primal=EqualityGamePrimal(
            node_states=state_matvec,
            offense_controls=flat_control_matvec,
            defense_controls=zero_aux_controls,
          ),
          dual=EqualityGameDual(
            node_multipliers=node_constraint_matvec,
            terminal_multipliers=terminal_constraint_matvec,
          ),
        )
      )

    curved_control_rhs = jnp.where(
      curved_control_mask,
      -residual.primal.offense_controls,
      jnp.zeros_like(residual.primal.offense_controls),
    )
    curved_control_rhs_step = curved_control_response(zero_node_duals, curved_control_rhs)
    curved_constraint_rhs = jnp.einsum("eij,ej->ei", linearization.dynamics_jac, curved_control_rhs_step)
    node_constraint_rhs = -residual.dual.node_multipliers
    node_constraint_rhs = node_constraint_rhs.at[self.topology.edge_children].add(curved_constraint_rhs)
    node_constraint_rhs = active_node_scale * node_constraint_rhs
    rhs = pack_reduced(
      EqualityGamePoint(
        primal=EqualityGamePrimal(
          node_states=-active_node_scale * residual.primal.node_states,
          offense_controls=-flat_control_scale * residual.primal.offense_controls,
          defense_controls=zero_aux_controls,
        ),
        dual=EqualityGameDual(
          node_multipliers=node_constraint_rhs,
          terminal_multipliers=-active_leaf_scale * residual.dual.terminal_multipliers,
        ),
      )
    )
    no_flat_controls = not bool(jnp.any(flat_control_mask))
    if prune_threshold <= 0.0 and no_flat_controls:
      tree_sweep_kernel = _drone3d_single_player_tree_sweep_kernel(
        self.topology.node_offsets_py,
        self.topology.edge_offsets_py,
        self.topology.total_depth,
        self.topology.node_count,
        self.topology.edge_count,
        self.topology.leaf_count,
        self.state_dim,
        self.control_dim,
        self.terminal_selector.shape[0],
      )
      node_states_step, controls_step, node_duals_step, terminal_duals_step = tree_sweep_kernel(
        -residual.primal.node_states,
        node_constraint_rhs,
        -residual.dual.terminal_multipliers,
        linearization.node_hdiag,
        linearization.dynamics_jac,
        effective_control_diag,
        curved_control_rhs,
        self.a_state,
        self.terminal_selector,
        self.topology.node_parents,
        self.topology.node_parent_edges,
        self.topology.edge_parents,
        self.topology.edge_children,
        self.topology.leaf_nodes,
        reg,
      )
      full_step = EqualityGamePoint(
        primal=EqualityGamePrimal(
          node_states=node_states_step,
          offense_controls=controls_step,
          defense_controls=zero_aux_controls,
        ),
        dual=EqualityGameDual(
          node_multipliers=node_duals_step,
          terminal_multipliers=terminal_duals_step,
        ),
      )
      full_product = self.kkt_matvec(full_step, theta, linearization)
      full_step_residual = EqualityGamePoint(
        primal=EqualityGamePrimal(
          node_states=full_product.primal.node_states + reg * full_step.primal.node_states + residual.primal.node_states,
          offense_controls=(
            full_product.primal.offense_controls
            + reg * full_step.primal.offense_controls
            + residual.primal.offense_controls
          ),
          defense_controls=zero_aux_controls,
        ),
        dual=EqualityGameDual(
          node_multipliers=(
            full_product.dual.node_multipliers
            + reg * full_step.dual.node_multipliers
            + residual.dual.node_multipliers
          ),
          terminal_multipliers=(
            full_product.dual.terminal_multipliers
            + reg * full_step.dual.terminal_multipliers
            + residual.dual.terminal_multipliers
          ),
        ),
      )
      full_step_residual_norm = float(_tree_max_abs(full_step_residual))
      residual_scale = max(1.0, float(_tree_max_abs(residual)))
      if (
        bool(jnp.isfinite(full_step_residual_norm))
        and full_step_residual_norm <= max(10.0 * cfg.gmres_tolerance * residual_scale, 1e-6)
      ):
        return (
          full_step,
          0.0,
          {
            "mode": "singularity_aware_partial_tree_sweep",
            "exact": True,
            "singularity_aware_partial_elimination": True,
            "approximate_pruning": False,
          },
        )

    reduced_dim = int(rhs.shape[0])
    reduced_matvec_kernel = _drone3d_single_player_reduced_matvec_kernel(
      self.topology.node_count,
      self.topology.edge_count,
      self.topology.leaf_count,
      self.state_dim,
      self.control_dim,
      self.terminal_selector.shape[0],
    )
    reduced_matvec = lambda flat_value: reduced_matvec_kernel(
      flat_value,
      active_node_scale,
      active_edge_scale,
      active_leaf_scale,
      flat_control_scale,
      curved_control_mask,
      effective_control_diag,
      linearization.dynamics_jac,
      linearization.node_hdiag,
      self.a_state,
      self.terminal_selector,
      self.topology.edge_parents,
      self.topology.edge_children,
      self.topology.leaf_nodes,
      reg,
    )
    if (
      bool(cfg.singularity_aware_dense_solve)
      and reduced_dim > 0
      and reduced_dim <= int(cfg.singularity_aware_dense_solve_max_dim)
    ):
      basis = jnp.eye(reduced_dim, dtype=dtype)
      dense_matrix = jax.vmap(reduced_matvec)(basis).T
      step_flat = jnp.linalg.solve(dense_matrix, rhs)
      linear_mode = "singularity_aware_partial_dense"
      linear_info = 0.0
    else:
      preconditioner = self.build_reduced_dual_preconditioner(
        theta,
        linearization,
        float(regularization),
        cfg.preconditioner_epsilon,
      )
      preconditioner_regularization = jnp.asarray(
        max(abs(float(regularization)), float(cfg.preconditioner_epsilon)),
        dtype=dtype,
      )
      state_diag = preconditioner_regularization * jnp.ones(
        (self.topology.node_count, self.state_dim),
        dtype=dtype,
      )
      state_diag = state_diag + linearization.node_hdiag
      state_inv = 1.0 / state_diag
      control_sign = jnp.where(linearization.control_hdiag >= 0.0, 1.0, -1.0).astype(dtype)
      control_safe = control_sign * jnp.maximum(
        jnp.abs(linearization.control_hdiag),
        preconditioner_regularization,
      )

      preconditioner_kernel = _drone3d_single_player_preconditioner_kernel(
        self.topology.node_offsets_py,
        self.topology.total_depth,
        self.topology.node_count,
        self.topology.edge_count,
        self.topology.leaf_count,
        self.state_dim,
        self.control_dim,
        self.terminal_selector.shape[0],
      )
      preconditioner_matvec = lambda flat_value: preconditioner_kernel(
        flat_value,
        active_node_scale,
        active_edge_scale,
        active_leaf_scale,
        flat_control_scale,
        flat_control_mask,
        linearization.dynamics_jac,
        self.a_state,
        self.terminal_selector,
        self.topology.edge_parents,
        self.topology.edge_children,
        self.topology.leaf_nodes,
        self.topology.node_parents,
        state_inv,
        control_safe,
        preconditioner.variable_mask,
        preconditioner.parent_coupling,
        preconditioner.schur_inv,
      )
      step_flat, gmres_info = jax.scipy.sparse.linalg.gmres(
        reduced_matvec,
        rhs,
        tol=cfg.gmres_tolerance,
        restart=cfg.gmres_restart,
        maxiter=cfg.gmres_maxiter,
        solve_method="incremental",
        M=preconditioner_matvec,
      )
      linear_mode = "singularity_aware_partial_pgmres"
      linear_info = 0.0 if gmres_info is None else float(gmres_info)
    reduced_step = unpack_reduced(step_flat)
    curved_controls = curved_control_response(
      reduced_step.dual.node_multipliers,
      curved_control_rhs,
    )
    full_step = EqualityGamePoint(
      primal=EqualityGamePrimal(
        node_states=active_node_scale * reduced_step.primal.node_states,
        offense_controls=flat_control_scale * reduced_step.primal.offense_controls + curved_controls,
        defense_controls=zero_aux_controls,
      ),
      dual=EqualityGameDual(
        node_multipliers=active_node_scale * reduced_step.dual.node_multipliers,
        terminal_multipliers=active_leaf_scale * reduced_step.dual.terminal_multipliers,
      ),
    )
    return (
      full_step,
      linear_info,
      {
        "mode": linear_mode,
        "exact": prune_threshold <= 0.0,
        "singularity_aware_partial_elimination": True,
        "approximate_pruning": prune_threshold > 0.0,
      },
    )

  def empty_alpha_logits(self) -> jnp.ndarray:
    return self.parent.empty_alpha_logits()

  def _zero_aux_controls(self, dtype: jnp.dtype) -> jnp.ndarray:
    return jnp.zeros((self.topology.edge_count, 0), dtype=dtype)

  def _edge_dynamics(
    self,
    parent_state: jnp.ndarray,
    control: jnp.ndarray,
  ) -> jnp.ndarray:
    accel, _, _ = self.parent._control_linearization(control)
    return self.a_state @ parent_state + self.control_matrix @ accel

  def _state_barrier_terms(self, node_states: jnp.ndarray) -> BoxBarrierTerms:
    return _box_barrier_terms(
      node_states,
      self.state_lower_bounds,
      self.state_upper_bounds,
      weight=self.inequality_barrier_weight,
    )

  def _control_barrier_terms(self, controls: jnp.ndarray) -> BoxBarrierTerms:
    return _box_barrier_terms(
      controls,
      self.control_lower_bounds,
      self.control_upper_bounds,
      weight=self.inequality_barrier_weight,
    )

  def _box_constraints_feasible(self, primal: EqualityGamePrimal) -> bool:
    if not self.has_box_inequality_constraints:
      return True
    state_terms = self._state_barrier_terms(primal.node_states)
    control_terms = self._control_barrier_terms(primal.offense_controls)
    return bool(state_terms.feasible and control_terms.feasible)

  def initial_point(self, theta: Any) -> EqualityGamePoint:
    del theta
    node_states = jnp.zeros((self.topology.node_count, self.state_dim), dtype=F32)
    node_states = node_states.at[0].set(self.x0)
    zero_control = jnp.zeros((self.control_dim,), dtype=F32)
    for edge_idx in range(self.topology.edge_count):
      parent_idx = int(self.topology.edge_parents[edge_idx])
      child_idx = int(self.topology.edge_children[edge_idx])
      child_state = self._edge_dynamics(node_states[parent_idx], zero_control)
      node_states = node_states.at[child_idx].set(child_state)
    zero_controls = jnp.zeros((self.topology.edge_count, self.control_dim), dtype=F32)
    point = EqualityGamePoint(
      primal=EqualityGamePrimal(
        node_states=node_states,
        offense_controls=zero_controls,
        defense_controls=self._zero_aux_controls(F32),
      ),
      dual=EqualityGameDual(
        node_multipliers=jnp.zeros((self.topology.node_count, self.state_dim), dtype=F32),
        terminal_multipliers=jnp.zeros((self.topology.leaf_count, self.terminal_selector.shape[0]), dtype=F32),
      ),
    )
    if self.has_box_inequality_constraints and not self._box_constraints_feasible(point.primal):
      raise ValueError(
        f"Active {self.player} box constraints exclude the default zero-control rollout. "
        "Provide a feasible warm start or widen the bounds.",
      )
    return point

  def node_constraints(
    self,
    primal: EqualityGamePrimal,
    theta: Any,
  ) -> jnp.ndarray:
    del theta
    constraints = jnp.zeros((self.topology.node_count, self.state_dim), dtype=primal.node_states.dtype)
    constraints = constraints.at[0].set(primal.node_states[0] - self.x0)
    for edge_idx in range(self.topology.edge_count):
      parent_idx = int(self.topology.edge_parents[edge_idx])
      child_idx = int(self.topology.edge_children[edge_idx])
      predicted = self._edge_dynamics(
        primal.node_states[parent_idx],
        primal.offense_controls[edge_idx],
      )
      constraints = constraints.at[child_idx].set(primal.node_states[child_idx] - predicted)
    return constraints

  def terminal_constraints(
    self,
    primal: EqualityGamePrimal,
    theta: Any,
  ) -> jnp.ndarray:
    del theta
    leaf_states = primal.node_states[self.topology.leaf_nodes]
    return leaf_states @ self.terminal_selector.T

  def objective(
    self,
    primal: EqualityGamePrimal,
    theta: Any,
  ) -> jnp.ndarray:
    alpha = build_alpha_from_logits(theta, self.tree)
    node_type_probs, _, _, edge_public_probs = propagate_type_probabilities(alpha, self.topology, self.prior)
    controls, _, _ = self.parent._control_linearization(primal.offense_controls)
    running_loss = 0.5 * jnp.sum(self.running_weights[None, :] * jnp.square(controls), axis=1)
    running_loss = self.parent.cfg.dt * jnp.sum(edge_public_probs * running_loss)

    leaf_type_probs = node_type_probs[self.topology.leaf_nodes]
    leaf_states = primal.node_states[self.topology.leaf_nodes]
    terminal_loss = jnp.sum(
      leaf_type_probs
      * (
        jnp.square(leaf_states) @ self.parent.terminal_type_diags.T
        + leaf_states @ self.terminal_r.T
        + self.terminal_c[None, :]
      ),
    )
    state_terms = self._state_barrier_terms(primal.node_states)
    control_terms = self._control_barrier_terms(primal.offense_controls)
    objective = running_loss + terminal_loss + state_terms.objective + control_terms.objective
    return jnp.where(
      state_terms.feasible & control_terms.feasible,
      objective,
      jnp.asarray(jnp.inf, dtype=objective.dtype),
    )

  def linearize_kkt(
    self,
    point: EqualityGamePoint,
    theta: Any,
  ) -> Drone3DSinglePlayerLinearization:
    alpha = build_alpha_from_logits(theta, self.tree)
    node_type_probs, node_public_probs, _, edge_public_probs = propagate_type_probabilities(alpha, self.topology, self.prior)
    leaf_type_probs = node_type_probs[self.topology.leaf_nodes]
    leaf_public_probs = node_public_probs[self.topology.leaf_nodes]

    controls, control_jac, control_second = self.parent._control_linearization(point.primal.offense_controls)
    dynamics_jac = self.control_matrix[None, :, :] * control_jac[:, None, :]

    node_states = point.primal.node_states
    parent_states = node_states[self.topology.edge_parents]
    predicted_states = parent_states @ self.a_state.T + controls @ self.control_matrix.T
    node_constraints = jnp.concatenate(
      [
        (node_states[0] - self.x0)[None, :],
        node_states[self.topology.edge_children] - predicted_states,
      ],
      axis=0,
    )
    leaf_states = node_states[self.topology.leaf_nodes]
    terminal_constraints = leaf_states @ self.terminal_selector.T

    weighted_pdiag = leaf_type_probs @ self.terminal_pdiag
    weighted_r = leaf_type_probs @ self.terminal_r
    leaf_grad = weighted_pdiag * leaf_states + weighted_r
    leaf_hdiag = weighted_pdiag
    state_terms = self._state_barrier_terms(node_states)
    control_terms = self._control_barrier_terms(point.primal.offense_controls)
    node_grad = state_terms.grad
    node_hdiag = state_terms.hdiag
    node_grad = node_grad.at[self.topology.leaf_nodes].add(leaf_grad)
    node_hdiag = node_hdiag.at[self.topology.leaf_nodes].add(leaf_hdiag)

    child_multipliers = point.dual.node_multipliers[self.topology.edge_children]
    control_coeff = child_multipliers @ self.control_matrix
    control_hdiag = (
      self.parent.cfg.dt
      * edge_public_probs[:, None]
      * self.running_weights[None, :]
      * (jnp.square(control_jac) + controls * control_second)
      - control_coeff * control_second
    )
    control_grad = (
      self.parent.cfg.dt
      * edge_public_probs[:, None]
      * self.running_weights[None, :]
      * controls
      * control_jac
      + control_terms.grad
    )
    control_hdiag = control_hdiag + control_terms.hdiag

    return Drone3DSinglePlayerLinearization(
      node_public_probs=node_public_probs,
      edge_public_probs=edge_public_probs,
      leaf_type_probs=leaf_type_probs,
      leaf_public_probs=leaf_public_probs,
      controls=controls,
      control_jac=control_jac,
      control_second=control_second,
      dynamics_jac=dynamics_jac,
      control_grad=control_grad,
      control_hdiag=control_hdiag,
      node_grad=node_grad,
      node_hdiag=node_hdiag,
      node_constraints=node_constraints,
      terminal_constraints=terminal_constraints,
    )

  def kkt_residual(
    self,
    point: EqualityGamePoint,
    theta: Any,
    linearization: Drone3DSinglePlayerLinearization | None = None,
  ) -> EqualityGamePoint:
    linearization = self.linearize_kkt(point, theta) if linearization is None else linearization
    node_residual = point.dual.node_multipliers
    child_multipliers = point.dual.node_multipliers[self.topology.edge_children]
    node_residual = node_residual.at[self.topology.edge_parents].add(-(child_multipliers @ self.a_state))
    node_residual = node_residual + linearization.node_grad
    node_residual = node_residual.at[self.topology.leaf_nodes].add(
      point.dual.terminal_multipliers @ self.terminal_selector,
    )
    control_constraint_grad = -jnp.einsum("ei,eij->ej", child_multipliers, linearization.dynamics_jac)
    return EqualityGamePoint(
      primal=EqualityGamePrimal(
        node_states=node_residual,
        offense_controls=linearization.control_grad + control_constraint_grad,
        defense_controls=jnp.zeros_like(point.primal.defense_controls),
      ),
      dual=EqualityGameDual(
        node_multipliers=linearization.node_constraints,
        terminal_multipliers=linearization.terminal_constraints,
      ),
    )

  def kkt_matvec(
    self,
    tangent: EqualityGamePoint,
    theta: Any,
    linearization: Drone3DSinglePlayerLinearization,
  ) -> EqualityGamePoint:
    del theta
    node_tangent = tangent.primal.node_states
    control_tangent = tangent.primal.offense_controls
    node_dual_tangent = tangent.dual.node_multipliers
    terminal_dual_tangent = tangent.dual.terminal_multipliers

    node_state_matvec = node_dual_tangent
    child_dual_tangent = node_dual_tangent[self.topology.edge_children]
    node_state_matvec = node_state_matvec.at[self.topology.edge_parents].add(-(child_dual_tangent @ self.a_state))
    node_state_matvec = node_state_matvec + linearization.node_hdiag * node_tangent
    node_state_matvec = node_state_matvec.at[self.topology.leaf_nodes].add(
      terminal_dual_tangent @ self.terminal_selector,
    )

    control_matvec = linearization.control_hdiag * control_tangent
    control_matvec = control_matvec - jnp.einsum(
      "ei,eij->ej",
      child_dual_tangent,
      linearization.dynamics_jac,
    )

    predicted_tangent = (
      node_tangent[self.topology.edge_parents] @ self.a_state.T
      + jnp.einsum("eij,ej->ei", linearization.dynamics_jac, control_tangent)
    )
    node_constraint_matvec = jnp.concatenate(
      [
        node_tangent[0][None, :],
        node_tangent[self.topology.edge_children] - predicted_tangent,
      ],
      axis=0,
    )
    terminal_constraint_matvec = node_tangent[self.topology.leaf_nodes] @ self.terminal_selector.T
    return EqualityGamePoint(
      primal=EqualityGamePrimal(
        node_states=node_state_matvec,
        offense_controls=control_matvec,
        defense_controls=jnp.zeros_like(tangent.primal.defense_controls),
      ),
      dual=EqualityGameDual(
        node_multipliers=node_constraint_matvec,
        terminal_multipliers=terminal_constraint_matvec,
      ),
    )

  def _primal_step_from_reduced_dual(
    self,
    dual_tangent: EqualityGameDual,
    neg_primal_residual: EqualityGamePrimal,
    linearization: Drone3DSinglePlayerLinearization,
    regularization: float,
  ) -> EqualityGamePrimal:
    reg = jnp.asarray(regularization, dtype=F32)
    child_dual_tangent = dual_tangent.node_multipliers[self.topology.edge_children]
    state_diag = reg + linearization.node_hdiag
    state_rhs = neg_primal_residual.node_states - dual_tangent.node_multipliers
    state_rhs = state_rhs.at[self.topology.edge_parents].add(child_dual_tangent @ self.a_state)
    state_rhs = state_rhs.at[self.topology.leaf_nodes].add(
      -(dual_tangent.terminal_multipliers @ self.terminal_selector),
    )
    node_states = state_rhs / state_diag

    control_diag = linearization.control_hdiag + reg
    control_rhs = neg_primal_residual.offense_controls + jnp.einsum(
      "ei,eij->ej",
      child_dual_tangent,
      linearization.dynamics_jac,
    )
    controls = control_rhs / control_diag
    return EqualityGamePrimal(
      node_states=node_states,
      offense_controls=controls,
      defense_controls=jnp.zeros_like(neg_primal_residual.defense_controls),
    )

  def _reduced_dual_from_primal_and_dual(
    self,
    primal_tangent: EqualityGamePrimal,
    dual_tangent: EqualityGameDual,
    linearization: Drone3DSinglePlayerLinearization,
    regularization: float,
  ) -> EqualityGameDual:
    reg = jnp.asarray(regularization, dtype=F32)
    predicted_tangent = (
      primal_tangent.node_states[self.topology.edge_parents] @ self.a_state.T
      + jnp.einsum("eij,ej->ei", linearization.dynamics_jac, primal_tangent.offense_controls)
    )
    node_constraint_matvec = jnp.concatenate(
      [
        primal_tangent.node_states[0][None, :],
        primal_tangent.node_states[self.topology.edge_children] - predicted_tangent,
      ],
      axis=0,
    ) + reg * dual_tangent.node_multipliers
    terminal_constraint_matvec = (
      primal_tangent.node_states[self.topology.leaf_nodes] @ self.terminal_selector.T
      + reg * dual_tangent.terminal_multipliers
    )
    return EqualityGameDual(
      node_multipliers=node_constraint_matvec,
      terminal_multipliers=terminal_constraint_matvec,
    )

  def reduced_dual_matvec(
    self,
    dual_tangent: EqualityGameDual,
    theta: Any,
    linearization: Drone3DSinglePlayerLinearization,
    regularization: float,
  ) -> EqualityGameDual:
    del theta
    zero_primal = EqualityGamePrimal(
      node_states=jnp.zeros((self.topology.node_count, self.state_dim), dtype=F32),
      offense_controls=jnp.zeros((self.topology.edge_count, self.control_dim), dtype=F32),
      defense_controls=self._zero_aux_controls(F32),
    )
    primal_tangent = self._primal_step_from_reduced_dual(
      dual_tangent,
      zero_primal,
      linearization,
      regularization,
    )
    return self._reduced_dual_from_primal_and_dual(
      primal_tangent,
      dual_tangent,
      linearization,
      regularization,
    )

  def reduced_dual_rhs(
    self,
    residual: EqualityGamePoint,
    theta: Any,
    linearization: Drone3DSinglePlayerLinearization,
    regularization: float,
  ) -> EqualityGameDual:
    del theta
    zero_dual = EqualityGameDual(
      node_multipliers=jnp.zeros_like(residual.dual.node_multipliers),
      terminal_multipliers=jnp.zeros_like(residual.dual.terminal_multipliers),
    )
    neg_primal_residual = jax.tree_util.tree_map(lambda value: -value, residual.primal)
    primal_response = self._primal_step_from_reduced_dual(
      zero_dual,
      neg_primal_residual,
      linearization,
      regularization,
    )
    dual_response = self._reduced_dual_from_primal_and_dual(
      primal_response,
      zero_dual,
      linearization,
      regularization,
    )
    return EqualityGameDual(
      node_multipliers=-residual.dual.node_multipliers - dual_response.node_multipliers,
      terminal_multipliers=-residual.dual.terminal_multipliers - dual_response.terminal_multipliers,
    )

  def recover_primal_step_from_reduced_dual(
    self,
    dual_step: EqualityGameDual,
    residual_primal: EqualityGamePrimal,
    theta: Any,
    linearization: Drone3DSinglePlayerLinearization,
    regularization: float,
  ) -> EqualityGamePrimal:
    del theta
    neg_primal_residual = jax.tree_util.tree_map(lambda value: -value, residual_primal)
    return self._primal_step_from_reduced_dual(
      dual_step,
      neg_primal_residual,
      linearization,
      regularization,
    )

  def build_tree_schur_preconditioner_data(
    self,
    theta: Any,
    linearization: Drone3DSinglePlayerLinearization,
    regularization: float,
    epsilon: float,
  ) -> TreeSchurBlockData:
    del theta
    dual_dtype = linearization.node_constraints.dtype
    preconditioner_regularization = jnp.asarray(max(abs(regularization), epsilon), dtype=dual_dtype)
    node_count = self.topology.node_count
    terminal_dim = self.terminal_selector.shape[0]
    max_variable_dim = self.state_dim + terminal_dim

    variable_dim = jnp.full((node_count,), self.state_dim, dtype=jnp.int32)
    variable_mask = jnp.zeros((node_count, max_variable_dim), dtype=dual_dtype)
    variable_mask = variable_mask.at[:, : self.state_dim].set(1.0)
    if terminal_dim > 0:
      variable_dim = variable_dim.at[self.topology.leaf_nodes].add(terminal_dim)
      variable_mask = variable_mask.at[
        self.topology.leaf_nodes,
        self.state_dim : self.state_dim + terminal_dim,
      ].set(1.0)

    state_diag = preconditioner_regularization + linearization.node_hdiag
    state_inv = 1.0 / state_diag
    state_diag_inv = jnp.eye(self.state_dim, dtype=dual_dtype)[None, :, :] * state_inv[:, None, :]

    parent_indices = jnp.maximum(self.topology.node_parents, 0)
    incoming_edge_indices = jnp.maximum(self.topology.node_parent_edges, 0)
    non_root_mask = (self.topology.node_parents >= 0).astype(dual_dtype)

    parent_coupling_state = (
      state_inv[parent_indices, :, None] * self.a_state.T[None, :, :]
    ) * non_root_mask[:, None, None]
    parent_state_term = jnp.einsum("ij,njk->nik", self.a_state, parent_coupling_state)

    dynamics_jac = linearization.dynamics_jac[incoming_edge_indices] * non_root_mask[:, None, None]
    control_diag_inv = (
      1.0 / (linearization.control_hdiag[incoming_edge_indices] + preconditioner_regularization)
    ) * non_root_mask[:, None]
    control_state_term = jnp.einsum(
      "nij,nj,nkj->nik",
      dynamics_jac,
      control_diag_inv,
      dynamics_jac,
    )

    state_self_block = (
      preconditioner_regularization * jnp.eye(self.state_dim, dtype=dual_dtype)[None, :, :]
      - state_diag_inv
      - parent_state_term
      - control_state_term
    )

    self_blocks = jnp.tile(
      jnp.eye(max_variable_dim, dtype=dual_dtype)[None, :, :],
      (node_count, 1, 1),
    )
    self_blocks = self_blocks.at[:, : self.state_dim, : self.state_dim].set(state_self_block)

    parent_coupling = jnp.zeros((node_count, max_variable_dim, max_variable_dim), dtype=dual_dtype)
    parent_coupling = parent_coupling.at[:, : self.state_dim, : self.state_dim].set(parent_coupling_state)

    if terminal_dim > 0:
      leaf_state_inv = state_inv[self.topology.leaf_nodes]
      lambda_mu_block = -leaf_state_inv[:, :, None] * self.terminal_selector.T[None, :, :]
      mu_lambda_block = -self.terminal_selector[None, :, :] * leaf_state_inv[:, None, :]
      mu_mu_block = (
        preconditioner_regularization * jnp.eye(terminal_dim, dtype=dual_dtype)[None, :, :]
        + jnp.einsum("ai,nib->nab", self.terminal_selector, lambda_mu_block)
      )
      self_blocks = self_blocks.at[
        self.topology.leaf_nodes,
        : self.state_dim,
        self.state_dim : self.state_dim + terminal_dim,
      ].set(lambda_mu_block)
      self_blocks = self_blocks.at[
        self.topology.leaf_nodes,
        self.state_dim : self.state_dim + terminal_dim,
        : self.state_dim,
      ].set(mu_lambda_block)
      self_blocks = self_blocks.at[
        self.topology.leaf_nodes,
        self.state_dim : self.state_dim + terminal_dim,
        self.state_dim : self.state_dim + terminal_dim,
      ].set(mu_mu_block)

    return TreeSchurBlockData(
      variable_dim=variable_dim,
      variable_mask=variable_mask,
      self_blocks=self_blocks,
      parent_coupling=parent_coupling,
    )

  def pack_reduced_dual_for_tree_schur(
    self,
    dual_value: EqualityGameDual,
    theta: Any,
    linearization: Drone3DSinglePlayerLinearization,
    regularization: float,
    preconditioner: TreeSchurPreconditioner,
  ) -> jnp.ndarray:
    del theta, linearization, regularization, preconditioner
    terminal_dim = self.terminal_selector.shape[0]
    packed = dual_value.node_multipliers
    if terminal_dim > 0:
      safe_leaf_indices = jnp.maximum(self.leaf_index_by_node, 0)
      node_terminal = dual_value.terminal_multipliers[safe_leaf_indices] * self.leaf_node_mask[:, None]
      packed = jnp.concatenate([packed, node_terminal], axis=1)
    return packed

  def unpack_reduced_dual_from_tree_schur(
    self,
    packed_value: jnp.ndarray,
    theta: Any,
    linearization: Drone3DSinglePlayerLinearization,
    regularization: float,
    preconditioner: TreeSchurPreconditioner,
  ) -> EqualityGameDual:
    del theta, linearization, regularization, preconditioner
    terminal_dim = self.terminal_selector.shape[0]
    terminal_result = jnp.zeros((self.topology.leaf_count, terminal_dim), dtype=packed_value.dtype)
    if terminal_dim > 0:
      terminal_result = packed_value[
        self.topology.leaf_nodes,
        self.state_dim : self.state_dim + terminal_dim,
      ]
    return EqualityGameDual(
      node_multipliers=packed_value[:, : self.state_dim],
      terminal_multipliers=terminal_result,
    )

  def build_reduced_dual_preconditioner(
    self,
    theta: Any,
    linearization: Drone3DSinglePlayerLinearization,
    regularization: float,
    epsilon: float,
  ) -> TreeSchurPreconditioner:
    return build_tree_schur_preconditioner(
      self.topology,
      self.build_tree_schur_preconditioner_data(
        theta,
        linearization,
        regularization,
        epsilon,
      ),
      epsilon,
    )

  def apply_reduced_dual_preconditioner(
    self,
    dual_value: EqualityGameDual,
    theta: Any,
    linearization: Drone3DSinglePlayerLinearization,
    regularization: float,
    preconditioner: TreeSchurPreconditioner,
  ) -> EqualityGameDual:
    rhs = self.pack_reduced_dual_for_tree_schur(
      dual_value,
      theta,
      linearization,
      regularization,
      preconditioner,
    )
    rhs = rhs * preconditioner.variable_mask
    solution = apply_tree_schur_preconditioner(
      self.topology,
      preconditioner.schur_inv,
      preconditioner.parent_coupling,
      rhs,
    )
    solution = solution * preconditioner.variable_mask
    return self.unpack_reduced_dual_from_tree_schur(
      solution,
      theta,
      linearization,
      regularization,
      preconditioner,
    )


def _make_drone3d_player_problem(
  problem: Drone3DTreeProblem,
  player: str,
) -> Drone3DSinglePlayerTreeProblem:
  return Drone3DSinglePlayerTreeProblem(problem, player)


def _make_drone3d_box_free_shadow_problem(problem: Drone3DTreeProblem) -> Drone3DTreeProblem:
  cfg = replace(
    problem.cfg,
    offense_state_lower_bounds=None,
    offense_state_upper_bounds=None,
    defense_state_lower_bounds=None,
    defense_state_upper_bounds=None,
    offense_control_lower_bounds=None,
    offense_control_upper_bounds=None,
    defense_control_lower_bounds=None,
    defense_control_upper_bounds=None,
    inequality_barrier_weight=0.0,
  )
  return Drone3DTreeProblem(problem.tree, cfg)


def _project_controls_to_box_interior(
  controls: jnp.ndarray,
  lower: jnp.ndarray,
  upper: jnp.ndarray,
  *,
  margin: float = 1e-3,
) -> jnp.ndarray:
  dtype = controls.dtype
  margin_value = jnp.asarray(max(0.0, float(margin)), dtype=dtype)
  lower_active = jnp.isfinite(lower)
  upper_active = jnp.isfinite(upper)
  span = upper - lower
  local_margin = jnp.where(
    jnp.logical_and(lower_active, upper_active),
    jnp.minimum(margin_value, 0.25 * jnp.maximum(span, 0.0)),
    margin_value,
  )
  effective_lower = jnp.where(lower_active, lower + local_margin, -jnp.inf)
  effective_upper = jnp.where(upper_active, upper - local_margin, jnp.inf)
  projected = jnp.maximum(controls, effective_lower[None, :])
  projected = jnp.minimum(projected, effective_upper[None, :])
  return projected


def _rollout_drone3d_player_states(
  player_problem: Drone3DSinglePlayerTreeProblem,
  controls: jnp.ndarray,
) -> jnp.ndarray:
  node_states = jnp.zeros((player_problem.topology.node_count, player_problem.state_dim), dtype=controls.dtype)
  node_states = node_states.at[0].set(player_problem.x0.astype(controls.dtype))
  for edge_idx in range(player_problem.topology.edge_count):
    parent_idx = int(player_problem.topology.edge_parents[edge_idx])
    child_idx = int(player_problem.topology.edge_children[edge_idx])
    child_state = player_problem._edge_dynamics(node_states[parent_idx], controls[edge_idx])
    node_states = node_states.at[child_idx].set(child_state)
  return node_states


def _project_drone3d_player_point_to_box_interior(
  player_problem: Drone3DSinglePlayerTreeProblem,
  point: EqualityGamePoint,
  *,
  margin: float = 1e-3,
) -> tuple[EqualityGamePoint, bool, float]:
  projected_controls = _project_controls_to_box_interior(
    point.primal.offense_controls,
    player_problem.control_lower_bounds,
    player_problem.control_upper_bounds,
    margin=margin,
  )
  projected_states = _rollout_drone3d_player_states(player_problem, projected_controls)
  projected_point = EqualityGamePoint(
    primal=EqualityGamePrimal(
      node_states=projected_states,
      offense_controls=projected_controls,
      defense_controls=player_problem._zero_aux_controls(projected_controls.dtype),
    ),
    dual=EqualityGameDual(
      node_multipliers=jnp.zeros_like(point.dual.node_multipliers),
      terminal_multipliers=jnp.zeros_like(point.dual.terminal_multipliers),
    ),
  )
  feasible = player_problem._box_constraints_feasible(projected_point.primal)
  delta_max = _tree_max_abs(projected_controls - point.primal.offense_controls)
  return projected_point, feasible, delta_max


def _summarize_player_control_range(
  player_problem: Drone3DSinglePlayerTreeProblem,
  point: EqualityGamePoint,
) -> dict[str, Any]:
  controls = point.primal.offense_controls
  if controls.size == 0:
    zero = jnp.zeros((player_problem.control_dim,), dtype=player_problem.x0.dtype)
    return {
      "control_min": zero,
      "control_max": zero,
      "control_max_abs": 0.0,
      "control_max_violation": 0.0,
      "controls_within_box": True,
    }
  lower_violation = jnp.where(
    jnp.isfinite(player_problem.control_lower_bounds)[None, :],
    jnp.maximum(player_problem.control_lower_bounds[None, :] - controls, 0.0),
    0.0,
  )
  upper_violation = jnp.where(
    jnp.isfinite(player_problem.control_upper_bounds)[None, :],
    jnp.maximum(controls - player_problem.control_upper_bounds[None, :], 0.0),
    0.0,
  )
  max_violation = max(
    _tree_max_abs(lower_violation),
    _tree_max_abs(upper_violation),
  )
  return {
    "control_min": jnp.min(controls, axis=0),
    "control_max": jnp.max(controls, axis=0),
    "control_max_abs": _tree_max_abs(controls),
    "control_max_violation": max_violation,
    "controls_within_box": bool(max_violation <= 0.0),
  }


def _make_drone3d_box_constrained_variant2_warm_start(
  problem: Drone3DTreeProblem,
  theta: jnp.ndarray,
  *,
  interior_margin: float = 1e-3,
) -> Drone3DBoxWarmStart:
  if not problem.has_box_inequality_constraints or problem.cfg.squash_controls:
    zero = jnp.zeros((3,), dtype=F32)
    return Drone3DBoxWarmStart(
      point=None,
      mode="disabled",
      prepass_sec=0.0,
      projected_control_delta_max=0.0,
      used_projection=False,
      feasible=False,
      fallback_reason="boxes_inactive_or_squashed",
      prepass_offense_control_min=zero,
      prepass_offense_control_max=zero,
      prepass_defense_control_min=zero,
      prepass_defense_control_max=zero,
      prepass_offense_control_max_abs=0.0,
      prepass_defense_control_max_abs=0.0,
      prepass_offense_control_max_violation=0.0,
      prepass_defense_control_max_violation=0.0,
      prepass_offense_controls_within_box=True,
      prepass_defense_controls_within_box=True,
    )

  prepass_start = time.perf_counter()
  shadow_problem = _make_drone3d_box_free_shadow_problem(problem)
  shadow_offense = _make_drone3d_player_problem(shadow_problem, "offense")
  shadow_defense = _make_drone3d_player_problem(shadow_problem, "defense")
  actual_offense = _make_drone3d_player_problem(problem, "offense")
  actual_defense = _make_drone3d_player_problem(problem, "defense")

  offense_exact = shadow_offense.solve_exact_lq(theta)
  defense_exact = shadow_defense.solve_exact_lq(theta)
  offense_range = _summarize_player_control_range(actual_offense, offense_exact.point)
  defense_range = _summarize_player_control_range(actual_defense, defense_exact.point)
  offense_projected, offense_feasible, offense_delta = _project_drone3d_player_point_to_box_interior(
    actual_offense,
    offense_exact.point,
    margin=interior_margin,
  )
  defense_projected, defense_feasible, defense_delta = _project_drone3d_player_point_to_box_interior(
    actual_defense,
    defense_exact.point,
    margin=interior_margin,
  )
  prepass_sec = time.perf_counter() - prepass_start

  if not offense_feasible or not defense_feasible:
    fallback_reasons: list[str] = []
    if not offense_feasible:
      fallback_reasons.append("offense_projection_infeasible")
    if not defense_feasible:
      fallback_reasons.append("defense_projection_infeasible")
    return Drone3DBoxWarmStart(
      point=None,
      mode="exact_lq_projected",
      prepass_sec=prepass_sec,
      projected_control_delta_max=max(offense_delta, defense_delta),
      used_projection=True,
      feasible=False,
      fallback_reason="+".join(fallback_reasons),
      prepass_offense_control_min=offense_range["control_min"],
      prepass_offense_control_max=offense_range["control_max"],
      prepass_defense_control_min=defense_range["control_min"],
      prepass_defense_control_max=defense_range["control_max"],
      prepass_offense_control_max_abs=offense_range["control_max_abs"],
      prepass_defense_control_max_abs=defense_range["control_max_abs"],
      prepass_offense_control_max_violation=offense_range["control_max_violation"],
      prepass_defense_control_max_violation=defense_range["control_max_violation"],
      prepass_offense_controls_within_box=offense_range["controls_within_box"],
      prepass_defense_controls_within_box=defense_range["controls_within_box"],
    )

  return Drone3DBoxWarmStart(
    point=_combine_drone3d_player_points(offense_projected, defense_projected),
    mode="exact_lq_projected",
    prepass_sec=prepass_sec,
    projected_control_delta_max=max(offense_delta, defense_delta),
    used_projection=bool(max(offense_delta, defense_delta) > 0.0),
    feasible=True,
    fallback_reason=None,
    prepass_offense_control_min=offense_range["control_min"],
    prepass_offense_control_max=offense_range["control_max"],
    prepass_defense_control_min=defense_range["control_min"],
    prepass_defense_control_max=defense_range["control_max"],
    prepass_offense_control_max_abs=offense_range["control_max_abs"],
    prepass_defense_control_max_abs=defense_range["control_max_abs"],
    prepass_offense_control_max_violation=offense_range["control_max_violation"],
    prepass_defense_control_max_violation=defense_range["control_max_violation"],
    prepass_offense_controls_within_box=offense_range["controls_within_box"],
    prepass_defense_controls_within_box=defense_range["controls_within_box"],
  )


def _initialize_drone3d_box_variant2_points(
  problem: Drone3DTreeProblem,
  theta: jnp.ndarray,
  *,
  initial_point: EqualityGamePoint | None = None,
  prefer_exact_lq_projected_warm_start: bool = True,
) -> tuple[EqualityGamePoint, dict[str, Any]]:
  def _default_initial() -> EqualityGamePoint:
    offense_problem = _make_drone3d_player_problem(problem, "offense")
    defense_problem = _make_drone3d_player_problem(problem, "defense")
    return _combine_drone3d_player_points(
      offense_problem.initial_point(theta),
      defense_problem.initial_point(theta),
    )

  if initial_point is not None:
    return initial_point, {
      "variant2_box_warm_start_mode": "user_initial_point",
      "variant2_box_warm_start_prepass_sec": 0.0,
      "variant2_box_warm_start_projected_control_delta_max": 0.0,
      "variant2_box_warm_start_used_projection": False,
      "variant2_box_warm_start_feasible": True,
      "variant2_box_warm_start_fallback_reason": None,
      "variant2_box_prepass_offense_control_min": jnp.zeros((3,), dtype=F32),
      "variant2_box_prepass_offense_control_max": jnp.zeros((3,), dtype=F32),
      "variant2_box_prepass_defense_control_min": jnp.zeros((3,), dtype=F32),
      "variant2_box_prepass_defense_control_max": jnp.zeros((3,), dtype=F32),
      "variant2_box_prepass_offense_control_max_abs": 0.0,
      "variant2_box_prepass_defense_control_max_abs": 0.0,
      "variant2_box_prepass_offense_control_max_violation": 0.0,
      "variant2_box_prepass_defense_control_max_violation": 0.0,
      "variant2_box_prepass_offense_controls_within_box": True,
      "variant2_box_prepass_defense_controls_within_box": True,
    }

  if problem.has_box_inequality_constraints and not problem.cfg.squash_controls:
    warm_start = _make_drone3d_box_constrained_variant2_warm_start(problem, theta)
    if warm_start.point is not None and prefer_exact_lq_projected_warm_start:
      return warm_start.point, {
        "variant2_box_warm_start_mode": warm_start.mode,
        "variant2_box_warm_start_prepass_sec": warm_start.prepass_sec,
        "variant2_box_warm_start_projected_control_delta_max": warm_start.projected_control_delta_max,
        "variant2_box_warm_start_used_projection": warm_start.used_projection,
        "variant2_box_warm_start_feasible": warm_start.feasible,
        "variant2_box_warm_start_fallback_reason": warm_start.fallback_reason,
        "variant2_box_prepass_offense_control_min": warm_start.prepass_offense_control_min,
        "variant2_box_prepass_offense_control_max": warm_start.prepass_offense_control_max,
        "variant2_box_prepass_defense_control_min": warm_start.prepass_defense_control_min,
        "variant2_box_prepass_defense_control_max": warm_start.prepass_defense_control_max,
        "variant2_box_prepass_offense_control_max_abs": warm_start.prepass_offense_control_max_abs,
        "variant2_box_prepass_defense_control_max_abs": warm_start.prepass_defense_control_max_abs,
        "variant2_box_prepass_offense_control_max_violation": warm_start.prepass_offense_control_max_violation,
        "variant2_box_prepass_defense_control_max_violation": warm_start.prepass_defense_control_max_violation,
        "variant2_box_prepass_offense_controls_within_box": warm_start.prepass_offense_controls_within_box,
        "variant2_box_prepass_defense_controls_within_box": warm_start.prepass_defense_controls_within_box,
      }
    return _default_initial(), {
      "variant2_box_warm_start_mode": "default_initial_point",
      "variant2_box_warm_start_prepass_sec": warm_start.prepass_sec,
      "variant2_box_warm_start_projected_control_delta_max": warm_start.projected_control_delta_max,
      "variant2_box_warm_start_used_projection": warm_start.used_projection,
      "variant2_box_warm_start_feasible": False,
      "variant2_box_warm_start_fallback_reason": warm_start.fallback_reason,
      "variant2_box_prepass_offense_control_min": warm_start.prepass_offense_control_min,
      "variant2_box_prepass_offense_control_max": warm_start.prepass_offense_control_max,
      "variant2_box_prepass_defense_control_min": warm_start.prepass_defense_control_min,
      "variant2_box_prepass_defense_control_max": warm_start.prepass_defense_control_max,
      "variant2_box_prepass_offense_control_max_abs": warm_start.prepass_offense_control_max_abs,
      "variant2_box_prepass_defense_control_max_abs": warm_start.prepass_defense_control_max_abs,
      "variant2_box_prepass_offense_control_max_violation": warm_start.prepass_offense_control_max_violation,
      "variant2_box_prepass_defense_control_max_violation": warm_start.prepass_defense_control_max_violation,
      "variant2_box_prepass_offense_controls_within_box": warm_start.prepass_offense_controls_within_box,
      "variant2_box_prepass_defense_controls_within_box": warm_start.prepass_defense_controls_within_box,
    }

  return _default_initial(), {
    "variant2_box_warm_start_mode": "not_applicable",
    "variant2_box_warm_start_prepass_sec": 0.0,
    "variant2_box_warm_start_projected_control_delta_max": 0.0,
    "variant2_box_warm_start_used_projection": False,
    "variant2_box_warm_start_feasible": True,
    "variant2_box_warm_start_fallback_reason": None,
    "variant2_box_prepass_offense_control_min": jnp.zeros((3,), dtype=F32),
    "variant2_box_prepass_offense_control_max": jnp.zeros((3,), dtype=F32),
    "variant2_box_prepass_defense_control_min": jnp.zeros((3,), dtype=F32),
    "variant2_box_prepass_defense_control_max": jnp.zeros((3,), dtype=F32),
    "variant2_box_prepass_offense_control_max_abs": 0.0,
    "variant2_box_prepass_defense_control_max_abs": 0.0,
    "variant2_box_prepass_offense_control_max_violation": 0.0,
    "variant2_box_prepass_defense_control_max_violation": 0.0,
    "variant2_box_prepass_offense_controls_within_box": True,
    "variant2_box_prepass_defense_controls_within_box": True,
  }


def _split_drone3d_full_point_by_player(
  problem: Drone3DTreeProblem,
  point: EqualityGamePoint,
) -> tuple[EqualityGamePoint, EqualityGamePoint]:
  control_dtype = point.primal.offense_controls.dtype
  empty_controls = jnp.zeros((problem.topology.edge_count, 0), dtype=control_dtype)
  terminal_dim = point.dual.terminal_multipliers.shape[1]
  half_terminal_dim = terminal_dim // 2
  offense_point = EqualityGamePoint(
    primal=EqualityGamePrimal(
      node_states=point.primal.node_states[:, 0:6],
      offense_controls=point.primal.offense_controls,
      defense_controls=empty_controls,
    ),
    dual=EqualityGameDual(
      node_multipliers=point.dual.node_multipliers[:, 0:6],
      terminal_multipliers=point.dual.terminal_multipliers[:, :half_terminal_dim],
    ),
  )
  defense_point = EqualityGamePoint(
    primal=EqualityGamePrimal(
      node_states=point.primal.node_states[:, 6:12],
      offense_controls=point.primal.defense_controls,
      defense_controls=empty_controls,
    ),
    dual=EqualityGameDual(
      node_multipliers=-point.dual.node_multipliers[:, 6:12],
      terminal_multipliers=-point.dual.terminal_multipliers[:, half_terminal_dim:],
    ),
  )
  return offense_point, defense_point


def _combine_drone3d_player_points(
  offense_point: EqualityGamePoint,
  defense_point: EqualityGamePoint,
) -> EqualityGamePoint:
  return EqualityGamePoint(
    primal=EqualityGamePrimal(
      node_states=jnp.concatenate(
        [offense_point.primal.node_states, defense_point.primal.node_states],
        axis=1,
      ),
      offense_controls=offense_point.primal.offense_controls,
      defense_controls=defense_point.primal.offense_controls,
    ),
    dual=EqualityGameDual(
      node_multipliers=jnp.concatenate(
        [offense_point.dual.node_multipliers, -defense_point.dual.node_multipliers],
        axis=1,
      ),
      terminal_multipliers=jnp.concatenate(
        [offense_point.dual.terminal_multipliers, -defense_point.dual.terminal_multipliers],
        axis=1,
      ),
    ),
  )


def _aggregate_drone3d_player_solve_result(
  problem: Drone3DTreeProblem,
  theta: jnp.ndarray,
  offense_result: Any,
  defense_result: Any,
  *,
  solver_variant: str = "player_separable_lq",
  extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
  offense_problem = _make_drone3d_player_problem(problem, "offense")
  defense_problem = _make_drone3d_player_problem(problem, "defense")
  full_point = _combine_drone3d_player_points(offense_result.point, defense_result.point)
  evaluation = problem.evaluate(full_point, theta)
  offense_linearization = offense_problem.linearize_kkt(offense_result.point, theta)
  defense_linearization = defense_problem.linearize_kkt(defense_result.point, theta)
  offense_residual = offense_problem.kkt_residual(offense_result.point, theta, offense_linearization)
  defense_residual = defense_problem.kkt_residual(defense_result.point, theta, defense_linearization)
  evaluation.update(
    {
      "point": full_point,
      "solver_variant": solver_variant,
      "residual_norm": max(_tree_max_abs(offense_residual), _tree_max_abs(defense_residual)),
      "num_iterations": max(int(offense_result.num_iterations), int(defense_result.num_iterations)),
      "num_iterations_total": int(offense_result.num_iterations) + int(defense_result.num_iterations),
      "num_iterations_offense": int(offense_result.num_iterations),
      "num_iterations_defense": int(defense_result.num_iterations),
      "status": _aggregate_status_codes(int(offense_result.status), int(defense_result.status)),
      "status_offense": int(offense_result.status),
      "status_defense": int(defense_result.status),
      "residual_history": jnp.maximum(offense_result.residual_history, defense_result.residual_history),
      "line_search_alphas": _aggregate_line_search_alphas(
        offense_result.line_search_alphas,
        defense_result.line_search_alphas,
      ),
      "gmres_infos": jnp.maximum(offense_result.gmres_infos, defense_result.gmres_infos),
      "residual_history_offense": offense_result.residual_history,
      "residual_history_defense": defense_result.residual_history,
      "line_search_alphas_offense": offense_result.line_search_alphas,
      "line_search_alphas_defense": defense_result.line_search_alphas,
      "gmres_infos_offense": offense_result.gmres_infos,
      "gmres_infos_defense": defense_result.gmres_infos,
    },
  )
  if extra_fields:
    evaluation.update(extra_fields)
  return evaluation


def _aggregate_drone3d_exact_lq_player_solve_result(
  problem: Drone3DTreeProblem,
  theta: jnp.ndarray,
  offense_result: Drone3DExactLQPlayerSolve,
  defense_result: Drone3DExactLQPlayerSolve,
  *,
  variant2_constraint_mode: str = "exact_lq",
  box_constraints_active: bool = False,
) -> dict[str, Any]:
  full_point = _combine_drone3d_player_points(offense_result.point, defense_result.point)
  evaluation = problem.evaluate(full_point, theta)
  objective_lq = float(problem.lq_objective_no_barrier(full_point.primal, theta))
  one_alpha = jnp.array([1.0], dtype=F32)
  evaluation.update(
    {
      "point": full_point,
      "solver_variant": "player_separable_lq",
      "variant2_constraint_mode": variant2_constraint_mode,
      "box_constraints_active": box_constraints_active,
      "objective": objective_lq,
      "residual_norm": 0.0,
      "num_iterations": max(int(offense_result.iterations), int(defense_result.iterations)),
      "num_iterations_total": int(offense_result.iterations) + int(defense_result.iterations),
      "num_iterations_offense": int(offense_result.iterations),
      "num_iterations_defense": int(defense_result.iterations),
      "status": _aggregate_status_codes(int(offense_result.status), int(defense_result.status)),
      "status_offense": int(offense_result.status),
      "status_defense": int(defense_result.status),
      "status_reason_offense": offense_result.status_reason,
      "status_reason_defense": defense_result.status_reason,
      "primal_bound_violation_offense": float(offense_result.primal_bound_violation),
      "primal_bound_violation_defense": float(defense_result.primal_bound_violation),
      "primal_terminal_violation_offense": float(offense_result.primal_terminal_violation),
      "primal_terminal_violation_defense": float(defense_result.primal_terminal_violation),
      "primal_dynamics_violation_offense": float(offense_result.primal_dynamics_violation),
      "primal_dynamics_violation_defense": float(defense_result.primal_dynamics_violation),
      "active_stationarity_violation_offense": float(offense_result.active_stationarity_violation),
      "active_stationarity_violation_defense": float(defense_result.active_stationarity_violation),
      "active_set_repeated_offense": bool(offense_result.repeated_active_set),
      "active_set_repeated_defense": bool(defense_result.repeated_active_set),
      "active_set_iteration_log_offense": list(offense_result.active_set_iteration_log),
      "active_set_iteration_log_defense": list(defense_result.active_set_iteration_log),
      "kkt_solve_log_offense": list(offense_result.kkt_solve_log),
      "kkt_solve_log_defense": list(defense_result.kkt_solve_log),
      "residual_history": jnp.array([0.0], dtype=F32),
      "line_search_alphas": one_alpha,
      "gmres_infos": jnp.array([0.0], dtype=F32),
      "residual_history_offense": jnp.array([0.0], dtype=F32),
      "residual_history_defense": jnp.array([0.0], dtype=F32),
      "line_search_alphas_offense": one_alpha,
      "line_search_alphas_defense": one_alpha,
      "gmres_infos_offense": jnp.array([0.0], dtype=F32),
      "gmres_infos_defense": jnp.array([0.0], dtype=F32),
      "objective_offense_lq": float(offense_result.objective),
      "objective_defense_lq": float(defense_result.objective),
      "objective_lq": objective_lq,
      "forward_linear_mode": f"player_separable[{offense_result.mode},{defense_result.mode}]",
      "forward_linear_mode_offense": offense_result.mode,
      "forward_linear_mode_defense": defense_result.mode,
      "forward_status": _aggregate_status_codes(int(offense_result.status), int(defense_result.status)),
      "forward_residual_norm": 0.0,
      "forward_initial_residual_norm": 0.0,
      "forward_num_iterations": int(offense_result.iterations) + int(defense_result.iterations),
      "forward_regularization_retry_count": 0,
      "forward_line_search_zero_count": 0,
      "variant2_box_warm_start_mode": "not_applicable",
      "variant2_box_warm_start_fallback_reason": None,
      "control_box_active_lower_count_offense": int(offense_result.active_lower_count),
      "control_box_active_upper_count_offense": int(offense_result.active_upper_count),
      "control_box_active_lower_count_defense": int(defense_result.active_lower_count),
      "control_box_active_upper_count_defense": int(defense_result.active_upper_count),
      "control_box_warm_start_active_lower_count_offense": int(offense_result.warm_start_active_lower_count),
      "control_box_warm_start_active_upper_count_offense": int(offense_result.warm_start_active_upper_count),
      "control_box_warm_start_active_lower_count_defense": int(defense_result.warm_start_active_lower_count),
      "control_box_warm_start_active_upper_count_defense": int(defense_result.warm_start_active_upper_count),
    },
  )
  return evaluation


def _solve_drone3d_fixed_alpha_player_separable(
  problem: Drone3DTreeProblem,
  inner_cfg: TreeDiffMPCConfig,
  *,
  alpha_logits: jnp.ndarray | None = None,
  initial_point: EqualityGamePoint | None = None,
  solver_variant_label: str = "player_separable_lq",
  extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
  theta = problem.empty_alpha_logits() if alpha_logits is None else alpha_logits
  offense_problem = _make_drone3d_player_problem(problem, "offense")
  defense_problem = _make_drone3d_player_problem(problem, "defense")
  offense_solver = TreeDiffMPCSolver(offense_problem, inner_cfg)
  defense_solver = TreeDiffMPCSolver(defense_problem, inner_cfg)
  full_initial, warm_start_fields = _initialize_drone3d_box_variant2_points(
    problem,
    theta,
    initial_point=initial_point,
    prefer_exact_lq_projected_warm_start=True,
  )
  offense_initial, defense_initial = _split_drone3d_full_point_by_player(problem, full_initial)

  offense_result = offense_solver.solve_with_metadata(theta, initial_point=offense_initial)
  defense_result = defense_solver.solve_with_metadata(theta, initial_point=defense_initial)
  offense_diag = offense_solver.get_last_solve_point_diagnostics()
  defense_diag = defense_solver.get_last_solve_point_diagnostics()
  player_diag = _aggregate_player_diagnostics(
    offense_diag.get("forward", {}),
    defense_diag.get("forward", {}),
  )
  combined_extra_fields = dict(extra_fields or {})
  combined_extra_fields.update(warm_start_fields)
  combined_extra_fields.update(
    {
      key: value
      for key, value in player_diag.items()
      if key.startswith("forward_")
    }
  )
  return _aggregate_drone3d_player_solve_result(
    problem,
    theta,
    offense_result,
    defense_result,
    solver_variant=solver_variant_label,
    extra_fields=combined_extra_fields,
  )


def _solve_drone3d_fixed_alpha_player_separable_lq(
  problem: Drone3DTreeProblem,
  inner_cfg: TreeDiffMPCConfig,
  *,
  alpha_logits: jnp.ndarray | None = None,
  initial_point: EqualityGamePoint | None = None,
) -> dict[str, Any]:
  offense_problem = _make_drone3d_player_problem(problem, "offense")
  defense_problem = _make_drone3d_player_problem(problem, "defense")
  if offense_problem.supports_exact_lq_solver() and defense_problem.supports_exact_lq_solver():
    del inner_cfg, initial_point
    theta = problem.empty_alpha_logits() if alpha_logits is None else alpha_logits
    offense_result = offense_problem.solve_exact_lq(theta)
    defense_result = defense_problem.solve_exact_lq(theta)
    return _aggregate_drone3d_exact_lq_player_solve_result(problem, theta, offense_result, defense_result)
  if (
    offense_problem.supports_exact_control_box_terminal_lq_solver()
    and defense_problem.supports_exact_control_box_terminal_lq_solver()
  ):
    theta = problem.empty_alpha_logits() if alpha_logits is None else alpha_logits
    if initial_point is None:
      offense_initial = None
      defense_initial = None
    else:
      offense_initial, defense_initial = _split_drone3d_full_point_by_player(problem, initial_point)
    backend = getattr(inner_cfg, "drone3d_box_terminal_backend", "riccati_first")
    if backend not in ("riccati_first", "sparse", "active_tree"):
      raise ValueError(
        "TreeDiffMPCConfig.drone3d_box_terminal_backend must be 'riccati_first', 'sparse', or 'active_tree'.",
      )
    if backend == "active_tree":
      offense_result = offense_problem.solve_exact_control_box_terminal_lq_active_tree(
        theta,
        max_active_set_iterations=inner_cfg.max_sqp_iterations,
        control_tolerance=inner_cfg.residual_tolerance,
        initial_point=offense_initial,
      )
      defense_result = defense_problem.solve_exact_control_box_terminal_lq_active_tree(
        theta,
        max_active_set_iterations=inner_cfg.max_sqp_iterations,
        control_tolerance=inner_cfg.residual_tolerance,
        initial_point=defense_initial,
      )
    else:
      prefer_riccati = backend == "riccati_first"
      offense_result = offense_problem.solve_exact_control_box_terminal_lq_sparse(
        theta,
        max_active_set_iterations=inner_cfg.max_sqp_iterations,
        control_tolerance=inner_cfg.residual_tolerance,
        prefer_riccati_active_set=prefer_riccati,
        initial_point=offense_initial,
      )
      defense_result = defense_problem.solve_exact_control_box_terminal_lq_sparse(
        theta,
        max_active_set_iterations=inner_cfg.max_sqp_iterations,
        control_tolerance=inner_cfg.residual_tolerance,
        prefer_riccati_active_set=prefer_riccati,
        initial_point=defense_initial,
      )
    terminal_mode = (
      "exact_control_box_terminal_lq_riccati"
      if (
        offense_result.mode == "player_separable_lq_box_terminal_riccati_active_set"
        and defense_result.mode == "player_separable_lq_box_terminal_riccati_active_set"
      )
      else "exact_control_box_terminal_lq_active_tree"
      if (
        offense_result.mode == "player_separable_lq_box_terminal_active_tree"
        and defense_result.mode == "player_separable_lq_box_terminal_active_tree"
      )
      else "exact_control_box_terminal_lq_sparse"
    )
    return _aggregate_drone3d_exact_lq_player_solve_result(
      problem,
      theta,
      offense_result,
      defense_result,
      variant2_constraint_mode=terminal_mode,
      box_constraints_active=True,
    )
  if (
    offense_problem.supports_exact_control_box_lq_solver()
    and defense_problem.supports_exact_control_box_lq_solver()
  ):
    theta = problem.empty_alpha_logits() if alpha_logits is None else alpha_logits
    if initial_point is None:
      offense_initial = None
      defense_initial = None
    else:
      offense_initial, defense_initial = _split_drone3d_full_point_by_player(problem, initial_point)
    box_lq_backend = getattr(inner_cfg, "drone3d_box_lq_backend", "active_set")
    if box_lq_backend not in ("active_set", "ipm"):
      raise ValueError("TreeDiffMPCConfig.drone3d_box_lq_backend must be 'active_set' or 'ipm'.")
    if box_lq_backend == "ipm":
      offense_result = offense_problem.solve_exact_control_box_lq_ipm(
        theta,
        max_iterations=inner_cfg.max_sqp_iterations,
        tolerance=inner_cfg.residual_tolerance,
        initial_point=offense_initial,
      )
      defense_result = defense_problem.solve_exact_control_box_lq_ipm(
        theta,
        max_iterations=inner_cfg.max_sqp_iterations,
        tolerance=inner_cfg.residual_tolerance,
        initial_point=defense_initial,
      )
      if int(offense_result.status) != int(SolverStatus.SUCCESS):
        offense_polish = offense_problem.solve_exact_control_box_lq(
          theta,
          max_active_set_iterations=inner_cfg.max_sqp_iterations,
          control_tolerance=inner_cfg.residual_tolerance,
          initial_point=offense_result.point,
        )
        if (
          int(offense_polish.status) == int(SolverStatus.SUCCESS)
          or float(offense_polish.active_stationarity_violation) < float(offense_result.active_stationarity_violation)
        ):
          offense_result = offense_polish
      if int(defense_result.status) != int(SolverStatus.SUCCESS):
        defense_polish = defense_problem.solve_exact_control_box_lq(
          theta,
          max_active_set_iterations=inner_cfg.max_sqp_iterations,
          control_tolerance=inner_cfg.residual_tolerance,
          initial_point=defense_result.point,
        )
        if (
          int(defense_polish.status) == int(SolverStatus.SUCCESS)
          or float(defense_polish.active_stationarity_violation) < float(defense_result.active_stationarity_violation)
        ):
          defense_result = defense_polish
      box_lq_mode = "exact_control_box_lq_ipm"
    else:
      offense_result = offense_problem.solve_exact_control_box_lq(
        theta,
        max_active_set_iterations=inner_cfg.max_sqp_iterations,
        control_tolerance=inner_cfg.residual_tolerance,
        initial_point=offense_initial,
      )
      defense_result = defense_problem.solve_exact_control_box_lq(
        theta,
        max_active_set_iterations=inner_cfg.max_sqp_iterations,
        control_tolerance=inner_cfg.residual_tolerance,
        initial_point=defense_initial,
      )
      box_lq_mode = "exact_control_box_lq"
    return _aggregate_drone3d_exact_lq_player_solve_result(
      problem,
      theta,
      offense_result,
      defense_result,
      variant2_constraint_mode=box_lq_mode,
      box_constraints_active=True,
    )
  if problem.has_box_inequality_constraints and not problem.cfg.squash_controls:
    return _solve_drone3d_fixed_alpha_player_separable(
      problem,
      inner_cfg,
      alpha_logits=alpha_logits,
      initial_point=initial_point,
      solver_variant_label="player_separable_lq",
      extra_fields={
        "variant2_constraint_mode": "box_barrier",
        "box_constraints_active": True,
      },
    )
  raise ValueError(
    "Drone3D Variant 2 currently requires squash_controls=False. "
    "Equality-only instances use the exact Riccati path; control-box-only instances with or without terminal "
    "velocity equalities use exact active-set paths; remaining constrained cases use the constrained barrier fallback.",
  )


def _solve_drone3d_bilevel_player_separable(
  problem: Drone3DTreeProblem,
  cfg: Drone3DBilevelConfig,
  *,
  alpha_logits: jnp.ndarray | None = None,
  warm_point: EqualityGamePoint | None = None,
  progress_callback: Callable[[dict[str, Any]], None] | None = None,
  solver_variant_label: str = "player_separable_lq",
  extra_history_fields: dict[str, Any] | None = None,
  extra_evaluation_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
  offense_problem = _make_drone3d_player_problem(problem, "offense")
  defense_problem = _make_drone3d_player_problem(problem, "defense")
  offense_solver = TreeDiffMPCSolver(offense_problem, cfg.inner)
  defense_solver = TreeDiffMPCSolver(defense_problem, cfg.inner)
  theta = (
    problem.init_alpha_logits(
      seed=cfg.outer.seed,
      init_scale=cfg.outer.alpha_init_scale,
      mode=cfg.outer.alpha_init_mode,
    )
    if alpha_logits is None
    else alpha_logits
  )
  full_warm, warm_start_fields = _initialize_drone3d_box_variant2_points(
    problem,
    theta,
    initial_point=warm_point,
    prefer_exact_lq_projected_warm_start=True,
  )
  offense_warm, defense_warm = _split_drone3d_full_point_by_player(problem, full_warm)

  optimizer = _make_optimizer(cfg.outer.optimizer, cfg.outer.lr_alpha)
  opt_state = optimizer.init(theta)
  history: list[dict[str, Any]] = []
  previous_loss: float | None = None
  outer_status = "max_steps"

  for outer_step in range(cfg.outer.steps):
    outer_step_start = time.perf_counter()

    def outer_objective(theta_value: jnp.ndarray) -> tuple[jnp.ndarray, tuple[EqualityGamePoint, EqualityGamePoint]]:
      offense_point = offense_solver.solve_point(theta_value, offense_warm)
      defense_point = defense_solver.solve_point(theta_value, defense_warm)
      combined_point = _combine_drone3d_player_points(offense_point, defense_point)
      return problem.objective(combined_point.primal, theta_value), (offense_point, defense_point)

    (loss_value, solved_points), grad_theta = jax.value_and_grad(outer_objective, has_aux=True)(theta)
    solved_offense, solved_defense = solved_points
    updates, opt_state = optimizer.update(grad_theta, opt_state, theta)
    theta = optax.apply_updates(theta, updates)
    if cfg.outer.alpha_logit_clip is not None:
      theta = jnp.clip(theta, -cfg.outer.alpha_logit_clip, cfg.outer.alpha_logit_clip)
    offense_warm = jax.tree_util.tree_map(jax.lax.stop_gradient, solved_offense)
    defense_warm = jax.tree_util.tree_map(jax.lax.stop_gradient, solved_defense)

    alpha = build_alpha_from_logits(theta, problem.tree)
    leaf_type_probs, node_public_probs, _, _ = propagate_type_probabilities(alpha, problem.topology, problem.prior)
    leaf_probs = node_public_probs[problem.topology.leaf_nodes]
    leaf_type_probs = leaf_type_probs[problem.topology.leaf_nodes]
    leaf_beliefs = leaf_type_probs / jnp.clip(leaf_probs[:, None], 1e-8, None)

    offense_diag = offense_solver.get_last_solve_point_diagnostics()
    defense_diag = defense_solver.get_last_solve_point_diagnostics()
    forward_diag = _aggregate_player_diagnostics(
      offense_diag.get("forward", {}),
      defense_diag.get("forward", {}),
    )
    backward_diag = _aggregate_player_diagnostics(
      offense_diag.get("backward", {}),
      defense_diag.get("backward", {}),
    )
    history_entry = {
      "outer_step": outer_step + 1,
      "loss": float(loss_value),
      "loss_change": None if previous_loss is None else float(loss_value) - previous_loss,
      "alpha_grad_norm": _tree_l2_norm(grad_theta),
      "step_elapsed_sec": time.perf_counter() - outer_step_start,
      "solver_variant": solver_variant_label,
      "forward_solve_sec": forward_diag["forward_solve_sec"],
      "forward_num_iterations": forward_diag["forward_num_iterations"],
      "forward_initial_residual_norm": forward_diag["forward_initial_residual_norm"],
      "forward_residual_norm": forward_diag["forward_residual_norm"],
      "forward_status": forward_diag["forward_status"],
      "forward_linear_mode": forward_diag["forward_linear_mode"],
      "forward_linear_mode_offense": forward_diag["forward_linear_mode_offense"],
      "forward_linear_mode_defense": forward_diag["forward_linear_mode_defense"],
      "forward_used_singularity_aware_partial_elimination": forward_diag["forward_used_singularity_aware_partial_elimination"],
      "forward_used_singularity_aware_partial_elimination_offense": forward_diag["forward_used_singularity_aware_partial_elimination_offense"],
      "forward_used_singularity_aware_partial_elimination_defense": forward_diag["forward_used_singularity_aware_partial_elimination_defense"],
      "forward_approximate_pruning_used": forward_diag["forward_approximate_pruning_used"],
      "forward_approximate_pruning_used_offense": forward_diag["forward_approximate_pruning_used_offense"],
      "forward_approximate_pruning_used_defense": forward_diag["forward_approximate_pruning_used_defense"],
      "forward_solve_sec_offense": forward_diag["forward_solve_sec_offense"],
      "forward_solve_sec_defense": forward_diag["forward_solve_sec_defense"],
      "forward_num_iterations_offense": forward_diag["forward_num_iterations_offense"],
      "forward_num_iterations_defense": forward_diag["forward_num_iterations_defense"],
      "forward_initial_residual_norm_offense": forward_diag["forward_initial_residual_norm_offense"],
      "forward_initial_residual_norm_defense": forward_diag["forward_initial_residual_norm_defense"],
      "forward_line_search_last_alpha": forward_diag["forward_line_search_last_alpha"],
      "forward_line_search_min_alpha": forward_diag["forward_line_search_min_alpha"],
      "forward_line_search_mean_alpha": forward_diag["forward_line_search_mean_alpha"],
      "forward_line_search_zero_count": forward_diag["forward_line_search_zero_count"],
      "forward_line_search_stalled": forward_diag["forward_line_search_stalled"],
      "forward_line_search_last_alpha_offense": forward_diag["forward_line_search_last_alpha_offense"],
      "forward_line_search_last_alpha_defense": forward_diag["forward_line_search_last_alpha_defense"],
      "forward_line_search_min_alpha_offense": forward_diag["forward_line_search_min_alpha_offense"],
      "forward_line_search_min_alpha_defense": forward_diag["forward_line_search_min_alpha_defense"],
      "forward_line_search_mean_alpha_offense": forward_diag["forward_line_search_mean_alpha_offense"],
      "forward_line_search_mean_alpha_defense": forward_diag["forward_line_search_mean_alpha_defense"],
      "forward_line_search_zero_count_offense": forward_diag["forward_line_search_zero_count_offense"],
      "forward_line_search_zero_count_defense": forward_diag["forward_line_search_zero_count_defense"],
      "forward_line_search_stalled_offense": forward_diag["forward_line_search_stalled_offense"],
      "forward_line_search_stalled_defense": forward_diag["forward_line_search_stalled_defense"],
      "forward_regularization_initial": forward_diag["forward_regularization_initial"],
      "forward_regularization_last": forward_diag["forward_regularization_last"],
      "forward_regularization_min": forward_diag["forward_regularization_min"],
      "forward_regularization_max": forward_diag["forward_regularization_max"],
      "forward_regularization_retry_count": forward_diag["forward_regularization_retry_count"],
      "forward_regularization_initial_offense": forward_diag["forward_regularization_initial_offense"],
      "forward_regularization_initial_defense": forward_diag["forward_regularization_initial_defense"],
      "forward_regularization_last_offense": forward_diag["forward_regularization_last_offense"],
      "forward_regularization_last_defense": forward_diag["forward_regularization_last_defense"],
      "forward_regularization_min_offense": forward_diag["forward_regularization_min_offense"],
      "forward_regularization_min_defense": forward_diag["forward_regularization_min_defense"],
      "forward_regularization_max_offense": forward_diag["forward_regularization_max_offense"],
      "forward_regularization_max_defense": forward_diag["forward_regularization_max_defense"],
      "forward_regularization_retry_count_offense": forward_diag["forward_regularization_retry_count_offense"],
      "forward_regularization_retry_count_defense": forward_diag["forward_regularization_retry_count_defense"],
      "forward_regularization_retry_last_offense": forward_diag["forward_regularization_retry_last_offense"],
      "forward_regularization_retry_last_defense": forward_diag["forward_regularization_retry_last_defense"],
      "backward_solve_sec": backward_diag["backward_solve_sec"],
      "backward_mode": backward_diag["backward_mode"],
      "backward_reduced_dual_attempted": backward_diag["backward_reduced_dual_attempted"],
      "backward_fallback": backward_diag["backward_fallback"],
      "backward_reduced_dual_residual_norm": backward_diag["backward_reduced_dual_residual_norm"],
      "backward_reduced_dual_regularization": backward_diag["backward_reduced_dual_regularization"],
      "backward_solve_sec_offense": backward_diag["backward_solve_sec_offense"],
      "backward_solve_sec_defense": backward_diag["backward_solve_sec_defense"],
      "backward_mode_offense": backward_diag["backward_mode_offense"],
      "backward_mode_defense": backward_diag["backward_mode_defense"],
      "backward_reduced_dual_regularization_offense": backward_diag["backward_reduced_dual_regularization_offense"],
      "backward_reduced_dual_regularization_defense": backward_diag["backward_reduced_dual_regularization_defense"],
      "leaf_belief_min": float(jnp.min(leaf_beliefs)) if leaf_beliefs.size else 1.0,
      "leaf_belief_max": float(jnp.max(leaf_beliefs)) if leaf_beliefs.size else 1.0,
      "frontier_belief_min": float(jnp.min(leaf_beliefs)) if leaf_beliefs.size else 1.0,
      "frontier_belief_max": float(jnp.max(leaf_beliefs)) if leaf_beliefs.size else 1.0,
    }
    if extra_history_fields:
      history_entry.update(extra_history_fields)
    history_entry.update(warm_start_fields)
    history.append(history_entry)
    if progress_callback is not None:
      progress_callback(history_entry)
    if outer_step + 1 >= max(1, cfg.outer.min_steps):
      stop_reasons: list[str] = []
      if (
        cfg.outer.grad_tolerance is not None
        and history_entry["alpha_grad_norm"] <= cfg.outer.grad_tolerance
      ):
        stop_reasons.append("grad_tolerance")
      loss_change = history_entry["loss_change"]
      if (
        cfg.outer.loss_change_tolerance is not None
        and loss_change is not None
        and abs(loss_change) <= cfg.outer.loss_change_tolerance
        and (
          cfg.outer.grad_tolerance is None
          or history_entry["alpha_grad_norm"] <= cfg.outer.grad_tolerance
        )
      ):
        stop_reasons.append("loss_change_tolerance")
      if stop_reasons:
        outer_status = "+".join(stop_reasons)
        break
    previous_loss = float(loss_value)

  offense_result = offense_solver.solve_with_metadata(theta, initial_point=offense_warm)
  defense_result = defense_solver.solve_with_metadata(theta, initial_point=defense_warm)
  evaluation = _aggregate_drone3d_player_solve_result(
    problem,
    theta,
    offense_result,
    defense_result,
    solver_variant=solver_variant_label,
    extra_fields={
      **warm_start_fields,
      **(extra_evaluation_fields or {}),
    },
  )
  evaluation.update(
    {
      "history": history,
      "outer_status": outer_status,
      "outer_steps_used": len(history),
    },
  )
  return evaluation


def _solve_drone3d_bilevel_exact_control_box_terminal_lq_sparse(
  problem: Drone3DTreeProblem,
  cfg: Drone3DBilevelConfig,
  *,
  alpha_logits: jnp.ndarray | None = None,
  progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
  theta = (
    problem.init_alpha_logits(
      seed=cfg.outer.seed,
      init_scale=cfg.outer.alpha_init_scale,
      mode=cfg.outer.alpha_init_mode,
    )
    if alpha_logits is None
    else alpha_logits
  )
  optimizer = _make_optimizer(cfg.outer.optimizer, cfg.outer.lr_alpha)
  opt_state = optimizer.init(theta)
  history: list[dict[str, Any]] = []
  previous_loss: float | None = None
  outer_status = "max_steps"
  last_result: dict[str, Any] | None = None
  warm_point: EqualityGamePoint | None = None

  def envelope_loss(theta_value: jnp.ndarray, point: EqualityGamePoint) -> jnp.ndarray:
    fixed_primal = jax.tree_util.tree_map(jax.lax.stop_gradient, point.primal)
    return problem.lq_objective_no_barrier(fixed_primal, theta_value)

  for outer_step in range(cfg.outer.steps):
    outer_step_start = time.perf_counter()
    solve_start = time.perf_counter()
    result = _solve_drone3d_fixed_alpha_player_separable_lq(
      problem,
      cfg.inner,
      alpha_logits=theta,
      initial_point=warm_point,
    )
    solve_elapsed = time.perf_counter() - solve_start
    last_result = result
    max_u = float(jnp.max(jnp.abs(result["u_seq"]))) if result["u_seq"].size else 0.0
    max_v = float(jnp.max(jnp.abs(result["v_seq"]))) if result["v_seq"].size else 0.0
    terminal_offense = float(jnp.max(jnp.abs(result["terminal_velocities_offense"]))) if result["terminal_velocities_offense"].size else 0.0
    terminal_defense = float(jnp.max(jnp.abs(result["terminal_velocities_defense"]))) if result["terminal_velocities_defense"].size else 0.0
    if int(result["status"]) != int(SolverStatus.SUCCESS):
      alpha = build_alpha_from_logits(theta, problem.tree)
      leaf_type_probs, node_public_probs, _, _ = propagate_type_probabilities(alpha, problem.topology, problem.prior)
      leaf_probs = node_public_probs[problem.topology.leaf_nodes]
      leaf_type_probs = leaf_type_probs[problem.topology.leaf_nodes]
      leaf_beliefs = leaf_type_probs / jnp.clip(leaf_probs[:, None], 1e-8, None)
      history_entry = {
        "outer_step": outer_step + 1,
        "loss": float(result.get("objective_lq", result["objective"])),
        "loss_change": None if previous_loss is None else float(result.get("objective_lq", result["objective"])) - previous_loss,
        "alpha_grad_norm": float("nan"),
        "step_elapsed_sec": time.perf_counter() - outer_step_start,
        "solver_variant": "player_separable_lq",
        "variant2_constraint_mode": result["variant2_constraint_mode"],
        "box_constraints_active": True,
        "forward_solve_sec": solve_elapsed,
        "forward_num_iterations": int(result["forward_num_iterations"]),
        "forward_initial_residual_norm": float(result["forward_initial_residual_norm"]),
        "forward_residual_norm": float(result["forward_residual_norm"]),
        "forward_status": int(result["forward_status"]),
        "forward_linear_mode": str(result["forward_linear_mode"]),
        "forward_linear_mode_offense": str(result["forward_linear_mode_offense"]),
        "forward_linear_mode_defense": str(result["forward_linear_mode_defense"]),
        "forward_solve_sec_offense": 0.5 * solve_elapsed,
        "forward_solve_sec_defense": 0.5 * solve_elapsed,
        "forward_num_iterations_offense": int(result["num_iterations_offense"]),
        "forward_num_iterations_defense": int(result["num_iterations_defense"]),
        "control_box_active_lower_count_offense": int(result["control_box_active_lower_count_offense"]),
        "control_box_active_upper_count_offense": int(result["control_box_active_upper_count_offense"]),
        "control_box_active_lower_count_defense": int(result["control_box_active_lower_count_defense"]),
        "control_box_active_upper_count_defense": int(result["control_box_active_upper_count_defense"]),
        "control_box_warm_start_active_lower_count_offense": int(result["control_box_warm_start_active_lower_count_offense"]),
        "control_box_warm_start_active_upper_count_offense": int(result["control_box_warm_start_active_upper_count_offense"]),
        "control_box_warm_start_active_lower_count_defense": int(result["control_box_warm_start_active_lower_count_defense"]),
        "control_box_warm_start_active_upper_count_defense": int(result["control_box_warm_start_active_upper_count_defense"]),
        "max_offense_accel_abs": max_u,
        "max_defense_accel_abs": max_v,
        "max_terminal_velocity_offense_abs": terminal_offense,
        "max_terminal_velocity_defense_abs": terminal_defense,
        "objective_lq": float(result.get("objective_lq", result["objective"])),
        "status": int(result["status"]),
        "status_offense": int(result["status_offense"]),
        "status_defense": int(result["status_defense"]),
        "status_reason_offense": str(result.get("status_reason_offense", "")),
        "status_reason_defense": str(result.get("status_reason_defense", "")),
        "primal_bound_violation_offense": float(result.get("primal_bound_violation_offense", 0.0)),
        "primal_bound_violation_defense": float(result.get("primal_bound_violation_defense", 0.0)),
        "primal_terminal_violation_offense": float(result.get("primal_terminal_violation_offense", 0.0)),
        "primal_terminal_violation_defense": float(result.get("primal_terminal_violation_defense", 0.0)),
        "primal_dynamics_violation_offense": float(result.get("primal_dynamics_violation_offense", 0.0)),
        "primal_dynamics_violation_defense": float(result.get("primal_dynamics_violation_defense", 0.0)),
        "active_stationarity_violation_offense": float(result.get("active_stationarity_violation_offense", 0.0)),
        "active_stationarity_violation_defense": float(result.get("active_stationarity_violation_defense", 0.0)),
        "active_set_repeated_offense": bool(result.get("active_set_repeated_offense", False)),
        "active_set_repeated_defense": bool(result.get("active_set_repeated_defense", False)),
        "active_set_iteration_log_offense": result.get("active_set_iteration_log_offense", []),
        "active_set_iteration_log_defense": result.get("active_set_iteration_log_defense", []),
        "kkt_solve_log_offense": result.get("kkt_solve_log_offense", []),
        "kkt_solve_log_defense": result.get("kkt_solve_log_defense", []),
        "leaf_belief_min": float(jnp.min(leaf_beliefs)) if leaf_beliefs.size else 1.0,
        "leaf_belief_max": float(jnp.max(leaf_beliefs)) if leaf_beliefs.size else 1.0,
        "frontier_belief_min": float(jnp.min(leaf_beliefs)) if leaf_beliefs.size else 1.0,
        "frontier_belief_max": float(jnp.max(leaf_beliefs)) if leaf_beliefs.size else 1.0,
        "path_probability_min": float(jnp.min(leaf_probs)) if leaf_probs.size else 1.0,
        "path_probability_max": float(jnp.max(leaf_probs)) if leaf_probs.size else 1.0,
        "alpha_logit_min": float(jnp.min(theta)) if theta.size else 0.0,
        "alpha_logit_max": float(jnp.max(theta)) if theta.size else 0.0,
        "alpha_logit_max_abs": float(jnp.max(jnp.abs(theta))) if theta.size else 0.0,
        "gradient_mode": "not_computed_inner_solve_failed",
        "backward_solve_sec": None,
        "backward_mode": "not_computed_inner_solve_failed",
        "backward_reduced_dual_attempted": False,
        "backward_fallback": False,
        "backward_reduced_dual_residual_norm": None,
        "backward_reduced_dual_regularization": None,
        "backward_solve_sec_offense": None,
        "backward_solve_sec_defense": None,
        "backward_mode_offense": "not_computed_inner_solve_failed",
        "backward_mode_defense": "not_computed_inner_solve_failed",
        "backward_reduced_dual_regularization_offense": None,
        "backward_reduced_dual_regularization_defense": None,
      }
      history.append(history_entry)
      if progress_callback is not None:
        progress_callback(history_entry)
      outer_status = "inner_solve_failed"
      break
    point = result["point"]
    warm_point = jax.tree_util.tree_map(jax.lax.stop_gradient, point)
    loss_value, grad_theta = jax.value_and_grad(lambda theta_value: envelope_loss(theta_value, point))(theta)
    updates, opt_state = optimizer.update(grad_theta, opt_state, theta)
    theta = optax.apply_updates(theta, updates)
    if cfg.outer.alpha_logit_clip is not None:
      theta = jnp.clip(theta, -cfg.outer.alpha_logit_clip, cfg.outer.alpha_logit_clip)

    alpha = build_alpha_from_logits(theta, problem.tree)
    leaf_type_probs, node_public_probs, _, _ = propagate_type_probabilities(alpha, problem.topology, problem.prior)
    leaf_probs = node_public_probs[problem.topology.leaf_nodes]
    leaf_type_probs = leaf_type_probs[problem.topology.leaf_nodes]
    leaf_beliefs = leaf_type_probs / jnp.clip(leaf_probs[:, None], 1e-8, None)
    history_entry = {
      "outer_step": outer_step + 1,
      "loss": float(loss_value),
      "loss_change": None if previous_loss is None else float(loss_value) - previous_loss,
      "alpha_grad_norm": _tree_l2_norm(grad_theta),
      "step_elapsed_sec": time.perf_counter() - outer_step_start,
      "solver_variant": "player_separable_lq",
      "variant2_constraint_mode": result["variant2_constraint_mode"],
      "box_constraints_active": True,
      "forward_solve_sec": solve_elapsed,
      "forward_num_iterations": int(result["forward_num_iterations"]),
      "forward_initial_residual_norm": float(result["forward_initial_residual_norm"]),
      "forward_residual_norm": float(result["forward_residual_norm"]),
      "forward_status": int(result["forward_status"]),
      "forward_linear_mode": str(result["forward_linear_mode"]),
      "forward_linear_mode_offense": str(result["forward_linear_mode_offense"]),
      "forward_linear_mode_defense": str(result["forward_linear_mode_defense"]),
      "forward_solve_sec_offense": 0.5 * solve_elapsed,
      "forward_solve_sec_defense": 0.5 * solve_elapsed,
      "forward_num_iterations_offense": int(result["num_iterations_offense"]),
      "forward_num_iterations_defense": int(result["num_iterations_defense"]),
      "control_box_active_lower_count_offense": int(result["control_box_active_lower_count_offense"]),
      "control_box_active_upper_count_offense": int(result["control_box_active_upper_count_offense"]),
      "control_box_active_lower_count_defense": int(result["control_box_active_lower_count_defense"]),
      "control_box_active_upper_count_defense": int(result["control_box_active_upper_count_defense"]),
      "control_box_warm_start_active_lower_count_offense": int(result["control_box_warm_start_active_lower_count_offense"]),
      "control_box_warm_start_active_upper_count_offense": int(result["control_box_warm_start_active_upper_count_offense"]),
      "control_box_warm_start_active_lower_count_defense": int(result["control_box_warm_start_active_lower_count_defense"]),
      "control_box_warm_start_active_upper_count_defense": int(result["control_box_warm_start_active_upper_count_defense"]),
      "max_offense_accel_abs": max_u,
      "max_defense_accel_abs": max_v,
      "max_terminal_velocity_offense_abs": terminal_offense,
      "max_terminal_velocity_defense_abs": terminal_defense,
      "objective_lq": float(result.get("objective_lq", result["objective"])),
      "status": int(result["status"]),
      "status_offense": int(result["status_offense"]),
      "status_defense": int(result["status_defense"]),
      "status_reason_offense": str(result.get("status_reason_offense", "")),
      "status_reason_defense": str(result.get("status_reason_defense", "")),
      "primal_bound_violation_offense": float(result.get("primal_bound_violation_offense", 0.0)),
      "primal_bound_violation_defense": float(result.get("primal_bound_violation_defense", 0.0)),
      "primal_terminal_violation_offense": float(result.get("primal_terminal_violation_offense", 0.0)),
      "primal_terminal_violation_defense": float(result.get("primal_terminal_violation_defense", 0.0)),
      "primal_dynamics_violation_offense": float(result.get("primal_dynamics_violation_offense", 0.0)),
      "primal_dynamics_violation_defense": float(result.get("primal_dynamics_violation_defense", 0.0)),
      "active_stationarity_violation_offense": float(result.get("active_stationarity_violation_offense", 0.0)),
      "active_stationarity_violation_defense": float(result.get("active_stationarity_violation_defense", 0.0)),
      "active_set_repeated_offense": bool(result.get("active_set_repeated_offense", False)),
      "active_set_repeated_defense": bool(result.get("active_set_repeated_defense", False)),
      "active_set_iteration_log_offense": result.get("active_set_iteration_log_offense", []),
      "active_set_iteration_log_defense": result.get("active_set_iteration_log_defense", []),
      "kkt_solve_log_offense": result.get("kkt_solve_log_offense", []),
      "kkt_solve_log_defense": result.get("kkt_solve_log_defense", []),
      "leaf_belief_min": float(jnp.min(leaf_beliefs)) if leaf_beliefs.size else 1.0,
      "leaf_belief_max": float(jnp.max(leaf_beliefs)) if leaf_beliefs.size else 1.0,
      "frontier_belief_min": float(jnp.min(leaf_beliefs)) if leaf_beliefs.size else 1.0,
      "frontier_belief_max": float(jnp.max(leaf_beliefs)) if leaf_beliefs.size else 1.0,
      "path_probability_min": float(jnp.min(leaf_probs)) if leaf_probs.size else 1.0,
      "path_probability_max": float(jnp.max(leaf_probs)) if leaf_probs.size else 1.0,
      "alpha_logit_min": float(jnp.min(theta)) if theta.size else 0.0,
      "alpha_logit_max": float(jnp.max(theta)) if theta.size else 0.0,
      "alpha_logit_max_abs": float(jnp.max(jnp.abs(theta))) if theta.size else 0.0,
      "gradient_mode": "envelope_fixed_active_set",
      "backward_solve_sec": None,
      "backward_mode": "envelope_fixed_active_set",
      "backward_reduced_dual_attempted": False,
      "backward_fallback": False,
      "backward_reduced_dual_residual_norm": None,
      "backward_reduced_dual_regularization": None,
      "backward_solve_sec_offense": None,
      "backward_solve_sec_defense": None,
      "backward_mode_offense": "envelope_fixed_active_set",
      "backward_mode_defense": "envelope_fixed_active_set",
      "backward_reduced_dual_regularization_offense": None,
      "backward_reduced_dual_regularization_defense": None,
    }
    history.append(history_entry)
    if progress_callback is not None:
      progress_callback(history_entry)
    if outer_step + 1 >= max(1, cfg.outer.min_steps):
      stop_reasons: list[str] = []
      if (
        cfg.outer.grad_tolerance is not None
        and history_entry["alpha_grad_norm"] <= cfg.outer.grad_tolerance
      ):
        stop_reasons.append("grad_tolerance")
      loss_change = history_entry["loss_change"]
      if (
        cfg.outer.loss_change_tolerance is not None
        and loss_change is not None
        and abs(loss_change) <= cfg.outer.loss_change_tolerance
        and (
          cfg.outer.grad_tolerance is None
          or history_entry["alpha_grad_norm"] <= cfg.outer.grad_tolerance
        )
      ):
        stop_reasons.append("loss_change_tolerance")
      if stop_reasons:
        outer_status = "+".join(stop_reasons)
        break
    previous_loss = float(loss_value)

  if outer_status == "inner_solve_failed" and last_result is not None:
    final_result = last_result
  else:
    final_result = _solve_drone3d_fixed_alpha_player_separable_lq(
      problem,
      cfg.inner,
      alpha_logits=theta,
      initial_point=warm_point,
    )
  final_result.update(
    {
      "history": history,
      "outer_status": outer_status,
      "outer_steps_used": len(history),
      "gradient_mode": "envelope_fixed_active_set",
    },
  )
  return final_result


def _solve_drone3d_bilevel_player_separable_lq(
  problem: Drone3DTreeProblem,
  cfg: Drone3DBilevelConfig,
  *,
  alpha_logits: jnp.ndarray | None = None,
  warm_point: EqualityGamePoint | None = None,
  progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
  offense_problem = _make_drone3d_player_problem(problem, "offense")
  defense_problem = _make_drone3d_player_problem(problem, "defense")
  if (
    offense_problem.supports_exact_control_box_terminal_lq_solver()
    and defense_problem.supports_exact_control_box_terminal_lq_solver()
  ):
    del warm_point
    return _solve_drone3d_bilevel_exact_control_box_terminal_lq_sparse(
      problem,
      cfg,
      alpha_logits=alpha_logits,
      progress_callback=progress_callback,
    )
  if (
    offense_problem.supports_exact_control_box_lq_solver()
    and defense_problem.supports_exact_control_box_lq_solver()
  ):
    del warm_point
    return _solve_drone3d_bilevel_exact_control_box_terminal_lq_sparse(
      problem,
      cfg,
      alpha_logits=alpha_logits,
      progress_callback=progress_callback,
    )
  if problem.has_box_inequality_constraints and not problem.cfg.squash_controls:
    return _solve_drone3d_bilevel_player_separable(
      problem,
      cfg,
      alpha_logits=alpha_logits,
      warm_point=warm_point,
      progress_callback=progress_callback,
      solver_variant_label="player_separable_lq",
      extra_history_fields={
        "variant2_constraint_mode": "box_barrier",
        "box_constraints_active": True,
      },
      extra_evaluation_fields={
        "variant2_constraint_mode": "box_barrier",
        "box_constraints_active": True,
      },
    )
  if not offense_problem.supports_exact_lq_solver() or not defense_problem.supports_exact_lq_solver():
    raise ValueError(
      "Drone3D Variant 2 exact LQ solve requires squash_controls=False and no active box constraints.",
    )
  del warm_point

  theta = (
    problem.init_alpha_logits(
      seed=cfg.outer.seed,
      init_scale=cfg.outer.alpha_init_scale,
      mode=cfg.outer.alpha_init_mode,
    )
    if alpha_logits is None
    else alpha_logits
  )
  optimizer = _make_optimizer(cfg.outer.optimizer, cfg.outer.lr_alpha)
  opt_state = optimizer.init(theta)
  history: list[dict[str, Any]] = []
  previous_loss: float | None = None
  outer_status = "max_steps"

  def outer_objective(theta_value: jnp.ndarray) -> jnp.ndarray:
    offense_point = offense_problem.solve_exact_lq(theta_value).point
    defense_point = defense_problem.solve_exact_lq(theta_value).point
    combined_point = _combine_drone3d_player_points(offense_point, defense_point)
    return problem.objective(combined_point.primal, theta_value)

  objective_and_grad = jax.jit(jax.value_and_grad(outer_objective))

  for outer_step in range(cfg.outer.steps):
    outer_step_start = time.perf_counter()
    loss_value, grad_theta = objective_and_grad(theta)
    updates, opt_state = optimizer.update(grad_theta, opt_state, theta)
    theta = optax.apply_updates(theta, updates)
    if cfg.outer.alpha_logit_clip is not None:
      theta = jnp.clip(theta, -cfg.outer.alpha_logit_clip, cfg.outer.alpha_logit_clip)

    alpha = build_alpha_from_logits(theta, problem.tree)
    leaf_type_probs, node_public_probs, _, _ = propagate_type_probabilities(alpha, problem.topology, problem.prior)
    leaf_probs = node_public_probs[problem.topology.leaf_nodes]
    leaf_type_probs = leaf_type_probs[problem.topology.leaf_nodes]
    leaf_beliefs = leaf_type_probs / jnp.clip(leaf_probs[:, None], 1e-8, None)
    total_elapsed = time.perf_counter() - outer_step_start
    history_entry = {
      "outer_step": outer_step + 1,
      "loss": float(loss_value),
      "loss_change": None if previous_loss is None else float(loss_value) - previous_loss,
      "alpha_grad_norm": _tree_l2_norm(grad_theta),
      "step_elapsed_sec": total_elapsed,
      "solver_variant": "player_separable_lq",
      "variant2_constraint_mode": "exact_lq",
      "box_constraints_active": False,
      "forward_solve_sec": total_elapsed,
      "forward_num_iterations": 1,
      "forward_initial_residual_norm": 0.0,
      "forward_residual_norm": 0.0,
      "forward_status": int(SolverStatus.SUCCESS),
      "forward_linear_mode": "player_separable[player_separable_lq_riccati,player_separable_lq_riccati]",
      "forward_linear_mode_offense": "player_separable_lq_riccati",
      "forward_linear_mode_defense": "player_separable_lq_riccati",
      "forward_used_singularity_aware_partial_elimination": False,
      "forward_used_singularity_aware_partial_elimination_offense": False,
      "forward_used_singularity_aware_partial_elimination_defense": False,
      "forward_approximate_pruning_used": False,
      "forward_approximate_pruning_used_offense": False,
      "forward_approximate_pruning_used_defense": False,
      "forward_solve_sec_offense": 0.5 * total_elapsed,
      "forward_solve_sec_defense": 0.5 * total_elapsed,
      "forward_num_iterations_offense": 1,
      "forward_num_iterations_defense": 1,
      "forward_initial_residual_norm_offense": 0.0,
      "forward_initial_residual_norm_defense": 0.0,
      "forward_line_search_last_alpha": 1.0,
      "forward_line_search_min_alpha": 1.0,
      "forward_line_search_mean_alpha": 1.0,
      "forward_line_search_zero_count": 0,
      "forward_line_search_stalled": False,
      "forward_line_search_last_alpha_offense": 1.0,
      "forward_line_search_last_alpha_defense": 1.0,
      "forward_line_search_min_alpha_offense": 1.0,
      "forward_line_search_min_alpha_defense": 1.0,
      "forward_line_search_mean_alpha_offense": 1.0,
      "forward_line_search_mean_alpha_defense": 1.0,
      "forward_line_search_zero_count_offense": 0,
      "forward_line_search_zero_count_defense": 0,
      "forward_line_search_stalled_offense": False,
      "forward_line_search_stalled_defense": False,
      "forward_regularization_initial": 0.0,
      "forward_regularization_last": 0.0,
      "forward_regularization_min": 0.0,
      "forward_regularization_max": 0.0,
      "forward_regularization_retry_count": 0,
      "forward_regularization_retry_last_offense": 0,
      "forward_regularization_retry_last_defense": 0,
      "forward_regularization_initial_offense": 0.0,
      "forward_regularization_initial_defense": 0.0,
      "forward_regularization_last_offense": 0.0,
      "forward_regularization_last_defense": 0.0,
      "forward_regularization_min_offense": 0.0,
      "forward_regularization_min_defense": 0.0,
      "forward_regularization_max_offense": 0.0,
      "forward_regularization_max_defense": 0.0,
      "forward_regularization_retry_count_offense": 0,
      "forward_regularization_retry_count_defense": 0,
      "backward_solve_sec": None,
      "backward_mode": None,
      "backward_reduced_dual_attempted": False,
      "backward_fallback": False,
      "backward_reduced_dual_residual_norm": None,
      "backward_reduced_dual_regularization": None,
      "backward_solve_sec_offense": None,
      "backward_solve_sec_defense": None,
      "backward_mode_offense": None,
      "backward_mode_defense": None,
      "backward_reduced_dual_regularization_offense": None,
      "backward_reduced_dual_regularization_defense": None,
      "leaf_belief_min": float(jnp.min(leaf_beliefs)) if leaf_beliefs.size else 1.0,
      "leaf_belief_max": float(jnp.max(leaf_beliefs)) if leaf_beliefs.size else 1.0,
      "frontier_belief_min": float(jnp.min(leaf_beliefs)) if leaf_beliefs.size else 1.0,
      "frontier_belief_max": float(jnp.max(leaf_beliefs)) if leaf_beliefs.size else 1.0,
    }
    history.append(history_entry)
    if progress_callback is not None:
      progress_callback(history_entry)
    if outer_step + 1 >= max(1, cfg.outer.min_steps):
      stop_reasons: list[str] = []
      if (
        cfg.outer.grad_tolerance is not None
        and history_entry["alpha_grad_norm"] <= cfg.outer.grad_tolerance
      ):
        stop_reasons.append("grad_tolerance")
      loss_change = history_entry["loss_change"]
      if (
        cfg.outer.loss_change_tolerance is not None
        and loss_change is not None
        and abs(loss_change) <= cfg.outer.loss_change_tolerance
        and (
          cfg.outer.grad_tolerance is None
          or history_entry["alpha_grad_norm"] <= cfg.outer.grad_tolerance
        )
      ):
        stop_reasons.append("loss_change_tolerance")
      if stop_reasons:
        outer_status = "+".join(stop_reasons)
        break
    previous_loss = float(loss_value)

  evaluation = _solve_drone3d_fixed_alpha_player_separable_lq(
    problem,
    cfg.inner,
    alpha_logits=theta,
  )
  evaluation.update(
    {
      "history": history,
      "outer_status": outer_status,
      "outer_steps_used": len(history),
    },
  )
  return evaluation


def solve_drone3d_fixed_alpha(
  problem: Drone3DTreeProblem,
  inner_cfg: TreeDiffMPCConfig,
  *,
  alpha_logits: jnp.ndarray | None = None,
  initial_point: EqualityGamePoint | None = None,
  solver_variant: str = "player_separable_lq",
) -> dict[str, Any]:
  if solver_variant != "player_separable_lq":
    raise ValueError(f"Unsupported Drone3D solver_variant: {solver_variant}")
  return _solve_drone3d_fixed_alpha_player_separable_lq(
    problem,
    inner_cfg,
    alpha_logits=alpha_logits,
    initial_point=initial_point,
  )


def solve_drone3d_bilevel(
  problem: Drone3DTreeProblem,
  cfg: Drone3DBilevelConfig,
  *,
  alpha_logits: jnp.ndarray | None = None,
  warm_point: EqualityGamePoint | None = None,
  progress_callback: Callable[[dict[str, Any]], None] | None = None,
  solver_variant: str = "player_separable_lq",
) -> dict[str, Any]:
  if solver_variant != "player_separable_lq":
    raise ValueError(f"Unsupported Drone3D solver_variant: {solver_variant}")
  return _solve_drone3d_bilevel_player_separable_lq(
    problem,
    cfg,
    alpha_logits=alpha_logits,
    warm_point=warm_point,
    progress_callback=progress_callback,
  )


def make_drone3d_problem_for_context(
  cfg_template: Drone3DProblemConfig,
  *,
  x0: jnp.ndarray,
  prior: jnp.ndarray,
  mixed_horizon_steps: int,
  tail_horizon_steps: int,
) -> Drone3DTreeProblem:
  cfg = replace(
    cfg_template,
    horizon_seconds=cfg_template.dt * float(mixed_horizon_steps + tail_horizon_steps),
    prior=tuple(float(value) for value in jnp.asarray(prior).tolist()),
    initial_state=tuple(float(value) for value in jnp.asarray(x0).tolist()),
  )
  tree = MixedPrefixTreeSpec(
    type_count=2,
    mixed_horizon_steps=int(mixed_horizon_steps),
    tail_horizon_steps=int(tail_horizon_steps),
  )
  return Drone3DTreeProblem(tree, cfg)


def drone3d_predictor_input(
  x0: jnp.ndarray,
  belief: jnp.ndarray,
  *,
  remaining_mixed_steps: int,
  remaining_tail_steps: int,
  max_mixed_steps: int,
  max_total_steps: int,
) -> jnp.ndarray:
  state = jnp.asarray(x0, dtype=F32)
  prior = jnp.asarray(belief, dtype=F32)
  extra = jnp.asarray(
    (
      float(remaining_mixed_steps) / float(max(1, max_mixed_steps)),
      float(remaining_tail_steps) / float(max(1, max_total_steps)),
      float(remaining_mixed_steps + remaining_tail_steps) / float(max(1, max_total_steps)),
    ),
    dtype=F32,
  )
  return jnp.concatenate([state, prior, extra], axis=0)


def drone3d_alpha_output_shape(max_mixed_steps: int, type_count: int = 2) -> tuple[int, int, int, int]:
  max_nodes = 1 if max_mixed_steps == 0 else type_count ** (max_mixed_steps - 1)
  return (max_mixed_steps, max_nodes, type_count, type_count)


def build_drone3d_alpha_training_example(
  x0: jnp.ndarray,
  belief: jnp.ndarray,
  *,
  remaining_mixed_steps: int,
  remaining_tail_steps: int,
  max_mixed_steps: int,
  max_total_steps: int,
  alpha: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
  output_shape = drone3d_alpha_output_shape(max_mixed_steps)
  target_alpha = jnp.zeros(output_shape, dtype=F32)
  mask = jnp.zeros(output_shape, dtype=F32)
  if remaining_mixed_steps > 0:
    target_alpha = target_alpha.at[:remaining_mixed_steps, : alpha.shape[1]].set(alpha)
    for depth in range(remaining_mixed_steps):
      node_count = 2**depth
      mask = mask.at[depth, :node_count].set(1.0)
  features = drone3d_predictor_input(
    x0,
    belief,
    remaining_mixed_steps=remaining_mixed_steps,
    remaining_tail_steps=remaining_tail_steps,
    max_mixed_steps=max_mixed_steps,
    max_total_steps=max_total_steps,
  )
  return {
    "features": features.astype(F32),
    "target_alpha": target_alpha,
    "mask": mask,
  }


def init_drone3d_alpha_model(
  *,
  key: jax.Array,
  input_dim: int,
  output_shape: tuple[int, int, int, int],
  hidden_sizes: tuple[int, ...],
) -> list[dict[str, jnp.ndarray]]:
  params: list[dict[str, jnp.ndarray]] = []
  dims = (input_dim, *hidden_sizes, int(jnp.prod(jnp.asarray(output_shape))))
  keys = jax.random.split(key, num=len(dims) - 1)
  for layer_key, fan_in, fan_out in zip(keys, dims[:-1], dims[1:], strict=False):
    weight_key, bias_key = jax.random.split(layer_key)
    limit = jnp.sqrt(jnp.asarray(6.0 / float(fan_in + fan_out), dtype=F32))
    params.append(
      {
        "w": jax.random.uniform(weight_key, (fan_in, fan_out), minval=-limit, maxval=limit, dtype=F32),
        "b": jnp.zeros((fan_out,), dtype=F32) + 0.0 * jax.random.normal(bias_key, (fan_out,), dtype=F32),
      },
    )
  return params


def drone3d_alpha_model_logits(
  params: list[dict[str, jnp.ndarray]],
  features: jnp.ndarray,
  *,
  output_shape: tuple[int, int, int, int],
) -> jnp.ndarray:
  activations = features
  for layer_index, layer in enumerate(params):
    activations = activations @ layer["w"] + layer["b"]
    if layer_index + 1 < len(params):
      activations = jax.nn.silu(activations)
  return jnp.reshape(activations, features.shape[:-1] + output_shape)


def drone3d_alpha_model_probs(
  params: list[dict[str, jnp.ndarray]],
  features: jnp.ndarray,
  *,
  output_shape: tuple[int, int, int, int],
) -> jnp.ndarray:
  logits = drone3d_alpha_model_logits(params, features, output_shape=output_shape)
  return jax.nn.softmax(logits, axis=-1)


def train_drone3d_alpha_model(
  examples: dict[str, jnp.ndarray],
  cfg: Drone3DAlphaModelConfig,
  *,
  output_shape: tuple[int, int, int, int],
) -> tuple[list[dict[str, jnp.ndarray]], list[dict[str, float]]]:
  features = jnp.asarray(examples["features"], dtype=F32)
  targets = jnp.asarray(examples["target_alpha"], dtype=F32)
  masks = jnp.asarray(examples["mask"], dtype=F32)
  if features.ndim != 2:
    raise ValueError("features must have shape (N, D).")
  if targets.shape[0] != features.shape[0] or masks.shape != targets.shape:
    raise ValueError("target_alpha and mask must align with features.")
  params = init_drone3d_alpha_model(
    key=jax.random.PRNGKey(cfg.seed),
    input_dim=int(features.shape[1]),
    output_shape=output_shape,
    hidden_sizes=cfg.hidden_sizes,
  )
  optimizer = optax.adam(cfg.learning_rate)
  opt_state = optimizer.init(params)
  history: list[dict[str, float]] = []

  @jax.jit
  def batch_loss(
    params_value: list[dict[str, jnp.ndarray]],
    batch_features: jnp.ndarray,
    batch_targets: jnp.ndarray,
    batch_masks: jnp.ndarray,
  ) -> jnp.ndarray:
    predicted = drone3d_alpha_model_probs(params_value, batch_features, output_shape=output_shape)
    squared_error = jnp.square(predicted - batch_targets)
    weighted_error = batch_masks * squared_error
    normalizer = jnp.maximum(1.0, jnp.sum(batch_masks))
    return jnp.sum(weighted_error) / normalizer

  @jax.jit
  def train_step(
    params_value: list[dict[str, jnp.ndarray]],
    opt_state_value: optax.OptState,
    batch_features: jnp.ndarray,
    batch_targets: jnp.ndarray,
    batch_masks: jnp.ndarray,
  ) -> tuple[list[dict[str, jnp.ndarray]], optax.OptState, jnp.ndarray]:
    loss_value, grads = jax.value_and_grad(batch_loss)(
      params_value,
      batch_features,
      batch_targets,
      batch_masks,
    )
    updates, new_opt_state = optimizer.update(grads, opt_state_value, params_value)
    new_params = optax.apply_updates(params_value, updates)
    return new_params, new_opt_state, loss_value

  key = jax.random.PRNGKey(cfg.seed + 1)
  sample_count = int(features.shape[0])
  batch_size = max(1, min(int(cfg.batch_size), sample_count))
  for step in range(cfg.steps):
    key, perm_key = jax.random.split(key)
    permutation = jax.random.permutation(perm_key, sample_count)
    batch_indices = permutation[:batch_size]
    params, opt_state, loss_value = train_step(
      params,
      opt_state,
      features[batch_indices],
      targets[batch_indices],
      masks[batch_indices],
    )
    if step == 0 or (step + 1) % 25 == 0 or step + 1 == cfg.steps:
      full_loss = float(batch_loss(params, features, targets, masks))
      history.append({"step": float(step + 1), "loss": full_loss})
  return params, history


def predict_drone3d_alpha_logits(
  params: list[dict[str, jnp.ndarray]],
  *,
  x0: jnp.ndarray,
  belief: jnp.ndarray,
  remaining_mixed_steps: int,
  remaining_tail_steps: int,
  max_mixed_steps: int,
  max_total_steps: int,
) -> jnp.ndarray:
  output_shape = drone3d_alpha_output_shape(max_mixed_steps)
  features = drone3d_predictor_input(
    x0,
    belief,
    remaining_mixed_steps=remaining_mixed_steps,
    remaining_tail_steps=remaining_tail_steps,
    max_mixed_steps=max_mixed_steps,
    max_total_steps=max_total_steps,
  )[None, :]
  logits = drone3d_alpha_model_logits(params, features, output_shape=output_shape)[0]
  if remaining_mixed_steps == 0:
    return jnp.zeros((0, 1, 2, 2), dtype=F32)
  max_nodes = 1 if remaining_mixed_steps == 0 else 2 ** (remaining_mixed_steps - 1)
  return logits[:remaining_mixed_steps, :max_nodes]
