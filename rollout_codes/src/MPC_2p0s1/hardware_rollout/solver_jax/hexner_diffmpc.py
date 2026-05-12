from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import time
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax.scipy.sparse.linalg import gmres

from .public_tree import PublicTreeTopology, build_public_tree_topology, propagate_type_probabilities
from .tree import F32, MixedPrefixTreeSpec, build_alpha_from_logits
from .tree_diffmpc import (
  EqualityGameDual,
  EqualityGamePoint,
  EqualityGamePrimal,
  SolverStatus,
  TreeSchurBlockData,
  TreeSchurPreconditioner,
  TreeDiffMPCConfig,
  TreeDiffMPCSolver,
  apply_tree_schur_preconditioner,
  build_tree_schur_preconditioner,
)


class HexnerLinearization(NamedTuple):
  edge_public_probs: jnp.ndarray
  leaf_type_probs: jnp.ndarray
  leaf_public_probs: jnp.ndarray
  offense_controls: jnp.ndarray
  defense_controls: jnp.ndarray
  offense_jac: jnp.ndarray
  defense_jac: jnp.ndarray
  offense_second: jnp.ndarray
  defense_second: jnp.ndarray
  offense_dynamics_jac: jnp.ndarray
  defense_dynamics_jac: jnp.ndarray
  offense_grad: jnp.ndarray
  defense_grad: jnp.ndarray
  offense_hdiag: jnp.ndarray
  defense_hdiag: jnp.ndarray
  node_grad: jnp.ndarray
  node_hdiag: jnp.ndarray
  node_constraints: jnp.ndarray
  terminal_constraints: jnp.ndarray


class HexnerReducedDualPreconditioner(NamedTuple):
  variable_dim: jnp.ndarray
  variable_mask: jnp.ndarray
  parent_coupling: jnp.ndarray
  schur_inv: jnp.ndarray


class HexnerSinglePlayerLinearization(NamedTuple):
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


class HexnerExactLQPlayerSolve(NamedTuple):
  point: EqualityGamePoint
  objective: jnp.ndarray
  mode: str


class BoxBarrierTerms(NamedTuple):
  objective: jnp.ndarray
  grad: jnp.ndarray
  hdiag: jnp.ndarray
  feasible: jnp.ndarray
  min_slack: jnp.ndarray


def _normalize_box_bounds(
  bounds: tuple[float, ...] | None,
  *,
  dim: int,
  name: str,
  default: float,
) -> jnp.ndarray:
  if bounds is None:
    return jnp.full((dim,), jnp.asarray(default, dtype=F32), dtype=F32)
  if len(bounds) != dim:
    raise ValueError(f"{name} must have length {dim}, got {len(bounds)}.")
  return jnp.asarray(bounds, dtype=F32)


def _validate_box_bounds(
  lower: jnp.ndarray,
  upper: jnp.ndarray,
  *,
  name: str,
) -> None:
  active = jnp.logical_and(jnp.isfinite(lower), jnp.isfinite(upper))
  if bool(jnp.any(jnp.logical_and(active, lower >= upper))):
    raise ValueError(f"{name} lower bounds must be strictly smaller than upper bounds.")


def _normalize_terminal_velocity_constraint_players(
  enforce_terminal_velocity_constraints: bool,
  terminal_velocity_constraint_players: tuple[str, ...] | None,
) -> tuple[str, ...]:
  if terminal_velocity_constraint_players is None:
    return ("offense", "defense") if enforce_terminal_velocity_constraints else ()
  allowed = {"offense", "defense"}
  normalized: list[str] = []
  for player in terminal_velocity_constraint_players:
    if player not in allowed:
      raise ValueError(
        "terminal_velocity_constraint_players entries must be 'offense' or 'defense'. "
        f"Got {player!r}.",
      )
    if player not in normalized:
      normalized.append(player)
  return tuple(normalized)


def _box_barrier_terms(
  values: jnp.ndarray,
  lower: jnp.ndarray,
  upper: jnp.ndarray,
  *,
  weight: float,
) -> BoxBarrierTerms:
  dtype = values.dtype
  zero = jnp.zeros_like(values)
  if weight <= 0.0:
    return BoxBarrierTerms(
      objective=jnp.asarray(0.0, dtype=dtype),
      grad=zero,
      hdiag=zero,
      feasible=jnp.asarray(True),
      min_slack=jnp.asarray(jnp.inf, dtype=dtype),
    )

  lower_active = jnp.isfinite(lower)
  upper_active = jnp.isfinite(upper)
  lower_slack = values - lower
  upper_slack = upper - values
  lower_feasible = jnp.logical_or(jnp.logical_not(lower_active), lower_slack > 0.0)
  upper_feasible = jnp.logical_or(jnp.logical_not(upper_active), upper_slack > 0.0)
  feasible = jnp.all(lower_feasible) & jnp.all(upper_feasible)

  epsilon = jnp.asarray(1e-8, dtype=dtype)
  safe_lower_slack = jnp.where(lower_active, jnp.maximum(lower_slack, epsilon), jnp.ones_like(values))
  safe_upper_slack = jnp.where(upper_active, jnp.maximum(upper_slack, epsilon), jnp.ones_like(values))
  weight_value = jnp.asarray(weight, dtype=dtype)

  objective = -weight_value * (
    jnp.sum(jnp.where(lower_active, jnp.log(safe_lower_slack), 0.0))
    + jnp.sum(jnp.where(upper_active, jnp.log(safe_upper_slack), 0.0))
  )
  grad = (
    -weight_value * jnp.where(lower_active, 1.0 / safe_lower_slack, 0.0)
    + weight_value * jnp.where(upper_active, 1.0 / safe_upper_slack, 0.0)
  )
  hdiag = weight_value * (
    jnp.where(lower_active, 1.0 / jnp.square(safe_lower_slack), 0.0)
    + jnp.where(upper_active, 1.0 / jnp.square(safe_upper_slack), 0.0)
  )
  min_lower_slack = jnp.min(jnp.where(lower_active, lower_slack, jnp.inf))
  min_upper_slack = jnp.min(jnp.where(upper_active, upper_slack, jnp.inf))
  min_slack = jnp.minimum(min_lower_slack, min_upper_slack)
  return BoxBarrierTerms(
    objective=objective,
    grad=grad,
    hdiag=hdiag,
    feasible=feasible,
    min_slack=min_slack,
  )


@lru_cache(maxsize=None)
def _hexner_single_player_reduced_matvec_kernel(
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
def _hexner_single_player_lq_tree_solve_kernel(
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
    leaf_target_mean: jnp.ndarray,
    leaf_target_sq_mean: jnp.ndarray,
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

    leaf_p = jnp.zeros((leaf_count, state_dim, state_dim), dtype=dtype)
    leaf_p = leaf_p.at[:, 0, 0].set(2.0)
    leaf_p = leaf_p.at[:, 1, 1].set(2.0)
    leaf_r = jnp.concatenate(
      [
        -2.0 * leaf_target_mean,
        jnp.zeros((leaf_count, state_dim - 2), dtype=dtype),
      ],
      axis=1,
    )
    p_nodes = p_nodes.at[leaf_nodes].set(leaf_p)
    r_nodes = r_nodes.at[leaf_nodes].set(leaf_r)
    c_nodes = c_nodes.at[leaf_nodes].set(leaf_target_sq_mean)

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
      child_states = (
        parent_states @ a_state.T
        + controls @ control_matrix.T
      )
      edge_controls = edge_controls.at[edge_slice].set(controls)
      node_states = node_states.at[edge_children[edge_slice]].set(child_states)

    if leaf_count > 0:
      leaf_states = node_states[leaf_nodes]
      leaf_grad = jnp.concatenate(
        [
          2.0 * leaf_public_probs[:, None] * (leaf_states[:, 0:2] - leaf_target_mean),
          jnp.zeros((leaf_count, state_dim - 2), dtype=dtype),
        ],
        axis=1,
      )
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
def _hexner_single_player_preconditioner_kernel(
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
def _hexner_single_player_tree_sweep_kernel(
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
    control_hdiag: jnp.ndarray,
    curved_control_rhs: jnp.ndarray,
    a_state: jnp.ndarray,
    terminal_selector: jnp.ndarray,
    node_parents: jnp.ndarray,
    node_parent_edges: jnp.ndarray,
    edge_parents: jnp.ndarray,
    edge_children: jnp.ndarray,
    leaf_nodes: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dtype = state_rhs.dtype
    eye_state = jnp.eye(state_dim, dtype=dtype)
    leaf_matrix_dim = state_dim + terminal_dim

    control_inv = 1.0 / control_hdiag
    edge_schur = jnp.einsum(
      "eik,ek,ejk->eij",
      dynamics_jac,
      control_inv,
      dynamics_jac,
    )

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
      h_block = node_hdiag[node_slice]
      h_matrix = eye_state[None, :, :] * h_block[:, None, :]
      b_block = state_rhs[node_slice]
      r_block = dual_rhs[node_slice]
      incoming_edges = node_parent_edges[node_slice]
      s_block = edge_schur[incoming_edges]

      if depth == total_depth:
        if terminal_dim > 0:
          k_block = eye_state[None, :, :] + h_block[:, :, None] * s_block
          upper = jnp.concatenate(
            [k_block, jnp.broadcast_to(terminal_selector.T, (node_count_depth, state_dim, terminal_dim))],
            axis=2,
          )
          lower = jnp.concatenate(
            [jnp.einsum("ij,njk->nik", terminal_selector, s_block), jnp.zeros((node_count_depth, terminal_dim, terminal_dim), dtype=dtype)],
            axis=2,
          )
          block_matrix = jnp.concatenate([upper, lower], axis=1)
          rhs_matrix = jnp.concatenate(
            [
              b_block[:, :, None] - h_block[:, :, None] * a_state[None, :, :] - jnp.einsum("nij,nj->ni", h_matrix, r_block)[:, :, None],
              terminal_rhs[:, :, None]
              - jnp.broadcast_to(terminal_selector @ a_state, (node_count_depth, terminal_dim, state_dim))
              - jnp.einsum("ij,nj->ni", terminal_selector, r_block)[:, :, None],
            ],
            axis=1,
          )
          rhs_vector = jnp.concatenate(
            [
              b_block - h_block * r_block,
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
          k_block = eye_state[None, :, :] + h_block[:, :, None] * s_block
          rhs_matrix = -h_block[:, :, None] * a_state[None, :, :]
          rhs_vector = b_block - h_block * r_block
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
        m_block = h_matrix - c_block
        k_block = eye_state[None, :, :] + jnp.einsum("nij,njk->nik", m_block, s_block)
        rhs_matrix = -jnp.einsum("nij,jk->nik", m_block, a_state)
        rhs_vector = b_block + d_block - jnp.einsum("nij,nj->ni", m_block, r_block)
        solution_matrix = jnp.linalg.solve(k_block, rhs_matrix)
        solution_vector = jnp.linalg.solve(k_block, rhs_vector[:, :, None]).squeeze(-1)
        p_matrix = p_matrix.at[node_slice].set(solution_matrix)
        q_vector = q_vector.at[node_slice].set(solution_vector)

    x_solution = jnp.zeros((node_count, state_dim), dtype=dtype)
    lambda_solution = jnp.zeros((node_count, state_dim), dtype=dtype)
    mu_solution = jnp.zeros((leaf_count, terminal_dim), dtype=dtype)

    root_slice = slice(0, 1)
    x_root = dual_rhs[root_slice]
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
    h_root = node_hdiag[root_slice][0]
    h_root_matrix = eye_state * h_root[None, :]
    lambda_root = state_rhs[root_slice][0] + d_root - jnp.einsum(
      "ij,j->i",
      h_root_matrix - c_root,
      x_root[0],
    )
    x_solution = x_solution.at[root_slice].set(x_root)
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
        + jnp.einsum("nij,nj->ni", edge_schur[node_parent_edges[node_slice]], lambda_block)
        + dual_rhs[node_slice]
      )
      x_solution = x_solution.at[node_slice].set(x_block)
      lambda_solution = lambda_solution.at[node_slice].set(lambda_block)
      if terminal_dim > 0 and depth == total_depth:
        mu_block = jnp.einsum("nij,nj->ni", mu_matrix[node_slice], x_parent) + mu_vector[node_slice]
        mu_solution = mu_solution.at[:].set(mu_block)

    child_duals = lambda_solution[edge_children]
    dual_term = jnp.einsum("ei,eij->ej", child_duals, dynamics_jac)
    controls = (curved_control_rhs + dual_term) / control_hdiag
    return x_solution, controls, lambda_solution, mu_solution

  return kernel


@dataclass(frozen=True)
class HexnerProblemConfig:
  horizon_seconds: float = 1.0
  dt: float = 0.1
  max_accel: float = 4.0
  squash_controls: bool = False
  use_forward_euler_dynamics: bool = False
  target_offset: float = 1.0
  prior: tuple[float, float] = (0.5, 0.5)
  running_r: tuple[float, float] = (0.05, 0.025)
  running_s: tuple[float, float] = (0.05, 0.10)
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
class HexnerOuterConfig:
  steps: int = 64
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
class HexnerBilevelConfig:
  inner: TreeDiffMPCConfig = TreeDiffMPCConfig()
  outer: HexnerOuterConfig = HexnerOuterConfig()


def build_reveal_schedule_alpha_logits(
  tree: MixedPrefixTreeSpec,
  *,
  reveal_depth: int | None,
  nonreveal_probs: tuple[float, float] = (0.5, 0.5),
  logit_scale: float = 12.0,
) -> jnp.ndarray:
  if tree.mixed_horizon_steps == 0:
    return jnp.zeros((0, 1, tree.type_count, tree.type_count), dtype=F32)
  if tree.type_count != 2:
    raise ValueError("Hexner reveal schedule helper currently assumes two types.")
  if reveal_depth is not None and not (0 <= reveal_depth < tree.mixed_horizon_steps):
    raise ValueError(f"reveal_depth must be in [0, {tree.mixed_horizon_steps}) or None.")

  max_nodes = tree.max_node_count
  logits = jnp.zeros((tree.mixed_horizon_steps, max_nodes, tree.type_count, tree.type_count), dtype=F32)
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


def _make_optimizer(name: str, learning_rate: float) -> optax.GradientTransformation:
  normalized = name.lower()
  if normalized == "adam":
    return optax.adam(learning_rate)
  if normalized == "sgd":
    return optax.sgd(learning_rate)
  raise ValueError(f"Unsupported optimizer: {name}")


def _tree_l2_norm(tree_like: Any) -> float:
  leaves = jax.tree_util.tree_leaves(tree_like)
  total = 0.0
  for leaf in leaves:
    total = total + float(jnp.sum(jnp.square(jnp.asarray(leaf, dtype=F32))))
  return total**0.5


@dataclass(frozen=True)
class HexnerTreeProblem:
  tree: MixedPrefixTreeSpec
  cfg: HexnerProblemConfig = HexnerProblemConfig()

  def __post_init__(self) -> None:
    topology = build_public_tree_topology(self.tree)
    expected_non_root = jnp.arange(1, topology.node_count, dtype=topology.edge_children.dtype)
    if not bool(jnp.all(topology.edge_children == expected_non_root)):
      raise ValueError("HexnerTreeProblem expects edge children to enumerate non-root nodes in order.")
    prior = jnp.array(self.cfg.prior, dtype=F32)
    targets = jnp.array(
      [
        [0.0, -self.cfg.target_offset],
        [0.0, self.cfg.target_offset],
      ],
      dtype=F32,
    )
    x0 = jnp.array([-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=F32)
    dt = self.cfg.dt
    a_state = jnp.eye(8, dtype=F32)
    a_state = a_state.at[0, 2].set(dt)
    a_state = a_state.at[1, 3].set(dt)
    a_state = a_state.at[4, 6].set(dt)
    a_state = a_state.at[5, 7].set(dt)
    pos_gain = 0.0 if self.cfg.use_forward_euler_dynamics else 0.5 * (dt**2)
    offense_matrix = jnp.zeros((8, 2), dtype=F32)
    offense_matrix = offense_matrix.at[0, 0].set(pos_gain)
    offense_matrix = offense_matrix.at[1, 1].set(pos_gain)
    offense_matrix = offense_matrix.at[2, 0].set(dt)
    offense_matrix = offense_matrix.at[3, 1].set(dt)
    defense_matrix = jnp.zeros((8, 2), dtype=F32)
    defense_matrix = defense_matrix.at[4, 0].set(pos_gain)
    defense_matrix = defense_matrix.at[5, 1].set(pos_gain)
    defense_matrix = defense_matrix.at[6, 0].set(dt)
    defense_matrix = defense_matrix.at[7, 1].set(dt)
    terminal_velocity_constraint_players = _normalize_terminal_velocity_constraint_players(
      self.cfg.enforce_terminal_velocity_constraints,
      self.cfg.terminal_velocity_constraint_players,
    )
    terminal_rows: list[jnp.ndarray] = []
    if "offense" in terminal_velocity_constraint_players:
      terminal_rows.append(jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=F32))
      terminal_rows.append(jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=F32))
    if "defense" in terminal_velocity_constraint_players:
      terminal_rows.append(jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=F32))
      terminal_rows.append(jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=F32))
    if terminal_rows:
      terminal_selector = jnp.stack(terminal_rows, axis=0)
    else:
      terminal_selector = jnp.zeros((0, 8), dtype=F32)
    if float(self.cfg.inequality_barrier_weight) < 0.0:
      raise ValueError("inequality_barrier_weight must be nonnegative.")
    offense_state_lower_bounds = _normalize_box_bounds(
      self.cfg.offense_state_lower_bounds,
      dim=4,
      name="offense_state_lower_bounds",
      default=-jnp.inf,
    )
    offense_state_upper_bounds = _normalize_box_bounds(
      self.cfg.offense_state_upper_bounds,
      dim=4,
      name="offense_state_upper_bounds",
      default=jnp.inf,
    )
    defense_state_lower_bounds = _normalize_box_bounds(
      self.cfg.defense_state_lower_bounds,
      dim=4,
      name="defense_state_lower_bounds",
      default=-jnp.inf,
    )
    defense_state_upper_bounds = _normalize_box_bounds(
      self.cfg.defense_state_upper_bounds,
      dim=4,
      name="defense_state_upper_bounds",
      default=jnp.inf,
    )
    offense_control_lower_bounds = _normalize_box_bounds(
      self.cfg.offense_control_lower_bounds,
      dim=2,
      name="offense_control_lower_bounds",
      default=-jnp.inf,
    )
    offense_control_upper_bounds = _normalize_box_bounds(
      self.cfg.offense_control_upper_bounds,
      dim=2,
      name="offense_control_upper_bounds",
      default=jnp.inf,
    )
    defense_control_lower_bounds = _normalize_box_bounds(
      self.cfg.defense_control_lower_bounds,
      dim=2,
      name="defense_control_lower_bounds",
      default=-jnp.inf,
    )
    defense_control_upper_bounds = _normalize_box_bounds(
      self.cfg.defense_control_upper_bounds,
      dim=2,
      name="defense_control_upper_bounds",
      default=jnp.inf,
    )
    _validate_box_bounds(
      offense_state_lower_bounds,
      offense_state_upper_bounds,
      name="offense_state_bounds",
    )
    _validate_box_bounds(
      defense_state_lower_bounds,
      defense_state_upper_bounds,
      name="defense_state_bounds",
    )
    _validate_box_bounds(
      offense_control_lower_bounds,
      offense_control_upper_bounds,
      name="offense_control_bounds",
    )
    _validate_box_bounds(
      defense_control_lower_bounds,
      defense_control_upper_bounds,
      name="defense_control_bounds",
    )
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
    object.__setattr__(self, "targets", targets)
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
    object.__setattr__(
      self,
      "state_lower_bounds",
      jnp.concatenate([offense_state_lower_bounds, defense_state_lower_bounds], axis=0),
    )
    object.__setattr__(
      self,
      "state_upper_bounds",
      jnp.concatenate([offense_state_upper_bounds, defense_state_upper_bounds], axis=0),
    )
    object.__setattr__(self, "inequality_barrier_weight", float(self.cfg.inequality_barrier_weight))
    object.__setattr__(self, "has_box_inequality_constraints", has_box_inequality_constraints)
    object.__setattr__(self, "state_dim", 8)
    object.__setattr__(self, "control_dim", 2)
    object.__setattr__(self, "edge_parents_py", tuple(int(value) for value in topology.edge_parents.tolist()))
    object.__setattr__(self, "edge_children_py", tuple(int(value) for value in topology.edge_children.tolist()))
    object.__setattr__(self, "node_parents_py", tuple(int(value) for value in topology.node_parents.tolist()))
    node_children_py = [[] for _ in range(topology.node_count)]
    for edge_idx, parent_idx in enumerate(topology.edge_parents.tolist()):
      node_children_py[int(parent_idx)].append(int(topology.edge_children[edge_idx]))
    leaf_index_by_node = -jnp.ones((topology.node_count,), dtype=jnp.int32)
    leaf_node_mask = jnp.zeros((topology.node_count,), dtype=F32)
    for leaf_idx, node_idx in enumerate(topology.leaf_nodes.tolist()):
      leaf_index_by_node = leaf_index_by_node.at[int(node_idx)].set(int(leaf_idx))
      leaf_node_mask = leaf_node_mask.at[int(node_idx)].set(1.0)
    object.__setattr__(self, "leaf_nodes_py", tuple(int(value) for value in topology.leaf_nodes.tolist()))
    object.__setattr__(self, "node_children_py", tuple(tuple(children) for children in node_children_py))
    object.__setattr__(self, "leaf_index_by_node", leaf_index_by_node)
    object.__setattr__(self, "leaf_node_mask", leaf_node_mask)
    object.__setattr__(self, "leaf_index_by_node_py", tuple(int(value) for value in leaf_index_by_node.tolist()))

  def kkt_operator_is_symmetric(self) -> bool:
    return True

  def reduced_dual_operator_is_spd(self, regularization: float) -> bool:
    del regularization
    return False

  def reduced_dual_operator_cg_sign(self, regularization: float) -> float | None:
    del regularization
    return None

  def empty_alpha_logits(self) -> jnp.ndarray:
    if self.tree.mixed_horizon_steps == 0:
      return jnp.zeros((0, 1, self.tree.type_count, self.tree.type_count), dtype=F32)
    return jnp.zeros(
      (
        self.tree.mixed_horizon_steps,
        self.tree.max_node_count,
        self.tree.type_count,
        self.tree.type_count,
      ),
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
    if mode == "random":
      key = jax.random.PRNGKey(seed)
      return init_scale * jax.random.normal(
        key,
        shape=(
          self.tree.mixed_horizon_steps,
          self.tree.max_node_count,
          self.tree.type_count,
          self.tree.type_count,
        ),
        dtype=F32,
      )
    if mode != "symmetry_breaking":
      raise ValueError(f"Unsupported alpha init mode: {mode}")
    if self.tree.type_count != 2:
      raise ValueError("Hexner symmetry-breaking alpha init currently assumes two types.")
    key = jax.random.PRNGKey(seed)
    jitter = 0.05 * init_scale * jax.random.normal(
      key,
      shape=(
        self.tree.mixed_horizon_steps,
        self.tree.max_node_count,
        self.tree.type_count,
        self.tree.type_count,
      ),
      dtype=F32,
    )
    logits = jitter
    type_action_pattern = jnp.array(
      [
        [1.0, -1.0],
        [-1.0, 1.0],
      ],
      dtype=F32,
    )
    horizon = max(1, self.tree.mixed_horizon_steps)
    for depth in range(self.tree.mixed_horizon_steps):
      node_count = self.tree.node_count(depth)
      depth_fraction = float(depth + 1) / float(horizon)
      # Weakly favor type-matching actions early, then strengthen that bias later.
      reveal_weight = 0.1 + 0.9 * (depth_fraction**2)
      node_block = (init_scale * reveal_weight) * type_action_pattern
      logits = logits.at[depth, :node_count].add(node_block[None, :, :])
    return logits

  def _edge_dynamics(
    self,
    parent_state: jnp.ndarray,
    offense_control: jnp.ndarray,
    defense_control: jnp.ndarray,
  ) -> jnp.ndarray:
    if self.cfg.squash_controls:
      offense_accel = self.cfg.max_accel * jnp.tanh(offense_control)
      defense_accel = self.cfg.max_accel * jnp.tanh(defense_control)
    else:
      offense_accel = offense_control
      defense_accel = defense_control
    return (
      self.a_state @ parent_state
      + self.offense_matrix @ offense_accel
      + self.defense_matrix @ defense_accel
    )

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

  def _state_barrier_terms(self, node_states: jnp.ndarray) -> tuple[BoxBarrierTerms, BoxBarrierTerms]:
    offense_terms = _box_barrier_terms(
      node_states[:, 0:4],
      self.offense_state_lower_bounds,
      self.offense_state_upper_bounds,
      weight=self.inequality_barrier_weight,
    )
    defense_terms = _box_barrier_terms(
      node_states[:, 4:8],
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

  def _box_constraints_feasible(self, primal: EqualityGamePrimal) -> bool:
    if not self.has_box_inequality_constraints:
      return True
    offense_state_terms, defense_state_terms = self._state_barrier_terms(primal.node_states)
    offense_control_terms, defense_control_terms = self._control_barrier_terms(
      primal.offense_controls,
      primal.defense_controls,
    )
    return bool(
      offense_state_terms.feasible
      and defense_state_terms.feasible
      and offense_control_terms.feasible
      and defense_control_terms.feasible
    )

  def initial_point(self, theta: Any) -> EqualityGamePoint:
    del theta
    node_states = jnp.zeros((self.topology.node_count, self.state_dim), dtype=F32)
    node_states = node_states.at[0].set(self.x0)
    zero_control = jnp.zeros((self.control_dim,), dtype=F32)
    for edge_idx in range(self.topology.edge_count):
      parent_idx = self.edge_parents_py[edge_idx]
      child_idx = self.edge_children_py[edge_idx]
      child_state = self._edge_dynamics(node_states[parent_idx], zero_control, zero_control)
      node_states = node_states.at[child_idx].set(child_state)
    point = EqualityGamePoint(
      primal=EqualityGamePrimal(
        node_states=node_states,
        offense_controls=jnp.zeros((self.topology.edge_count, self.control_dim), dtype=F32),
        defense_controls=jnp.zeros((self.topology.edge_count, self.control_dim), dtype=F32),
      ),
      dual=EqualityGameDual(
        node_multipliers=jnp.zeros((self.topology.node_count, self.state_dim), dtype=F32),
        terminal_multipliers=jnp.zeros((self.topology.leaf_count, self.terminal_selector.shape[0]), dtype=F32),
      ),
    )
    if self.has_box_inequality_constraints and not self._box_constraints_feasible(point.primal):
      raise ValueError(
        "Active Hexner box constraints exclude the default zero-control rollout. "
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
      parent_idx = self.edge_parents_py[edge_idx]
      child_idx = self.edge_children_py[edge_idx]
      predicted = self._edge_dynamics(
        primal.node_states[parent_idx],
        primal.offense_controls[edge_idx],
        primal.defense_controls[edge_idx],
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

    if self.cfg.squash_controls:
      offense_controls = self.cfg.max_accel * jnp.tanh(primal.offense_controls)
      defense_controls = self.cfg.max_accel * jnp.tanh(primal.defense_controls)
    else:
      offense_controls = primal.offense_controls
      defense_controls = primal.defense_controls
    running_loss = 0.5 * (
      self.cfg.running_r[0] * jnp.square(offense_controls[:, 0])
      + self.cfg.running_r[1] * jnp.square(offense_controls[:, 1])
      - self.cfg.running_s[0] * jnp.square(defense_controls[:, 0])
      - self.cfg.running_s[1] * jnp.square(defense_controls[:, 1])
    )
    running_loss = self.cfg.dt * jnp.sum(edge_public_probs * running_loss)

    leaf_type_probs = node_type_probs[self.topology.leaf_nodes]
    leaf_states = primal.node_states[self.topology.leaf_nodes]
    pos1 = leaf_states[:, 0:2]
    pos2 = leaf_states[:, 4:6]
    delta1 = pos1[:, None, :] - self.targets[None, :, :]
    delta2 = pos2[:, None, :] - self.targets[None, :, :]
    terminal_loss = jnp.sum(jnp.square(delta1), axis=-1) - jnp.sum(jnp.square(delta2), axis=-1)
    terminal_loss = jnp.sum(leaf_type_probs * terminal_loss)
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
    pos1 = leaf_states[:, 0:2]
    pos2 = leaf_states[:, 4:6]
    delta1 = pos1[:, None, :] - self.targets[None, :, :]
    delta2 = pos2[:, None, :] - self.targets[None, :, :]

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
      "u_seq": self.cfg.max_accel * jnp.tanh(edge_u_paths) if self.cfg.squash_controls else edge_u_paths,
      "v_seq": self.cfg.max_accel * jnp.tanh(edge_v_paths) if self.cfg.squash_controls else edge_v_paths,
      "terminal_velocities_offense": leaf_states[:, 2:4],
      "terminal_velocities_defense": leaf_states[:, 6:8],
      "terminal_distances_offense": jnp.linalg.norm(delta1, axis=-1),
      "terminal_distances_defense": jnp.linalg.norm(delta2, axis=-1),
      "targets": self.targets,
      "prior": self.prior,
      "box_constraints_active": self.has_box_inequality_constraints,
      "type_count": self.tree.type_count,
      "mixed_horizon_steps": self.tree.mixed_horizon_steps,
      "topology_leaf_nodes": self.topology.leaf_nodes,
    }

  def linearize_kkt(
    self,
    point: EqualityGamePoint,
    theta: Any,
  ) -> HexnerLinearization:
    alpha = build_alpha_from_logits(theta, self.tree)
    node_type_probs, node_public_probs, _, edge_public_probs = propagate_type_probabilities(alpha, self.topology, self.prior)
    leaf_type_probs = node_type_probs[self.topology.leaf_nodes]
    leaf_public_probs = node_public_probs[self.topology.leaf_nodes]

    offense_accel, offense_jac, offense_second = self._control_linearization(point.primal.offense_controls)
    defense_accel, defense_jac, defense_second = self._control_linearization(point.primal.defense_controls)

    offense_dynamics_jac = self.offense_matrix[None, :, :] * offense_jac[:, None, :]
    defense_dynamics_jac = self.defense_matrix[None, :, :] * defense_jac[:, None, :]

    node_states = point.primal.node_states
    parent_states = node_states[self.topology.edge_parents]
    predicted_states = (
      parent_states @ self.a_state.T
      + offense_accel @ self.offense_matrix.T
      + defense_accel @ self.defense_matrix.T
    )
    node_constraints = jnp.concatenate(
      [
        (node_states[0] - self.x0)[None, :],
        node_states[self.topology.edge_children] - predicted_states,
      ],
      axis=0,
    )

    leaf_states = node_states[self.topology.leaf_nodes]
    terminal_constraints = leaf_states @ self.terminal_selector.T

    target_mean = leaf_type_probs @ self.targets
    offense_leaf_grad = 2.0 * (leaf_public_probs[:, None] * leaf_states[:, 0:2] - target_mean)
    defense_leaf_grad = -2.0 * (leaf_public_probs[:, None] * leaf_states[:, 4:6] - target_mean)
    zero_block = jnp.zeros((self.topology.leaf_count, 2), dtype=F32)
    leaf_grad = jnp.concatenate([offense_leaf_grad, zero_block, defense_leaf_grad, zero_block], axis=1)
    leaf_hdiag = jnp.concatenate(
      [
        2.0 * leaf_public_probs[:, None] * jnp.ones((self.topology.leaf_count, 2), dtype=F32),
        zero_block,
        -2.0 * leaf_public_probs[:, None] * jnp.ones((self.topology.leaf_count, 2), dtype=F32),
        zero_block,
      ],
      axis=1,
    )
    offense_state_terms, defense_state_terms = self._state_barrier_terms(node_states)
    offense_control_terms, defense_control_terms = self._control_barrier_terms(
      point.primal.offense_controls,
      point.primal.defense_controls,
    )
    node_grad = jnp.concatenate(
      [offense_state_terms.grad, -defense_state_terms.grad],
      axis=1,
    )
    node_hdiag = jnp.concatenate(
      [offense_state_terms.hdiag, -defense_state_terms.hdiag],
      axis=1,
    )
    node_grad = node_grad.at[self.topology.leaf_nodes].add(leaf_grad)
    node_hdiag = node_hdiag.at[self.topology.leaf_nodes].add(leaf_hdiag)

    child_multipliers = point.dual.node_multipliers[self.topology.edge_children]
    offense_coeff = child_multipliers @ self.offense_matrix
    defense_coeff = child_multipliers @ self.defense_matrix
    running_r = jnp.array(self.cfg.running_r, dtype=F32)
    running_s = jnp.array(self.cfg.running_s, dtype=F32)
    offense_hdiag = (
      self.cfg.dt * edge_public_probs[:, None] * running_r[None, :] * (jnp.square(offense_jac) + offense_accel * offense_second)
      - offense_coeff * offense_second
    )
    defense_hdiag = (
      -self.cfg.dt * edge_public_probs[:, None] * running_s[None, :] * (jnp.square(defense_jac) + defense_accel * defense_second)
      - defense_coeff * defense_second
    )
    offense_grad = (
      self.cfg.dt
      * edge_public_probs[:, None]
      * running_r[None, :]
      * offense_accel
      * offense_jac
      + offense_control_terms.grad
    )
    defense_grad = (
      -self.cfg.dt
      * edge_public_probs[:, None]
      * running_s[None, :]
      * defense_accel
      * defense_jac
      - defense_control_terms.grad
    )
    offense_hdiag = offense_hdiag + offense_control_terms.hdiag
    defense_hdiag = defense_hdiag - defense_control_terms.hdiag

    return HexnerLinearization(
      edge_public_probs=edge_public_probs,
      leaf_type_probs=leaf_type_probs,
      leaf_public_probs=leaf_public_probs,
      offense_controls=offense_accel,
      defense_controls=defense_accel,
      offense_jac=offense_jac,
      defense_jac=defense_jac,
      offense_second=offense_second,
      defense_second=defense_second,
      offense_dynamics_jac=offense_dynamics_jac,
      defense_dynamics_jac=defense_dynamics_jac,
      offense_grad=offense_grad,
      defense_grad=defense_grad,
      offense_hdiag=offense_hdiag,
      defense_hdiag=defense_hdiag,
      node_grad=node_grad,
      node_hdiag=node_hdiag,
      node_constraints=node_constraints,
      terminal_constraints=terminal_constraints,
    )

  def kkt_residual(
    self,
    point: EqualityGamePoint,
    theta: Any,
    linearization: HexnerLinearization | None = None,
  ) -> EqualityGamePoint:
    linearization = self.linearize_kkt(point, theta) if linearization is None else linearization

    node_residual = point.dual.node_multipliers
    child_multipliers = point.dual.node_multipliers[self.topology.edge_children]
    parent_updates = -(child_multipliers @ self.a_state)
    node_residual = node_residual.at[self.topology.edge_parents].add(parent_updates)
    node_residual = node_residual + linearization.node_grad
    terminal_dual_contrib = point.dual.terminal_multipliers @ self.terminal_selector
    node_residual = node_residual.at[self.topology.leaf_nodes].add(terminal_dual_contrib)
    offense_constraint_grad = -jnp.einsum("ei,eij->ej", child_multipliers, linearization.offense_dynamics_jac)
    defense_constraint_grad = -jnp.einsum("ei,eij->ej", child_multipliers, linearization.defense_dynamics_jac)

    return EqualityGamePoint(
      primal=EqualityGamePrimal(
        node_states=node_residual,
        offense_controls=linearization.offense_grad + offense_constraint_grad,
        defense_controls=linearization.defense_grad + defense_constraint_grad,
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
    linearization: HexnerLinearization,
  ) -> EqualityGamePoint:
    del theta
    node_tangent = tangent.primal.node_states
    offense_tangent = tangent.primal.offense_controls
    defense_tangent = tangent.primal.defense_controls
    node_dual_tangent = tangent.dual.node_multipliers
    terminal_dual_tangent = tangent.dual.terminal_multipliers

    node_state_matvec = node_dual_tangent
    child_dual_tangent = node_dual_tangent[self.topology.edge_children]
    node_state_matvec = node_state_matvec.at[self.topology.edge_parents].add(-(child_dual_tangent @ self.a_state))
    node_state_matvec = node_state_matvec + linearization.node_hdiag * node_tangent
    node_state_matvec = node_state_matvec.at[self.topology.leaf_nodes].add(
      terminal_dual_tangent @ self.terminal_selector,
    )

    offense_control_matvec = linearization.offense_hdiag * offense_tangent
    defense_control_matvec = linearization.defense_hdiag * defense_tangent
    offense_control_matvec = offense_control_matvec - jnp.einsum(
      "ei,eij->ej",
      child_dual_tangent,
      linearization.offense_dynamics_jac,
    )
    defense_control_matvec = defense_control_matvec - jnp.einsum(
      "ei,eij->ej",
      child_dual_tangent,
      linearization.defense_dynamics_jac,
    )

    node_constraint_matvec = jnp.zeros_like(node_tangent)
    predicted_tangent = (
      node_tangent[self.topology.edge_parents] @ self.a_state.T
      + jnp.einsum("eij,ej->ei", linearization.offense_dynamics_jac, offense_tangent)
      + jnp.einsum("eij,ej->ei", linearization.defense_dynamics_jac, defense_tangent)
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
        offense_controls=offense_control_matvec,
        defense_controls=defense_control_matvec,
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
    linearization: HexnerLinearization,
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

    offense_diag = linearization.offense_hdiag + reg
    defense_diag = linearization.defense_hdiag + reg
    offense_rhs = neg_primal_residual.offense_controls + jnp.einsum(
      "ei,eij->ej",
      child_dual_tangent,
      linearization.offense_dynamics_jac,
    )
    defense_rhs = neg_primal_residual.defense_controls + jnp.einsum(
      "ei,eij->ej",
      child_dual_tangent,
      linearization.defense_dynamics_jac,
    )
    offense_controls = offense_rhs / offense_diag
    defense_controls = defense_rhs / defense_diag

    return EqualityGamePrimal(
      node_states=node_states,
      offense_controls=offense_controls,
      defense_controls=defense_controls,
    )

  def _reduced_dual_from_primal_and_dual(
    self,
    primal_tangent: EqualityGamePrimal,
    dual_tangent: EqualityGameDual,
    linearization: HexnerLinearization,
    regularization: float,
  ) -> EqualityGameDual:
    reg = jnp.asarray(regularization, dtype=F32)
    predicted_tangent = (
      primal_tangent.node_states[self.topology.edge_parents] @ self.a_state.T
      + jnp.einsum("eij,ej->ei", linearization.offense_dynamics_jac, primal_tangent.offense_controls)
      + jnp.einsum("eij,ej->ei", linearization.defense_dynamics_jac, primal_tangent.defense_controls)
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
    linearization: HexnerLinearization,
    regularization: float,
  ) -> EqualityGameDual:
    del theta
    zero_primal = EqualityGamePrimal(
      node_states=jnp.zeros((self.topology.node_count, self.state_dim), dtype=F32),
      offense_controls=jnp.zeros((self.topology.edge_count, self.control_dim), dtype=F32),
      defense_controls=jnp.zeros((self.topology.edge_count, self.control_dim), dtype=F32),
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
    linearization: HexnerLinearization,
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
    linearization: HexnerLinearization,
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
    linearization: HexnerLinearization,
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

    offense_dynamics_jac = linearization.offense_dynamics_jac[incoming_edge_indices] * non_root_mask[:, None, None]
    offense_diag_inv = (
      1.0 / (linearization.offense_hdiag[incoming_edge_indices] + preconditioner_regularization)
    ) * non_root_mask[:, None]
    offense_state_term = jnp.einsum(
      "nij,nj,nkj->nik",
      offense_dynamics_jac,
      offense_diag_inv,
      offense_dynamics_jac,
    )

    defense_dynamics_jac = linearization.defense_dynamics_jac[incoming_edge_indices] * non_root_mask[:, None, None]
    defense_diag_inv = (
      1.0 / (linearization.defense_hdiag[incoming_edge_indices] + preconditioner_regularization)
    ) * non_root_mask[:, None]
    defense_state_term = jnp.einsum(
      "nij,nj,nkj->nik",
      defense_dynamics_jac,
      defense_diag_inv,
      defense_dynamics_jac,
    )

    state_self_block = (
      preconditioner_regularization * jnp.eye(self.state_dim, dtype=dual_dtype)[None, :, :]
      - state_diag_inv
      - parent_state_term
      - offense_state_term
      - defense_state_term
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
    linearization: HexnerLinearization,
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
    linearization: HexnerLinearization,
    regularization: float,
    preconditioner: TreeSchurPreconditioner,
  ) -> EqualityGameDual:
    del theta, linearization, regularization, preconditioner
    terminal_dim = self.terminal_selector.shape[0]
    terminal_result = jnp.zeros(
      (self.topology.leaf_count, terminal_dim),
      dtype=packed_value.dtype,
    )
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
    linearization: HexnerLinearization,
    regularization: float,
    epsilon: float,
  ) -> HexnerReducedDualPreconditioner:
    tree_schur = build_tree_schur_preconditioner(
      self.topology,
      self.build_tree_schur_preconditioner_data(
        theta,
        linearization,
        regularization,
        epsilon,
      ),
      epsilon,
    )
    return HexnerReducedDualPreconditioner(
      variable_dim=tree_schur.variable_dim,
      variable_mask=tree_schur.variable_mask,
      parent_coupling=tree_schur.parent_coupling,
      schur_inv=tree_schur.schur_inv,
    )

  def apply_reduced_dual_preconditioner(
    self,
    dual_value: EqualityGameDual,
    theta: Any,
    linearization: HexnerLinearization,
    regularization: float,
    preconditioner: HexnerReducedDualPreconditioner,
  ) -> EqualityGameDual:
    rhs = self.pack_reduced_dual_for_tree_schur(
      dual_value,
      theta,
      linearization,
      regularization,
      TreeSchurPreconditioner(
        variable_dim=preconditioner.variable_dim,
        variable_mask=preconditioner.variable_mask,
        parent_coupling=preconditioner.parent_coupling,
        schur_inv=preconditioner.schur_inv,
      ),
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
      TreeSchurPreconditioner(
        variable_dim=preconditioner.variable_dim,
        variable_mask=preconditioner.variable_mask,
        parent_coupling=preconditioner.parent_coupling,
        schur_inv=preconditioner.schur_inv,
      ),
    )


@dataclass(frozen=True)
class HexnerSinglePlayerTreeProblem:
  parent: HexnerTreeProblem
  player: str

  def __post_init__(self) -> None:
    if self.player not in ("offense", "defense"):
      raise ValueError("player must be 'offense' or 'defense'.")
    state_slice = slice(0, 4) if self.player == "offense" else slice(4, 8)
    control_matrix = (
      self.parent.offense_matrix[state_slice, :]
      if self.player == "offense"
      else self.parent.defense_matrix[state_slice, :]
    )
    running_weights = (
      jnp.array(self.parent.cfg.running_r, dtype=F32)
      if self.player == "offense"
      else jnp.array(self.parent.cfg.running_s, dtype=F32)
    )
    terminal_selector = (
      jnp.array(
        [
          [0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=F32,
      )
      if self.player in self.parent.terminal_velocity_constraint_players
      else jnp.zeros((0, 4), dtype=F32)
    )
    object.__setattr__(self, "tree", self.parent.tree)
    object.__setattr__(self, "topology", self.parent.topology)
    object.__setattr__(self, "prior", self.parent.prior)
    object.__setattr__(self, "targets", self.parent.targets)
    object.__setattr__(self, "state_dim", 4)
    object.__setattr__(self, "control_dim", 2)
    object.__setattr__(self, "a_state", self.parent.a_state[state_slice, state_slice])
    object.__setattr__(self, "control_matrix", control_matrix)
    object.__setattr__(self, "x0", self.parent.x0[state_slice])
    object.__setattr__(self, "running_weights", running_weights)
    object.__setattr__(self, "terminal_selector", terminal_selector)
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
    object.__setattr__(self, "state_lower_bounds", state_lower_bounds)
    object.__setattr__(self, "state_upper_bounds", state_upper_bounds)
    object.__setattr__(self, "control_lower_bounds", control_lower_bounds)
    object.__setattr__(self, "control_upper_bounds", control_upper_bounds)
    object.__setattr__(self, "inequality_barrier_weight", self.parent.inequality_barrier_weight)
    object.__setattr__(self, "has_box_inequality_constraints", has_box_inequality_constraints)

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

  def solve_exact_lq(self, theta: Any) -> HexnerExactLQPlayerSolve:
    if not self.supports_exact_lq_solver():
      raise ValueError(
        "Exact Hexner Variant 2 LQ solve requires squash_controls=False and no active box constraints.",
      )

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
    target_mean = leaf_beliefs @ self.targets
    target_sq = jnp.sum(
      leaf_beliefs * jnp.sum(jnp.square(self.targets), axis=1)[None, :],
      axis=1,
    )

    kernel = _hexner_single_player_lq_tree_solve_kernel(
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
      target_mean.astype(F32),
      target_sq.astype(F32),
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
    return HexnerExactLQPlayerSolve(
      point=point,
      objective=value,
      mode="player_separable_lq_riccati",
    )

  def _branch_activity_masks(
    self,
    linearization: HexnerSinglePlayerLinearization,
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
    linearization: HexnerSinglePlayerLinearization,
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
    curved_control_scale = curved_control_mask.astype(dtype)
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
      curved_controls = curved_control_response(
        node_duals,
        zero_curved_control_rhs,
      )
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
    curved_control_rhs_step = curved_control_response(
      zero_node_duals,
      curved_control_rhs,
    )
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
      tree_sweep_kernel = _hexner_single_player_tree_sweep_kernel(
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
        linearization.control_hdiag,
        curved_control_rhs,
        self.a_state,
        self.terminal_selector,
        self.topology.node_parents,
        self.topology.node_parent_edges,
        self.topology.edge_parents,
        self.topology.edge_children,
        self.topology.leaf_nodes,
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
    reduced_matvec_kernel = _hexner_single_player_reduced_matvec_kernel(
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

      preconditioner_kernel = _hexner_single_player_preconditioner_kernel(
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
      step_flat, gmres_info = gmres(
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
      parent_idx = self.parent.edge_parents_py[edge_idx]
      child_idx = self.parent.edge_children_py[edge_idx]
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
      parent_idx = self.parent.edge_parents_py[edge_idx]
      child_idx = self.parent.edge_children_py[edge_idx]
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
    delta = leaf_states[:, None, 0:2] - self.targets[None, :, :]
    terminal_loss = jnp.sum(leaf_type_probs * jnp.sum(jnp.square(delta), axis=-1))
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
  ) -> HexnerSinglePlayerLinearization:
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

    target_mean = leaf_type_probs @ self.targets
    zero_block = jnp.zeros((self.topology.leaf_count, 2), dtype=F32)
    leaf_grad = jnp.concatenate(
      [
        2.0 * (leaf_public_probs[:, None] * leaf_states[:, 0:2] - target_mean),
        zero_block,
      ],
      axis=1,
    )
    leaf_hdiag = jnp.concatenate(
      [
        2.0 * leaf_public_probs[:, None] * jnp.ones((self.topology.leaf_count, 2), dtype=F32),
        zero_block,
      ],
      axis=1,
    )
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

    return HexnerSinglePlayerLinearization(
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
    linearization: HexnerSinglePlayerLinearization | None = None,
  ) -> EqualityGamePoint:
    linearization = self.linearize_kkt(point, theta) if linearization is None else linearization
    node_residual = point.dual.node_multipliers
    child_multipliers = point.dual.node_multipliers[self.topology.edge_children]
    node_residual = node_residual.at[self.topology.edge_parents].add(-(child_multipliers @ self.a_state))
    node_residual = node_residual + linearization.node_grad
    node_residual = node_residual.at[self.topology.leaf_nodes].add(
      point.dual.terminal_multipliers @ self.terminal_selector,
    )
    control_constraint_grad = -jnp.einsum(
      "ei,eij->ej",
      child_multipliers,
      linearization.dynamics_jac,
    )
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
    linearization: HexnerSinglePlayerLinearization,
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
    linearization: HexnerSinglePlayerLinearization,
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
    linearization: HexnerSinglePlayerLinearization,
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
    linearization: HexnerSinglePlayerLinearization,
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
    linearization: HexnerSinglePlayerLinearization,
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
    linearization: HexnerSinglePlayerLinearization,
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
    linearization: HexnerSinglePlayerLinearization,
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
    linearization: HexnerSinglePlayerLinearization,
    regularization: float,
    preconditioner: TreeSchurPreconditioner,
  ) -> jnp.ndarray:
    del theta, linearization, regularization, preconditioner
    terminal_dim = self.terminal_selector.shape[0]
    packed = dual_value.node_multipliers
    if terminal_dim > 0:
      safe_leaf_indices = jnp.maximum(self.parent.leaf_index_by_node, 0)
      node_terminal = dual_value.terminal_multipliers[safe_leaf_indices] * self.parent.leaf_node_mask[:, None]
      packed = jnp.concatenate([packed, node_terminal], axis=1)
    return packed

  def unpack_reduced_dual_from_tree_schur(
    self,
    packed_value: jnp.ndarray,
    theta: Any,
    linearization: HexnerSinglePlayerLinearization,
    regularization: float,
    preconditioner: TreeSchurPreconditioner,
  ) -> EqualityGameDual:
    del theta, linearization, regularization, preconditioner
    terminal_dim = self.terminal_selector.shape[0]
    terminal_result = jnp.zeros(
      (self.topology.leaf_count, terminal_dim),
      dtype=packed_value.dtype,
    )
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
    linearization: HexnerSinglePlayerLinearization,
    regularization: float,
    epsilon: float,
  ) -> HexnerReducedDualPreconditioner:
    tree_schur = build_tree_schur_preconditioner(
      self.topology,
      self.build_tree_schur_preconditioner_data(
        theta,
        linearization,
        regularization,
        epsilon,
      ),
      epsilon,
    )
    return HexnerReducedDualPreconditioner(
      variable_dim=tree_schur.variable_dim,
      variable_mask=tree_schur.variable_mask,
      parent_coupling=tree_schur.parent_coupling,
      schur_inv=tree_schur.schur_inv,
    )

  def apply_reduced_dual_preconditioner(
    self,
    dual_value: EqualityGameDual,
    theta: Any,
    linearization: HexnerSinglePlayerLinearization,
    regularization: float,
    preconditioner: HexnerReducedDualPreconditioner,
  ) -> EqualityGameDual:
    rhs = self.pack_reduced_dual_for_tree_schur(
      dual_value,
      theta,
      linearization,
      regularization,
      TreeSchurPreconditioner(
        variable_dim=preconditioner.variable_dim,
        variable_mask=preconditioner.variable_mask,
        parent_coupling=preconditioner.parent_coupling,
        schur_inv=preconditioner.schur_inv,
      ),
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
      TreeSchurPreconditioner(
        variable_dim=preconditioner.variable_dim,
        variable_mask=preconditioner.variable_mask,
        parent_coupling=preconditioner.parent_coupling,
        schur_inv=preconditioner.schur_inv,
      ),
    )


def _make_hexner_player_problem(problem: HexnerTreeProblem, player: str) -> HexnerSinglePlayerTreeProblem:
  return HexnerSinglePlayerTreeProblem(problem, player)


def _split_hexner_full_point_by_player(
  problem: HexnerTreeProblem,
  point: EqualityGamePoint,
) -> tuple[EqualityGamePoint, EqualityGamePoint]:
  control_dtype = point.primal.offense_controls.dtype
  empty_controls = jnp.zeros((problem.topology.edge_count, 0), dtype=control_dtype)
  terminal_dim = point.dual.terminal_multipliers.shape[1]
  half_terminal_dim = terminal_dim // 2
  offense_point = EqualityGamePoint(
    primal=EqualityGamePrimal(
      node_states=point.primal.node_states[:, 0:4],
      offense_controls=point.primal.offense_controls,
      defense_controls=empty_controls,
    ),
    dual=EqualityGameDual(
      node_multipliers=point.dual.node_multipliers[:, 0:4],
      terminal_multipliers=point.dual.terminal_multipliers[:, :half_terminal_dim],
    ),
  )
  defense_point = EqualityGamePoint(
    primal=EqualityGamePrimal(
      node_states=point.primal.node_states[:, 4:8],
      offense_controls=point.primal.defense_controls,
      defense_controls=empty_controls,
    ),
    dual=EqualityGameDual(
      node_multipliers=-point.dual.node_multipliers[:, 4:8],
      terminal_multipliers=-point.dual.terminal_multipliers[:, half_terminal_dim:],
    ),
  )
  return offense_point, defense_point


def _combine_hexner_player_points(
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


def _aggregate_status_codes(*statuses: int) -> int:
  if any(status == int(SolverStatus.NONFINITE) for status in statuses):
    return int(SolverStatus.NONFINITE)
  if all(status == int(SolverStatus.SUCCESS) for status in statuses):
    return int(SolverStatus.SUCCESS)
  return int(SolverStatus.MAX_ITERATIONS)


def _aggregate_line_search_alphas(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
  left_mask = left > 0.0
  right_mask = right > 0.0
  both = left_mask & right_mask
  result = jnp.where(both, jnp.minimum(left, right), jnp.where(left_mask, left, right))
  return result


def _tree_max_abs(tree_like: Any) -> float:
  leaves = jax.tree_util.tree_leaves(tree_like)
  if not leaves:
    return 0.0
  max_abs = 0.0
  for leaf in leaves:
    array = jnp.asarray(leaf)
    if array.size == 0:
      continue
    max_abs = max(max_abs, float(jnp.max(jnp.abs(array))))
  return max_abs


def _aggregate_player_solve_result(
  problem: HexnerTreeProblem,
  theta: jnp.ndarray,
  offense_result: Any,
  defense_result: Any,
  *,
  solver_variant: str = "player_separable",
  extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
  full_point = _combine_hexner_player_points(offense_result.point, defense_result.point)
  evaluation = problem.evaluate(full_point, theta)
  linearization = problem.linearize_kkt(full_point, theta)
  residual = problem.kkt_residual(full_point, theta, linearization)
  evaluation.update(
    {
      "point": full_point,
      "solver_variant": solver_variant,
      "residual_norm": _tree_max_abs(residual),
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


def _aggregate_player_diagnostics(
  offense_diag: dict[str, Any],
  defense_diag: dict[str, Any],
) -> dict[str, Any]:
  def _safe_float(diag: dict[str, Any], key: str) -> float:
    return float(diag.get(key, float("nan")))

  def _safe_optional_float(diag: dict[str, Any], key: str) -> float | None:
    value = diag.get(key)
    return None if value is None else float(value)

  def _safe_int(diag: dict[str, Any], key: str) -> int:
    return int(diag.get(key, -1))

  def _safe_bool(diag: dict[str, Any], key: str) -> bool:
    return bool(diag.get(key, False))

  def _safe_step_count(diag: dict[str, Any]) -> int:
    return int(diag.get("num_step_iterations", diag.get("num_iterations", -1)))

  offense_mode = str(offense_diag.get("mode", "unknown"))
  defense_mode = str(defense_diag.get("mode", "unknown"))
  offense_linear_mode = str(offense_diag.get("linear_mode_last", "unknown"))
  defense_linear_mode = str(defense_diag.get("linear_mode_last", "unknown"))
  offense_resid = offense_diag.get("reduced_dual_residual_norm")
  defense_resid = defense_diag.get("reduced_dual_residual_norm")
  offense_reg = offense_diag.get("reduced_dual_regularization")
  defense_reg = defense_diag.get("reduced_dual_regularization")
  offense_ls_last = _safe_optional_float(offense_diag, "line_search_last_alpha")
  defense_ls_last = _safe_optional_float(defense_diag, "line_search_last_alpha")
  offense_ls_min = _safe_optional_float(offense_diag, "line_search_min_alpha")
  defense_ls_min = _safe_optional_float(defense_diag, "line_search_min_alpha")
  offense_ls_mean = _safe_optional_float(offense_diag, "line_search_mean_alpha")
  defense_ls_mean = _safe_optional_float(defense_diag, "line_search_mean_alpha")
  offense_stalled = bool(offense_diag.get("line_search_stalled", False))
  defense_stalled = bool(defense_diag.get("line_search_stalled", False))
  offense_reg_initial = _safe_optional_float(offense_diag, "regularization_initial")
  defense_reg_initial = _safe_optional_float(defense_diag, "regularization_initial")
  offense_reg_last = _safe_optional_float(offense_diag, "regularization_last")
  defense_reg_last = _safe_optional_float(defense_diag, "regularization_last")
  offense_reg_min = _safe_optional_float(offense_diag, "regularization_min")
  defense_reg_min = _safe_optional_float(defense_diag, "regularization_min")
  offense_reg_max = _safe_optional_float(offense_diag, "regularization_max")
  defense_reg_max = _safe_optional_float(defense_diag, "regularization_max")
  offense_reg_retry = max(0, _safe_int(offense_diag, "regularization_retry_count"))
  defense_reg_retry = max(0, _safe_int(defense_diag, "regularization_retry_count"))
  offense_reg_retry_last = max(0, _safe_int(offense_diag, "regularization_retry_last"))
  defense_reg_retry_last = max(0, _safe_int(defense_diag, "regularization_retry_last"))
  offense_partial = _safe_bool(offense_diag, "used_singularity_aware_partial_elimination")
  defense_partial = _safe_bool(defense_diag, "used_singularity_aware_partial_elimination")
  offense_pruned = _safe_bool(offense_diag, "approximate_pruning_used")
  defense_pruned = _safe_bool(defense_diag, "approximate_pruning_used")
  line_search_last_alpha = None
  last_values = [value for value in (offense_ls_last, defense_ls_last) if value is not None]
  if last_values:
    line_search_last_alpha = float(min(last_values))
  line_search_min_alpha = None
  min_values = [value for value in (offense_ls_min, defense_ls_min) if value is not None]
  if min_values:
    line_search_min_alpha = float(min(min_values))
  line_search_mean_alpha = None
  mean_values = [value for value in (offense_ls_mean, defense_ls_mean) if value is not None]
  if mean_values:
    line_search_mean_alpha = float(sum(mean_values) / len(mean_values))
  regularization_initial = None
  initial_reg_values = [value for value in (offense_reg_initial, defense_reg_initial) if value is not None]
  if initial_reg_values:
    regularization_initial = float(max(initial_reg_values))
  regularization_last = None
  last_reg_values = [value for value in (offense_reg_last, defense_reg_last) if value is not None]
  if last_reg_values:
    regularization_last = float(max(last_reg_values))
  regularization_min = None
  min_reg_values = [value for value in (offense_reg_min, defense_reg_min) if value is not None]
  if min_reg_values:
    regularization_min = float(min(min_reg_values))
  regularization_max = None
  max_reg_values = [value for value in (offense_reg_max, defense_reg_max) if value is not None]
  if max_reg_values:
    regularization_max = float(max(max_reg_values))
  reduced_resid = None
  reduced_reg = None
  if offense_resid is not None or defense_resid is not None:
    values = [value for value in (offense_resid, defense_resid) if value is not None]
    reduced_resid = float(max(values))
  if offense_reg is not None or defense_reg is not None:
    values = [value for value in (offense_reg, defense_reg) if value is not None]
    reduced_reg = float(max(values))
  return {
    "forward_solve_sec": _safe_float(offense_diag, "elapsed_sec") + _safe_float(defense_diag, "elapsed_sec"),
    "forward_num_iterations": _safe_step_count(offense_diag) + _safe_step_count(defense_diag),
    "forward_initial_residual_norm": max(
      _safe_float(offense_diag, "initial_residual_norm"),
      _safe_float(defense_diag, "initial_residual_norm"),
    ),
    "forward_residual_norm": max(
      _safe_float(offense_diag, "residual_norm"),
      _safe_float(defense_diag, "residual_norm"),
    ),
    "forward_status": _aggregate_status_codes(
      _safe_int(offense_diag, "status"),
      _safe_int(defense_diag, "status"),
    ),
    "forward_linear_mode": f"player_separable[{offense_linear_mode},{defense_linear_mode}]",
    "forward_linear_mode_offense": offense_linear_mode,
    "forward_linear_mode_defense": defense_linear_mode,
    "forward_used_singularity_aware_partial_elimination": offense_partial or defense_partial,
    "forward_used_singularity_aware_partial_elimination_offense": offense_partial,
    "forward_used_singularity_aware_partial_elimination_defense": defense_partial,
    "forward_approximate_pruning_used": offense_pruned or defense_pruned,
    "forward_approximate_pruning_used_offense": offense_pruned,
    "forward_approximate_pruning_used_defense": defense_pruned,
    "forward_solve_sec_offense": _safe_float(offense_diag, "elapsed_sec"),
    "forward_solve_sec_defense": _safe_float(defense_diag, "elapsed_sec"),
    "forward_num_iterations_offense": _safe_step_count(offense_diag),
    "forward_num_iterations_defense": _safe_step_count(defense_diag),
    "forward_initial_residual_norm_offense": _safe_float(offense_diag, "initial_residual_norm"),
    "forward_initial_residual_norm_defense": _safe_float(defense_diag, "initial_residual_norm"),
    "forward_line_search_last_alpha": line_search_last_alpha,
    "forward_line_search_min_alpha": line_search_min_alpha,
    "forward_line_search_mean_alpha": line_search_mean_alpha,
    "forward_line_search_zero_count": max(0, _safe_int(offense_diag, "line_search_zero_count")) + max(0, _safe_int(defense_diag, "line_search_zero_count")),
    "forward_line_search_stalled": offense_stalled or defense_stalled,
    "forward_line_search_last_alpha_offense": offense_ls_last,
    "forward_line_search_last_alpha_defense": defense_ls_last,
    "forward_line_search_min_alpha_offense": offense_ls_min,
    "forward_line_search_min_alpha_defense": defense_ls_min,
    "forward_line_search_mean_alpha_offense": offense_ls_mean,
    "forward_line_search_mean_alpha_defense": defense_ls_mean,
    "forward_line_search_zero_count_offense": max(0, _safe_int(offense_diag, "line_search_zero_count")),
    "forward_line_search_zero_count_defense": max(0, _safe_int(defense_diag, "line_search_zero_count")),
    "forward_line_search_stalled_offense": offense_stalled,
    "forward_line_search_stalled_defense": defense_stalled,
    "forward_regularization_initial": regularization_initial,
    "forward_regularization_last": regularization_last,
    "forward_regularization_min": regularization_min,
    "forward_regularization_max": regularization_max,
    "forward_regularization_retry_count": offense_reg_retry + defense_reg_retry,
    "forward_regularization_retry_last_offense": offense_reg_retry_last,
    "forward_regularization_retry_last_defense": defense_reg_retry_last,
    "forward_regularization_initial_offense": offense_reg_initial,
    "forward_regularization_initial_defense": defense_reg_initial,
    "forward_regularization_last_offense": offense_reg_last,
    "forward_regularization_last_defense": defense_reg_last,
    "forward_regularization_min_offense": offense_reg_min,
    "forward_regularization_min_defense": defense_reg_min,
    "forward_regularization_max_offense": offense_reg_max,
    "forward_regularization_max_defense": defense_reg_max,
    "forward_regularization_retry_count_offense": offense_reg_retry,
    "forward_regularization_retry_count_defense": defense_reg_retry,
    "backward_solve_sec": _safe_float(offense_diag, "elapsed_sec") + _safe_float(defense_diag, "elapsed_sec"),
    "backward_mode": f"player_separable[{offense_mode},{defense_mode}]",
    "backward_reduced_dual_attempted": bool(offense_diag.get("reduced_dual_attempted", False)) or bool(defense_diag.get("reduced_dual_attempted", False)),
    "backward_fallback": bool(offense_diag.get("fallback", False)) or bool(defense_diag.get("fallback", False)),
    "backward_reduced_dual_residual_norm": reduced_resid,
    "backward_reduced_dual_regularization": reduced_reg,
    "backward_solve_sec_offense": _safe_float(offense_diag, "elapsed_sec"),
    "backward_solve_sec_defense": _safe_float(defense_diag, "elapsed_sec"),
    "backward_mode_offense": offense_mode,
    "backward_mode_defense": defense_mode,
    "backward_reduced_dual_regularization_offense": None if offense_reg is None else float(offense_reg),
    "backward_reduced_dual_regularization_defense": None if defense_reg is None else float(defense_reg),
  }


def _aggregate_exact_lq_player_solve_result(
  problem: HexnerTreeProblem,
  theta: jnp.ndarray,
  offense_result: HexnerExactLQPlayerSolve,
  defense_result: HexnerExactLQPlayerSolve,
) -> dict[str, Any]:
  full_point = _combine_hexner_player_points(offense_result.point, defense_result.point)
  evaluation = problem.evaluate(full_point, theta)
  linearization = problem.linearize_kkt(full_point, theta)
  residual = problem.kkt_residual(full_point, theta, linearization)
  residual_norm = float(_tree_max_abs(residual))
  residual_history = jnp.array([residual_norm], dtype=F32)
  one_alpha = jnp.array([1.0], dtype=F32)
  evaluation.update(
    {
      "point": full_point,
      "solver_variant": "player_separable_lq",
      "variant2_constraint_mode": "exact_lq",
      "box_constraints_active": False,
      "residual_norm": residual_norm,
      "num_iterations": 1,
      "num_iterations_total": 2,
      "num_iterations_offense": 1,
      "num_iterations_defense": 1,
      "status": int(SolverStatus.SUCCESS),
      "status_offense": int(SolverStatus.SUCCESS),
      "status_defense": int(SolverStatus.SUCCESS),
      "residual_history": residual_history,
      "line_search_alphas": one_alpha,
      "gmres_infos": jnp.array([0.0], dtype=F32),
      "residual_history_offense": residual_history,
      "residual_history_defense": residual_history,
      "line_search_alphas_offense": one_alpha,
      "line_search_alphas_defense": one_alpha,
      "gmres_infos_offense": jnp.array([0.0], dtype=F32),
      "gmres_infos_defense": jnp.array([0.0], dtype=F32),
      "objective_offense_lq": float(offense_result.objective),
      "objective_defense_lq": float(defense_result.objective),
      "forward_linear_mode": f"player_separable[{offense_result.mode},{defense_result.mode}]",
    },
  )
  return evaluation


def _solve_hexner_fixed_alpha_player_separable_lq(
  problem: HexnerTreeProblem,
  inner_cfg: TreeDiffMPCConfig,
  *,
  alpha_logits: jnp.ndarray | None = None,
  initial_point: EqualityGamePoint | None = None,
) -> dict[str, Any]:
  offense_problem = _make_hexner_player_problem(problem, "offense")
  defense_problem = _make_hexner_player_problem(problem, "defense")
  if offense_problem.supports_exact_lq_solver() and defense_problem.supports_exact_lq_solver():
    del inner_cfg, initial_point
    theta = problem.empty_alpha_logits() if alpha_logits is None else alpha_logits
    offense_result = offense_problem.solve_exact_lq(theta)
    defense_result = defense_problem.solve_exact_lq(theta)
    return _aggregate_exact_lq_player_solve_result(problem, theta, offense_result, defense_result)
  if problem.has_box_inequality_constraints and not problem.cfg.squash_controls:
    return _solve_hexner_fixed_alpha_player_separable(
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
    "Hexner Variant 2 currently requires squash_controls=False. "
    "Equality-only instances use the exact Riccati path; box constraints use the constrained barrier fallback.",
  )


def _solve_hexner_bilevel_player_separable_lq(
  problem: HexnerTreeProblem,
  cfg: HexnerBilevelConfig,
  *,
  alpha_logits: jnp.ndarray | None = None,
  warm_point: EqualityGamePoint | None = None,
  progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
  offense_problem = _make_hexner_player_problem(problem, "offense")
  defense_problem = _make_hexner_player_problem(problem, "defense")
  if problem.has_box_inequality_constraints and not problem.cfg.squash_controls:
    return _solve_hexner_bilevel_player_separable(
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
      "Hexner Variant 2 exact LQ solve requires squash_controls=False and no active box constraints.",
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
    combined_point = _combine_hexner_player_points(offense_point, defense_point)
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

  evaluation = _solve_hexner_fixed_alpha_player_separable_lq(
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


def _solve_hexner_fixed_alpha_player_separable(
  problem: HexnerTreeProblem,
  inner_cfg: TreeDiffMPCConfig,
  *,
  alpha_logits: jnp.ndarray | None = None,
  initial_point: EqualityGamePoint | None = None,
  solver_variant_label: str = "player_separable",
  extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
  theta = problem.empty_alpha_logits() if alpha_logits is None else alpha_logits
  offense_problem = _make_hexner_player_problem(problem, "offense")
  defense_problem = _make_hexner_player_problem(problem, "defense")
  offense_solver = TreeDiffMPCSolver(offense_problem, inner_cfg)
  defense_solver = TreeDiffMPCSolver(defense_problem, inner_cfg)

  if initial_point is None:
    offense_initial = offense_problem.initial_point(theta)
    defense_initial = defense_problem.initial_point(theta)
  else:
    offense_initial, defense_initial = _split_hexner_full_point_by_player(problem, initial_point)

  offense_result = offense_solver.solve_with_metadata(theta, initial_point=offense_initial)
  defense_result = defense_solver.solve_with_metadata(theta, initial_point=defense_initial)
  return _aggregate_player_solve_result(
    problem,
    theta,
    offense_result,
    defense_result,
    solver_variant=solver_variant_label,
    extra_fields=extra_fields,
  )


def _solve_hexner_bilevel_player_separable(
  problem: HexnerTreeProblem,
  cfg: HexnerBilevelConfig,
  *,
  alpha_logits: jnp.ndarray | None = None,
  warm_point: EqualityGamePoint | None = None,
  progress_callback: Callable[[dict[str, Any]], None] | None = None,
  solver_variant_label: str = "player_separable",
  extra_history_fields: dict[str, Any] | None = None,
  extra_evaluation_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
  offense_problem = _make_hexner_player_problem(problem, "offense")
  defense_problem = _make_hexner_player_problem(problem, "defense")
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
  if warm_point is None:
    offense_warm = offense_problem.initial_point(theta)
    defense_warm = defense_problem.initial_point(theta)
  else:
    offense_warm, defense_warm = _split_hexner_full_point_by_player(problem, warm_point)

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
      combined_point = _combine_hexner_player_points(offense_point, defense_point)
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
  evaluation = _aggregate_player_solve_result(
    problem,
    theta,
    offense_result,
    defense_result,
    solver_variant=solver_variant_label,
    extra_fields=extra_evaluation_fields,
  )
  evaluation.update(
    {
      "history": history,
      "outer_status": outer_status,
      "outer_steps_used": len(history),
    },
  )
  return evaluation


def solve_hexner_fixed_alpha(
  problem: HexnerTreeProblem,
  inner_cfg: TreeDiffMPCConfig,
  *,
  alpha_logits: jnp.ndarray | None = None,
  initial_point: EqualityGamePoint | None = None,
  solver_variant: str = "coupled",
) -> dict[str, Any]:
  if solver_variant == "player_separable":
    return _solve_hexner_fixed_alpha_player_separable(
      problem,
      inner_cfg,
      alpha_logits=alpha_logits,
      initial_point=initial_point,
    )
  if solver_variant == "player_separable_lq":
    return _solve_hexner_fixed_alpha_player_separable_lq(
      problem,
      inner_cfg,
      alpha_logits=alpha_logits,
      initial_point=initial_point,
    )
  if solver_variant != "coupled":
    raise ValueError(f"Unsupported Hexner solver_variant: {solver_variant}")
  solver = TreeDiffMPCSolver(problem, inner_cfg)
  theta = problem.empty_alpha_logits() if alpha_logits is None else alpha_logits
  result = solver.solve_with_metadata(theta, initial_point=initial_point)
  evaluation = problem.evaluate(result.point, theta)
  evaluation.update(
    {
      "point": result.point,
      "solver_variant": "coupled",
      "residual_norm": float(result.residual_norm),
      "num_iterations": int(result.num_iterations),
      "status": int(result.status),
      "residual_history": result.residual_history,
      "line_search_alphas": result.line_search_alphas,
      "gmres_infos": result.gmres_infos,
    },
  )
  return evaluation


def solve_hexner_bilevel(
  problem: HexnerTreeProblem,
  cfg: HexnerBilevelConfig,
  *,
  alpha_logits: jnp.ndarray | None = None,
  warm_point: EqualityGamePoint | None = None,
  progress_callback: Callable[[dict[str, Any]], None] | None = None,
  solver_variant: str = "coupled",
) -> dict[str, Any]:
  if solver_variant == "player_separable":
    return _solve_hexner_bilevel_player_separable(
      problem,
      cfg,
      alpha_logits=alpha_logits,
      warm_point=warm_point,
      progress_callback=progress_callback,
    )
  if solver_variant == "player_separable_lq":
    return _solve_hexner_bilevel_player_separable_lq(
      problem,
      cfg,
      alpha_logits=alpha_logits,
      warm_point=warm_point,
      progress_callback=progress_callback,
    )
  if solver_variant != "coupled":
    raise ValueError(f"Unsupported Hexner solver_variant: {solver_variant}")
  solver = TreeDiffMPCSolver(problem, cfg.inner)
  theta = (
    problem.init_alpha_logits(
      seed=cfg.outer.seed,
      init_scale=cfg.outer.alpha_init_scale,
      mode=cfg.outer.alpha_init_mode,
    )
    if alpha_logits is None
    else alpha_logits
  )
  point_warm = problem.initial_point(theta) if warm_point is None else warm_point

  optimizer = _make_optimizer(cfg.outer.optimizer, cfg.outer.lr_alpha)
  opt_state = optimizer.init(theta)
  history: list[dict[str, Any]] = []
  previous_loss: float | None = None
  outer_status = "max_steps"

  for outer_step in range(cfg.outer.steps):
    outer_step_start = time.perf_counter()

    def outer_objective(theta_value: jnp.ndarray) -> tuple[jnp.ndarray, EqualityGamePoint]:
      point_value = solver.solve_point(theta_value, point_warm)
      return problem.objective(point_value.primal, theta_value), point_value

    (loss_value, solved_point), grad_theta = jax.value_and_grad(outer_objective, has_aux=True)(theta)
    updates, opt_state = optimizer.update(grad_theta, opt_state, theta)
    theta = optax.apply_updates(theta, updates)
    if cfg.outer.alpha_logit_clip is not None:
      theta = jnp.clip(theta, -cfg.outer.alpha_logit_clip, cfg.outer.alpha_logit_clip)
    point_warm = jax.tree_util.tree_map(jax.lax.stop_gradient, solved_point)

    alpha = build_alpha_from_logits(theta, problem.tree)
    (
      leaf_type_probs,
      node_public_probs,
      _,
      _,
    ) = propagate_type_probabilities(alpha, problem.topology, problem.prior)
    leaf_probs = node_public_probs[problem.topology.leaf_nodes]
    leaf_type_probs = leaf_type_probs[problem.topology.leaf_nodes]
    leaf_beliefs = leaf_type_probs / jnp.clip(leaf_probs[:, None], 1e-8, None)
    diagnostics = solver.get_last_solve_point_diagnostics()
    forward_diag = diagnostics.get("forward", {})
    backward_diag = diagnostics.get("backward", {})
    history_entry = {
      "outer_step": outer_step + 1,
      "loss": float(loss_value),
      "loss_change": None if previous_loss is None else float(loss_value) - previous_loss,
      "alpha_grad_norm": _tree_l2_norm(grad_theta),
      "step_elapsed_sec": time.perf_counter() - outer_step_start,
      "solver_variant": "coupled",
      "forward_solve_sec": float(forward_diag.get("elapsed_sec", float("nan"))),
      "forward_num_iterations": int(
        forward_diag.get("num_step_iterations", forward_diag.get("num_iterations", -1))
      ),
      "forward_initial_residual_norm": float(forward_diag.get("initial_residual_norm", float("nan"))),
      "forward_residual_norm": float(forward_diag.get("residual_norm", float("nan"))),
      "forward_status": int(forward_diag.get("status", -1)),
      "forward_linear_mode": str(forward_diag.get("linear_mode_last", "unknown")),
      "forward_used_singularity_aware_partial_elimination": bool(
        forward_diag.get("used_singularity_aware_partial_elimination", False)
      ),
      "forward_approximate_pruning_used": bool(forward_diag.get("approximate_pruning_used", False)),
      "forward_line_search_last_alpha": (
        None
        if forward_diag.get("line_search_last_alpha") is None
        else float(forward_diag["line_search_last_alpha"])
      ),
      "forward_line_search_min_alpha": (
        None
        if forward_diag.get("line_search_min_alpha") is None
        else float(forward_diag["line_search_min_alpha"])
      ),
      "forward_line_search_mean_alpha": (
        None
        if forward_diag.get("line_search_mean_alpha") is None
        else float(forward_diag["line_search_mean_alpha"])
      ),
      "forward_line_search_zero_count": int(forward_diag.get("line_search_zero_count", 0)),
      "forward_line_search_stalled": bool(forward_diag.get("line_search_stalled", False)),
      "forward_regularization_initial": (
        None
        if forward_diag.get("regularization_initial") is None
        else float(forward_diag["regularization_initial"])
      ),
      "forward_regularization_last": (
        None
        if forward_diag.get("regularization_last") is None
        else float(forward_diag["regularization_last"])
      ),
      "forward_regularization_min": (
        None
        if forward_diag.get("regularization_min") is None
        else float(forward_diag["regularization_min"])
      ),
      "forward_regularization_max": (
        None
        if forward_diag.get("regularization_max") is None
        else float(forward_diag["regularization_max"])
      ),
      "forward_regularization_retry_count": int(forward_diag.get("regularization_retry_count", 0)),
      "forward_regularization_retry_last": int(forward_diag.get("regularization_retry_last", 0)),
      "backward_solve_sec": float(backward_diag.get("elapsed_sec", float("nan"))),
      "backward_mode": str(backward_diag.get("mode", "unknown")),
      "backward_reduced_dual_attempted": bool(backward_diag.get("reduced_dual_attempted", False)),
      "backward_fallback": bool(backward_diag.get("fallback", False)),
      "backward_reduced_dual_residual_norm": (
        None
        if backward_diag.get("reduced_dual_residual_norm") is None
        else float(backward_diag["reduced_dual_residual_norm"])
      ),
      "backward_reduced_dual_regularization": (
        None
        if backward_diag.get("reduced_dual_regularization") is None
        else float(backward_diag["reduced_dual_regularization"])
      ),
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

  fixed_result = solver.solve_with_metadata(theta, initial_point=point_warm)
  evaluation = problem.evaluate(fixed_result.point, theta)
  evaluation.update(
    {
      "point": fixed_result.point,
      "solver_variant": "coupled",
      "residual_norm": float(fixed_result.residual_norm),
      "num_iterations": int(fixed_result.num_iterations),
      "status": int(fixed_result.status),
      "residual_history": fixed_result.residual_history,
      "line_search_alphas": fixed_result.line_search_alphas,
      "gmres_infos": fixed_result.gmres_infos,
      "history": history,
      "outer_status": outer_status,
      "outer_steps_used": len(history),
    },
  )
  return evaluation
