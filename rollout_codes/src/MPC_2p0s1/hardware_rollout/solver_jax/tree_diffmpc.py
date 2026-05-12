from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import cg, gmres

from .public_tree import PublicTreeTopology


class EqualityGamePrimal(NamedTuple):
  node_states: jnp.ndarray
  offense_controls: jnp.ndarray
  defense_controls: jnp.ndarray


class EqualityGameDual(NamedTuple):
  node_multipliers: jnp.ndarray
  terminal_multipliers: jnp.ndarray


class EqualityGamePoint(NamedTuple):
  primal: EqualityGamePrimal
  dual: EqualityGameDual


class SolverStatus(enum.IntEnum):
  SUCCESS = 0
  MAX_ITERATIONS = 1
  NONFINITE = -1


class EqualityGameSolveResult(NamedTuple):
  point: EqualityGamePoint
  objective: jnp.ndarray
  residual_norm: jnp.ndarray
  num_iterations: jnp.ndarray
  num_step_iterations: jnp.ndarray
  status: jnp.ndarray
  residual_history: jnp.ndarray
  line_search_alphas: jnp.ndarray
  regularization_history: jnp.ndarray
  regularization_retry_counts: jnp.ndarray
  gmres_infos: jnp.ndarray


class TreeSchurBlockData(NamedTuple):
  variable_dim: jnp.ndarray
  variable_mask: jnp.ndarray
  self_blocks: jnp.ndarray
  parent_coupling: jnp.ndarray


class TreeSchurPreconditioner(NamedTuple):
  variable_dim: jnp.ndarray
  variable_mask: jnp.ndarray
  parent_coupling: jnp.ndarray
  schur_inv: jnp.ndarray


def stabilized_symmetric_inverse(blocks: jnp.ndarray, epsilon: float) -> jnp.ndarray:
  epsilon_value = jnp.asarray(epsilon, dtype=blocks.dtype)
  sym_blocks = 0.5 * (blocks + jnp.swapaxes(blocks, -1, -2))
  eigenvalues, eigenvectors = jnp.linalg.eigh(sym_blocks)
  signs = jnp.where(eigenvalues >= 0.0, 1.0, -1.0).astype(blocks.dtype)
  safe_eigenvalues = signs * jnp.maximum(jnp.abs(eigenvalues), epsilon_value)
  inverse_eigenvalues = 1.0 / safe_eigenvalues
  scaled_eigenvectors = eigenvectors * inverse_eigenvalues[..., None, :]
  return scaled_eigenvectors @ jnp.swapaxes(eigenvectors, -1, -2)


def factor_tree_schur_blocks(
  topology: PublicTreeTopology,
  self_blocks: jnp.ndarray,
  parent_coupling: jnp.ndarray,
  epsilon: float,
) -> jnp.ndarray:
  schur_blocks = self_blocks
  schur_inv = jnp.zeros_like(self_blocks)
  for depth in range(topology.total_depth, -1, -1):
    start = topology.node_offsets_py[depth]
    end = topology.node_offsets_py[depth + 1] if depth < topology.total_depth else topology.node_count
    local_inv = stabilized_symmetric_inverse(schur_blocks[start:end], epsilon)
    schur_inv = schur_inv.at[start:end].set(local_inv)
    if depth > 0:
      local_coupling = parent_coupling[start:end]
      parents = topology.node_parents[start:end]
      schur_update = jnp.einsum("nij,njk,nlk->nil", local_coupling, local_inv, local_coupling)
      schur_blocks = schur_blocks.at[parents].add(-schur_update)
  return schur_inv


def build_tree_schur_preconditioner(
  topology: PublicTreeTopology,
  block_data: TreeSchurBlockData,
  epsilon: float,
) -> TreeSchurPreconditioner:
  return TreeSchurPreconditioner(
    variable_dim=block_data.variable_dim,
    variable_mask=block_data.variable_mask,
    parent_coupling=block_data.parent_coupling,
    schur_inv=factor_tree_schur_blocks(
      topology,
      block_data.self_blocks,
      block_data.parent_coupling,
      epsilon,
    ),
  )


def apply_tree_schur_preconditioner(
  topology: PublicTreeTopology,
  schur_inv: jnp.ndarray,
  parent_coupling: jnp.ndarray,
  rhs: jnp.ndarray,
) -> jnp.ndarray:
  reduced_rhs = rhs
  for depth in range(topology.total_depth, 0, -1):
    start = topology.node_offsets_py[depth]
    end = topology.node_offsets_py[depth + 1] if depth < topology.total_depth else topology.node_count
    local_coupling = parent_coupling[start:end]
    parents = topology.node_parents[start:end]
    child_response = jnp.einsum("nij,nj->ni", schur_inv[start:end], reduced_rhs[start:end])
    parent_update = -jnp.einsum("nij,nj->ni", local_coupling, child_response)
    reduced_rhs = reduced_rhs.at[parents].add(parent_update)

  solution = jnp.zeros_like(rhs)
  solution = solution.at[0:1].set(jnp.einsum("nij,nj->ni", schur_inv[0:1], reduced_rhs[0:1]))
  for depth in range(1, topology.total_depth + 1):
    start = topology.node_offsets_py[depth]
    end = topology.node_offsets_py[depth + 1] if depth < topology.total_depth else topology.node_count
    local_coupling = parent_coupling[start:end]
    parent_solution = solution[topology.node_parents[start:end]]
    local_rhs = reduced_rhs[start:end] - jnp.einsum("nji,nj->ni", local_coupling, parent_solution)
    local_solution = jnp.einsum("nij,nj->ni", schur_inv[start:end], local_rhs)
    solution = solution.at[start:end].set(local_solution)
  return solution


@dataclass(frozen=True)
class TreeDiffMPCConfig:
  max_sqp_iterations: int = 24
  residual_tolerance: float = 1e-6
  gmres_tolerance: float = 1e-7
  gmres_restart: int = 40
  gmres_maxiter: int = 200
  regularization: float = 1e-6
  adaptive_regularization: bool = True
  regularization_min: float = 1e-8
  regularization_max: float = 1.0
  regularization_increase_factor: float = 10.0
  regularization_decrease_factor: float = 0.5
  regularization_retry_limit: int = 3
  singularity_aware_partial_elimination: bool = True
  partial_elimination_curvature_threshold: float = 1e-10
  near_zero_branch_prune_threshold: float = 0.0
  singularity_aware_dense_solve: bool = True
  singularity_aware_dense_solve_max_dim: int = 1024
  singularity_aware_state_dual_preconditioner_max_dim: int = 1536
  linear_solver: str = "full_gmres"
  drone3d_box_terminal_backend: str = "riccati_first"
  drone3d_box_lq_backend: str = "active_set"
  preconditioner_epsilon: float = 1e-4
  adjoint_regularization_epsilon: float = 1e-8
  linesearch_eta: float = 1e-4
  linesearch_mu_floor: float = 1e-1
  line_search_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.05)
  singularity_aware_nonlinear_residual_guard: bool = True


class TreeDiffMPCSolver:
  def __init__(self, problem: Any, cfg: TreeDiffMPCConfig = TreeDiffMPCConfig()) -> None:
    self.problem = problem
    self.cfg = cfg
    self._last_solve_point_forward_stats: dict[str, Any] = {}
    self._last_solve_point_backward_stats: dict[str, Any] = {}
    self._last_forward_regularization: float | None = None
    self.solve_point = self._make_solve_point()

  def get_last_solve_point_diagnostics(self) -> dict[str, dict[str, Any]]:
    return {
      "forward": dict(self._last_solve_point_forward_stats),
      "backward": dict(self._last_solve_point_backward_stats),
    }

  def _forward_stats_from_result(self, result: EqualityGameSolveResult) -> dict[str, Any]:
    num_iterations = int(result.num_iterations)
    num_step_iterations = int(result.num_step_iterations)
    stats = {
      "num_iterations": num_iterations,
      "num_step_iterations": num_step_iterations,
      "residual_norm": float(result.residual_norm),
      "status": int(result.status),
    }
    if num_iterations <= 0:
      stats.update(
        {
          "initial_residual_norm": float(result.residual_norm),
          "line_search_last_alpha": None,
          "line_search_min_alpha": None,
          "line_search_mean_alpha": None,
          "line_search_zero_count": 0,
          "line_search_stalled": False,
          "regularization_initial": float(self.cfg.regularization),
          "regularization_last": float(self.cfg.regularization),
          "regularization_min": float(self.cfg.regularization),
          "regularization_max": float(self.cfg.regularization),
          "regularization_retry_count": 0,
          "regularization_retry_last": 0,
        },
      )
      return stats

    residual_history = jnp.asarray(result.residual_history)[:num_iterations]
    alpha_history = jnp.asarray(result.line_search_alphas)[:num_step_iterations]
    regularization_history = jnp.asarray(result.regularization_history)[:num_step_iterations]
    regularization_retry_history = jnp.asarray(result.regularization_retry_counts)[:num_step_iterations]
    alpha_values = [float(value) for value in alpha_history.tolist()]
    positive_alphas = [value for value in alpha_values if value > 0.0]
    regularization_values = [float(value) for value in regularization_history.tolist()]
    regularization_retry_values = [int(value) for value in regularization_retry_history.tolist()]
    stats.update(
      {
        "initial_residual_norm": float(residual_history[0]),
        "line_search_last_alpha": alpha_values[-1] if alpha_values else None,
        "line_search_min_alpha": min(positive_alphas) if positive_alphas else None,
        "line_search_mean_alpha": (
          sum(positive_alphas) / len(positive_alphas) if positive_alphas else None
        ),
        "line_search_zero_count": sum(1 for value in alpha_values if value == 0.0),
        "line_search_stalled": bool(alpha_values[-1] == 0.0) if alpha_values else False,
        "regularization_initial": (
          regularization_values[0] if regularization_values else float(self.cfg.regularization)
        ),
        "regularization_last": (
          regularization_values[-1] if regularization_values else float(self.cfg.regularization)
        ),
        "regularization_min": (
          min(regularization_values) if regularization_values else float(self.cfg.regularization)
        ),
        "regularization_max": (
          max(regularization_values) if regularization_values else float(self.cfg.regularization)
        ),
        "regularization_retry_count": sum(regularization_retry_values),
        "regularization_retry_last": regularization_retry_values[-1] if regularization_retry_values else 0,
      },
    )
    return stats

  def _increase_forward_regularization(self, regularization: float) -> float:
    minimum = max(0.0, float(self.cfg.regularization_min))
    maximum = max(minimum, float(self.cfg.regularization_max))
    factor = max(1.0, float(self.cfg.regularization_increase_factor))
    current = abs(float(regularization))
    scaled = max(current, minimum)
    if scaled == 0.0:
      scaled = minimum
    return min(maximum, scaled * factor if scaled > 0.0 else minimum)

  def _update_forward_regularization_after_accept(
    self,
    regularization: float,
    accepted_alpha: float,
    retry_count: int,
  ) -> float:
    current = abs(float(regularization))
    if not self.cfg.adaptive_regularization:
      return current
    if retry_count > 0:
      return current

    minimum = max(0.0, float(self.cfg.regularization_min))
    decrease_factor = min(1.0, max(0.0, float(self.cfg.regularization_decrease_factor)))
    full_step_alpha = max(float(alpha) for alpha in self.cfg.line_search_alphas)
    minimum_step_alpha = min(float(alpha) for alpha in self.cfg.line_search_alphas)

    if accepted_alpha >= full_step_alpha:
      if current == 0.0:
        return 0.0
      return max(minimum, current * decrease_factor)
    if accepted_alpha <= minimum_step_alpha:
      return self._increase_forward_regularization(current)
    return current

  def constraints(
    self,
    primal: EqualityGamePrimal,
    theta: Any,
  ) -> EqualityGameDual:
    return EqualityGameDual(
      node_multipliers=self.problem.node_constraints(primal, theta),
      terminal_multipliers=self.problem.terminal_constraints(primal, theta),
    )

  def _supports_structured_kkt(self) -> bool:
    return all(
      hasattr(self.problem, name)
      for name in ("linearize_kkt", "kkt_residual", "kkt_matvec")
    )

  def _supports_reduced_dual_kkt(self) -> bool:
    return all(
      hasattr(self.problem, name)
      for name in (
        "reduced_dual_matvec",
        "reduced_dual_rhs",
        "recover_primal_step_from_reduced_dual",
      )
    )

  def _supports_reduced_dual_preconditioner(self) -> bool:
    return self._supports_explicit_tree_schur_preconditioner() or all(
      hasattr(self.problem, name)
      for name in (
        "build_reduced_dual_preconditioner",
        "apply_reduced_dual_preconditioner",
      )
    )

  def _supports_explicit_tree_schur_preconditioner(self) -> bool:
    return all(
      hasattr(self.problem, name)
      for name in (
        "build_tree_schur_preconditioner_data",
        "pack_reduced_dual_for_tree_schur",
        "unpack_reduced_dual_from_tree_schur",
      )
    )

  def _supports_singularity_aware_structured_linear_solve(self) -> bool:
    return (
      self.cfg.singularity_aware_partial_elimination
      and hasattr(self.problem, "solve_singularity_aware_linearized_system")
    )

  def _singularity_aware_structured_linear_solve_uses_regularization(self) -> bool:
    indicator = getattr(
      self.problem,
      "singularity_aware_linearized_system_uses_regularization",
      None,
    )
    if indicator is None:
      return True
    if callable(indicator):
      return bool(indicator())
    return bool(indicator)

  def _structured_kkt_is_symmetric(self) -> bool:
    indicator = getattr(self.problem, "kkt_operator_is_symmetric", False)
    return bool(indicator() if callable(indicator) else indicator)

  def _reduced_dual_operator_is_spd(self, regularization: float) -> bool:
    indicator = getattr(self.problem, "reduced_dual_operator_is_spd", False)
    if callable(indicator):
      return bool(indicator(regularization))
    return bool(indicator)

  def _reduced_dual_operator_cg_sign(self, regularization: float) -> float | None:
    sign_indicator = getattr(self.problem, "reduced_dual_operator_cg_sign", None)
    sign = sign_indicator(regularization) if callable(sign_indicator) else sign_indicator
    if sign is not None:
      sign_value = float(sign)
      if sign_value not in (-1.0, 1.0):
        raise ValueError("Reduced-dual CG sign must be +1 or -1 when provided.")
      return sign_value
    if self._reduced_dual_operator_is_spd(regularization):
      return 1.0
    return None

  def _supports_reduced_dual_adjoint(self, regularization: float) -> bool:
    indicator = getattr(self.problem, "supports_reduced_dual_adjoint", None)
    if indicator is None:
      return True
    if callable(indicator):
      return bool(indicator(regularization))
    return bool(indicator)

  def _reduced_dual_adjoint_regularization(self, regularization: float) -> float | None:
    requested_regularization = float(regularization)
    if self._supports_reduced_dual_adjoint(requested_regularization):
      return requested_regularization
    if requested_regularization == 0.0:
      stabilized_regularization = float(self.cfg.adjoint_regularization_epsilon)
      if stabilized_regularization > 0.0 and self._supports_reduced_dual_adjoint(stabilized_regularization):
        return stabilized_regularization
    return None

  def lagrangian(
    self,
    primal: EqualityGamePrimal,
    dual: EqualityGameDual,
    theta: Any,
  ) -> jnp.ndarray:
    constraints = self.constraints(primal, theta)
    return (
      self.problem.objective(primal, theta)
      + jnp.sum(dual.node_multipliers * constraints.node_multipliers)
      + jnp.sum(dual.terminal_multipliers * constraints.terminal_multipliers)
    )

  def residual(
    self,
    point: EqualityGamePoint,
    theta: Any,
  ) -> EqualityGamePoint:
    if hasattr(self.problem, "kkt_residual"):
      return self.problem.kkt_residual(point, theta)
    primal_grad = jax.grad(lambda primal: self.lagrangian(primal, point.dual, theta))(point.primal)
    constraints = self.constraints(point.primal, theta)
    return EqualityGamePoint(primal=primal_grad, dual=constraints)

  def initial_point(self, theta: Any) -> EqualityGamePoint:
    return self.problem.initial_point(theta)

  def _solve_linearized_system(
    self,
    *,
    flat_current: jnp.ndarray,
    point_current: EqualityGamePoint,
    theta: Any,
    residual_value: jnp.ndarray,
    residual_point: EqualityGamePoint | None,
    linearization: Any,
    unravel_point: Any,
    residual_from_flat: Any,
    regularization: float | None = None,
  ) -> tuple[EqualityGamePoint, jnp.ndarray, float, dict[str, Any]]:
    structured_kkt = linearization is not None and residual_point is not None
    reg = self.cfg.regularization if regularization is None else regularization
    singularity_aware_partial_failed = False

    def solve_full_system(
      *,
      mode: str = "full_gmres",
      exact: bool = True,
      singularity_aware_partial_fallback: bool = False,
    ) -> tuple[EqualityGamePoint, jnp.ndarray, float, dict[str, Any]]:
      if structured_kkt:
        def matvec(vec: jnp.ndarray) -> jnp.ndarray:
          tangent = unravel_point(vec)
          product = self.problem.kkt_matvec(tangent, theta, linearization)
          return ravel_pytree(product)[0] + reg * vec
      else:
        _, linearized_residual = jax.linearize(residual_from_flat, flat_current)

        def matvec(vec: jnp.ndarray) -> jnp.ndarray:
          return linearized_residual(vec) + reg * vec

      matvec = jax.jit(matvec)
      step_flat, gmres_info = gmres(
        matvec,
        -residual_value,
        tol=self.cfg.gmres_tolerance,
        restart=self.cfg.gmres_restart,
        maxiter=self.cfg.gmres_maxiter,
      )
      return (
        unravel_point(step_flat),
        step_flat,
        float(gmres_info),
        {
          "mode": mode,
          "exact": exact,
          "singularity_aware_partial_fallback": singularity_aware_partial_fallback,
          "dynamics_mode": (
            getattr(linearization, "dynamics_mode", "unknown")
            if structured_kkt and linearization is not None
            else "unknown"
          ),
          "fastpath_max_contact_weight": (
            getattr(linearization, "fastpath_max_contact_weight", None)
            if structured_kkt and linearization is not None
            else None
          ),
          "fastpath_min_speed_margin": (
            getattr(linearization, "fastpath_min_speed_margin", None)
            if structured_kkt and linearization is not None
            else None
          ),
        },
      )

    if (
      structured_kkt
      and self.cfg.linear_solver != "full_gmres"
      and self._supports_singularity_aware_structured_linear_solve()
    ):
      step_point, linear_solver_info, solver_meta = self.problem.solve_singularity_aware_linearized_system(
        residual=residual_point,
        point_current=point_current,
        theta=theta,
        linearization=linearization,
        regularization=reg,
        cfg=self.cfg,
      )
      step_flat, _ = ravel_pytree(step_point)
      if bool(jnp.all(jnp.isfinite(step_flat))):
        full_product = self.problem.kkt_matvec(step_point, theta, linearization)
        full_step_residual = ravel_pytree(full_product)[0] + reg * step_flat + residual_value
        full_step_residual_norm = float(jnp.max(jnp.abs(full_step_residual)))
        residual_scale = max(1.0, float(jnp.max(jnp.abs(residual_value))))
        exact_step = bool(solver_meta.get("exact", True))
        if (
          not exact_step
          or (
            bool(jnp.isfinite(full_step_residual_norm))
            and full_step_residual_norm <= max(10.0 * self.cfg.gmres_tolerance * residual_scale, 1e-6)
          )
        ):
          return step_point, step_flat, linear_solver_info, solver_meta
      singularity_aware_partial_failed = True

    if structured_kkt and self.cfg.linear_solver in (
      "reduced_dual_gmres",
      "reduced_dual_pgmres",
      "reduced_dual_cg",
      "reduced_dual_pcg",
    ):
      if not self._supports_reduced_dual_kkt():
        raise ValueError("Configured reduced-dual linear solve, but the problem does not implement it.")
      use_preconditioner = self.cfg.linear_solver in ("reduced_dual_pgmres", "reduced_dual_pcg")
      use_cg = self.cfg.linear_solver in ("reduced_dual_cg", "reduced_dual_pcg")
      if use_preconditioner and not self._supports_reduced_dual_preconditioner():
        raise ValueError("Configured reduced-dual preconditioned solve, but the problem does not implement a reduced-dual preconditioner.")
      cg_sign = None if not use_cg else self._reduced_dual_operator_cg_sign(reg)
      if use_cg and cg_sign is None:
        if singularity_aware_partial_failed:
          return solve_full_system(
            mode="full_gmres_fallback_from_singularity_aware_partial",
            exact=True,
            singularity_aware_partial_fallback=True,
          )
        raise ValueError(
          "Configured reduced-dual CG solve, but the problem does not declare a symmetric definite reduced-dual operator and sign for this regularization.",
        )
      dual_template = jax.tree_util.tree_map(jnp.asarray, point_current.dual)
      _, unravel_dual = ravel_pytree(dual_template)
      dual_rhs = self.problem.reduced_dual_rhs(
        residual_point,
        theta,
        linearization,
        reg,
      )
      apply_preconditioner = None
      if use_preconditioner:
        if self._supports_explicit_tree_schur_preconditioner():
          block_data = self.problem.build_tree_schur_preconditioner_data(
            theta,
            linearization,
            reg,
            self.cfg.preconditioner_epsilon,
          )
          preconditioner = build_tree_schur_preconditioner(
            self.problem.topology,
            block_data,
            self.cfg.preconditioner_epsilon,
          )

          def apply_preconditioner(dual_value: EqualityGameDual) -> EqualityGameDual:
            packed_rhs = self.problem.pack_reduced_dual_for_tree_schur(
              dual_value,
              theta,
              linearization,
              reg,
              preconditioner,
            )
            packed_rhs = packed_rhs * preconditioner.variable_mask
            packed_solution = apply_tree_schur_preconditioner(
              self.problem.topology,
              preconditioner.schur_inv,
              preconditioner.parent_coupling,
              packed_rhs,
            )
            packed_solution = packed_solution * preconditioner.variable_mask
            return self.problem.unpack_reduced_dual_from_tree_schur(
              packed_solution,
              theta,
              linearization,
              reg,
              preconditioner,
            )
        else:
          preconditioner = self.problem.build_reduced_dual_preconditioner(
            theta,
            linearization,
            reg,
            self.cfg.preconditioner_epsilon,
          )

          def apply_preconditioner(dual_value: EqualityGameDual) -> EqualityGameDual:
            return self.problem.apply_reduced_dual_preconditioner(
              dual_value,
              theta,
              linearization,
              reg,
              preconditioner,
            )

      if use_cg:
        assert cg_sign is not None
        flat_dual_rhs, _ = ravel_pytree(dual_rhs)
        flat_dual_rhs = cg_sign * flat_dual_rhs

        def reduced_matvec(vec: jnp.ndarray) -> jnp.ndarray:
          dual_tangent = unravel_dual(vec)
          product = self.problem.reduced_dual_matvec(
            dual_tangent,
            theta,
            linearization,
            reg,
          )
          return cg_sign * ravel_pytree(product)[0]

        reduced_matvec = jax.jit(reduced_matvec)
        preconditioner_matvec = None
        if use_preconditioner:
          assert apply_preconditioner is not None

          def preconditioner_matvec(vec: jnp.ndarray) -> jnp.ndarray:
            dual_value = unravel_dual(vec)
            product = apply_preconditioner(dual_value)
            return cg_sign * ravel_pytree(product)[0]

          preconditioner_matvec = jax.jit(preconditioner_matvec)
        dual_step_flat, cg_info = cg(
          reduced_matvec,
          flat_dual_rhs,
          tol=self.cfg.gmres_tolerance,
          maxiter=self.cfg.gmres_maxiter,
          M=preconditioner_matvec,
        )
        dual_step = unravel_dual(dual_step_flat)
        linear_solver_info = 0.0 if cg_info is None else float(cg_info)
      else:
        assert self.cfg.linear_solver in ("reduced_dual_gmres", "reduced_dual_pgmres")
        if use_preconditioner:
          assert apply_preconditioner is not None
          dual_rhs = apply_preconditioner(dual_rhs)
        flat_dual_rhs, _ = ravel_pytree(dual_rhs)

        def reduced_matvec(vec: jnp.ndarray) -> jnp.ndarray:
          dual_tangent = unravel_dual(vec)
          product = self.problem.reduced_dual_matvec(
            dual_tangent,
            theta,
            linearization,
            reg,
          )
          if use_preconditioner:
            assert apply_preconditioner is not None
            product = apply_preconditioner(product)
          return ravel_pytree(product)[0]

        reduced_matvec = jax.jit(reduced_matvec)
        dual_step_flat, gmres_info = gmres(
          reduced_matvec,
          flat_dual_rhs,
          tol=self.cfg.gmres_tolerance,
          restart=self.cfg.gmres_restart,
          maxiter=self.cfg.gmres_maxiter,
        )
        dual_step = unravel_dual(dual_step_flat)
        linear_solver_info = float(gmres_info)
      primal_step = self.problem.recover_primal_step_from_reduced_dual(
        dual_step,
        residual_point.primal,
        theta,
        linearization,
        reg,
      )
      step_point = EqualityGamePoint(primal=primal_step, dual=dual_step)
      step_flat, _ = ravel_pytree(step_point)
      if linear_solver_info >= 0.0 and bool(jnp.all(jnp.isfinite(step_flat))):
        full_product = self.problem.kkt_matvec(step_point, theta, linearization)
        full_step_residual = ravel_pytree(full_product)[0] + reg * step_flat + residual_value
        full_step_residual_norm = float(jnp.max(jnp.abs(full_step_residual)))
        residual_scale = max(1.0, float(jnp.max(jnp.abs(residual_value))))
        if (
          bool(jnp.isfinite(full_step_residual_norm))
          and full_step_residual_norm <= max(10.0 * self.cfg.gmres_tolerance * residual_scale, 1e-6)
        ):
          return step_point, step_flat, linear_solver_info, {"mode": self.cfg.linear_solver, "exact": True}

    return solve_full_system()

  def _solve_impl(
    self,
    theta: Any,
    initial_point: EqualityGamePoint,
  ) -> EqualityGameSolveResult:
    point_template = jax.tree_util.tree_map(jnp.asarray, initial_point)
    flat_point, unravel_point = ravel_pytree(point_template)
    structured_kkt = self._supports_structured_kkt()

    def residual_from_flat(flat: jnp.ndarray) -> jnp.ndarray:
      point = unravel_point(flat)
      residual = self.residual(point, theta)
      return ravel_pytree(residual)[0]

    residual_from_flat = jax.jit(residual_from_flat)

    flat_current = flat_point
    residual_history = []
    line_search_history = []
    regularization_history = []
    regularization_retry_history = []
    gmres_infos = []
    status = SolverStatus.MAX_ITERATIONS
    last_linear_mode = "full_gmres"
    last_preconditioner_mode = "none"
    last_dynamics_mode = "unknown"
    last_fastpath_max_contact_weight = None
    last_fastpath_min_speed_margin = None
    used_singularity_aware_partial_elimination = False
    approximate_pruning_used = False
    candidate_alphas = tuple(float(alpha) for alpha in self.cfg.line_search_alphas)
    use_structured_partial_solver = (
      structured_kkt
      and self.cfg.linear_solver != "full_gmres"
      and self._supports_singularity_aware_structured_linear_solve()
    )
    adaptive_regularization_enabled = (
      self.cfg.adaptive_regularization
      and (
        not use_structured_partial_solver
        or self._singularity_aware_structured_linear_solve_uses_regularization()
      )
    )
    initial_regularization = (
      abs(float(self._last_forward_regularization))
      if adaptive_regularization_enabled and self._last_forward_regularization is not None
      else abs(float(self.cfg.regularization))
    )
    if use_structured_partial_solver and not self._singularity_aware_structured_linear_solve_uses_regularization():
      initial_regularization = 0.0
    current_regularization = min(
      initial_regularization,
      max(0.0, float(self.cfg.regularization_max)),
    )

    for iteration in range(self.cfg.max_sqp_iterations):
      point_current = unravel_point(flat_current)
      linearization = self.problem.linearize_kkt(point_current, theta) if structured_kkt else None
      if structured_kkt:
        residual_point = self.problem.kkt_residual(point_current, theta, linearization)
        residual_value = ravel_pytree(residual_point)[0]
      else:
        residual_point = None
        residual_value = residual_from_flat(flat_current)
      residual_norm = float(jnp.max(jnp.abs(residual_value)))
      residual_history.append(residual_norm)

      if not jnp.isfinite(residual_norm):
        status = SolverStatus.NONFINITE
        break

      if residual_norm <= self.cfg.residual_tolerance:
        status = SolverStatus.SUCCESS
        break

      objective = self.problem.objective(point_current.primal, theta)
      constraints_current = self.constraints(point_current.primal, theta)
      flat_constraints_current = ravel_pytree(constraints_current)[0]
      constraints_l1_norm = float(jnp.sum(jnp.abs(flat_constraints_current)))
      constraints_inf_norm = float(jnp.max(jnp.abs(flat_constraints_current)))
      nearly_feasible_iterate = constraints_inf_norm <= max(10.0 * self.cfg.residual_tolerance, 1e-8)
      objective_grad = ravel_pytree(
        jax.grad(lambda primal: self.problem.objective(primal, theta))(point_current.primal)
      )[0]
      merit_base_value = float(objective)
      retry_count = 0
      iteration_regularization = current_regularization
      best_alpha = 0.0
      best_flat = flat_current
      while True:
        step_point, step, gmres_info, linear_meta = self._solve_linearized_system(
          flat_current=flat_current,
          point_current=point_current,
          theta=theta,
          residual_value=residual_value,
          residual_point=residual_point,
          linearization=linearization,
          unravel_point=unravel_point,
          residual_from_flat=residual_from_flat,
          regularization=iteration_regularization,
        )
        last_linear_mode = str(linear_meta.get("mode", last_linear_mode))
        last_preconditioner_mode = str(
          linear_meta.get("preconditioner_mode", last_preconditioner_mode)
        )
        last_dynamics_mode = str(linear_meta.get("dynamics_mode", last_dynamics_mode))
        last_fastpath_max_contact_weight = linear_meta.get(
          "fastpath_max_contact_weight",
          last_fastpath_max_contact_weight,
        )
        last_fastpath_min_speed_margin = linear_meta.get(
          "fastpath_min_speed_margin",
          last_fastpath_min_speed_margin,
        )
        used_singularity_aware_partial_elimination = (
          used_singularity_aware_partial_elimination
          or bool(linear_meta.get("singularity_aware_partial_elimination", False))
        )
        approximate_pruning_used = (
          approximate_pruning_used
          or bool(linear_meta.get("approximate_pruning", False))
        )

        if not bool(jnp.all(jnp.isfinite(step))):
          if (
            adaptive_regularization_enabled
            and retry_count < self.cfg.regularization_retry_limit
          ):
            next_regularization = self._increase_forward_regularization(iteration_regularization)
            if next_regularization > iteration_regularization:
              retry_count += 1
              iteration_regularization = next_regularization
              continue
          status = SolverStatus.NONFINITE
          best_alpha = 0.0
          best_flat = flat_current
          break

        flat_primal_step = ravel_pytree(step_point.primal)[0]
        objective_directional_derivative = float(jnp.dot(objective_grad, flat_primal_step))
        if constraints_l1_norm > 1e-12:
          linesearch_mu_min = objective_directional_derivative / (0.5 * constraints_l1_norm)
        else:
          linesearch_mu_min = float("-inf")
        linesearch_mu = max(self.cfg.linesearch_mu_floor, linesearch_mu_min)
        merit_value = merit_base_value + linesearch_mu * constraints_l1_norm
        merit_directional_derivative = (
          objective_directional_derivative - linesearch_mu * constraints_l1_norm
        )
        linearized_residual_direction = None
        if nearly_feasible_iterate:
          if structured_kkt:
            linearized_residual_direction = ravel_pytree(
              self.problem.kkt_matvec(step_point, theta, linearization)
            )[0]
          else:
            _, linearized_residual = jax.linearize(residual_from_flat, flat_current)
            linearized_residual_direction = linearized_residual(step)

        best_alpha = 0.0
        best_flat = flat_current
        use_nonlinear_residual_guard = (
          structured_kkt
          and self.cfg.singularity_aware_nonlinear_residual_guard
          and bool(linear_meta.get("singularity_aware_partial_elimination", False))
        )
        for alpha in candidate_alphas:
          candidate = flat_current + alpha * step
          candidate_point = None
          candidate_accepted = False
          if linearized_residual_direction is not None:
            candidate_linearized_residual = residual_value + alpha * linearized_residual_direction
            candidate_linearized_norm = float(jnp.max(jnp.abs(candidate_linearized_residual)))
            residual_decrease = (
              candidate_linearized_norm - (1.0 - self.cfg.linesearch_eta * alpha) * residual_norm
            )
            if residual_decrease <= 0.0:
              candidate_accepted = True
          if not candidate_accepted:
            candidate_point = unravel_point(candidate)
            candidate_objective = float(self.problem.objective(candidate_point.primal, theta))
            candidate_constraints = self.constraints(candidate_point.primal, theta)
            flat_candidate_constraints = ravel_pytree(candidate_constraints)[0]
            candidate_constraints_l1_norm = float(jnp.sum(jnp.abs(flat_candidate_constraints)))
            merit_value_candidate = candidate_objective + linesearch_mu * candidate_constraints_l1_norm
            merit_decrease = merit_value_candidate - (
              merit_value + self.cfg.linesearch_eta * alpha * merit_directional_derivative
            )
            candidate_accepted = merit_decrease <= 0.0
          if not candidate_accepted:
            continue
          if use_nonlinear_residual_guard:
            if candidate_point is None:
              candidate_point = unravel_point(candidate)
            candidate_linearization = self.problem.linearize_kkt(candidate_point, theta)
            candidate_residual_point = self.problem.kkt_residual(
              candidate_point,
              theta,
              candidate_linearization,
            )
            candidate_residual_value = ravel_pytree(candidate_residual_point)[0]
            candidate_residual_norm = float(jnp.max(jnp.abs(candidate_residual_value)))
            residual_acceptance_threshold = max(
              (1.0 - self.cfg.linesearch_eta * alpha) * residual_norm,
              self.cfg.residual_tolerance,
            )
            if (
              not bool(jnp.isfinite(candidate_residual_norm))
              or candidate_residual_norm > residual_acceptance_threshold
            ):
              continue
          best_alpha = alpha
          best_flat = candidate
          break

        if best_alpha > 0.0:
          flat_current = best_flat
          gmres_infos.append(gmres_info)
          line_search_history.append(best_alpha)
          regularization_history.append(iteration_regularization)
          regularization_retry_history.append(retry_count)
          current_regularization = self._update_forward_regularization_after_accept(
            iteration_regularization,
            best_alpha,
            retry_count,
          )
          break

        if (
          not adaptive_regularization_enabled
          or retry_count >= self.cfg.regularization_retry_limit
        ):
          gmres_infos.append(gmres_info)
          line_search_history.append(0.0)
          regularization_history.append(iteration_regularization)
          regularization_retry_history.append(retry_count)
          current_regularization = iteration_regularization
          break

        next_regularization = self._increase_forward_regularization(iteration_regularization)
        if next_regularization <= iteration_regularization:
          gmres_infos.append(gmres_info)
          line_search_history.append(0.0)
          regularization_history.append(iteration_regularization)
          regularization_retry_history.append(retry_count)
          current_regularization = iteration_regularization
          break
        retry_count += 1
        iteration_regularization = next_regularization

      if status == SolverStatus.NONFINITE or best_alpha == 0.0:
        break

    point = unravel_point(flat_current)
    objective = self.problem.objective(point.primal, theta)
    final_residual = residual_from_flat(flat_current)
    final_residual_norm = jnp.max(jnp.abs(final_residual))

    residual_history_arr = jnp.zeros((self.cfg.max_sqp_iterations,), dtype=flat_current.dtype)
    line_search_arr = jnp.zeros((self.cfg.max_sqp_iterations,), dtype=flat_current.dtype)
    regularization_arr = jnp.zeros((self.cfg.max_sqp_iterations,), dtype=flat_current.dtype)
    regularization_retry_arr = jnp.zeros((self.cfg.max_sqp_iterations,), dtype=jnp.int32)
    gmres_arr = jnp.zeros((self.cfg.max_sqp_iterations,), dtype=flat_current.dtype)
    if residual_history:
      residual_history_arr = residual_history_arr.at[: len(residual_history)].set(jnp.array(residual_history, dtype=flat_current.dtype))
    if line_search_history:
      line_search_arr = line_search_arr.at[: len(line_search_history)].set(jnp.array(line_search_history, dtype=flat_current.dtype))
    if regularization_history:
      regularization_arr = regularization_arr.at[: len(regularization_history)].set(jnp.array(regularization_history, dtype=flat_current.dtype))
    if regularization_retry_history:
      regularization_retry_arr = regularization_retry_arr.at[: len(regularization_retry_history)].set(jnp.array(regularization_retry_history, dtype=jnp.int32))
    if gmres_infos:
      gmres_arr = gmres_arr.at[: len(gmres_infos)].set(jnp.array(gmres_infos, dtype=flat_current.dtype))

    self._last_solve_point_forward_stats = {
      "linear_mode_last": last_linear_mode,
      "preconditioner_mode_last": last_preconditioner_mode,
      "dynamics_mode_last": last_dynamics_mode,
      "fastpath_max_contact_weight_last": last_fastpath_max_contact_weight,
      "fastpath_min_speed_margin_last": last_fastpath_min_speed_margin,
      "used_singularity_aware_partial_elimination": used_singularity_aware_partial_elimination,
      "approximate_pruning_used": approximate_pruning_used,
    }

    return EqualityGameSolveResult(
      point=point,
      objective=objective,
      residual_norm=final_residual_norm,
      num_iterations=jnp.array(len(residual_history), dtype=jnp.int32),
      num_step_iterations=jnp.array(len(line_search_history), dtype=jnp.int32),
      status=jnp.array(int(status), dtype=jnp.int32),
      residual_history=residual_history_arr,
      line_search_alphas=line_search_arr,
      regularization_history=regularization_arr,
      regularization_retry_counts=regularization_retry_arr,
      gmres_infos=gmres_arr,
    )

  def solve_with_metadata(
    self,
    theta: Any,
    initial_point: EqualityGamePoint | None = None,
  ) -> EqualityGameSolveResult:
    point0 = self.initial_point(theta) if initial_point is None else initial_point
    forward_start = time.perf_counter()
    result = self._solve_impl(theta, point0)
    forward_stats = self._forward_stats_from_result(result)
    forward_stats.update(self._last_solve_point_forward_stats)
    self._last_solve_point_forward_stats = forward_stats
    self._last_forward_regularization = self._last_solve_point_forward_stats.get("regularization_last")
    self._last_solve_point_forward_stats["elapsed_sec"] = time.perf_counter() - forward_start
    self._last_solve_point_backward_stats = {}
    return result

  def _make_solve_point(self):
    @partial(jax.custom_vjp, nondiff_argnums=(0,))
    def solve_point(
      _: TreeDiffMPCSolver,
      theta: Any,
      initial_point: EqualityGamePoint,
    ) -> EqualityGamePoint:
      forward_start = time.perf_counter()
      result = self._solve_impl(theta, initial_point)
      forward_stats = self._forward_stats_from_result(result)
      forward_stats.update(self._last_solve_point_forward_stats)
      self._last_solve_point_forward_stats = forward_stats
      self._last_forward_regularization = self._last_solve_point_forward_stats.get("regularization_last")
      self._last_solve_point_forward_stats["elapsed_sec"] = time.perf_counter() - forward_start
      self._last_solve_point_backward_stats = {}
      return result.point

    def solve_fwd(
      _: TreeDiffMPCSolver,
      theta: Any,
      initial_point: EqualityGamePoint,
    ) -> tuple[EqualityGamePoint, tuple[Any, EqualityGamePoint]]:
      forward_start = time.perf_counter()
      result = self._solve_impl(theta, initial_point)
      forward_stats = self._forward_stats_from_result(result)
      forward_stats.update(self._last_solve_point_forward_stats)
      self._last_solve_point_forward_stats = forward_stats
      self._last_forward_regularization = self._last_solve_point_forward_stats.get("regularization_last")
      self._last_solve_point_forward_stats["elapsed_sec"] = time.perf_counter() - forward_start
      self._last_solve_point_backward_stats = {}
      return result.point, (theta, result.point)

    def solve_bwd(
      _: TreeDiffMPCSolver,
      residuals: tuple[Any, EqualityGamePoint],
      cotangent: EqualityGamePoint,
    ) -> tuple[Any, None]:
      backward_start = time.perf_counter()
      theta, point = residuals
      flat_point, unravel_point = ravel_pytree(point)
      flat_cotangent, _ = ravel_pytree(cotangent)
      structured_kkt = self._supports_structured_kkt()
      use_reduced_dual_adjoint = False
      reduced_dual_attempted = False
      backward_fallback = False
      reduced_backward_residual_norm: float | None = None
      backward_mode = "full_transpose_gmres"

      if structured_kkt:
        linearization = self.problem.linearize_kkt(point, theta)

        def raw_matvec(vec: jnp.ndarray) -> jnp.ndarray:
          tangent = unravel_point(vec)
          product = self.problem.kkt_matvec(tangent, theta, linearization)
          return ravel_pytree(product)[0]

        reduced_dual_linear_solver = self.cfg.linear_solver in (
          "reduced_dual_gmres",
          "reduced_dual_pgmres",
          "reduced_dual_cg",
          "reduced_dual_pcg",
        )
        reduced_dual_cg_solver = self.cfg.linear_solver in ("reduced_dual_cg", "reduced_dual_pcg")
        uses_structured_partial = (
          reduced_dual_linear_solver
          and self._supports_singularity_aware_structured_linear_solve()
          and self._structured_kkt_is_symmetric()
        )
        reduced_dual_adjoint_regularization = (
          0.0
          if uses_structured_partial and not self._singularity_aware_structured_linear_solve_uses_regularization()
          else self._reduced_dual_adjoint_regularization(0.0)
          if (
            reduced_dual_linear_solver
            and (
              uses_structured_partial
              or self._supports_reduced_dual_kkt()
            )
            and self._structured_kkt_is_symmetric()
          )
          else None
        )
        use_reduced_dual_adjoint = (
          reduced_dual_adjoint_regularization is not None
          and (
            uses_structured_partial
            or not reduced_dual_cg_solver
            or self._reduced_dual_operator_cg_sign(reduced_dual_adjoint_regularization) is not None
          )
        )
        reduced_dual_attempted = use_reduced_dual_adjoint
        if use_reduced_dual_adjoint:
          adjoint_residual = jax.tree_util.tree_map(lambda value: -value, cotangent)
          adjoint_point, adjoint, _, linear_meta = self._solve_linearized_system(
            flat_current=flat_point,
            point_current=point,
            theta=theta,
            residual_value=-flat_cotangent,
            residual_point=adjoint_residual,
            linearization=linearization,
            unravel_point=unravel_point,
            residual_from_flat=None,
            regularization=reduced_dual_adjoint_regularization,
          )
          del adjoint_point
          if bool(jnp.all(jnp.isfinite(adjoint))):
            reduced_backward_residual = raw_matvec(adjoint) - flat_cotangent
            reduced_backward_residual_norm = float(jnp.max(jnp.abs(reduced_backward_residual)))
            use_reduced_dual_adjoint = (
              bool(jnp.isfinite(reduced_backward_residual_norm))
              and reduced_backward_residual_norm <= max(10.0 * self.cfg.gmres_tolerance, 1e-6)
            )
          else:
            use_reduced_dual_adjoint = False
          backward_fallback = reduced_dual_attempted and not use_reduced_dual_adjoint
          if use_reduced_dual_adjoint:
            backward_mode = (
              "singularity_aware_partial_adjoint"
              if bool(linear_meta.get("singularity_aware_partial_elimination", False))
              else "reduced_dual_adjoint"
            )
      else:
        def residual_from_flat(flat: jnp.ndarray) -> jnp.ndarray:
          residual_point = self.residual(unravel_point(flat), theta)
          return ravel_pytree(residual_point)[0]

        residual_from_flat = jax.jit(residual_from_flat)
        _, linearized_residual = jax.linearize(residual_from_flat, flat_point)

        def raw_matvec(vec: jnp.ndarray) -> jnp.ndarray:
          return linearized_residual(vec)

      if not use_reduced_dual_adjoint:
        raw_matvec = jax.jit(raw_matvec)
        linear_transpose = jax.linear_transpose(raw_matvec, jnp.zeros_like(flat_point))

        def transpose_matvec(vec: jnp.ndarray) -> jnp.ndarray:
          return linear_transpose(vec)[0]

        transpose_matvec = jax.jit(transpose_matvec)

        adjoint, _ = gmres(
          transpose_matvec,
          flat_cotangent,
          tol=self.cfg.gmres_tolerance,
          restart=self.cfg.gmres_restart,
          maxiter=self.cfg.gmres_maxiter,
        )
        backward_mode = "full_transpose_gmres"

      def residual_wrt_theta(theta_value: Any) -> jnp.ndarray:
        return ravel_pytree(self.residual(point, theta_value))[0]

      _, theta_vjp = jax.vjp(residual_wrt_theta, theta)
      theta_grad = theta_vjp(adjoint)[0]
      theta_grad = jax.tree_util.tree_map(lambda value: -value, theta_grad)
      self._last_solve_point_backward_stats = {
        "elapsed_sec": time.perf_counter() - backward_start,
        "mode": backward_mode if use_reduced_dual_adjoint else "full_transpose_gmres",
        "reduced_dual_attempted": reduced_dual_attempted,
        "fallback": backward_fallback,
        "reduced_dual_residual_norm": reduced_backward_residual_norm,
        "reduced_dual_regularization": (
          reduced_dual_adjoint_regularization if structured_kkt and reduced_dual_attempted else None
        ),
      }
      return theta_grad, None

    solve_point.defvjp(solve_fwd, solve_bwd)
    return partial(solve_point, self)
