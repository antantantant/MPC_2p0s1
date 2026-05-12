from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp

from . import drone_3d_diffmpc as base
from .tree_diffmpc import EqualityGamePoint


def _default_theta_for_bilevel(
  problem: base.Drone3DTreeProblem,
  cfg: base.Drone3DBilevelConfig,
  alpha_logits: jnp.ndarray | None,
) -> jnp.ndarray:
  if alpha_logits is not None:
    return alpha_logits
  return problem.init_alpha_logits(
    seed=cfg.outer.seed,
    init_scale=cfg.outer.alpha_init_scale,
    mode=cfg.outer.alpha_init_mode,
  )


def _default_theta_for_fixed_alpha(
  problem: base.Drone3DTreeProblem,
  alpha_logits: jnp.ndarray | None,
) -> jnp.ndarray:
  return problem.empty_alpha_logits() if alpha_logits is None else alpha_logits


def _empty_warm_start(
  *,
  mode: str,
  feasible: bool,
  fallback_reason: str | None,
) -> base.Drone3DBoxWarmStart:
  zero = jnp.zeros((3,), dtype=base.F32)
  return base.Drone3DBoxWarmStart(
    point=None,
    mode=mode,
    prepass_sec=0.0,
    projected_control_delta_max=0.0,
    used_projection=False,
    feasible=feasible,
    fallback_reason=fallback_reason,
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


def _zero_initial_box_warm_start(
  problem: base.Drone3DTreeProblem,
  theta: jnp.ndarray,
) -> base.Drone3DBoxWarmStart:
  if not problem.has_box_inequality_constraints or problem.cfg.squash_controls:
    return _empty_warm_start(
      mode="disabled",
      feasible=False,
      fallback_reason="boxes_inactive_or_squashed",
    )

  offense_problem = base._make_drone3d_player_problem(problem, "offense")
  defense_problem = base._make_drone3d_player_problem(problem, "defense")
  try:
    offense_point = offense_problem.initial_point(theta)
    defense_point = defense_problem.initial_point(theta)
  except ValueError as exc:
    return _empty_warm_start(
      mode="zero_initial_point",
      feasible=False,
      fallback_reason=str(exc),
    )

  offense_range = base._summarize_player_control_range(offense_problem, offense_point)
  defense_range = base._summarize_player_control_range(defense_problem, defense_point)
  return base.Drone3DBoxWarmStart(
    point=base._combine_drone3d_player_points(offense_point, defense_point),
    mode="zero_initial_point",
    prepass_sec=0.0,
    projected_control_delta_max=0.0,
    used_projection=False,
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


def prepare_box_gpu_warm_start(
  problem: base.Drone3DTreeProblem,
  theta: jnp.ndarray,
  *,
  mode: str = "auto",
  interior_margin: float = 1e-3,
) -> base.Drone3DBoxWarmStart:
  normalized = mode.strip().lower()
  if normalized not in ("auto", "zero", "exact_lq_projected"):
    raise ValueError(
      f"Unsupported box warm-start mode {mode!r}. Expected auto, zero, or exact_lq_projected.",
    )
  if not problem.has_box_inequality_constraints or problem.cfg.squash_controls:
    return _empty_warm_start(
      mode="not_applicable",
      feasible=True,
      fallback_reason=None,
    )

  if normalized == "zero":
    return _zero_initial_box_warm_start(problem, theta)

  if normalized == "exact_lq_projected":
    return base._make_drone3d_box_constrained_variant2_warm_start(
      problem,
      theta,
      interior_margin=interior_margin,
    )

  zero_warm_start = _zero_initial_box_warm_start(problem, theta)
  if zero_warm_start.point is not None:
    return zero_warm_start

  projected_warm_start = base._make_drone3d_box_constrained_variant2_warm_start(
    problem,
    theta,
    interior_margin=interior_margin,
  )
  if projected_warm_start.point is not None:
    return projected_warm_start

  fallback_reason_parts = []
  if zero_warm_start.fallback_reason:
    fallback_reason_parts.append(f"zero:{zero_warm_start.fallback_reason}")
  if projected_warm_start.fallback_reason:
    fallback_reason_parts.append(f"exact:{projected_warm_start.fallback_reason}")
  return base.Drone3DBoxWarmStart(
    point=None,
    mode=projected_warm_start.mode,
    prepass_sec=projected_warm_start.prepass_sec,
    projected_control_delta_max=projected_warm_start.projected_control_delta_max,
    used_projection=projected_warm_start.used_projection,
    feasible=False,
    fallback_reason="+".join(fallback_reason_parts) if fallback_reason_parts else "auto_failed",
    prepass_offense_control_min=projected_warm_start.prepass_offense_control_min,
    prepass_offense_control_max=projected_warm_start.prepass_offense_control_max,
    prepass_defense_control_min=projected_warm_start.prepass_defense_control_min,
    prepass_defense_control_max=projected_warm_start.prepass_defense_control_max,
    prepass_offense_control_max_abs=projected_warm_start.prepass_offense_control_max_abs,
    prepass_defense_control_max_abs=projected_warm_start.prepass_defense_control_max_abs,
    prepass_offense_control_max_violation=projected_warm_start.prepass_offense_control_max_violation,
    prepass_defense_control_max_violation=projected_warm_start.prepass_defense_control_max_violation,
    prepass_offense_controls_within_box=projected_warm_start.prepass_offense_controls_within_box,
    prepass_defense_controls_within_box=projected_warm_start.prepass_defense_controls_within_box,
  )


def _warm_start_fields(
  warm_start: base.Drone3DBoxWarmStart,
  *,
  policy: str,
) -> dict[str, Any]:
  return {
    "variant2_box_warm_start_policy": policy,
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


def _inject_warm_start_fields(result: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
  patched = dict(result)
  patched.update(fields)
  history = result.get("history")
  if isinstance(history, list):
    patched["history"] = [dict(entry, **fields) if isinstance(entry, dict) else entry for entry in history]
  return patched


def solve_drone3d_fixed_alpha_box_gpu(
  problem: base.Drone3DTreeProblem,
  inner_cfg: base.TreeDiffMPCConfig,
  *,
  alpha_logits: jnp.ndarray | None = None,
  initial_point: EqualityGamePoint | None = None,
  solver_variant: str = "player_separable_lq",
  box_warm_start_mode: str = "auto",
  box_warm_start_margin: float = 1e-3,
) -> dict[str, Any]:
  if solver_variant != "player_separable_lq":
    raise ValueError(f"Unsupported Drone3D solver_variant: {solver_variant}")
  if initial_point is not None or not problem.has_box_inequality_constraints or problem.cfg.squash_controls:
    return base.solve_drone3d_fixed_alpha(
      problem,
      inner_cfg,
      alpha_logits=alpha_logits,
      initial_point=initial_point,
      solver_variant=solver_variant,
    )

  theta = _default_theta_for_fixed_alpha(problem, alpha_logits)
  warm_start = prepare_box_gpu_warm_start(
    problem,
    theta,
    mode=box_warm_start_mode,
    interior_margin=box_warm_start_margin,
  )
  if warm_start.point is None:
    raise ValueError(
      "Could not construct a feasible boxed warm start for the GPU wrapper. "
      f"policy={box_warm_start_mode!r} reason={warm_start.fallback_reason!r}",
    )
  result = base.solve_drone3d_fixed_alpha(
    problem,
    inner_cfg,
    alpha_logits=alpha_logits,
    initial_point=warm_start.point,
    solver_variant=solver_variant,
  )
  return _inject_warm_start_fields(
    result,
    _warm_start_fields(warm_start, policy=box_warm_start_mode),
  )


def solve_drone3d_bilevel_box_gpu(
  problem: base.Drone3DTreeProblem,
  cfg: base.Drone3DBilevelConfig,
  *,
  alpha_logits: jnp.ndarray | None = None,
  warm_point: EqualityGamePoint | None = None,
  progress_callback: Callable[[dict[str, Any]], None] | None = None,
  solver_variant: str = "player_separable_lq",
  box_warm_start_mode: str = "auto",
  box_warm_start_margin: float = 1e-3,
) -> dict[str, Any]:
  if solver_variant != "player_separable_lq":
    raise ValueError(f"Unsupported Drone3D solver_variant: {solver_variant}")
  if warm_point is not None or not problem.has_box_inequality_constraints or problem.cfg.squash_controls:
    return base.solve_drone3d_bilevel(
      problem,
      cfg,
      alpha_logits=alpha_logits,
      warm_point=warm_point,
      progress_callback=progress_callback,
      solver_variant=solver_variant,
    )

  theta0 = _default_theta_for_bilevel(problem, cfg, alpha_logits)
  warm_start = prepare_box_gpu_warm_start(
    problem,
    theta0,
    mode=box_warm_start_mode,
    interior_margin=box_warm_start_margin,
  )
  if warm_start.point is None:
    raise ValueError(
      "Could not construct a feasible boxed warm start for the GPU wrapper. "
      f"policy={box_warm_start_mode!r} reason={warm_start.fallback_reason!r}",
    )
  result = base.solve_drone3d_bilevel(
    problem,
    cfg,
    alpha_logits=alpha_logits,
    warm_point=warm_start.point,
    progress_callback=progress_callback,
    solver_variant=solver_variant,
  )
  return _inject_warm_start_fields(
    result,
    _warm_start_fields(warm_start, policy=box_warm_start_mode),
  )
