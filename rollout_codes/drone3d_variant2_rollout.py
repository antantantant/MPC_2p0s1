from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional for visualization
    plt = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:  # pragma: no cover - optional for visualization
    go = None
    make_subplots = None

if __package__ in (None, ""):
    _FILE = Path(__file__).resolve()
    _SEARCH_ROOTS = [_FILE.parent / "src"]
    for _root in _SEARCH_ROOTS:
        if (_root / "MPC_2p0s1").exists() and str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

from MPC_2p0s1.hardware_rollout.drone3d_feedback import (
    Drone3DAffinePolicy,
    Drone3DBeliefTables,
    build_belief_tables,
    child_node_index,
    extract_affine_policy,
    make_context_problem,
    node_index_from_prototypes,
    rebase_alpha_logits_to_fixed_padded,
    rebase_alpha_logits_to_remaining,
    solve_bilevel_with_policy,
    solve_fixed_alpha_with_policy,
    solve_full_horizon_unconstrained_bilevel_with_policy,
    solve_padded_unconstrained_bilevel_with_policy,
)
from MPC_2p0s1.hardware_rollout.solver_jax import (
    Drone3DBilevelConfig,
    Drone3DOuterConfig,
    Drone3DProblemConfig,
    Drone3DTreeProblem,
    F32,
    MixedPrefixTreeSpec,
    TreeDiffMPCConfig,
    build_alpha_from_logits,
    build_drone3d_tree_spec,
)


LINEAR_SOLVER_CHOICES = (
    "full_gmres",
    "reduced_dual_gmres",
    "reduced_dual_pgmres",
    "reduced_dual_cg",
    "reduced_dual_pcg",
)
BOX_TERMINAL_BACKEND_CHOICES = (
    "riccati_first",
    "sparse",
    "active_tree",
)
BOX_LQ_BACKEND_CHOICES = (
    "active_set",
    "ipm",
)


@dataclass(frozen=True)
class LoadedSetup:
    args_dict: Dict[str, Any]
    output_device: torch.device
    output_dtype: torch.dtype
    problem_cfg_template: Drone3DProblemConfig
    root_tree: MixedPrefixTreeSpec
    bilevel_cfg: Drone3DBilevelConfig
    use_box_gpu_wrapper: bool
    control_tolerance: float


@dataclass
class PolicyRuntime:
    setup: LoadedSetup
    problem: Drone3DTreeProblem
    x0: torch.Tensor
    p0: torch.Tensor
    alpha_full: torch.Tensor
    alpha_logits_full: np.ndarray
    belief_tables: Drone3DBeliefTables
    policy: Drone3DAffinePolicy
    solve_result: Dict[str, Any]
    solve_timing: Dict[str, float]


@dataclass
class OnlineSolveConfig:
    enabled: bool = False
    skip_solve_at_t0: bool = True
    fixed_shape_padding: bool = True
    fixed_shape_padding_logit: float = 20.0
    precompile_fixed_shape: bool = True
    full_horizon_receding: bool = False
    outer_steps: Optional[int] = None
    outer_min_steps: Optional[int] = None
    inner_iters: Optional[int] = None
    grad_tolerance: Optional[float] = None
    loss_change_tolerance: Optional[float] = None


@dataclass
class PolicyStepResult:
    time_step: int
    node_idx: int
    child_node_idx: int
    prototype_index: int
    action_probs: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    current_belief: torch.Tensor
    next_belief: torch.Tensor
    control_compute_ms: float
    online_solve_ms: float
    control_total_ms: float
    policy_source: str
    solve_diag: Dict[str, Any]


def _parse_csv_floats(spec: str) -> Tuple[float, ...]:
    vals = [v.strip() for v in str(spec).split(",") if v.strip()]
    if not vals:
        raise ValueError(f"Expected comma-separated floats, got '{spec}'.")
    return tuple(float(v) for v in vals)


def _parse_csv_ints(spec: str) -> Tuple[int, ...]:
    text = str(spec).strip()
    if text == "":
        return tuple()
    vals = [v.strip() for v in text.split(",") if v.strip()]
    return tuple(int(v) for v in vals)


def _parse_xyz(text: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected three comma-separated coordinates, got {text!r}.",
        )
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Could not parse coordinates from {text!r}.",
        ) from exc


def _parse_diag6_list(text: str) -> tuple[tuple[float, ...], ...]:
    rows = [row.strip() for row in str(text).split(";") if row.strip()]
    if not rows:
        raise argparse.ArgumentTypeError("Expected one or more semicolon-separated 6D diagonal rows.")
    parsed_rows: list[tuple[float, ...]] = []
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        if len(parts) != 6:
            raise argparse.ArgumentTypeError(
                f"Expected each terminal diagonal row to have six comma-separated entries, got {row!r}.",
            )
        try:
            parsed_rows.append(tuple(float(part) for part in parts))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Could not parse terminal diagonal row {row!r}.") from exc
    return tuple(parsed_rows)


def _coerce_xyz(value: Any, *, name: str) -> tuple[float, float, float]:
    if value is None:
        raise ValueError(f"{name} may not be None.")
    if isinstance(value, (tuple, list)):
        if len(value) != 3:
            raise ValueError(f"{name} must have three entries.")
        return tuple(float(v) for v in value)
    return _parse_xyz(str(value))


def _coerce_diag6_list(value: Any) -> Optional[tuple[tuple[float, ...], ...]]:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        rows = []
        for row in value:
            row_vals = tuple(float(v) for v in row)
            if len(row_vals) != 6:
                raise ValueError(f"Expected 6 entries per terminal diagonal row, got {len(row_vals)}.")
            rows.append(row_vals)
        return tuple(rows)
    return _parse_diag6_list(str(value))


def _coerce_prior(value: Any) -> Optional[tuple[float, ...]]:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return tuple(float(v) for v in value)
    text = str(value).strip()
    if text.lower() == "none" or text == "":
        return None
    return _parse_csv_floats(text)


def _parse_state_csv(text: Optional[str], expected_dim: int, name: str) -> Optional[Tuple[float, ...]]:
    if text is None:
        return None
    vals = _parse_csv_floats(text)
    if len(vals) != expected_dim:
        raise ValueError(f"{name} expected {expected_dim} floats, got {len(vals)}.")
    return vals


def _parse_belief_csv(text: Optional[str], I: int) -> Optional[Tuple[float, ...]]:
    if text is None:
        return None
    vals = _parse_csv_floats(text)
    if len(vals) != I:
        raise ValueError(f"Belief expected {I} floats, got {len(vals)}.")
    total = sum(vals)
    if total <= 0.0:
        raise ValueError("Belief must have positive mass.")
    return tuple(v / total for v in vals)


def _tensor_to_list(tensor: torch.Tensor) -> Any:
    return tensor.detach().cpu().tolist()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _tensor_to_list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _choose_prototype(
    action_probs: torch.Tensor,
    sample_actions: bool,
    generator: Optional[torch.Generator],
) -> Tuple[int, torch.Tensor]:
    probs = action_probs / action_probs.sum().clamp_min(1e-12)
    if sample_actions:
        cpu_generator = None
        if generator is not None and str(getattr(generator, "device", "cpu")) == "cpu":
            cpu_generator = generator
        proto = int(
            torch.multinomial(
                probs.detach().cpu(),
                num_samples=1,
                replacement=True,
                generator=cpu_generator,
            ).item()
        )
    else:
        proto = int(torch.argmax(probs).item())
    return proto, probs


def _time_block(fn):
    t0 = time.perf_counter_ns()
    result = fn()
    t1 = time.perf_counter_ns()
    return result, (t1 - t0) / 1e6


def _build_progress_logger(prefix: str):
    def callback(entry: Dict[str, Any]) -> None:
        step = int(entry.get("outer_step", 0))
        loss = float(entry.get("loss", float("nan")))
        grad = float(entry.get("alpha_grad_norm", float("nan")))
        fwd = float(entry.get("forward_solve_sec", float("nan")))
        status = str(entry.get("status_reason_offense", "")) or str(entry.get("outer_status", ""))
        print(
            f"[{prefix}] outer={step:03d} loss={loss:.9f} grad={grad:.3e} "
            f"solve_sec={fwd:.3f} {status}".rstrip(),
            flush=True,
        )

    return callback


def _namespace_from_args_or_config(
    args_or_config: Optional[argparse.Namespace | Mapping[str, Any]],
    overrides: Mapping[str, Any],
) -> argparse.Namespace:
    parser = build_parser()
    ns = parser.parse_args([])
    if args_or_config is not None:
        if isinstance(args_or_config, argparse.Namespace):
            source = vars(args_or_config)
        else:
            source = dict(args_or_config)
        for key, value in source.items():
            setattr(ns, key, value)
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def _sample_type_index(prior: torch.Tensor, generator: Optional[torch.Generator]) -> int:
    probs = prior / prior.sum().clamp_min(1e-12)
    return int(torch.multinomial(probs, 1, replacement=True, generator=generator).item())


def _control_bounds(problem: Drone3DTreeProblem) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not problem.has_box_inequality_constraints:
        return None, None, None, None
    to_tensor = lambda vals: torch.tensor(np.asarray(vals, dtype=np.float32))
    return (
        to_tensor(problem.cfg.offense_control_lower_bounds),
        to_tensor(problem.cfg.offense_control_upper_bounds),
        to_tensor(problem.cfg.defense_control_lower_bounds),
        to_tensor(problem.cfg.defense_control_upper_bounds),
    )


def _clip_controls(problem: Drone3DTreeProblem, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lower_u, upper_u, lower_v, upper_v = _control_bounds(problem)
    if lower_u is not None and upper_u is not None:
        u = torch.maximum(torch.minimum(u, upper_u.to(device=u.device, dtype=u.dtype)), lower_u.to(device=u.device, dtype=u.dtype))
    if lower_v is not None and upper_v is not None:
        v = torch.maximum(torch.minimum(v, upper_v.to(device=v.device, dtype=v.dtype)), lower_v.to(device=v.device, dtype=v.dtype))
    return u, v


def _build_problem_from_args(
    args: argparse.Namespace,
    *,
    x0_override: Optional[torch.Tensor] = None,
    p0_override: Optional[torch.Tensor] = None,
) -> Drone3DTreeProblem:
    if not (0 <= int(args.reveal_step) <= int(args.total_steps)):
        raise ValueError("reveal-step must lie in [0, total-steps].")

    p1_max_accel = float(args.max_accel if args.p1_max_accel is None else args.p1_max_accel)
    p2_max_accel = float(args.max_accel if args.p2_max_accel is None else args.p2_max_accel)
    p1_control_lower = None if bool(args.unconstrained) else (-p1_max_accel, -p1_max_accel, -p1_max_accel)
    p1_control_upper = None if bool(args.unconstrained) else (p1_max_accel, p1_max_accel, p1_max_accel)
    p2_control_lower = None if bool(args.unconstrained) else (-p2_max_accel, -p2_max_accel, -p2_max_accel)
    p2_control_upper = None if bool(args.unconstrained) else (p2_max_accel, p2_max_accel, p2_max_accel)

    target0 = args.target0_pos
    target1 = args.target1_pos
    target_positions = None
    if target0 is not None or target1 is not None:
        if target0 is None or target1 is None:
            raise ValueError("--target0-pos and --target1-pos must be provided together.")
        target_positions = (
            _coerce_xyz(target0, name="target0_pos"),
            _coerce_xyz(target1, name="target1_pos"),
        )

    if x0_override is None:
        p1_init_pos = _coerce_xyz(args.p1_init_pos, name="p1_init_pos")
        p2_init_pos = _coerce_xyz(args.p2_init_pos, name="p2_init_pos")
        initial_state = (
            float(p1_init_pos[0]),
            float(p1_init_pos[1]),
            float(p1_init_pos[2]),
            0.0,
            0.0,
            0.0,
            float(p2_init_pos[0]),
            float(p2_init_pos[1]),
            float(p2_init_pos[2]),
            0.0,
            0.0,
            0.0,
        )
    else:
        state_vals = tuple(float(v) for v in x0_override.detach().cpu().tolist())
        if len(state_vals) != 12:
            raise ValueError("x0_override must have shape (12,).")
        initial_state = state_vals

    prior_override = None
    if p0_override is not None:
        prior_override = tuple(float(v) for v in p0_override.detach().cpu().tolist())
    elif getattr(args, "prior", None) is not None:
        prior_override = _coerce_prior(args.prior)

    problem_cfg_kwargs = dict(
        horizon_seconds=float(args.dt) * float(args.total_steps),
        dt=float(args.dt),
        mass_kg=1.3,
        max_accel=float(max(p1_max_accel, p2_max_accel)),
        running_r=tuple(float(v) for v in (_coerce_xyz(args.running_r, name="running_r") if args.running_r is not None else Drone3DProblemConfig.running_r)),
        running_s=tuple(float(v) for v in (_coerce_xyz(args.running_s, name="running_s") if args.running_s is not None else Drone3DProblemConfig.running_s)),
        target_positions=target_positions,
        terminal_type_diags=_coerce_diag6_list(args.terminal_type_diags),
        offense_control_lower_bounds=p1_control_lower,
        offense_control_upper_bounds=p1_control_upper,
        defense_control_lower_bounds=p2_control_lower,
        defense_control_upper_bounds=p2_control_upper,
        inequality_barrier_weight=0.0 if bool(args.unconstrained) else float(args.barrier_weight),
        enforce_terminal_velocity_constraints=not bool(args.no_terminal_velocity_constraints),
        initial_state=initial_state,
    )
    if prior_override is not None:
        problem_cfg_kwargs["prior"] = prior_override

    tree = build_drone3d_tree_spec(
        total_steps=int(args.total_steps),
        reveal_step=int(args.reveal_step),
        force_identity_reveal=not bool(args.no_forced_reveal),
    )
    return Drone3DTreeProblem(
        tree,
        cfg=Drone3DProblemConfig(**problem_cfg_kwargs),
    )


def _build_bilevel_cfg_from_args(args: argparse.Namespace) -> Drone3DBilevelConfig:
    return Drone3DBilevelConfig(
        inner=TreeDiffMPCConfig(
            max_sqp_iterations=int(args.inner_iters),
            residual_tolerance=float(args.inner_residual_tol),
            gmres_tolerance=min(float(args.inner_residual_tol), 1e-6),
            gmres_restart=int(args.gmres_restart),
            gmres_maxiter=int(args.gmres_maxiter),
            regularization=1e-6,
            linear_solver=str(args.linear_solver),
            drone3d_box_terminal_backend=str(args.box_terminal_backend),
            drone3d_box_lq_backend=str(args.box_lq_backend),
        ),
        outer=Drone3DOuterConfig(
            steps=int(args.outer_steps),
            min_steps=int(args.outer_min_steps),
            lr_alpha=float(args.outer_lr),
            alpha_logit_clip=float(args.outer_alpha_clip),
            seed=int(args.seed),
        ),
    )


def build_setup(
    args_or_config: Optional[argparse.Namespace | Mapping[str, Any]] = None,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    **overrides: Any,
) -> LoadedSetup:
    args = _namespace_from_args_or_config(args_or_config, overrides)
    output_device = torch.device(device)
    problem = _build_problem_from_args(args)
    return LoadedSetup(
        args_dict={key: _to_jsonable(value) for key, value in vars(args).items()},
        output_device=output_device,
        output_dtype=dtype,
        problem_cfg_template=problem.cfg,
        root_tree=problem.tree,
        bilevel_cfg=_build_bilevel_cfg_from_args(args),
        use_box_gpu_wrapper=not bool(args.unconstrained),
        control_tolerance=float(args.inner_residual_tol),
    )


def _solve_problem_with_policy(
    problem: Drone3DTreeProblem,
    setup: LoadedSetup,
    *,
    alpha_logits: Optional[np.ndarray] = None,
    warm_point=None,
    progress_callback=None,
) -> tuple[dict, Drone3DAffinePolicy, Drone3DBeliefTables, Dict[str, float]]:
    def _solve():
        if int(problem.tree.mixed_horizon_steps) > 0:
            # The root solve is the high-quality reference solve matching the
            # standalone Variant-2 solver. Fast full-horizon receding solves are
            # only used inside query_policy_action for online MPC updates.
            return solve_bilevel_with_policy(
                problem,
                setup.bilevel_cfg,
                alpha_logits=None if alpha_logits is None else np.asarray(alpha_logits, dtype=np.float32),
                warm_point=warm_point,
                use_box_gpu_wrapper=setup.use_box_gpu_wrapper,
                progress_callback=progress_callback,
                control_tolerance=setup.control_tolerance,
            )
        return solve_fixed_alpha_with_policy(
            problem,
            setup.bilevel_cfg.inner,
            alpha_logits=None if alpha_logits is None else np.asarray(alpha_logits, dtype=np.float32),
            initial_point=warm_point,
            use_box_gpu_wrapper=setup.use_box_gpu_wrapper,
            control_tolerance=setup.control_tolerance,
        )

    (result, policy, beliefs), total_ms = _time_block(_solve)
    return result, policy, beliefs, {
        "solver_ms": float(total_ms),
        "belief_tree_ms": 0.0,
        "feedback_extract_ms": 0.0,
        "offline_total_ms": float(total_ms),
    }


def build_policy_runtime(
    setup: LoadedSetup,
    *,
    x0: Optional[torch.Tensor] = None,
    p0: Optional[torch.Tensor] = None,
    progress_callback=None,
) -> PolicyRuntime:
    problem = _build_problem_from_args(
        argparse.Namespace(**setup.args_dict),
        x0_override=x0,
        p0_override=p0,
    )
    solve_result, policy, beliefs, solve_timing = _solve_problem_with_policy(
        problem,
        setup,
        progress_callback=progress_callback,
    )

    alpha_logits = np.asarray(solve_result["alpha_logits"], dtype=np.float32)
    alpha = build_alpha_from_logits(alpha_logits, problem.tree)
    x0_tensor = torch.tensor(np.asarray(problem.x0, dtype=np.float32), device=setup.output_device, dtype=setup.output_dtype)
    p0_tensor = torch.tensor(np.asarray(problem.prior, dtype=np.float32), device=setup.output_device, dtype=setup.output_dtype)
    alpha_tensor = torch.tensor(np.asarray(alpha, dtype=np.float32), device=setup.output_device, dtype=setup.output_dtype)

    return PolicyRuntime(
        setup=setup,
        problem=problem,
        x0=x0_tensor,
        p0=p0_tensor,
        alpha_full=alpha_tensor,
        alpha_logits_full=alpha_logits,
        belief_tables=beliefs,
        policy=policy,
        solve_result=solve_result,
        solve_timing=solve_timing,
    )


def load_policy_runtime(
    args_or_config: Optional[argparse.Namespace | Mapping[str, Any]] = None,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    x0: Optional[torch.Tensor] = None,
    p0: Optional[torch.Tensor] = None,
    **overrides: Any,
) -> PolicyRuntime:
    setup = build_setup(args_or_config, device=device, dtype=dtype, **overrides)
    progress_callback = _build_progress_logger("root") if bool(setup.args_dict.get("verbose_history", False)) else None
    return build_policy_runtime(setup, x0=x0, p0=p0, progress_callback=progress_callback)


def infer_belief_from_prototypes(
    runtime: PolicyRuntime,
    prototypes_so_far: Sequence[int],
) -> tuple[int, torch.Tensor]:
    node_idx = node_index_from_prototypes(runtime.problem.topology, prototypes_so_far)
    belief = runtime.belief_tables.node_beliefs[node_idx]
    belief_tensor = torch.tensor(np.asarray(belief, dtype=np.float32), device=runtime.setup.output_device, dtype=runtime.setup.output_dtype)
    return node_idx, belief_tensor


def _action_probs_for_node(
    runtime: PolicyRuntime,
    node_idx: int,
    time_step: int,
    type_index: int,
) -> torch.Tensor:
    if int(time_step) >= runtime.problem.tree.mixed_horizon_steps:
        probs = torch.zeros((runtime.problem.tree.type_count,), device=runtime.setup.output_device, dtype=runtime.setup.output_dtype)
        probs[0] = 1.0
        return probs
    local_idx = int(runtime.problem.topology.node_local_indices[int(node_idx)])
    return runtime.alpha_full[int(time_step), local_idx, int(type_index)]


def _edge_index_from_step(
    topology: MixedPrefixTreeSpec | PublicTreeTopology,
    full_topology,
    time_step: int,
    node_idx: int,
    prototype_index: int,
) -> int:
    del topology
    local_idx = int(full_topology.node_local_indices[int(node_idx)])
    if int(time_step) < full_topology.mixed_depth:
        return int(full_topology.edge_offsets_py[int(time_step)] + local_idx * full_topology.branch_factor + int(prototype_index))
    return int(full_topology.edge_offsets_py[int(time_step)] + local_idx)


def _policy_tensors_for_runtime(runtime: PolicyRuntime) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = runtime.setup.output_device
    dtype = runtime.setup.output_dtype
    return (
        torch.tensor(np.asarray(runtime.policy.offense_feedback, dtype=np.float32), device=device, dtype=dtype),
        torch.tensor(np.asarray(runtime.policy.offense_bias, dtype=np.float32), device=device, dtype=dtype),
        torch.tensor(np.asarray(runtime.policy.defense_feedback, dtype=np.float32), device=device, dtype=dtype),
        torch.tensor(np.asarray(runtime.policy.defense_bias, dtype=np.float32), device=device, dtype=dtype),
    )


def _edge_policy_tensors_for_runtime(
    runtime: PolicyRuntime,
    edge_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = runtime.setup.output_device
    dtype = runtime.setup.output_dtype
    edge = int(edge_idx)
    return (
        torch.tensor(np.asarray(runtime.policy.offense_feedback[edge], dtype=np.float32), device=device, dtype=dtype),
        torch.tensor(np.asarray(runtime.policy.offense_bias[edge], dtype=np.float32), device=device, dtype=dtype),
        torch.tensor(np.asarray(runtime.policy.defense_feedback[edge], dtype=np.float32), device=device, dtype=dtype),
        torch.tensor(np.asarray(runtime.policy.defense_bias[edge], dtype=np.float32), device=device, dtype=dtype),
    )


def _summarize_solve_result(
    problem: Drone3DTreeProblem,
    result: Dict[str, Any],
    bilevel_cfg: Optional[Drone3DBilevelConfig] = None,
) -> Dict[str, Any]:
    summary = {
        "objective": float(result.get("objective", float("nan"))),
        "outer_status": result.get("outer_status"),
        "outer_steps_used": int(result.get("outer_steps_used", 0)),
        "num_iterations": int(result.get("num_iterations", 0)),
        "residual_norm": float(result.get("residual_norm", 0.0)),
        "variant2_constraint_mode": str(result.get("variant2_constraint_mode", "unknown")),
        "forward_linear_mode": str(result.get("forward_linear_mode", "unknown")),
        "box_constraints_active": bool(result.get("box_constraints_active", problem.has_box_inequality_constraints)),
        "mixed_horizon_steps": int(problem.tree.mixed_horizon_steps),
        "tail_horizon_steps": int(problem.tree.tail_horizon_steps),
        "fixed_shape_padded": bool(result.get("fixed_shape_padded", False)),
        "full_horizon_receding": bool(result.get("full_horizon_receding", False)),
    }
    if "active_mixed_horizon_steps" in result:
        summary["active_mixed_horizon_steps"] = int(result["active_mixed_horizon_steps"])
    if "active_total_horizon_steps" in result:
        summary["active_total_horizon_steps"] = int(result["active_total_horizon_steps"])
    if bilevel_cfg is not None:
        summary.update(
            {
                "configured_outer_steps": int(bilevel_cfg.outer.steps),
                "configured_outer_min_steps": int(bilevel_cfg.outer.min_steps),
                "configured_inner_iters": int(bilevel_cfg.inner.max_sqp_iterations),
                "configured_grad_tolerance": None if bilevel_cfg.outer.grad_tolerance is None else float(bilevel_cfg.outer.grad_tolerance),
                "configured_loss_change_tolerance": None if bilevel_cfg.outer.loss_change_tolerance is None else float(bilevel_cfg.outer.loss_change_tolerance),
            }
        )
    return summary


def _remaining_horizons(problem: Drone3DTreeProblem, time_step: int) -> tuple[int, int]:
    if int(time_step) < problem.tree.mixed_horizon_steps:
        return (
            int(problem.tree.mixed_horizon_steps - time_step),
            int(problem.tree.tail_horizon_steps),
        )
    return (
        0,
        int(problem.tree.total_horizon_steps - time_step),
    )


def _make_online_setup(
    setup: LoadedSetup,
    online_cfg: OnlineSolveConfig,
) -> LoadedSetup:
    inner_cfg = setup.bilevel_cfg.inner
    outer_cfg = setup.bilevel_cfg.outer

    if online_cfg.inner_iters is not None:
        inner_cfg = replace(
            inner_cfg,
            max_sqp_iterations=max(1, int(online_cfg.inner_iters)),
        )

    resolved_outer_steps = int(outer_cfg.steps if online_cfg.outer_steps is None else max(1, int(online_cfg.outer_steps)))
    resolved_outer_min_steps = int(
        outer_cfg.min_steps if online_cfg.outer_min_steps is None else max(1, int(online_cfg.outer_min_steps))
    )
    resolved_outer_min_steps = min(resolved_outer_min_steps, resolved_outer_steps)
    resolved_grad_tolerance = outer_cfg.grad_tolerance if online_cfg.grad_tolerance is None else float(online_cfg.grad_tolerance)
    resolved_loss_change_tolerance = (
        outer_cfg.loss_change_tolerance
        if online_cfg.loss_change_tolerance is None
        else float(online_cfg.loss_change_tolerance)
    )

    if (
        resolved_outer_steps != int(outer_cfg.steps)
        or resolved_outer_min_steps != int(outer_cfg.min_steps)
        or resolved_grad_tolerance != outer_cfg.grad_tolerance
        or resolved_loss_change_tolerance != outer_cfg.loss_change_tolerance
    ):
        outer_cfg = replace(
            outer_cfg,
            steps=resolved_outer_steps,
            min_steps=resolved_outer_min_steps,
            grad_tolerance=resolved_grad_tolerance,
            loss_change_tolerance=resolved_loss_change_tolerance,
        )

    bilevel_cfg = replace(setup.bilevel_cfg, inner=inner_cfg, outer=outer_cfg)
    if bilevel_cfg == setup.bilevel_cfg:
        return setup
    return replace(setup, bilevel_cfg=bilevel_cfg)


def precompile_online_solver(
    runtime: PolicyRuntime,
    online_solve: Optional[OnlineSolveConfig],
) -> tuple[float, Dict[str, Any]]:
    online_cfg = online_solve or OnlineSolveConfig(enabled=False)
    if (
        not bool(online_cfg.enabled)
        or not bool(online_cfg.precompile_fixed_shape)
        or runtime.problem.has_box_inequality_constraints
        or bool(runtime.problem.cfg.squash_controls)
        or int(runtime.problem.tree.mixed_horizon_steps) <= 0
    ):
        return 0.0, {}

    precompile_cfg = replace(
        online_cfg,
        outer_steps=1,
        outer_min_steps=1,
        grad_tolerance=None,
        loss_change_tolerance=None,
    )
    online_setup = _make_online_setup(runtime.setup, precompile_cfg)
    padded_problem = make_context_problem(
        runtime.setup.problem_cfg_template,
        x0=runtime.x0.detach().cpu().numpy().astype(np.float32),
        prior=runtime.p0.detach().cpu().numpy().astype(np.float32),
        mixed_horizon_steps=runtime.problem.tree.mixed_horizon_steps,
        tail_horizon_steps=runtime.problem.tree.tail_horizon_steps,
        force_identity_reveal=runtime.problem.tree.force_identity_reveal,
    )
    if bool(online_cfg.full_horizon_receding):
        alpha_logits = np.asarray(runtime.alpha_logits_full, dtype=np.float32)

        def _compile_once():
            return solve_full_horizon_unconstrained_bilevel_with_policy(
                padded_problem,
                online_setup.bilevel_cfg,
                alpha_logits=np.asarray(alpha_logits, dtype=np.float32),
                control_tolerance=online_setup.control_tolerance,
            )

        policy_source = "online_full_horizon_receding_precompile"
    elif bool(online_cfg.fixed_shape_padding):
        alpha_logits = rebase_alpha_logits_to_fixed_padded(
            runtime.problem.tree,
            np.asarray(runtime.alpha_logits_full, dtype=np.float32),
            (),
            active_mixed_steps=runtime.problem.tree.mixed_horizon_steps,
            deterministic_logit=float(online_cfg.fixed_shape_padding_logit),
        )

        def _compile_once():
            return solve_padded_unconstrained_bilevel_with_policy(
                padded_problem,
                online_setup.bilevel_cfg,
                active_mixed_steps=runtime.problem.tree.mixed_horizon_steps,
                active_total_steps=runtime.problem.tree.total_horizon_steps,
                alpha_logits=np.asarray(alpha_logits, dtype=np.float32),
                deterministic_logit=float(online_cfg.fixed_shape_padding_logit),
            )

        policy_source = "online_padded_fixed_shape_precompile"
    else:
        return 0.0, {}

    (solve_result, _, _), precompile_ms = _time_block(_compile_once)
    diag = _summarize_solve_result(
        padded_problem,
        solve_result,
        bilevel_cfg=online_setup.bilevel_cfg,
    )
    diag["policy_source"] = policy_source
    diag["solve_ms"] = float(precompile_ms)
    return float(precompile_ms), diag


def query_policy_action(
    runtime: PolicyRuntime,
    type_index: int,
    x_current: torch.Tensor,
    prototypes_so_far: Sequence[int] = (),
    time_step: Optional[int] = None,
    belief_current: Optional[torch.Tensor] = None,
    sample_actions: bool = False,
    use_action_clip: bool = False,
    online_solve: Optional[OnlineSolveConfig] = None,
    generator: Optional[torch.Generator] = None,
) -> PolicyStepResult:
    query_t0 = time.perf_counter_ns()
    t = len(prototypes_so_far)
    if time_step is not None and int(time_step) != t:
        raise ValueError(f"time_step={time_step} does not match len(prototypes_so_far)={t}.")
    if not (0 <= int(type_index) < runtime.problem.tree.type_count):
        raise ValueError(f"type_index={type_index} out of range.")
    if t >= runtime.problem.tree.total_horizon_steps:
        raise ValueError(f"No control available at t={t}; horizon={runtime.problem.tree.total_horizon_steps}.")

    x = x_current.to(device=runtime.setup.output_device, dtype=runtime.setup.output_dtype)
    node_idx, inferred_belief = infer_belief_from_prototypes(runtime, prototypes_so_far)
    if belief_current is None:
        p = inferred_belief
    else:
        p = belief_current.to(device=runtime.setup.output_device, dtype=runtime.setup.output_dtype)
        p = p / p.sum().clamp_min(1e-12)

    online_cfg = online_solve or OnlineSolveConfig(enabled=False)
    action_probs = _action_probs_for_node(runtime, node_idx, t, int(type_index))
    current_policy_runtime = runtime
    online_solve_ms = 0.0
    solve_diag: Dict[str, Any] = {}
    policy_source = "offline_root_tree"
    policy_node_idx = int(node_idx)
    policy_time_step = int(t)

    if bool(online_cfg.enabled) and not (t == 0 and bool(online_cfg.skip_solve_at_t0)):
        remaining_mixed, remaining_tail = _remaining_horizons(runtime.problem, t)
        online_setup = _make_online_setup(runtime.setup, online_cfg)
        use_padded_online = (
            bool(online_cfg.fixed_shape_padding)
            and not bool(online_cfg.full_horizon_receding)
            and not runtime.problem.has_box_inequality_constraints
            and not bool(runtime.problem.cfg.squash_controls)
            and int(runtime.problem.tree.mixed_horizon_steps) > 0
        )
        use_full_horizon_online = (
            bool(online_cfg.full_horizon_receding)
            and not runtime.problem.has_box_inequality_constraints
            and not bool(runtime.problem.cfg.squash_controls)
            and int(runtime.problem.tree.mixed_horizon_steps) > 0
        )

        if use_full_horizon_online:
            remaining_problem = make_context_problem(
                runtime.setup.problem_cfg_template,
                x0=x.detach().cpu().numpy().astype(np.float32),
                prior=p.detach().cpu().numpy().astype(np.float32),
                mixed_horizon_steps=runtime.problem.tree.mixed_horizon_steps,
                tail_horizon_steps=runtime.problem.tree.tail_horizon_steps,
                force_identity_reveal=runtime.problem.tree.force_identity_reveal,
            )
            alpha_logits_rem = np.asarray(runtime.alpha_logits_full, dtype=np.float32)

            def _solve_remaining():
                solve_result, policy, beliefs = solve_full_horizon_unconstrained_bilevel_with_policy(
                    remaining_problem,
                    online_setup.bilevel_cfg,
                    alpha_logits=np.asarray(alpha_logits_rem, dtype=np.float32),
                    control_tolerance=online_setup.control_tolerance,
                )
                return solve_result, policy, beliefs, {}

            policy_source = "online_full_horizon_receding_solve"
        elif use_padded_online:
            remaining_problem = make_context_problem(
                runtime.setup.problem_cfg_template,
                x0=x.detach().cpu().numpy().astype(np.float32),
                prior=p.detach().cpu().numpy().astype(np.float32),
                mixed_horizon_steps=runtime.problem.tree.mixed_horizon_steps,
                tail_horizon_steps=runtime.problem.tree.tail_horizon_steps,
                force_identity_reveal=runtime.problem.tree.force_identity_reveal,
            )
            alpha_logits_rem = rebase_alpha_logits_to_fixed_padded(
                runtime.problem.tree,
                np.asarray(runtime.alpha_logits_full, dtype=np.float32),
                prototypes_so_far,
                active_mixed_steps=remaining_mixed,
                deterministic_logit=float(online_cfg.fixed_shape_padding_logit),
            )

            def _solve_remaining():
                solve_result, policy, beliefs = solve_padded_unconstrained_bilevel_with_policy(
                    remaining_problem,
                    online_setup.bilevel_cfg,
                    active_mixed_steps=remaining_mixed,
                    active_total_steps=remaining_mixed + remaining_tail,
                    alpha_logits=np.asarray(alpha_logits_rem, dtype=np.float32),
                    deterministic_logit=float(online_cfg.fixed_shape_padding_logit),
                )
                return solve_result, policy, beliefs, {}

            policy_source = "online_padded_fixed_shape_solve"
        else:
            remaining_problem = make_context_problem(
                runtime.setup.problem_cfg_template,
                x0=x.detach().cpu().numpy().astype(np.float32),
                prior=p.detach().cpu().numpy().astype(np.float32),
                mixed_horizon_steps=remaining_mixed,
                tail_horizon_steps=remaining_tail,
                force_identity_reveal=runtime.problem.tree.force_identity_reveal,
            )
            alpha_logits_rem = rebase_alpha_logits_to_remaining(
                runtime.problem.tree,
                np.asarray(runtime.alpha_logits_full, dtype=np.float32),
                prototypes_so_far,
            )

            def _solve_remaining():
                return _solve_problem_with_policy(
                    remaining_problem,
                    online_setup,
                    alpha_logits=np.asarray(alpha_logits_rem, dtype=np.float32),
                )

            policy_source = "online_remaining_solve"

        (solve_result, rem_policy, rem_beliefs, _), online_solve_ms = _time_block(_solve_remaining)
        if "alpha" in solve_result:
            rem_alpha = np.asarray(solve_result["alpha"], dtype=np.float32)
        else:
            rem_alpha = build_alpha_from_logits(np.asarray(solve_result["alpha_logits"], dtype=np.float32), remaining_problem.tree)
        current_policy_runtime = PolicyRuntime(
            setup=online_setup,
            problem=remaining_problem,
            x0=x,
            p0=p,
            alpha_full=torch.tensor(np.asarray(rem_alpha, dtype=np.float32), device=runtime.setup.output_device, dtype=runtime.setup.output_dtype),
            alpha_logits_full=np.asarray(solve_result["alpha_logits"], dtype=np.float32),
            belief_tables=rem_beliefs,
            policy=rem_policy,
            solve_result=solve_result,
            solve_timing={},
        )
        action_probs = _action_probs_for_node(current_policy_runtime, 0, 0, int(type_index))
        solve_diag = _summarize_solve_result(
            remaining_problem,
            solve_result,
            bilevel_cfg=online_setup.bilevel_cfg,
        )
        policy_node_idx = 0
        policy_time_step = 0
    elif bool(online_cfg.enabled) and t == 0 and bool(online_cfg.skip_solve_at_t0):
        policy_source = "offline_root_tree_skip_solve_at_t0"

    control_t0 = time.perf_counter_ns()
    proto_idx, action_probs = _choose_prototype(action_probs, sample_actions=sample_actions, generator=generator)
    edge_idx = _edge_index_from_step(
        current_policy_runtime.problem.tree,
        current_policy_runtime.problem.topology,
        policy_time_step,
        policy_node_idx,
        proto_idx,
    )
    offense_feedback, offense_bias, defense_feedback, defense_bias = _edge_policy_tensors_for_runtime(
        current_policy_runtime,
        edge_idx,
    )
    offense_state = x[0:6]
    defense_state = x[6:12]
    u = offense_feedback @ offense_state + offense_bias
    v = defense_feedback @ defense_state + defense_bias
    if use_action_clip:
        u, v = _clip_controls(runtime.problem, u, v)
    control_t1 = time.perf_counter_ns()
    control_compute_ms = (control_t1 - control_t0) / 1e6

    child_node_idx = child_node_index(runtime.problem.topology, t, node_idx, proto_idx if t < runtime.problem.tree.mixed_horizon_steps else 0)
    if policy_source in ("online_remaining_solve", "online_padded_fixed_shape_solve", "online_full_horizon_receding_solve"):
        rem_child_idx = child_node_index(
            current_policy_runtime.problem.topology,
            policy_time_step,
            policy_node_idx,
            proto_idx if current_policy_runtime.problem.tree.mixed_horizon_steps > 0 else 0,
        )
        next_belief_np = np.asarray(current_policy_runtime.belief_tables.node_beliefs[rem_child_idx], dtype=np.float32)
    else:
        next_belief_np = np.asarray(runtime.belief_tables.node_beliefs[child_node_idx], dtype=np.float32)
    next_belief = torch.tensor(next_belief_np, device=runtime.setup.output_device, dtype=runtime.setup.output_dtype)
    query_t1 = time.perf_counter_ns()
    control_total_ms = (query_t1 - query_t0) / 1e6

    return PolicyStepResult(
        time_step=t,
        node_idx=int(node_idx),
        child_node_idx=int(child_node_idx),
        prototype_index=int(proto_idx if t < runtime.problem.tree.mixed_horizon_steps else 0),
        action_probs=action_probs.detach().clone(),
        u=u.detach().clone(),
        v=v.detach().clone(),
        current_belief=p.detach().clone(),
        next_belief=next_belief,
        control_compute_ms=float(control_compute_ms),
        online_solve_ms=float(online_solve_ms),
        control_total_ms=float(control_total_ms),
        policy_source=policy_source,
        solve_diag=solve_diag,
    )


def rollout_total_cost(
    problem: Drone3DTreeProblem,
    x_traj: torch.Tensor,
    u_traj: torch.Tensor,
    v_traj: torch.Tensor,
    type_index: int,
) -> float:
    x_np = x_traj.detach().cpu().numpy()
    u_np = u_traj.detach().cpu().numpy()
    v_np = v_traj.detach().cpu().numpy()
    running_r = np.asarray(problem.cfg.running_r, dtype=np.float64)
    running_s = np.asarray(problem.cfg.running_s, dtype=np.float64)
    target = np.asarray(problem.targets[int(type_index)], dtype=np.float64)
    terminal_diags = np.asarray(problem.terminal_type_diags[int(type_index)], dtype=np.float64)
    running_cost = 0.5 * float(problem.cfg.dt) * float(
        np.sum(running_r[None, :] * np.square(u_np)) - np.sum(running_s[None, :] * np.square(v_np))
    )
    offense_delta = x_np[-1, 0:6] - target
    defense_delta = x_np[-1, 6:12] - target
    terminal_cost = float(np.sum(terminal_diags * np.square(offense_delta)) - np.sum(terminal_diags * np.square(defense_delta)))
    return running_cost + terminal_cost


def terminal_metrics(problem: Drone3DTreeProblem, x_terminal: torch.Tensor) -> Dict[str, float]:
    x_np = x_terminal.detach().cpu().numpy()
    offense = x_np[0:6]
    defense = x_np[6:12]
    targets = np.asarray(problem.targets, dtype=np.float64)
    offense_dists = np.linalg.norm(targets[:, :3] - offense[None, :3], axis=1)
    defense_dists = np.linalg.norm(targets[:, :3] - defense[None, :3], axis=1)
    return {
        "offense_pos_norm": float(np.linalg.norm(offense[:3])),
        "offense_vel_norm": float(np.linalg.norm(offense[3:6])),
        "defense_pos_norm": float(np.linalg.norm(defense[:3])),
        "defense_vel_norm": float(np.linalg.norm(defense[3:6])),
        "offense_min_target_dist": float(np.min(offense_dists)),
        "defense_min_target_dist": float(np.min(defense_dists)),
    }


def rollout_one_game(
    runtime: PolicyRuntime,
    type_index: int,
    sample_actions: bool,
    use_action_clip: bool,
    generator: Optional[torch.Generator],
    online_solve: Optional[OnlineSolveConfig] = None,
) -> Dict[str, Any]:
    K = runtime.problem.tree.total_horizon_steps
    state_dim = int(runtime.problem.state_dim)
    control_dim = int(runtime.problem.control_dim)
    type_count = int(runtime.problem.tree.type_count)
    device = runtime.setup.output_device
    dtype = runtime.setup.output_dtype

    x_traj = torch.empty((K + 1, state_dim), device=device, dtype=dtype)
    u_traj = torch.empty((K, control_dim), device=device, dtype=dtype)
    v_traj = torch.empty((K, control_dim), device=device, dtype=dtype)
    belief_traj = torch.empty((K + 1, type_count), device=device, dtype=dtype)
    proto_indices = torch.empty((K,), dtype=torch.long, device=device)
    action_probs_traj = torch.empty((K, type_count), device=device, dtype=dtype)

    x = runtime.x0.clone()
    p = runtime.p0.clone()
    prototypes_so_far: list[int] = []
    node_indices = [0]
    control_ms: list[float] = []
    control_total_ms: list[float] = []
    step_ms: list[float] = []
    online_solve_ms: list[float] = []
    step_policy_sources: list[str] = []
    online_solve_diag: list[Dict[str, Any]] = []
    online_precompile_ms, online_precompile_diag = precompile_online_solver(runtime, online_solve)

    x_traj[0] = x
    belief_traj[0] = p

    a_state = torch.tensor(np.asarray(runtime.problem.a_state, dtype=np.float32), device=device, dtype=dtype)
    offense_matrix = torch.tensor(np.asarray(runtime.problem.offense_matrix, dtype=np.float32), device=device, dtype=dtype)
    defense_matrix = torch.tensor(np.asarray(runtime.problem.defense_matrix, dtype=np.float32), device=device, dtype=dtype)

    for _ in range(K):
        step_t0 = time.perf_counter_ns()
        step = query_policy_action(
            runtime=runtime,
            type_index=int(type_index),
            x_current=x,
            prototypes_so_far=prototypes_so_far,
            time_step=len(prototypes_so_far),
            belief_current=p,
            sample_actions=sample_actions,
            use_action_clip=use_action_clip,
            online_solve=online_solve,
            generator=generator,
        )
        u_traj[step.time_step] = step.u
        v_traj[step.time_step] = step.v
        proto_indices[step.time_step] = step.prototype_index
        action_probs_traj[step.time_step] = step.action_probs

        x = a_state @ x + offense_matrix @ step.u + defense_matrix @ step.v
        p = step.next_belief
        x_traj[step.time_step + 1] = x
        belief_traj[step.time_step + 1] = p

        prototypes_so_far.append(int(step.prototype_index))
        node_indices.append(int(step.child_node_idx))
        control_ms.append(float(step.control_compute_ms))
        control_total_ms.append(float(step.control_total_ms))
        step_policy_sources.append(step.policy_source)
        online_solve_ms.append(float(step.online_solve_ms))
        if step.solve_diag:
            diag = dict(step.solve_diag)
            diag["time_step"] = int(step.time_step)
            diag["policy_source"] = step.policy_source
            diag["solve_ms"] = float(step.online_solve_ms)
            online_solve_diag.append(diag)
        step_t1 = time.perf_counter_ns()
        step_ms.append((step_t1 - step_t0) / 1e6)

    total_cost = rollout_total_cost(runtime.problem, x_traj, u_traj, v_traj, int(type_index))
    terminal = terminal_metrics(runtime.problem, x_traj[-1])
    return {
        "x_traj": x_traj,
        "u_traj": u_traj,
        "v_traj": v_traj,
        "belief_traj": belief_traj,
        "proto_indices": proto_indices,
        "node_indices": node_indices,
        "action_probs_traj": action_probs_traj,
        "control_compute_ms": control_ms,
        "control_total_ms": control_total_ms,
        "step_total_ms": step_ms,
        "online_solve_ms": online_solve_ms,
        "online_precompile_ms": float(online_precompile_ms),
        "online_precompile_diag": online_precompile_diag,
        "online_solve_diag": online_solve_diag,
        "step_policy_sources": step_policy_sources,
        "rollout_cost": total_cost,
        "terminal": terminal,
    }


def _root_posteriors(root_alpha: torch.Tensor, prior: torch.Tensor) -> list[list[float]]:
    posteriors = []
    for action_idx in range(root_alpha.shape[-1]):
        numer = prior * root_alpha[:, action_idx]
        denom = numer.sum()
        if float(denom.item()) <= 1e-12:
            post = torch.zeros_like(numer)
        else:
            post = numer / denom
        posteriors.append([float(v) for v in post.detach().cpu().tolist()])
    return posteriors


def build_rollout_result_payload(
    runtime: PolicyRuntime,
    true_type: int,
    rollout: Dict[str, Any],
) -> Dict[str, Any]:
    root_alpha = runtime.alpha_full[0, 0] if runtime.problem.tree.mixed_horizon_steps > 0 else torch.zeros(
        (runtime.problem.tree.type_count, runtime.problem.tree.type_count),
        device=runtime.setup.output_device,
        dtype=runtime.setup.output_dtype,
    )
    return {
        "mode": "rollout",
        "sampled_context": {
            "x0": _tensor_to_list(runtime.x0),
            "prior": _tensor_to_list(runtime.p0),
            "true_type": int(true_type),
        },
        "policy": {
            "root_alpha": _tensor_to_list(root_alpha),
            "root_posteriors": _root_posteriors(root_alpha=root_alpha, prior=runtime.p0) if runtime.problem.tree.mixed_horizon_steps > 0 else [],
            "proto_indices": rollout["proto_indices"].detach().cpu().tolist(),
            "node_indices": [int(v) for v in rollout["node_indices"]],
            "action_probs_traj": _tensor_to_list(rollout["action_probs_traj"]),
            "step_policy_sources": list(rollout["step_policy_sources"]),
        },
        "timings_ms": {
            **runtime.solve_timing,
            "control_compute_per_step_ms": [float(v) for v in rollout["control_compute_ms"]],
            "control_total_per_step_ms": [float(v) for v in rollout["control_total_ms"]],
            "step_total_per_step_ms": [float(v) for v in rollout["step_total_ms"]],
            "online_solve_per_step_ms": [float(v) for v in rollout["online_solve_ms"]],
            "online_precompile_ms": float(rollout.get("online_precompile_ms", 0.0)),
        },
        "solver": _summarize_solve_result(runtime.problem, runtime.solve_result, bilevel_cfg=runtime.setup.bilevel_cfg),
        "costs": {
            "root_objective": float(runtime.solve_result.get("objective", float("nan"))),
            "rollout_cost": float(rollout["rollout_cost"]),
        },
        "terminal": rollout["terminal"],
        "game": {
            "total_steps": int(runtime.problem.tree.total_horizon_steps),
            "reveal_step": int(runtime.problem.tree.mixed_horizon_steps),
            "dt": float(runtime.problem.cfg.dt),
            "targets": np.asarray(runtime.problem.targets, dtype=np.float32).tolist(),
            "force_identity_reveal": bool(runtime.problem.tree.force_identity_reveal),
        },
        "trajectories": {
            "x_traj": _tensor_to_list(rollout["x_traj"]),
            "u_traj": _tensor_to_list(rollout["u_traj"]),
            "v_traj": _tensor_to_list(rollout["v_traj"]),
            "belief_traj": _tensor_to_list(rollout["belief_traj"]),
        },
        "online_precompile_diag": rollout.get("online_precompile_diag", {}),
        "online_solve_diag": rollout["online_solve_diag"],
    }


def build_query_result_payload(
    runtime: PolicyRuntime,
    type_index: int,
    x_current: torch.Tensor,
    belief_current: torch.Tensor,
    prototypes_so_far: Sequence[int],
    step_result: PolicyStepResult,
) -> Dict[str, Any]:
    return {
        "mode": "query",
        "query": {
            "time_step": int(step_result.time_step),
            "type_index": int(type_index),
            "prototypes_so_far": [int(v) for v in prototypes_so_far],
            "node_idx": int(step_result.node_idx),
            "child_node_idx": int(step_result.child_node_idx),
            "policy_source": step_result.policy_source,
        },
        "state": {
            "x_current": _tensor_to_list(x_current),
            "belief_current": _tensor_to_list(belief_current),
            "belief_next": _tensor_to_list(step_result.next_belief),
        },
        "policy": {
            "action_probs": _tensor_to_list(step_result.action_probs),
            "prototype_index": int(step_result.prototype_index),
            "u": _tensor_to_list(step_result.u),
            "v": _tensor_to_list(step_result.v),
        },
        "timings_ms": {
            "offline_total_ms": float(runtime.solve_timing.get("offline_total_ms", 0.0)),
            "control_compute_ms": float(step_result.control_compute_ms),
            "online_solve_ms": float(step_result.online_solve_ms),
            "control_total_ms": float(step_result.control_total_ms),
        },
        "solver": _summarize_solve_result(runtime.problem, runtime.solve_result, bilevel_cfg=runtime.setup.bilevel_cfg),
        "online_solve_diag": step_result.solve_diag,
    }


def _set_equal_axes_3d(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 1e-3)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def save_trajectory_png(
    runtime: PolicyRuntime,
    rollout: Dict[str, Any],
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    if plt is None:
        raise ImportError("matplotlib is required to save a trajectory PNG.")

    x_np = rollout["x_traj"].detach().cpu().numpy()
    p1 = x_np[:, 0:3]
    p2 = x_np[:, 6:9]
    points = np.concatenate([p1, p2], axis=0)

    fig = plt.figure(figsize=(8.0, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(p1[:, 0], p1[:, 1], p1[:, 2], color="#d62728", linewidth=2.2, label="P1")
    ax.plot(p2[:, 0], p2[:, 1], p2[:, 2], color="#1f77b4", linewidth=2.2, label="P2")
    ax.scatter(p1[0, 0], p1[0, 1], p1[0, 2], color="#d62728", marker="o", s=50)
    ax.scatter(p2[0, 0], p2[0, 1], p2[0, 2], color="#1f77b4", marker="o", s=50)
    ax.scatter(p1[-1, 0], p1[-1, 1], p1[-1, 2], color="#d62728", marker="^", s=60)
    ax.scatter(p2[-1, 0], p2[-1, 1], p2[-1, 2], color="#1f77b4", marker="^", s=60)
    targets = np.asarray(runtime.problem.targets, dtype=np.float64)
    for i in range(targets.shape[0]):
        ax.scatter(
            targets[i, 0],
            targets[i, 1],
            targets[i, 2],
            color="#2ca02c" if i == 0 else "#ff7f0e",
            marker="D",
            s=60,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title or "Drone3D Variant-2 rollout")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    _set_equal_axes_3d(ax, points if targets.size == 0 else np.concatenate([points, targets[:, :3]], axis=0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_animation_html(
    runtime: PolicyRuntime,
    rollout: Dict[str, Any],
    output_path: Path,
    title: Optional[str] = None,
    true_type: Optional[int] = None,
) -> None:
    if go is None or make_subplots is None:
        raise ImportError("plotly is required to save an interactive animation HTML.")

    def _type_colors() -> list[str]:
        return [
            "#2ca02c",
            "#ff7f0e",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
        ]

    def _make_single_type_figure(
        type_idx: int,
        type_rollout: Dict[str, Any],
        selected_type: Optional[int],
    ):
        x_np = type_rollout["x_traj"].detach().cpu().numpy()
        p1 = x_np[:, 0:3]
        p2 = x_np[:, 6:9]
        belief_np = type_rollout["belief_traj"].detach().cpu().numpy()
        num_types = belief_np.shape[1]
        time_idx = np.arange(belief_np.shape[0], dtype=float)
        step_idx = np.arange(len(type_rollout["control_total_ms"]), dtype=float)
        control_total_ms = np.asarray(type_rollout["control_total_ms"], dtype=np.float64)
        online_ms = np.asarray(type_rollout["online_solve_ms"], dtype=np.float64)
        control_compute_ms = np.asarray(type_rollout["control_compute_ms"], dtype=np.float64)
        step_policy_sources = list(type_rollout.get("step_policy_sources", []))
        targets = np.asarray(runtime.problem.targets, dtype=np.float64)[:, :3]
        points = np.concatenate([p1, p2], axis=0)
        if targets.size > 0:
            points = np.concatenate([points, targets], axis=0)
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        pad = np.maximum(0.15 * (maxs - mins), 0.2)
        type_colors = _type_colors()

        title_text = title or "Drone3D Variant-2 rollout animation"
        title_text = f"{title_text} | P1 type {int(type_idx)}"
        if selected_type is not None and int(selected_type) == int(type_idx):
            title_text = f"{title_text} | selected rollout"
        root_solve_ms = float(runtime.solve_timing.get("offline_total_ms", 0.0))
        online_precompile_ms = float(type_rollout.get("online_precompile_ms", 0.0))
        num_online_solves = int(np.count_nonzero(online_ms > 0.0))
        first_policy_source = step_policy_sources[0] if step_policy_sources else "n/a"
        root_budget_text = (
            f"Root budget: outer<={int(runtime.setup.bilevel_cfg.outer.steps)}, "
            f"min={int(runtime.setup.bilevel_cfg.outer.min_steps)}, "
            f"inner={int(runtime.setup.bilevel_cfg.inner.max_sqp_iterations)}"
        )
        root_tol_text = (
            f"Root conv: grad<={runtime.setup.bilevel_cfg.outer.grad_tolerance}, "
            f"loss<={runtime.setup.bilevel_cfg.outer.loss_change_tolerance}"
        )
        online_cfg_display = OnlineSolveConfig(
            enabled=True,
            fixed_shape_padding=bool(runtime.setup.args_dict.get("online_fixed_shape_padding", True)),
            fixed_shape_padding_logit=float(runtime.setup.args_dict.get("online_fixed_shape_padding_logit", 20.0)),
            precompile_fixed_shape=bool(runtime.setup.args_dict.get("online_precompile_fixed_shape", True)),
            full_horizon_receding=bool(runtime.setup.args_dict.get("online_full_horizon_receding", False)),
            outer_steps=runtime.setup.args_dict.get("online_outer_steps"),
            outer_min_steps=runtime.setup.args_dict.get("online_outer_min_steps"),
            inner_iters=runtime.setup.args_dict.get("online_inner_iters"),
            grad_tolerance=runtime.setup.args_dict.get("online_grad_tolerance"),
            loss_change_tolerance=runtime.setup.args_dict.get("online_loss_change_tolerance"),
        )
        online_setup_display = _make_online_setup(runtime.setup, online_cfg_display)
        online_budget_text = (
            f"Online budget: outer<={int(online_setup_display.bilevel_cfg.outer.steps)}, "
            f"min={int(online_setup_display.bilevel_cfg.outer.min_steps)}, "
            f"inner={int(online_setup_display.bilevel_cfg.inner.max_sqp_iterations)}"
        )
        online_tol_text = (
            f"Online conv: grad<={online_setup_display.bilevel_cfg.outer.grad_tolerance}, "
            f"loss<={online_setup_display.bilevel_cfg.outer.loss_change_tolerance}"
        )
        online_shape_supported = (
            online_cfg_display.fixed_shape_padding
            and not online_cfg_display.full_horizon_receding
            and not runtime.problem.has_box_inequality_constraints
            and not bool(runtime.problem.cfg.squash_controls)
            and int(runtime.problem.tree.mixed_horizon_steps) > 0
        )
        if online_cfg_display.full_horizon_receding:
            online_shape_text = "Online shape: full-horizon receding graph"
        else:
            online_shape_text = (
                "Online shape: padded fixed tree"
                if online_shape_supported
                else "Online shape: shrinking remaining tree"
            )
        summary_text = (
            f"Initial root solve: {root_solve_ms:.1f} ms"
            f"<br>{root_budget_text}"
            f"<br>{root_tol_text}"
            f"<br>{online_budget_text}"
            f"<br>{online_tol_text}"
            f"<br>{online_shape_text}"
            f"<br>Online precompile: {online_precompile_ms:.1f} ms"
            f"<br>Step 0 source: {first_policy_source}"
            f"<br>Online re-solves used: {num_online_solves}"
        )

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
                [None, {"type": "xy"}],
            ],
            column_widths=[0.62, 0.38],
            row_heights=[0.62, 0.38],
            vertical_spacing=0.10,
            subplot_titles=("3D Trajectories", "Belief Evolution", "Control Query Latency"),
        )

        fig.add_trace(
            go.Scatter3d(
                x=p1[:1, 0],
                y=p1[:1, 1],
                z=p1[:1, 2],
                mode="lines+markers",
                line=dict(color="#d62728", width=6),
                marker=dict(size=4, color="#d62728"),
                name="P1",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=p2[:1, 0],
                y=p2[:1, 1],
                z=p2[:1, 2],
                mode="lines+markers",
                line=dict(color="#1f77b4", width=6),
                marker=dict(size=4, color="#1f77b4"),
                name="P2",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=[p1[0, 0]],
                y=[p1[0, 1]],
                z=[p1[0, 2]],
                mode="markers+text",
                marker=dict(size=7, color="#d62728", symbol="circle"),
                text=["P1 start"],
                textposition="top center",
                name="P1 start",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=[p2[0, 0]],
                y=[p2[0, 1]],
                z=[p2[0, 2]],
                mode="markers+text",
                marker=dict(size=7, color="#1f77b4", symbol="circle"),
                text=["P2 start"],
                textposition="top center",
                name="P2 start",
            ),
            row=1,
            col=1,
        )

        for i in range(targets.shape[0]):
            color = type_colors[i % len(type_colors)]
            marker_symbol = "diamond"
            marker_size = 8
            label = f"Target type {i}"
            if int(type_idx) == i:
                marker_symbol = "diamond-open"
                marker_size = 11
                label = f"Chosen target type {i}"
            fig.add_trace(
                go.Scatter3d(
                    x=[targets[i, 0]],
                    y=[targets[i, 1]],
                    z=[targets[i, 2]],
                    mode="markers+text",
                    marker=dict(size=marker_size, color=color, symbol=marker_symbol),
                    text=[label],
                    textposition="top center",
                    name=label,
                ),
                row=1,
                col=1,
            )

        belief_marker_trace_indices: list[int] = []
        for i in range(num_types):
            color = type_colors[i % len(type_colors)]
            fig.add_trace(
                go.Scatter(
                    x=time_idx,
                    y=belief_np[:, i],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=f"Belief type {i}",
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=[time_idx[0]],
                    y=[belief_np[0, i]],
                    mode="markers",
                    marker=dict(color=color, size=10),
                    name=f"Current belief type {i}",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            belief_marker_trace_indices.append(len(fig.data) - 1)

        fig.add_trace(
            go.Scatter(
                x=[time_idx[0], time_idx[0]],
                y=[0.0, 1.0],
                mode="lines",
                line=dict(color="#444444", width=1, dash="dot"),
                name="Belief step",
            ),
            row=1,
            col=2,
        )
        belief_step_line_trace_index = len(fig.data) - 1

        latency_max = float(
            max(
                float(control_total_ms.max()) if control_total_ms.size else 0.0,
                float(online_ms.max()) if online_ms.size else 0.0,
                float(control_compute_ms.max()) if control_compute_ms.size else 0.0,
                1.0,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=step_idx,
                y=control_total_ms,
                mode="lines",
                line=dict(color="#111111", width=3),
                name="Control total ms",
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=step_idx,
                y=online_ms,
                mode="lines",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
                name="Online solve ms",
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=step_idx,
                y=control_compute_ms,
                mode="lines",
                line=dict(color="#2ca02c", width=2, dash="dot"),
                name="Affine control ms",
            ),
            row=2,
            col=2,
        )
        latency_total_marker_idx = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(color="#111111", size=10),
                name="Current total ms",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        latency_online_marker_idx = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(color="#ff7f0e", size=9),
                name="Current online solve ms",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        latency_compute_marker_idx = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(color="#2ca02c", size=9),
                name="Current affine control ms",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[0.0, 0.0],
                y=[0.0, latency_max * 1.05],
                mode="lines",
                line=dict(color="#444444", width=1, dash="dot"),
                name="Latency step",
            ),
            row=2,
            col=2,
        )
        latency_step_line_trace_index = len(fig.data) - 1

        frames = []
        for k in range(p1.shape[0]):
            frame_data: list[Any] = [
                go.Scatter3d(
                    x=p1[: k + 1, 0],
                    y=p1[: k + 1, 1],
                    z=p1[: k + 1, 2],
                    mode="lines+markers",
                    line=dict(color="#d62728", width=6),
                    marker=dict(size=4, color="#d62728"),
                    name="P1",
                ),
                go.Scatter3d(
                    x=p2[: k + 1, 0],
                    y=p2[: k + 1, 1],
                    z=p2[: k + 1, 2],
                    mode="lines+markers",
                    line=dict(color="#1f77b4", width=6),
                    marker=dict(size=4, color="#1f77b4"),
                    name="P2",
                ),
            ]
            frame_traces = [0, 1]
            for i in range(num_types):
                color = type_colors[i % len(type_colors)]
                frame_data.append(
                    go.Scatter(
                        x=[time_idx[k]],
                        y=[belief_np[k, i]],
                        mode="markers",
                        marker=dict(color=color, size=10),
                        showlegend=False,
                    )
                )
                frame_traces.append(belief_marker_trace_indices[i])
            frame_data.append(
                go.Scatter(
                    x=[time_idx[k], time_idx[k]],
                    y=[0.0, 1.0],
                    mode="lines",
                    line=dict(color="#444444", width=1, dash="dot"),
                    showlegend=False,
                )
            )
            frame_traces.append(belief_step_line_trace_index)

            if control_total_ms.size and k > 0:
                latency_step = min(k - 1, control_total_ms.shape[0] - 1)
                latency_x = [float(step_idx[latency_step])]
                latency_total_y = [float(control_total_ms[latency_step])]
                latency_online_y = [float(online_ms[latency_step])]
                latency_compute_y = [float(control_compute_ms[latency_step])]
                latency_line_x = [float(step_idx[latency_step]), float(step_idx[latency_step])]
            else:
                latency_x = []
                latency_total_y = []
                latency_online_y = []
                latency_compute_y = []
                latency_line_x = [0.0, 0.0]
            frame_data.extend(
                [
                    go.Scatter(
                        x=latency_x,
                        y=latency_total_y,
                        mode="markers",
                        marker=dict(color="#111111", size=10),
                        showlegend=False,
                    ),
                    go.Scatter(
                        x=latency_x,
                        y=latency_online_y,
                        mode="markers",
                        marker=dict(color="#ff7f0e", size=9),
                        showlegend=False,
                    ),
                    go.Scatter(
                        x=latency_x,
                        y=latency_compute_y,
                        mode="markers",
                        marker=dict(color="#2ca02c", size=9),
                        showlegend=False,
                    ),
                    go.Scatter(
                        x=latency_line_x,
                        y=[0.0, latency_max * 1.05],
                        mode="lines",
                        line=dict(color="#444444", width=1, dash="dot"),
                        showlegend=False,
                    ),
                ]
            )
            frame_traces.extend(
                [
                    latency_total_marker_idx,
                    latency_online_marker_idx,
                    latency_compute_marker_idx,
                    latency_step_line_trace_index,
                ]
            )
            frames.append(go.Frame(name=str(k), data=frame_data, traces=frame_traces))

        fig.frames = frames
        belief_xmax = float(time_idx[-1]) if time_idx.size else 1.0
        latency_xmax = float(step_idx[-1]) if step_idx.size else 1.0
        fig.update_layout(
            title=title_text,
            scene=dict(
                xaxis=dict(title="x", range=[float(mins[0] - pad[0]), float(maxs[0] + pad[0])]),
                yaxis=dict(title="y", range=[float(mins[1] - pad[1]), float(maxs[1] + pad[1])]),
                zaxis=dict(title="z", range=[float(mins[2] - pad[2]), float(maxs[2] + pad[2])]),
                aspectmode="cube",
            ),
            xaxis=dict(title="Step", range=[0.0, belief_xmax]),
            yaxis=dict(title="Belief", range=[-0.02, 1.02]),
            xaxis2=dict(title="Control step", range=[0.0, latency_xmax]),
            yaxis2=dict(title="Latency (ms)", range=[0.0, latency_max * 1.10]),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.02,
                    y=1.12,
                    xanchor="left",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True}],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    steps=[
                        dict(
                            method="animate",
                            args=[[str(k)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                            label=str(k),
                        )
                        for k in range(len(frames))
                    ],
                    currentvalue={"prefix": "Step "},
                )
            ],
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=20, r=20, t=130, b=30),
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.60,
            y=1.18,
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            bordercolor="#999999",
            borderwidth=1,
            borderpad=6,
            bgcolor="rgba(255,255,255,0.92)",
            font=dict(size=12, color="#222222"),
            text=summary_text,
        )
        return fig

    displayed_type = int(true_type) if true_type is not None else 0
    figure = _make_single_type_figure(
        type_idx=displayed_type,
        type_rollout=rollout,
        selected_type=true_type,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output_path), include_plotlyjs="cdn")


def _save_json(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_to_jsonable(payload), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Direct Variant-2 rollout and query runner for the 3D drone signaling game. "
            "It preserves the current rollout/query contract but solves the game online with "
            "the exact JAX tree solver instead of using a neural checkpoint."
        )
    )
    parser.add_argument("--mode", type=str, default="rollout", choices=["rollout", "query"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--save-trajectory-png", type=Path, default=None)
    parser.add_argument("--save-animation-html", type=Path, default=None)
    parser.add_argument("--figure-title", type=str, default=None)

    parser.add_argument("--total-steps", type=int, default=16)
    parser.add_argument("--reveal-step", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.3)
    parser.add_argument("--max-accel", type=float, default=8.0)
    parser.add_argument("--p1-max-accel", type=float, default=None)
    parser.add_argument("--p2-max-accel", type=float, default=None)
    parser.add_argument("--barrier-weight", type=float, default=1e-4)
    parser.add_argument("--p1-init-pos", type=_parse_xyz, default=(-1.0, 0.0, 1.0))
    parser.add_argument("--p2-init-pos", type=_parse_xyz, default=(1.0, 0.0, 1.0))
    parser.add_argument("--target0-pos", type=_parse_xyz, default=None)
    parser.add_argument("--target1-pos", type=_parse_xyz, default=None)
    parser.add_argument("--running-r", type=_parse_xyz, default=None)
    parser.add_argument("--running-s", type=_parse_xyz, default=None)
    parser.add_argument("--terminal-type-diags", type=_parse_diag6_list, default=None)
    parser.add_argument("--prior", type=str, default=None, help="Optional prior override as CSV, e.g. '0.5,0.5'.")
    parser.add_argument("--unconstrained", action="store_true")
    parser.add_argument("--no-forced-reveal", action="store_true")
    parser.add_argument("--no-terminal-velocity-constraints", action="store_true")
    parser.add_argument("--inner-iters", type=int, default=4)
    parser.add_argument("--inner-residual-tol", type=float, default=1e-6)
    parser.add_argument("--gmres-restart", type=int, default=20)
    parser.add_argument("--gmres-maxiter", type=int, default=80)
    parser.add_argument("--linear-solver", type=str, default="full_gmres", choices=LINEAR_SOLVER_CHOICES)
    parser.add_argument(
        "--box-terminal-backend",
        type=str,
        default="riccati_first",
        choices=BOX_TERMINAL_BACKEND_CHOICES,
    )
    parser.add_argument(
        "--box-lq-backend",
        type=str,
        default="active_set",
        choices=BOX_LQ_BACKEND_CHOICES,
    )
    parser.add_argument("--outer-steps", type=int, default=50)
    parser.add_argument("--outer-min-steps", type=int, default=4)
    parser.add_argument("--outer-lr", type=float, default=0.03)
    parser.add_argument("--outer-alpha-clip", type=float, default=8.0)
    parser.add_argument("--verbose-history", action="store_true")
    parser.add_argument("--significant-threshold", type=float, default=0.05)
    parser.add_argument("--strict-significant-threshold", action="store_true")
    parser.add_argument("--max-plotted-branches", type=int, default=8)
    parser.add_argument("--plot-all-branches", action="store_true")

    parser.add_argument("--type-index", type=int, default=None)
    parser.add_argument("--sample-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-clip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--x-current", type=str, default=None)
    parser.add_argument("--belief-current", type=str, default=None)
    parser.add_argument("--time-step", type=int, default=None)
    parser.add_argument("--prototypes-so-far", type=str, default="")
    parser.add_argument("--online-solve", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--online-skip-solve-at-t0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--online-full-horizon-receding", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--online-fixed-shape-padding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--online-fixed-shape-padding-logit", type=float, default=20.0)
    parser.add_argument(
        "--online-precompile-fixed-shape",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronously precompile the online solve graph during startup so control steps do not pay JAX compile time.",
    )
    parser.add_argument("--online-outer-steps", type=int, default=None)
    parser.add_argument("--online-outer-min-steps", type=int, default=None)
    parser.add_argument("--online-inner-iters", type=int, default=None)
    parser.add_argument("--online-grad-tolerance", type=float, default=None)
    parser.add_argument("--online-loss-change-tolerance", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime = load_policy_runtime(args, device=args.device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))

    if args.type_index is None:
        type_index = _sample_type_index(runtime.p0, generator)
    else:
        type_index = int(args.type_index)

    online_cfg = OnlineSolveConfig(
        enabled=bool(args.online_solve),
        skip_solve_at_t0=bool(args.online_skip_solve_at_t0),
        fixed_shape_padding=bool(args.online_fixed_shape_padding),
        fixed_shape_padding_logit=float(args.online_fixed_shape_padding_logit),
        precompile_fixed_shape=bool(args.online_precompile_fixed_shape),
        full_horizon_receding=bool(args.online_full_horizon_receding),
        outer_steps=args.online_outer_steps,
        outer_min_steps=args.online_outer_min_steps,
        inner_iters=args.online_inner_iters,
        grad_tolerance=args.online_grad_tolerance,
        loss_change_tolerance=args.online_loss_change_tolerance,
    )

    if args.mode == "rollout":
        rollout = rollout_one_game(
            runtime=runtime,
            type_index=type_index,
            sample_actions=bool(args.sample_actions),
            use_action_clip=bool(args.action_clip),
            generator=generator,
            online_solve=online_cfg,
        )
        payload = build_rollout_result_payload(runtime, type_index, rollout)
        if args.save_trajectory_png is not None:
            save_trajectory_png(
                runtime=runtime,
                rollout=rollout,
                output_path=args.save_trajectory_png,
                title=args.figure_title,
            )
            print(f"saved_png={args.save_trajectory_png}", flush=True)
        if args.save_animation_html is not None:
            save_animation_html(
                runtime=runtime,
                rollout=rollout,
                output_path=args.save_animation_html,
                title=args.figure_title,
                true_type=type_index,
            )
            print(f"saved_html={args.save_animation_html}", flush=True)
        if args.output_json is not None:
            _save_json(payload, args.output_json)
        elif args.output_dir is not None:
            _save_json(payload, args.output_dir / "rollout_result.json")
        print(
            f"mode=rollout type_index={type_index} "
            f"root_objective={payload['costs']['root_objective']:.6f} "
            f"rollout_cost={payload['costs']['rollout_cost']:.6f} "
            f"offline_total_ms={payload['timings_ms']['offline_total_ms']:.3f}",
            flush=True,
        )
        return

    x_current_vals = _parse_state_csv(args.x_current, expected_dim=int(runtime.problem.state_dim), name="--x-current")
    if x_current_vals is None:
        x_current = runtime.x0.clone()
    else:
        x_current = torch.tensor(x_current_vals, device=runtime.setup.output_device, dtype=runtime.setup.output_dtype)
    belief_vals = _parse_belief_csv(args.belief_current, I=int(runtime.problem.tree.type_count))
    belief_current = None
    if belief_vals is not None:
        belief_current = torch.tensor(belief_vals, device=runtime.setup.output_device, dtype=runtime.setup.output_dtype)
    prototypes_so_far = list(_parse_csv_ints(args.prototypes_so_far))
    query_precompile_ms, query_precompile_diag = precompile_online_solver(runtime, online_cfg)

    step = query_policy_action(
        runtime=runtime,
        type_index=type_index,
        x_current=x_current,
        prototypes_so_far=prototypes_so_far,
        time_step=args.time_step,
        belief_current=belief_current,
        sample_actions=bool(args.sample_actions),
        use_action_clip=bool(args.action_clip),
        online_solve=online_cfg,
        generator=generator,
    )
    if belief_current is None:
        belief_current = step.current_belief
    payload = build_query_result_payload(
        runtime,
        type_index=type_index,
        x_current=x_current,
        belief_current=belief_current,
        prototypes_so_far=prototypes_so_far,
        step_result=step,
    )
    if args.output_json is not None:
        payload["timings_ms"]["online_precompile_ms"] = float(query_precompile_ms)
        payload["online_precompile_diag"] = query_precompile_diag
        _save_json(payload, args.output_json)
    elif args.output_dir is not None:
        payload["timings_ms"]["online_precompile_ms"] = float(query_precompile_ms)
        payload["online_precompile_diag"] = query_precompile_diag
        _save_json(payload, args.output_dir / "query_result.json")
    print(
        f"mode=query step={step.time_step} proto={step.prototype_index} "
        f"u={step.u.detach().cpu().tolist()} v={step.v.detach().cpu().tolist()} "
        f"policy_source={step.policy_source}",
        flush=True,
    )


if __name__ == "__main__":
    main()
