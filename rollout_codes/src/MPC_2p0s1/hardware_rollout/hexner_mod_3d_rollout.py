from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch

if __package__ in (None, ""):
    _FILE = Path(__file__).resolve()
    _PACKAGE_ROOT = _FILE.parents[1]
    _PARENT = _PACKAGE_ROOT.parent
    if str(_PARENT) not in sys.path:
        sys.path.insert(0, str(_PARENT))

from MPC_2p0s1.config.base_config import GameConfig
from MPC_2p0s1.core.action_spaces import BoxActionSpace
from MPC_2p0s1.games.base_lq_game import BaseLQGame
from MPC_2p0s1.games.hexner_mod_3d_game import HexnerMod3DGame, HexnerMod3DParams
from MPC_2p0s1.games.hexner_mod_3d_game_original import (
    HexnerMod3DOriginalGame,
    HexnerMod3DOriginalParams,
)
from MPC_2p0s1.outer_opt.amortized_alpha import AmortizedAlphaConfig, AmortizedAlphaParam
from MPC_2p0s1.tree.averaged_costs import compute_averaged_costs
from MPC_2p0s1.tree.belief_tree import BeliefTree, build_belief_tree
from MPC_2p0s1.tree.indexing import FullIaryTreeIndexer
from MPC_2p0s1.tree.riccati_tree import RiccatiSolution, riccati_backward


@dataclass
class LoadedSetup:
    checkpoint_path: Path
    game_type: str
    payload: Dict[str, Any]
    meta: Dict[str, Any]
    args_ckpt: Dict[str, Any]
    game_cfg: GameConfig
    game: BaseLQGame
    model: AmortizedAlphaParam
    indexer: FullIaryTreeIndexer
    action_space: BoxActionSpace


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parse_csv_floats(spec: str) -> Tuple[float, ...]:
    vals = [v.strip() for v in str(spec).split(",") if v.strip()]
    if not vals:
        raise ValueError(f"Expected comma-separated floats, got '{spec}'.")
    return tuple(float(v) for v in vals)


def _parse_csv_ints(spec: str) -> Tuple[int, ...]:
    vals = [v.strip() for v in str(spec).split(",") if v.strip()]
    if not vals:
        raise ValueError(f"Expected comma-separated ints, got '{spec}'.")
    return tuple(int(v) for v in vals)


def _maybe_parse_tuple_floats(value: Any) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return tuple(float(v) for v in value)
    text = str(value).strip()
    if text.lower() == "none" or text == "":
        return None
    return _parse_csv_floats(text)


def _parse_type_k_diags(value: Any, I: int, dx1: int) -> Optional[Tuple[Tuple[float, ...], ...]]:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        rows = []
        for row in value:
            row_vals = tuple(float(v) for v in row)
            if len(row_vals) != dx1:
                raise ValueError(
                    f"Expected dx1={dx1} values per type-k row, got {len(row_vals)}."
                )
            rows.append(row_vals)
        if len(rows) != I:
            raise ValueError(f"Expected I={I} rows in type-k diagonals, got {len(rows)}.")
        return tuple(rows)

    rows_text = [row.strip() for row in str(value).split(";") if row.strip()]
    if len(rows_text) != I:
        raise ValueError(f"Expected I={I} rows in type-k diagonals, got {len(rows_text)}.")
    rows = []
    for row in rows_text:
        vals = tuple(float(v.strip()) for v in row.split(",") if v.strip())
        if len(vals) != dx1:
            raise ValueError(f"Expected dx1={dx1} values per row, got {len(vals)} in '{row}'.")
        rows.append(vals)
    return tuple(rows)


def _parse_matrix(value: Any, size: int) -> Optional[Tuple[Tuple[float, ...], ...]]:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        rows = []
        for row in value:
            row_vals = tuple(float(v) for v in row)
            if len(row_vals) != size:
                raise ValueError(
                    f"Expected {size} values per terminal-matrix row, got {len(row_vals)}."
                )
            rows.append(row_vals)
        if len(rows) != size:
            raise ValueError(f"Expected {size} rows in terminal-matrix, got {len(rows)}.")
        return tuple(rows)

    rows_text = [row.strip() for row in str(value).split(";") if row.strip()]
    if len(rows_text) != size:
        raise ValueError(f"Expected {size} rows in terminal-matrix, got {len(rows_text)}.")
    rows = []
    for row in rows_text:
        vals = tuple(float(v.strip()) for v in row.split(",") if v.strip())
        if len(vals) != size:
            raise ValueError(
                f"Expected {size} values per terminal-matrix row, got {len(vals)} in '{row}'."
            )
        rows.append(vals)
    return tuple(rows)


def _coerce_dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    text = str(value).strip().lower()
    if "float64" in text or text == "torch.double":
        return torch.float64
    if "float32" in text or text == "torch.float":
        return torch.float32
    raise ValueError(f"Unsupported dtype value '{value}'.")


def _rebuild_game_config(meta_dict: Dict[str, Any], device: torch.device) -> GameConfig:
    cfg = GameConfig()
    for key, value in meta_dict.items():
        setattr(cfg, key, value)

    cfg.device = str(device)
    cfg.device_resolved = device

    if not hasattr(cfg, "dtype"):
        cfg.dtype = torch.float32
    else:
        cfg.dtype = _coerce_dtype(cfg.dtype)

    if not hasattr(cfg, "dx"):
        cfg.dx = cfg.dx1 + cfg.dx2
    if not hasattr(cfg, "tau"):
        cfg.tau = cfg.T / cfg.K

    return cfg


def _infer_game_type(args_ckpt: Dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    coupled_markers = {"target_state", "R_diag", "S_diag", "terminal_matrix", "terminal_scale"}
    if any(key in args_ckpt for key in coupled_markers):
        return "coupled"
    return "original"


def _build_original_game(game_cfg: GameConfig, args_ckpt: Dict[str, Any]) -> BaseLQGame:
    params = HexnerMod3DOriginalParams(
        theta_values=_parse_csv_floats(args_ckpt.get("theta_values", "-1.0,1.0")),
        target_z=_maybe_parse_tuple_floats(args_ckpt.get("target_z", None)),
        R1_scale=float(args_ckpt.get("R1_scale", 1.0)),
        R2_scale=float(args_ckpt.get("R2_scale", 0.4)),
        K1_scale=float(args_ckpt.get("K1_scale", 1.0)),
        K2_scale=float(args_ckpt.get("K2_scale", 1.0)),
        extra_position_scale=float(args_ckpt.get("extra_position_scale", 20.0)),
        terminal_velocity_scale=float(args_ckpt.get("terminal_velocity_scale", 1.0)),
        type_k_diags=_parse_type_k_diags(
            args_ckpt.get("type_k_diags", None),
            I=int(game_cfg.I),
            dx1=int(game_cfg.dx1),
        ),
        default_x0=_maybe_parse_tuple_floats(args_ckpt.get("default_x0", None)),
    )
    return HexnerMod3DOriginalGame(cfg=game_cfg, params=params, prior=None)


def _build_coupled_game(game_cfg: GameConfig, args_ckpt: Dict[str, Any]) -> BaseLQGame:
    params = HexnerMod3DParams(
        theta_values=_parse_csv_floats(args_ckpt.get("theta_values", "-1.0,1.0")),
        target_state=_maybe_parse_tuple_floats(args_ckpt.get("target_state", None)),
        R_diag=_parse_csv_floats(args_ckpt.get("R_diag", "1.30,0.55,1.80")),
        S_diag=_parse_csv_floats(args_ckpt.get("S_diag", "3.60,2.40,3.00")),
        terminal_K=_parse_matrix(args_ckpt.get("terminal_matrix", None), size=int(game_cfg.dx1)),
        terminal_scale=float(args_ckpt.get("terminal_scale", 1.0)),
        default_x0=_maybe_parse_tuple_floats(args_ckpt.get("default_x0", None)),
    )
    return HexnerMod3DGame(cfg=game_cfg, params=params, prior=None)


def load_setup(
    checkpoint_path: Path,
    device: torch.device,
    game_type: str = "auto",
) -> LoadedSetup:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    meta = payload.get("meta", {})
    args_ckpt = meta.get("args", {})
    game_cfg = _rebuild_game_config(meta.get("game_config", {}), device=device)
    action_space = BoxActionSpace.from_config(game_cfg)
    inferred_game_type = _infer_game_type(args_ckpt=args_ckpt, requested=game_type)

    if inferred_game_type == "original":
        game = _build_original_game(game_cfg=game_cfg, args_ckpt=args_ckpt)
    elif inferred_game_type == "coupled":
        game = _build_coupled_game(game_cfg=game_cfg, args_ckpt=args_ckpt)
    else:
        raise ValueError(f"Unsupported game type '{inferred_game_type}'.")

    hidden_dims = _parse_csv_ints(args_ckpt.get("hidden_dims", "256,256"))
    model = AmortizedAlphaParam(
        indexer=FullIaryTreeIndexer(I=game_cfg.I, K=game_cfg.K),
        dx=game_cfg.dx,
        I=game_cfg.I,
        cfg=AmortizedAlphaConfig(
            hidden_dims=hidden_dims,
            activation=str(args_ckpt.get("activation", "silu")),
            dropout=float(args_ckpt.get("dropout", 0.0)),
        ),
        dtype=game_cfg.dtype,
        device=device,
    )
    if "model_state_dict" not in payload:
        raise RuntimeError("Checkpoint is missing 'model_state_dict'.")
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    return LoadedSetup(
        checkpoint_path=checkpoint_path,
        game_type=inferred_game_type,
        payload=payload,
        meta=meta,
        args_ckpt=args_ckpt,
        game_cfg=game_cfg,
        game=game,
        model=model,
        indexer=model.indexer,
        action_space=action_space,
    )


def _sample_uniform_box(
    ref: torch.Tensor,
    jitter: float,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if float(jitter) <= 0.0:
        return ref.clone()
    noise = (2.0 * torch.rand(ref.shape, generator=generator, device=ref.device, dtype=ref.dtype) - 1.0)
    return ref + float(jitter) * noise


def _sample_prior(
    I: int,
    prior_min: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if I == 2:
        lo = float(prior_min)
        hi = 1.0 - lo
        if not (0.0 <= lo < 0.5):
            raise ValueError(f"prior_min must satisfy 0 <= prior_min < 0.5, got {prior_min}.")
        p_type0 = lo + (hi - lo) * torch.rand((), generator=generator, device=device, dtype=dtype)
        return torch.stack([p_type0, 1.0 - p_type0], dim=0)

    raw = torch.rand(I, generator=generator, device=device, dtype=dtype).clamp_min(1e-6)
    return raw / raw.sum()


def sample_context(
    setup: LoadedSetup,
    args: argparse.Namespace,
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = setup.game.device_resolved
    dtype = setup.game.dtype

    if args.x0 is not None:
        vals = _parse_csv_floats(args.x0)
        if len(vals) != setup.game.dx:
            raise ValueError(f"--x0 must provide {setup.game.dx} values, got {len(vals)}.")
        x0 = torch.tensor(vals, device=device, dtype=dtype)
    else:
        x0 = setup.game.default_initial_state().to(device=device, dtype=dtype)
        if setup.game_type == "original":
            p1_pos_jitter = (
                float(args.p1_pos_jitter)
                if args.p1_pos_jitter is not None
                else float(setup.args_ckpt.get("p1_pos_jitter", 0.40))
            )
            p2_pos_jitter = (
                float(args.p2_pos_jitter)
                if args.p2_pos_jitter is not None
                else float(setup.args_ckpt.get("p2_pos_jitter", 0.40))
            )
            p1_vel_jitter = (
                float(args.p1_vel_jitter)
                if args.p1_vel_jitter is not None
                else float(setup.args_ckpt.get("p1_vel_jitter", 0.0))
            )
            p2_vel_jitter = (
                float(args.p2_vel_jitter)
                if args.p2_vel_jitter is not None
                else float(setup.args_ckpt.get("p2_vel_jitter", 0.0))
            )
            x0[0:3] = _sample_uniform_box(x0[0:3], p1_pos_jitter, generator)
            x0[6:9] = _sample_uniform_box(x0[6:9], p2_pos_jitter, generator)
            x0[3:6] = _sample_uniform_box(x0[3:6], p1_vel_jitter, generator)
            x0[9:12] = _sample_uniform_box(x0[9:12], p2_vel_jitter, generator)
        else:
            pos_jitter = (
                float(args.pos_jitter)
                if args.pos_jitter is not None
                else float(setup.args_ckpt.get("pos_jitter", 0.40))
            )
            vel_jitter = (
                float(args.vel_jitter)
                if args.vel_jitter is not None
                else float(setup.args_ckpt.get("vel_jitter", 0.0))
            )
            x0[0:3] = _sample_uniform_box(x0[0:3], pos_jitter, generator)
            x0[6:9] = _sample_uniform_box(x0[6:9], pos_jitter, generator)
            x0[3:6] = _sample_uniform_box(x0[3:6], vel_jitter, generator)
            x0[9:12] = _sample_uniform_box(x0[9:12], vel_jitter, generator)

    if args.prior is not None:
        vals = _parse_csv_floats(args.prior)
        if len(vals) != setup.game.I:
            raise ValueError(f"--prior must provide {setup.game.I} values, got {len(vals)}.")
        p0 = torch.tensor(vals, device=device, dtype=dtype)
        p0 = p0 / p0.sum()
    else:
        p0 = _sample_prior(
            I=setup.game.I,
            prior_min=float(args.prior_min),
            device=device,
            dtype=dtype,
            generator=generator,
        )

    return x0, p0


def sample_true_type(
    p0: torch.Tensor,
    generator: Optional[torch.Generator],
    forced_type_index: Optional[int],
) -> int:
    if forced_type_index is not None:
        return int(forced_type_index)
    return int(torch.multinomial(p0, num_samples=1, replacement=True, generator=generator).item())


def rollout_total_cost(game: BaseLQGame, x_traj: torch.Tensor, u_traj: torch.Tensor, v_traj: torch.Tensor, type_index: int) -> float:
    R_i = game.R[type_index]
    S_i = game.S[type_index]
    tau = float(game.cfg.tau)
    run_u = 0.5 * tau * torch.einsum("kd,dd,kd->k", u_traj, R_i, u_traj)
    run_v = 0.5 * tau * torch.einsum("kd,dd,kd->k", v_traj, S_i, v_traj)
    running = (run_u - run_v).sum()
    terminal = game.terminal_cost_type(type_index, x_traj[-1])
    return float((running + terminal).detach().cpu().item())


def terminal_metrics(game: BaseLQGame, xT: torch.Tensor) -> Dict[str, float]:
    dx1 = game.cfg.dx1
    p1_pos = xT[:3]
    p1_vel = xT[3:6]
    p2_pos = xT[dx1 : dx1 + 3]
    p2_vel = xT[dx1 + 3 : dx1 + 6]
    return {
        "p1_xT": float(p1_pos[0].item()),
        "p1_yT": float(p1_pos[1].item()),
        "p1_zT": float(p1_pos[2].item()),
        "p1_speed_T": float(torch.linalg.norm(p1_vel).item()),
        "p2_xT": float(p2_pos[0].item()),
        "p2_yT": float(p2_pos[1].item()),
        "p2_zT": float(p2_pos[2].item()),
        "p2_speed_T": float(torch.linalg.norm(p2_vel).item()),
    }


def _time_block(device: torch.device, fn) -> Tuple[Any, float]:
    _sync_if_needed(device)
    t0 = time.perf_counter_ns()
    out = fn()
    _sync_if_needed(device)
    t1 = time.perf_counter_ns()
    return out, (t1 - t0) / 1e6


def solve_from_checkpoint_context(
    setup: LoadedSetup,
    x0: torch.Tensor,
    p0: torch.Tensor,
    use_action_clip: bool,
    one_hot_alpha: bool,
) -> Tuple[torch.Tensor, BeliefTree, RiccatiSolution, Dict[str, float]]:
    device = setup.game.device_resolved
    solve_action_space = setup.action_space if use_action_clip else None

    def _eval_alpha():
        with torch.inference_mode():
            alpha_out = setup.model.single_alpha(x0=x0, p0=p0)
            if one_hot_alpha:
                idx = torch.argmax(alpha_out, dim=-1, keepdim=True)
                alpha_proj = torch.zeros_like(alpha_out)
                alpha_proj.scatter_(-1, idx, 1.0)
                return alpha_proj
            return alpha_out

    alpha, alpha_ms = _time_block(device, _eval_alpha)
    with torch.inference_mode():
        belief_tree, belief_ms = _time_block(
            device, lambda: build_belief_tree(alpha=alpha, p0=p0, indexer=setup.indexer)
        )
        avg_costs, avg_ms = _time_block(
            device, lambda: compute_averaged_costs(game=setup.game, belief_tree=belief_tree)
        )
        riccati_sol, riccati_ms = _time_block(
            device,
            lambda: riccati_backward(
                game=setup.game,
                belief_tree=belief_tree,
                avg_costs=avg_costs,
                action_space=solve_action_space,
            ),
        )

    return alpha, belief_tree, riccati_sol, {
        "alpha_eval_ms": alpha_ms,
        "belief_tree_ms": belief_ms,
        "avg_costs_ms": avg_ms,
        "riccati_ms": riccati_ms,
        "offline_total_ms": alpha_ms + belief_ms + avg_ms + riccati_ms,
    }


def rollout_one_game(
    setup: LoadedSetup,
    alpha: torch.Tensor,
    belief_tree: BeliefTree,
    riccati_sol: RiccatiSolution,
    x0: torch.Tensor,
    type_index: int,
    sample_actions: bool,
    use_action_clip: bool,
    generator: Optional[torch.Generator],
) -> Dict[str, Any]:
    K = setup.indexer.K
    I = setup.indexer.I
    dx = setup.game.dx
    du = setup.game.du
    dv = setup.game.dv
    device = setup.game.device_resolved
    dtype = setup.game.dtype
    rollout_action_space = setup.action_space if use_action_clip else None

    x_traj = torch.empty(K + 1, dx, device=device, dtype=dtype)
    u_traj = torch.empty(K, du, device=device, dtype=dtype)
    v_traj = torch.empty(K, dv, device=device, dtype=dtype)
    belief_traj = torch.empty(K + 1, I, device=device, dtype=dtype)
    proto_indices = torch.empty(K, dtype=torch.long, device=device)

    node_indices = [0]
    control_ms: list[float] = []
    step_ms: list[float] = []

    x = x0.to(device=device, dtype=dtype)
    node_idx = 0
    belief = belief_tree.beliefs[0][node_idx]
    x_traj[0] = x
    belief_traj[0] = belief

    with torch.inference_mode():
        for k in range(K):
            step_t0 = time.perf_counter_ns()

            _sync_if_needed(device)
            control_t0 = time.perf_counter_ns()
            alpha_row = alpha[k, node_idx, type_index]
            if sample_actions:
                a = torch.multinomial(
                    alpha_row,
                    num_samples=1,
                    replacement=True,
                    generator=generator,
                ).squeeze(0)
            else:
                a = torch.argmax(alpha_row)

            a_int = int(a.item())
            K_u_edge = riccati_sol.K_u[k][node_idx, a_int]
            K_v_edge = riccati_sol.K_v[k][node_idx, a_int]
            kappa_u_edge = riccati_sol.kappa_u[k][node_idx, a_int]
            kappa_v_edge = riccati_sol.kappa_v[k][node_idx, a_int]
            u = K_u_edge @ x + kappa_u_edge
            v = K_v_edge @ x + kappa_v_edge
            if rollout_action_space is not None:
                u = rollout_action_space.clip_u(u)
                v = rollout_action_space.clip_v(v)
            _sync_if_needed(device)
            control_t1 = time.perf_counter_ns()

            u_traj[k] = u
            v_traj[k] = v
            proto_indices[k] = a

            x = setup.game.step_dynamics(x, u, v)
            x_traj[k + 1] = x

            child_idx = setup.indexer.child_index(k, node_idx, a_int)
            belief = belief_tree.beliefs[k + 1][child_idx]
            belief_traj[k + 1] = belief
            node_idx = child_idx
            node_indices.append(node_idx)

            step_t1 = time.perf_counter_ns()
            control_ms.append((control_t1 - control_t0) / 1e6)
            step_ms.append((step_t1 - step_t0) / 1e6)

    total_cost = rollout_total_cost(
        game=setup.game,
        x_traj=x_traj,
        u_traj=u_traj,
        v_traj=v_traj,
        type_index=type_index,
    )
    terminal = terminal_metrics(setup.game, x_traj[-1])

    return {
        "x_traj": x_traj,
        "u_traj": u_traj,
        "v_traj": v_traj,
        "belief_traj": belief_traj,
        "proto_indices": proto_indices,
        "node_indices": node_indices,
        "control_compute_ms": control_ms,
        "step_total_ms": step_ms,
        "rollout_cost": total_cost,
        "terminal": terminal,
    }


def _tensor_to_list(x: torch.Tensor) -> Any:
    return x.detach().cpu().tolist()


def _root_posteriors(root_alpha: torch.Tensor, prior: torch.Tensor) -> list[list[float]]:
    posteriors: list[list[float]] = []
    for action_idx in range(root_alpha.shape[-1]):
        numer = prior * root_alpha[:, action_idx]
        denom = numer.sum()
        if float(denom.item()) <= 1e-12:
            post = torch.zeros_like(numer)
        else:
            post = numer / denom
        posteriors.append([float(v) for v in post.detach().cpu().tolist()])
    return posteriors


def build_result_payload(
    setup: LoadedSetup,
    x0: torch.Tensor,
    p0: torch.Tensor,
    true_type: int,
    alpha: torch.Tensor,
    riccati_sol: RiccatiSolution,
    solve_timing: Dict[str, float],
    rollout: Dict[str, Any],
) -> Dict[str, Any]:
    root_alpha = alpha[0, 0]
    x_traj = rollout["x_traj"]
    root_value = float(riccati_sol.value_at_root(x0).detach().cpu().item())
    control_times = rollout["control_compute_ms"]
    step_times = rollout["step_total_ms"]

    return {
        "checkpoint_path": str(setup.checkpoint_path),
        "game_type": setup.game_type,
        "checkpoint_step": setup.meta.get("step"),
        "sampled_context": {
            "x0": _tensor_to_list(x0),
            "prior": _tensor_to_list(p0),
            "true_type": int(true_type),
        },
        "policy": {
            "root_alpha": _tensor_to_list(root_alpha),
            "root_posteriors": _root_posteriors(root_alpha=root_alpha, prior=p0),
            "proto_indices": rollout["proto_indices"].detach().cpu().tolist(),
            "node_indices": [int(v) for v in rollout["node_indices"]],
        },
        "timings_ms": {
            **solve_timing,
            "control_compute_per_step_ms": [float(v) for v in control_times],
            "step_total_per_step_ms": [float(v) for v in step_times],
            "control_compute_mean_ms": float(sum(control_times) / max(len(control_times), 1)),
            "control_compute_max_ms": float(max(control_times) if control_times else 0.0),
            "step_total_mean_ms": float(sum(step_times) / max(len(step_times), 1)),
            "step_total_max_ms": float(max(step_times) if step_times else 0.0),
        },
        "costs": {
            "root_objective": root_value,
            "rollout_cost": float(rollout["rollout_cost"]),
        },
        "terminal": rollout["terminal"],
        "game": {
            "K": int(setup.game_cfg.K),
            "T": float(setup.game_cfg.T),
            "tau": float(setup.game_cfg.tau),
            "targets": _tensor_to_list(setup.game.type_targets()),
        },
        "trajectories": {
            "x_traj": _tensor_to_list(x_traj),
            "u_traj": _tensor_to_list(rollout["u_traj"]),
            "v_traj": _tensor_to_list(rollout["v_traj"]),
            "belief_traj": _tensor_to_list(rollout["belief_traj"]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one checkpoint-driven rollout for the 3D Hexner-mod amortized policy, "
            "with random context sampling and per-step control timing."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a trained amortized checkpoint (.pt).")
    parser.add_argument(
        "--game-type",
        type=str,
        default="auto",
        choices=["auto", "original", "coupled"],
        help="How to rebuild the game from checkpoint metadata.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--x0", type=str, default=None, help="Optional fixed 12D initial state as CSV.")
    parser.add_argument("--prior", type=str, default=None, help="Optional fixed prior as CSV.")
    parser.add_argument("--type-index", type=int, default=None, help="Optional realized type index. If omitted, sample from the prior.")
    parser.add_argument("--prior-min", type=float, default=0.05, help="Minimum type probability when randomly sampling a belief for I=2.")
    parser.add_argument("--sample-actions", action=argparse.BooleanOptionalAction, default=True, help="Whether to sample public signals/actions from alpha at rollout time.")
    parser.add_argument("--action-clip", action=argparse.BooleanOptionalAction, default=False, help="Whether to enforce box action clipping during solve and rollout.")
    parser.add_argument("--one-hot-alpha", action=argparse.BooleanOptionalAction, default=False, help="Project alpha to one-hot before solving the tree.")
    parser.add_argument("--p1-pos-jitter", type=float, default=None, help="Original-game P1 position jitter if x0 is sampled.")
    parser.add_argument("--p2-pos-jitter", type=float, default=None, help="Original-game P2 position jitter if x0 is sampled.")
    parser.add_argument("--p1-vel-jitter", type=float, default=None, help="Original-game P1 velocity jitter if x0 is sampled.")
    parser.add_argument("--p2-vel-jitter", type=float, default=None, help="Original-game P2 velocity jitter if x0 is sampled.")
    parser.add_argument("--pos-jitter", type=float, default=None, help="Coupled-game shared position jitter if x0 is sampled.")
    parser.add_argument("--vel-jitter", type=float, default=None, help="Coupled-game shared velocity jitter if x0 is sampled.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write rollout results as JSON.")
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False, help="Suppress summary printing.")
    return parser.parse_args()


def _print_summary(result: Dict[str, Any]) -> None:
    ctx = result["sampled_context"]
    root_alpha = result["policy"]["root_alpha"]
    times = result["timings_ms"]
    terminal = result["terminal"]
    print(f"checkpoint       : {result['checkpoint_path']}")
    print(f"game_type        : {result['game_type']}")
    print(f"checkpoint_step  : {result['checkpoint_step']}")
    print(f"true_type        : {ctx['true_type']}")
    print(f"prior            : {ctx['prior']}")
    print(f"x0               : {ctx['x0']}")
    print(f"root_alpha       : {root_alpha}")
    print(
        "offline_ms       : "
        f"alpha={times['alpha_eval_ms']:.3f}, "
        f"belief={times['belief_tree_ms']:.3f}, "
        f"avg={times['avg_costs_ms']:.3f}, "
        f"riccati={times['riccati_ms']:.3f}, "
        f"total={times['offline_total_ms']:.3f}"
    )
    print(
        "control_ms       : "
        f"mean={times['control_compute_mean_ms']:.6f}, "
        f"max={times['control_compute_max_ms']:.6f}, "
        f"per_step={times['control_compute_per_step_ms']}"
    )
    print(
        "terminal         : "
        f"p1_z={terminal['p1_zT']:.6f}, "
        f"p1_speed={terminal['p1_speed_T']:.6f}, "
        f"p2_z={terminal['p2_zT']:.6f}, "
        f"p2_speed={terminal['p2_speed_T']:.6f}"
    )
    print(
        "costs            : "
        f"root={result['costs']['root_objective']:.6f}, "
        f"rollout={result['costs']['rollout_cost']:.6f}"
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    torch.manual_seed(int(args.seed))
    generator = None

    setup = load_setup(
        checkpoint_path=args.checkpoint,
        device=device,
        game_type=args.game_type,
    )
    x0, p0 = sample_context(setup=setup, args=args, generator=generator)
    true_type = sample_true_type(
        p0=p0.detach().cpu(),
        generator=generator,
        forced_type_index=args.type_index,
    )

    alpha, belief_tree, riccati_sol, solve_timing = solve_from_checkpoint_context(
        setup=setup,
        x0=x0,
        p0=p0,
        use_action_clip=bool(args.action_clip),
        one_hot_alpha=bool(args.one_hot_alpha),
    )

    rollout = rollout_one_game(
        setup=setup,
        alpha=alpha,
        belief_tree=belief_tree,
        riccati_sol=riccati_sol,
        x0=x0,
        type_index=true_type,
        sample_actions=bool(args.sample_actions),
        use_action_clip=bool(args.action_clip),
        generator=generator,
    )

    result = build_result_payload(
        setup=setup,
        x0=x0,
        p0=p0,
        true_type=true_type,
        alpha=alpha,
        riccati_sol=riccati_sol,
        solve_timing=solve_timing,
        rollout=rollout,
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2))
        if not args.quiet:
            print(f"saved_json       : {args.output_json}")

    if not args.quiet:
        _print_summary(result)


if __name__ == "__main__":
    main()
