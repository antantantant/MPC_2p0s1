from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    _SEARCH_ROOTS = [
        _FILE.parent / "src",
        _FILE.parents[2],
    ]
    for _root in _SEARCH_ROOTS:
        if (_root / "MPC_2p0s1").exists() and str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

from MPC_2p0s1.config.base_config import GameConfig
from MPC_2p0s1.core.action_spaces import BoxActionSpace
from MPC_2p0s1.games.base_lq_game import BaseLQGame
from MPC_2p0s1.games.hexner_mod_3d_game import HexnerMod3DGame, HexnerMod3DParams
from MPC_2p0s1.games.hexner_mod_3d_game_original import (
    HexnerMod3DOriginalGame,
    HexnerMod3DOriginalParams,
)
from MPC_2p0s1.outer_opt.amortized_alpha import (
    AmortizedAlphaConfig,
    AmortizedAlphaParam,
    primal_objective_from_alpha,
)
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


@dataclass
class PolicyRuntime:
    setup: LoadedSetup
    x0: torch.Tensor
    p0: torch.Tensor
    alpha_full: torch.Tensor
    belief_tree: BeliefTree
    riccati_sol: RiccatiSolution
    solve_timing: Dict[str, float]


@dataclass
class OnlineSolveConfig:
    enabled: bool = False
    alpha_iters: int = 0
    alpha_lr: float = 5e-2
    alpha_early_stop: bool = True
    alpha_min_iters: int = 5
    alpha_patience: int = 3
    alpha_min_delta: float = 1e-5
    skip_solve_at_t0: bool = True


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
    policy_source: str
    solve_diag: Dict[str, Any]


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def rollout_total_cost(
    game: BaseLQGame,
    x_traj: torch.Tensor,
    u_traj: torch.Tensor,
    v_traj: torch.Tensor,
    type_index: int,
) -> float:
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


def _tensor_to_list(x: torch.Tensor) -> Any:
    return x.detach().cpu().tolist()


def _summarize_ms(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "mean_ms": 0.0,
            "max_ms": 0.0,
            "min_ms": 0.0,
            "total_ms": 0.0,
            "count": 0,
        }
    return {
        "mean_ms": float(arr.mean()),
        "max_ms": float(arr.max()),
        "min_ms": float(arr.min()),
        "total_ms": float(arr.sum()),
        "count": int(arr.size),
    }


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


def build_policy_runtime(
    setup: LoadedSetup,
    x0: torch.Tensor,
    p0: torch.Tensor,
    use_action_clip: bool = False,
    one_hot_alpha: bool = False,
) -> PolicyRuntime:
    alpha, belief_tree, riccati_sol, solve_timing = solve_from_checkpoint_context(
        setup=setup,
        x0=x0,
        p0=p0,
        use_action_clip=use_action_clip,
        one_hot_alpha=one_hot_alpha,
    )
    return PolicyRuntime(
        setup=setup,
        x0=x0,
        p0=p0,
        alpha_full=alpha,
        belief_tree=belief_tree,
        riccati_sol=riccati_sol,
        solve_timing=solve_timing,
    )


def load_policy_runtime(
    checkpoint_path: Path,
    device: torch.device,
    game_type: str = "auto",
    x0: Optional[torch.Tensor] = None,
    p0: Optional[torch.Tensor] = None,
    use_action_clip: bool = False,
    one_hot_alpha: bool = False,
) -> PolicyRuntime:
    setup = load_setup(checkpoint_path=checkpoint_path, device=device, game_type=game_type)
    if x0 is None:
        x0 = setup.game.default_initial_state().to(device=setup.game.device_resolved, dtype=setup.game.dtype)
    if p0 is None:
        p0 = setup.game.default_prior().to(device=setup.game.device_resolved, dtype=setup.game.dtype)
    return build_policy_runtime(
        setup=setup,
        x0=x0,
        p0=p0,
        use_action_clip=use_action_clip,
        one_hot_alpha=one_hot_alpha,
    )


def _node_index_from_prototypes(indexer: FullIaryTreeIndexer, prototypes_so_far: Sequence[int]) -> int:
    node_idx = 0
    for t, proto in enumerate(prototypes_so_far):
        if not (0 <= int(proto) < indexer.I):
            raise ValueError(f"Prototype index {proto} at step {t} is out of range [0, {indexer.I}).")
        if t >= indexer.K:
            raise ValueError(f"Prototype history length {len(prototypes_so_far)} exceeds horizon K={indexer.K}.")
        node_idx = indexer.child_index(t, node_idx, int(proto))
    return int(node_idx)


def infer_belief_from_prototypes(
    runtime: PolicyRuntime,
    prototypes_so_far: Sequence[int],
) -> Tuple[int, torch.Tensor]:
    node_idx = _node_index_from_prototypes(runtime.setup.indexer, prototypes_so_far)
    belief = runtime.belief_tree.beliefs[len(prototypes_so_far)][node_idx]
    return node_idx, belief


def rebase_alpha_to_remaining(
    alpha_full: torch.Tensor,
    indexer_full: FullIaryTreeIndexer,
    prototypes_so_far: Sequence[int],
) -> Tuple[torch.Tensor, FullIaryTreeIndexer]:
    I = indexer_full.I
    K_full = indexer_full.K
    t = len(prototypes_so_far)
    K_rem = K_full - t
    if K_rem <= 0:
        raise ValueError(f"No remaining horizon: K_full={K_full}, prototypes={t}")

    indexer_rem = FullIaryTreeIndexer(I=I, K=K_rem)
    alpha_rem = alpha_full.new_zeros((K_rem, indexer_rem.max_nodes_per_depth, I, I))

    base = 0
    for proto in prototypes_so_far:
        base = base * I + int(proto)

    for d in range(K_rem):
        num_nodes = indexer_rem.node_count(d)
        for new_node in range(num_nodes):
            old_node = base * (I ** d) + new_node
            alpha_rem[d, new_node] = alpha_full[t + d, old_node]

    return alpha_rem, indexer_rem


def optimize_remaining_alpha(
    game: BaseLQGame,
    alpha_init: torch.Tensor,
    indexer_rem: FullIaryTreeIndexer,
    action_space: Optional[BoxActionSpace],
    x0: torch.Tensor,
    p0: torch.Tensor,
    num_iters: int,
    lr: float,
    use_early_stop: bool,
    min_iters: int,
    patience: int,
    min_delta: float,
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
    with torch.no_grad():
        warm_loss_t = primal_objective_from_alpha(
            game=game,
            alpha=alpha_init,
            indexer=indexer_rem,
            x0=x0,
            p0=p0,
            action_space=action_space,
            return_details=False,
        )
    warm_loss = float(warm_loss_t.detach().cpu().item())

    if int(num_iters) <= 0:
        with torch.no_grad():
            final_loss_t, final_details = primal_objective_from_alpha(
                game=game,
                alpha=alpha_init,
                indexer=indexer_rem,
                x0=x0,
                p0=p0,
                action_space=action_space,
                return_details=True,
            )
        final_loss = float(final_loss_t.detach().cpu().item())
        diag = {
            "warm_loss": warm_loss,
            "final_loss": final_loss,
            "iters_run": 0,
            "early_stopped": False,
            "converged": False,
            "loss_curve": [final_loss],
        }
        return alpha_init, final_details, diag

    logits = torch.nn.Parameter(torch.log(alpha_init.clamp_min(1e-8)))
    optimizer = torch.optim.Adam([logits], lr=float(lr))

    losses: List[float] = []
    prev = None
    stable_steps = 0
    early_stopped = False

    for _ in range(int(num_iters)):
        optimizer.zero_grad()
        alpha_it = torch.softmax(logits, dim=-1)
        loss_it = primal_objective_from_alpha(
            game=game,
            alpha=alpha_it,
            indexer=indexer_rem,
            x0=x0,
            p0=p0,
            action_space=action_space,
            return_details=False,
        )
        loss_it.backward()
        optimizer.step()

        cur = float(loss_it.detach().cpu().item())
        losses.append(cur)

        if prev is not None and abs(cur - prev) <= float(min_delta):
            stable_steps += 1
        else:
            stable_steps = 0
        prev = cur

        if bool(use_early_stop) and len(losses) >= int(min_iters) and stable_steps >= int(patience):
            early_stopped = True
            break

    alpha_final = torch.softmax(logits.detach(), dim=-1)
    with torch.no_grad():
        final_loss_t, final_details = primal_objective_from_alpha(
            game=game,
            alpha=alpha_final,
            indexer=indexer_rem,
            x0=x0,
            p0=p0,
            action_space=action_space,
            return_details=True,
        )
    final_loss = float(final_loss_t.detach().cpu().item())
    converged = bool(
        early_stopped
        or (len(losses) >= 2 and abs(losses[-1] - losses[-2]) <= float(min_delta))
    )

    diag = {
        "warm_loss": warm_loss,
        "final_loss": final_loss,
        "iters_run": len(losses),
        "early_stopped": bool(early_stopped),
        "converged": converged,
        "loss_curve": losses,
    }
    return alpha_final, final_details, diag


def _choose_prototype(
    alpha_row: torch.Tensor,
    sample_actions: bool,
    generator: Optional[torch.Generator],
) -> Tuple[int, torch.Tensor]:
    if sample_actions:
        proto = int(
            torch.multinomial(alpha_row, num_samples=1, replacement=True, generator=generator).item()
        )
    else:
        proto = int(torch.argmax(alpha_row).item())
    return proto, alpha_row


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
    setup = runtime.setup
    game = setup.game
    device = game.device_resolved
    dtype = game.dtype
    indexer_full = setup.indexer
    prototypes = [int(v) for v in prototypes_so_far]
    t = len(prototypes)

    if time_step is not None and int(time_step) != t:
        raise ValueError(
            f"time_step={time_step} does not match len(prototypes_so_far)={len(prototypes)}."
        )
    if not (0 <= int(type_index) < setup.game.I):
        raise ValueError(f"type_index={type_index} out of range [0, {setup.game.I}).")
    if t >= indexer_full.K:
        raise ValueError(f"No control available at t={t}; horizon K={indexer_full.K}.")

    x = x_current.to(device=device, dtype=dtype)
    node_idx, inferred_belief = infer_belief_from_prototypes(runtime=runtime, prototypes_so_far=prototypes)
    if belief_current is None:
        p = inferred_belief
    else:
        p = belief_current.to(device=device, dtype=dtype)
        p = p / p.sum().clamp_min(1e-12)

    online_cfg = online_solve or OnlineSolveConfig(enabled=False)
    rollout_action_space = setup.action_space if use_action_clip else None
    solve_diag: Dict[str, Any] = {}
    online_solve_ms = 0.0

    if bool(online_cfg.enabled) and not (t == 0 and bool(online_cfg.skip_solve_at_t0)):
        alpha_rem_init, indexer_rem = rebase_alpha_to_remaining(
            alpha_full=runtime.alpha_full,
            indexer_full=indexer_full,
            prototypes_so_far=prototypes,
        )

        def _solve_remaining():
            return optimize_remaining_alpha(
                game=game,
                alpha_init=alpha_rem_init,
                indexer_rem=indexer_rem,
                action_space=rollout_action_space,
                x0=x,
                p0=p,
                num_iters=int(online_cfg.alpha_iters),
                lr=float(online_cfg.alpha_lr),
                use_early_stop=bool(online_cfg.alpha_early_stop),
                min_iters=int(online_cfg.alpha_min_iters),
                patience=int(online_cfg.alpha_patience),
                min_delta=float(online_cfg.alpha_min_delta),
            )

        (alpha_rem_opt, details, solve_diag), online_solve_ms = _time_block(device, _solve_remaining)
        action_probs = alpha_rem_opt[0, 0, type_index]
        riccati_sol = details["riccati_solution"]
        belief_tree = details["belief_tree"]
        policy_source = "online_remaining_solve"
        gain_k = 0
        gain_node_idx = 0
    else:
        action_probs = runtime.alpha_full[t, node_idx, type_index]
        riccati_sol = runtime.riccati_sol
        belief_tree = runtime.belief_tree
        policy_source = "offline_root_tree"
        if bool(online_cfg.enabled) and t == 0 and bool(online_cfg.skip_solve_at_t0):
            policy_source = "offline_root_tree_skip_solve_at_t0"
        gain_k = t
        gain_node_idx = node_idx

    _sync_if_needed(device)
    control_t0 = time.perf_counter_ns()
    proto_idx, action_probs = _choose_prototype(
        alpha_row=action_probs,
        sample_actions=sample_actions,
        generator=generator,
    )
    K_u_edge = riccati_sol.K_u[gain_k][gain_node_idx, proto_idx]
    K_v_edge = riccati_sol.K_v[gain_k][gain_node_idx, proto_idx]
    kappa_u_edge = riccati_sol.kappa_u[gain_k][gain_node_idx, proto_idx]
    kappa_v_edge = riccati_sol.kappa_v[gain_k][gain_node_idx, proto_idx]
    u = K_u_edge @ x + kappa_u_edge
    v = K_v_edge @ x + kappa_v_edge
    if rollout_action_space is not None:
        u = rollout_action_space.clip_u(u)
        v = rollout_action_space.clip_v(v)
    _sync_if_needed(device)
    control_t1 = time.perf_counter_ns()
    control_compute_ms = (control_t1 - control_t0) / 1e6

    child_node_idx = indexer_full.child_index(t, node_idx, proto_idx)
    if policy_source == "online_remaining_solve":
        next_belief = belief_tree.beliefs[1][proto_idx]
    else:
        next_belief = belief_tree.beliefs[t + 1][child_node_idx]

    return PolicyStepResult(
        time_step=t,
        node_idx=node_idx,
        child_node_idx=int(child_node_idx),
        prototype_index=int(proto_idx),
        action_probs=action_probs.detach().clone(),
        u=u.detach().clone(),
        v=v.detach().clone(),
        current_belief=p.detach().clone(),
        next_belief=next_belief.detach().clone(),
        control_compute_ms=float(control_compute_ms),
        online_solve_ms=float(online_solve_ms),
        policy_source=policy_source,
        solve_diag=solve_diag,
    )


def rollout_one_game(
    runtime: PolicyRuntime,
    type_index: int,
    sample_actions: bool,
    use_action_clip: bool,
    generator: Optional[torch.Generator],
    online_solve: Optional[OnlineSolveConfig] = None,
) -> Dict[str, Any]:
    setup = runtime.setup
    game = setup.game
    K = setup.indexer.K
    dx = game.dx
    du = game.du
    dv = game.dv
    I = game.I
    device = game.device_resolved
    dtype = game.dtype

    x_traj = torch.empty(K + 1, dx, device=device, dtype=dtype)
    u_traj = torch.empty(K, du, device=device, dtype=dtype)
    v_traj = torch.empty(K, dv, device=device, dtype=dtype)
    belief_traj = torch.empty(K + 1, I, device=device, dtype=dtype)
    proto_indices = torch.empty(K, dtype=torch.long, device=device)
    action_probs_traj = torch.empty(K, I, device=device, dtype=dtype)

    node_indices = [0]
    control_ms: List[float] = []
    step_ms: List[float] = []
    online_solve_ms: List[float] = []
    step_policy_sources: List[str] = []
    online_solve_diag: List[Dict[str, Any]] = []

    x = runtime.x0.to(device=device, dtype=dtype)
    p = runtime.p0.to(device=device, dtype=dtype)
    prototypes_so_far: List[int] = []

    x_traj[0] = x
    belief_traj[0] = p

    for _ in range(K):
        step_t0 = time.perf_counter_ns()
        step_result = query_policy_action(
            runtime=runtime,
            type_index=type_index,
            x_current=x,
            prototypes_so_far=prototypes_so_far,
            time_step=len(prototypes_so_far),
            belief_current=p,
            sample_actions=sample_actions,
            use_action_clip=use_action_clip,
            online_solve=online_solve,
            generator=generator,
        )
        u_traj[step_result.time_step] = step_result.u
        v_traj[step_result.time_step] = step_result.v
        proto_indices[step_result.time_step] = step_result.prototype_index
        action_probs_traj[step_result.time_step] = step_result.action_probs

        x = game.step_dynamics(x, step_result.u, step_result.v)
        p = step_result.next_belief
        x_traj[step_result.time_step + 1] = x
        belief_traj[step_result.time_step + 1] = p

        prototypes_so_far.append(step_result.prototype_index)
        node_indices.append(step_result.child_node_idx)
        control_ms.append(step_result.control_compute_ms)
        online_solve_ms.append(step_result.online_solve_ms)
        step_policy_sources.append(step_result.policy_source)
        if step_result.solve_diag:
            diag = dict(step_result.solve_diag)
            diag["time_step"] = int(step_result.time_step)
            diag["policy_source"] = step_result.policy_source
            diag["solve_ms"] = float(step_result.online_solve_ms)
            online_solve_diag.append(diag)

        step_t1 = time.perf_counter_ns()
        step_ms.append((step_t1 - step_t0) / 1e6)

    total_cost = rollout_total_cost(
        game=game,
        x_traj=x_traj,
        u_traj=u_traj,
        v_traj=v_traj,
        type_index=type_index,
    )
    terminal = terminal_metrics(game, x_traj[-1])

    return {
        "x_traj": x_traj,
        "u_traj": u_traj,
        "v_traj": v_traj,
        "belief_traj": belief_traj,
        "proto_indices": proto_indices,
        "node_indices": node_indices,
        "action_probs_traj": action_probs_traj,
        "control_compute_ms": control_ms,
        "step_total_ms": step_ms,
        "online_solve_ms": online_solve_ms,
        "online_solve_diag": online_solve_diag,
        "step_policy_sources": step_policy_sources,
        "rollout_cost": total_cost,
        "terminal": terminal,
    }


def _root_posteriors(root_alpha: torch.Tensor, prior: torch.Tensor) -> List[List[float]]:
    posteriors: List[List[float]] = []
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
    root_alpha = runtime.alpha_full[0, 0]
    x_traj = rollout["x_traj"]
    root_value = float(runtime.riccati_sol.value_at_root(runtime.x0).detach().cpu().item())
    control_times = rollout["control_compute_ms"]
    step_times = rollout["step_total_ms"]
    online_times = rollout["online_solve_ms"]
    online_summary = _summarize_ms(ms for ms in online_times if float(ms) > 0.0)

    return {
        "mode": "rollout",
        "checkpoint_path": str(runtime.setup.checkpoint_path),
        "game_type": runtime.setup.game_type,
        "checkpoint_step": runtime.setup.meta.get("step"),
        "sampled_context": {
            "x0": _tensor_to_list(runtime.x0),
            "prior": _tensor_to_list(runtime.p0),
            "true_type": int(true_type),
        },
        "policy": {
            "root_alpha": _tensor_to_list(root_alpha),
            "root_posteriors": _root_posteriors(root_alpha=root_alpha, prior=runtime.p0),
            "proto_indices": rollout["proto_indices"].detach().cpu().tolist(),
            "node_indices": [int(v) for v in rollout["node_indices"]],
            "action_probs_traj": _tensor_to_list(rollout["action_probs_traj"]),
            "step_policy_sources": list(rollout["step_policy_sources"]),
        },
        "timings_ms": {
            **runtime.solve_timing,
            "control_compute_per_step_ms": [float(v) for v in control_times],
            "step_total_per_step_ms": [float(v) for v in step_times],
            "online_solve_per_step_ms": [float(v) for v in online_times],
            "control_compute_mean_ms": float(sum(control_times) / max(len(control_times), 1)),
            "control_compute_max_ms": float(max(control_times) if control_times else 0.0),
            "step_total_mean_ms": float(sum(step_times) / max(len(step_times), 1)),
            "step_total_max_ms": float(max(step_times) if step_times else 0.0),
            "online_solve_mean_ms": float(online_summary["mean_ms"]),
            "online_solve_max_ms": float(online_summary["max_ms"]),
            "online_solve_total_ms": float(online_summary["total_ms"]),
            "num_online_solves": int(online_summary["count"]),
        },
        "costs": {
            "root_objective": root_value,
            "rollout_cost": float(rollout["rollout_cost"]),
        },
        "terminal": rollout["terminal"],
        "game": {
            "K": int(runtime.setup.game_cfg.K),
            "T": float(runtime.setup.game_cfg.T),
            "tau": float(runtime.setup.game_cfg.tau),
            "targets": _tensor_to_list(runtime.setup.game.type_targets()),
        },
        "trajectories": {
            "x_traj": _tensor_to_list(x_traj),
            "u_traj": _tensor_to_list(rollout["u_traj"]),
            "v_traj": _tensor_to_list(rollout["v_traj"]),
            "belief_traj": _tensor_to_list(rollout["belief_traj"]),
        },
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
        "checkpoint_path": str(runtime.setup.checkpoint_path),
        "game_type": runtime.setup.game_type,
        "checkpoint_step": runtime.setup.meta.get("step"),
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
            "offline_total_ms": float(runtime.solve_timing["offline_total_ms"]),
            "control_compute_ms": float(step_result.control_compute_ms),
            "online_solve_ms": float(step_result.online_solve_ms),
        },
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
    dx1 = runtime.setup.game_cfg.dx1
    p1 = x_np[:, 0:3]
    p2 = x_np[:, dx1 : dx1 + 3]
    points = np.concatenate([p1, p2], axis=0)

    fig = plt.figure(figsize=(8.0, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(p1[:, 0], p1[:, 1], p1[:, 2], color="#1f77b4", linewidth=2.2, label="P1")
    ax.plot(p2[:, 0], p2[:, 1], p2[:, 2], color="#d62728", linewidth=2.2, label="P2")
    ax.scatter(p1[0, 0], p1[0, 1], p1[0, 2], color="#1f77b4", marker="o", s=50)
    ax.scatter(p2[0, 0], p2[0, 1], p2[0, 2], color="#d62728", marker="o", s=50)
    ax.scatter(p1[-1, 0], p1[-1, 1], p1[-1, 2], color="#1f77b4", marker="^", s=60)
    ax.scatter(p2[-1, 0], p2[-1, 1], p2[-1, 2], color="#d62728", marker="^", s=60)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title or "Hexner 3D rollout")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    _set_equal_axes_3d(ax, points)

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

    x_np = rollout["x_traj"].detach().cpu().numpy()
    dx1 = runtime.setup.game_cfg.dx1
    p1 = x_np[:, 0:3]
    p2 = x_np[:, dx1 : dx1 + 3]
    belief_np = rollout["belief_traj"].detach().cpu().numpy()
    num_types = belief_np.shape[1]
    time_idx = np.arange(belief_np.shape[0], dtype=float)
    targets = runtime.setup.game.type_targets().detach().cpu().numpy()[:, :3]
    points = np.concatenate([p1, p2], axis=0)
    if targets.size > 0:
        points = np.concatenate([points, targets], axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    pad = np.maximum(0.15 * (maxs - mins), 0.2)

    type_colors = [
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    title_text = title or "Hexner 3D rollout animation"
    if true_type is not None:
        type_desc = f"type {int(true_type)}"
        if hasattr(runtime.setup.game, "theta_vals"):
            theta_vals = runtime.setup.game.theta_vals.detach().cpu().tolist()
            if 0 <= int(true_type) < len(theta_vals):
                type_desc = f"type {int(true_type)} (theta={theta_vals[int(true_type)]:.3f})"
        title_text = f"{title_text} | True {type_desc}"

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[0.66, 0.34],
        subplot_titles=("3D Trajectories", "Belief Evolution"),
    )

    fig.add_trace(
        go.Scatter3d(
            x=p1[:1, 0],
            y=p1[:1, 1],
            z=p1[:1, 2],
            mode="lines+markers",
            line=dict(color="#1f77b4", width=6),
            marker=dict(size=4, color="#1f77b4"),
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
            line=dict(color="#d62728", width=6),
            marker=dict(size=4, color="#d62728"),
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
            marker=dict(size=7, color="#1f77b4", symbol="circle"),
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
            marker=dict(size=7, color="#d62728", symbol="circle"),
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
        if true_type is not None and int(true_type) == i:
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
            name="Current step",
        ),
        row=1,
        col=2,
    )
    step_line_trace_index = len(fig.data) - 1

    frames = []
    for k in range(p1.shape[0]):
        frame_data: list[Any] = [
            go.Scatter3d(
                x=p1[: k + 1, 0],
                y=p1[: k + 1, 1],
                z=p1[: k + 1, 2],
                mode="lines+markers",
                line=dict(color="#1f77b4", width=6),
                marker=dict(size=4, color="#1f77b4"),
                name="P1",
            ),
            go.Scatter3d(
                x=p2[: k + 1, 0],
                y=p2[: k + 1, 1],
                z=p2[: k + 1, 2],
                mode="lines+markers",
                line=dict(color="#d62728", width=6),
                marker=dict(size=4, color="#d62728"),
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
        frame_traces.append(step_line_trace_index)
        frames.append(go.Frame(name=str(k), data=frame_data, traces=frame_traces))

    fig.frames = frames
    fig.update_layout(
        title=title_text,
        scene=dict(
            xaxis=dict(title="x", range=[float(mins[0] - pad[0]), float(maxs[0] + pad[0])]),
            yaxis=dict(title="y", range=[float(mins[1] - pad[1]), float(maxs[1] + pad[1])]),
            zaxis=dict(title="z", range=[float(mins[2] - pad[2]), float(maxs[2] + pad[2])]),
            aspectmode="cube",
        ),
        xaxis2=dict(title="Step", range=[float(time_idx[0]), float(time_idx[-1])]),
        yaxis2=dict(title="Belief", range=[-0.02, 1.02]),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
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
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Checkpoint-driven rollout and action-query runner for the 3D Hexner-mod amortized policy."
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
    parser.add_argument("--mode", type=str, default="rollout", choices=["rollout", "query"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--x0", type=str, default=None, help="Optional fixed initial state as CSV.")
    parser.add_argument("--prior", type=str, default=None, help="Optional fixed prior as CSV.")
    parser.add_argument("--type-index", type=int, default=None, help="Optional realized type index. If omitted, sample from the prior.")
    parser.add_argument("--prior-min", type=float, default=0.05, help="Minimum type probability when randomly sampling a belief for I=2.")
    parser.add_argument("--sample-actions", action=argparse.BooleanOptionalAction, default=True, help="Whether to sample public prototypes from alpha.")
    parser.add_argument("--action-clip", action=argparse.BooleanOptionalAction, default=False, help="Whether to enforce box action clipping during solve and rollout.")
    parser.add_argument("--one-hot-alpha", action=argparse.BooleanOptionalAction, default=False, help="Project alpha to one-hot before solving the tree.")
    parser.add_argument("--p1-pos-jitter", type=float, default=None, help="Original-game P1 position jitter if x0 is sampled.")
    parser.add_argument("--p2-pos-jitter", type=float, default=None, help="Original-game P2 position jitter if x0 is sampled.")
    parser.add_argument("--p1-vel-jitter", type=float, default=None, help="Original-game P1 velocity jitter if x0 is sampled.")
    parser.add_argument("--p2-vel-jitter", type=float, default=None, help="Original-game P2 velocity jitter if x0 is sampled.")
    parser.add_argument("--pos-jitter", type=float, default=None, help="Coupled-game shared position jitter if x0 is sampled.")
    parser.add_argument("--vel-jitter", type=float, default=None, help="Coupled-game shared velocity jitter if x0 is sampled.")
    parser.add_argument("--x-current", type=str, default=None, help="Query mode: current state as CSV. Defaults to sampled x0.")
    parser.add_argument("--belief-current", type=str, default=None, help="Query mode: current public belief as CSV. Defaults to inference from prototype history.")
    parser.add_argument("--time-step", type=int, default=None, help="Query mode: optional time-step sanity check.")
    parser.add_argument("--prototypes-so-far", type=str, default="", help="Query mode: realized prototype indices as CSV, e.g. '0,1,1'.")
    parser.add_argument("--online-solve", action=argparse.BooleanOptionalAction, default=False, help="If enabled, re-solve the remaining subtree online from the current state and belief.")
    parser.add_argument("--online-alpha-iters", type=int, default=0, help="Online solve: number of Adam steps for remaining alpha optimization.")
    parser.add_argument("--online-alpha-lr", type=float, default=5e-2, help="Online solve: Adam learning rate.")
    parser.add_argument("--online-alpha-early-stop", action=argparse.BooleanOptionalAction, default=True, help="Online solve: enable simple early stopping.")
    parser.add_argument("--online-alpha-min-iters", type=int, default=5, help="Online solve: minimum iterations before early stop.")
    parser.add_argument("--online-alpha-patience", type=int, default=3, help="Online solve: stable-step patience.")
    parser.add_argument("--online-alpha-min-delta", type=float, default=1e-5, help="Online solve: convergence delta on objective.")
    parser.add_argument("--online-skip-solve-at-t0", action=argparse.BooleanOptionalAction, default=True, help="Online solve: reuse the offline root solve at t=0.")
    parser.add_argument("--save-trajectory-png", type=Path, default=None, help="Rollout mode: optional static 3D trajectory image path.")
    parser.add_argument("--save-animation-html", type=Path, default=None, help="Rollout mode: optional interactive Plotly animation HTML path.")
    parser.add_argument("--figure-title", type=str, default=None, help="Optional title for saved trajectory/animation figures.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write results as JSON.")
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False, help="Suppress summary printing.")
    return parser.parse_args()


def _print_rollout_summary(result: Dict[str, Any]) -> None:
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
        f"max={times['control_compute_max_ms']:.6f}"
    )
    print(
        "online_solve_ms  : "
        f"num={times['num_online_solves']}, "
        f"mean={times['online_solve_mean_ms']:.6f}, "
        f"max={times['online_solve_max_ms']:.6f}, "
        f"total={times['online_solve_total_ms']:.6f}"
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


def _print_query_summary(result: Dict[str, Any]) -> None:
    q = result["query"]
    s = result["state"]
    p = result["policy"]
    t = result["timings_ms"]
    print(f"checkpoint       : {result['checkpoint_path']}")
    print(f"game_type        : {result['game_type']}")
    print(f"time_step        : {q['time_step']}")
    print(f"type_index       : {q['type_index']}")
    print(f"prototypes_so_far: {q['prototypes_so_far']}")
    print(f"policy_source    : {q['policy_source']}")
    print(f"belief_current   : {s['belief_current']}")
    print(f"action_probs     : {p['action_probs']}")
    print(f"prototype_index  : {p['prototype_index']}")
    print(f"u                : {p['u']}")
    print(f"v                : {p['v']}")
    print(
        "timings_ms       : "
        f"control={t['control_compute_ms']:.6f}, "
        f"online_solve={t['online_solve_ms']:.6f}"
    )


def _parse_state_csv(value: Optional[str], expected_dim: int, name: str) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None
    vals = _parse_csv_floats(value)
    if len(vals) != expected_dim:
        raise ValueError(f"{name} must provide {expected_dim} values, got {len(vals)}.")
    return vals


def _parse_belief_csv(value: Optional[str], I: int) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None
    vals = _parse_csv_floats(value)
    if len(vals) != I:
        raise ValueError(f"--belief-current must provide {I} values, got {len(vals)}.")
    return vals


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

    runtime = build_policy_runtime(
        setup=setup,
        x0=x0,
        p0=p0,
        use_action_clip=bool(args.action_clip),
        one_hot_alpha=bool(args.one_hot_alpha),
    )
    online_cfg = OnlineSolveConfig(
        enabled=bool(args.online_solve),
        alpha_iters=int(args.online_alpha_iters),
        alpha_lr=float(args.online_alpha_lr),
        alpha_early_stop=bool(args.online_alpha_early_stop),
        alpha_min_iters=int(args.online_alpha_min_iters),
        alpha_patience=int(args.online_alpha_patience),
        alpha_min_delta=float(args.online_alpha_min_delta),
        skip_solve_at_t0=bool(args.online_skip_solve_at_t0),
    )

    if args.mode == "rollout":
        rollout = rollout_one_game(
            runtime=runtime,
            type_index=true_type,
            sample_actions=bool(args.sample_actions),
            use_action_clip=bool(args.action_clip),
            generator=generator,
            online_solve=online_cfg,
        )
        result = build_rollout_result_payload(
            runtime=runtime,
            true_type=true_type,
            rollout=rollout,
        )

        if args.save_trajectory_png is not None:
            save_trajectory_png(
                runtime=runtime,
                rollout=rollout,
                output_path=args.save_trajectory_png,
                title=args.figure_title,
            )
            if not args.quiet:
                print(f"saved_png        : {args.save_trajectory_png}")

        if args.save_animation_html is not None:
            save_animation_html(
                runtime=runtime,
                rollout=rollout,
                output_path=args.save_animation_html,
                title=args.figure_title,
                true_type=true_type,
            )
            if not args.quiet:
                print(f"saved_html       : {args.save_animation_html}")

        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(result, indent=2))
            if not args.quiet:
                print(f"saved_json       : {args.output_json}")

        if not args.quiet:
            _print_rollout_summary(result)
        return

    x_current_vals = _parse_state_csv(args.x_current, expected_dim=setup.game.dx, name="--x-current")
    x_current = (
        torch.tensor(x_current_vals, device=setup.game.device_resolved, dtype=setup.game.dtype)
        if x_current_vals is not None
        else runtime.x0.clone()
    )
    prototypes_so_far = list(_parse_csv_ints(args.prototypes_so_far))
    belief_vals = _parse_belief_csv(args.belief_current, I=setup.game.I)
    belief_current = (
        torch.tensor(belief_vals, device=setup.game.device_resolved, dtype=setup.game.dtype)
        if belief_vals is not None
        else None
    )
    step_result = query_policy_action(
        runtime=runtime,
        type_index=true_type,
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
        belief_current = step_result.current_belief

    result = build_query_result_payload(
        runtime=runtime,
        type_index=true_type,
        x_current=x_current,
        belief_current=belief_current,
        prototypes_so_far=prototypes_so_far,
        step_result=step_result,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2))
        if not args.quiet:
            print(f"saved_json       : {args.output_json}")

    if not args.quiet:
        _print_query_summary(result)


if __name__ == "__main__":
    main()
