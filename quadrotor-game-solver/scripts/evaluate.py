# scripts/evaluate.py
"""Evaluate a trained checkpoint on the quadrotor signaling game.

Canonical evaluation mode is cold-start SQP from hover controls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

sys.path.append(str(Path(__file__).parent.parent))  # for absolute imports

from src.game.quadrotor_game import Hexner3DQuadrotorGame
from src.optimization.objective_primal_sqp import primal_objective_sqp
from src.rollout.trajectory import RolloutResult, rollout_trajectory
from src.solvers.action_spaces import BoxActionSpace
from src.tree.indexing import FullIaryTreeIndexer
from src.tree.signaling import AlphaParam, AlphaParamConfig
from src.utils.config import GameConfig
from src.utils.correctness import (
    compute_config_fingerprint,
    passes_correctness_gates,
    rollout_correctness_metrics,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate a trained checkpoint on the 3-D Hexner quadrotor game."
        )
    )

    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="Config sidecar for bare alpha files (best_alpha.pt/final_alpha.pt).",
    )
    p.add_argument("--output-dir", type=str, default="reports/quad_eval")
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--num-rollouts-per-type", type=int, default=1)
    p.add_argument("--sample-actions", action="store_true")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--animate", action="store_true")

    p.add_argument("--warm-start", action="store_true",
                   help="Run an additional warm-start SQP solve for comparison.")
    p.add_argument("--cold-start", dest="warm_start", action="store_false",
                   help="Use canonical cold-start SQP solve (default).")
    p.set_defaults(warm_start=False)

    p.add_argument("--sqp-iters", type=int, default=3)
    p.add_argument("--sqp-step-size", type=float, default=1.0)
    p.add_argument("--riccati-reg", type=float, default=1e-3)
    p.add_argument("--ls-alpha-min", type=float, default=0.01)
    p.add_argument("--ls-backtrack", type=float, default=0.5)
    p.add_argument("--ls-max-steps", type=int, default=8)
    p.add_argument("--allow-worse-step", action="store_true")

    p.add_argument("--u-max", type=float, default=20.0)
    p.add_argument("--v-max", type=float, default=20.0)
    p.add_argument("--no-action-clamp", action="store_true")
    p.add_argument("--prior", type=float, default=0.5)

    p.add_argument("--gate-min-separation", type=float, default=0.05)
    p.add_argument("--gate-min-terminal-consistency", type=float, default=0.5)
    p.add_argument("--altitude-floor", type=float, default=-5.0)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _rebuild_game_config(meta: Dict[str, object]) -> GameConfig:
    dtype_str = str(meta.get("dtype", "torch.float64"))
    dtype = torch.float64 if "64" in dtype_str else torch.float32

    return GameConfig(
        I=int(meta.get("I", 2)),
        T=float(meta.get("T", 2.0)),
        K=int(meta.get("K", 5)),
        integrator=str(meta.get("integrator", "euler")),
        control_cost_mode=str(meta.get("control_cost_mode", "hover_relative")),
        line_search_accept_worse=bool(meta.get("line_search_accept_worse", False)),
        device=str(meta.get("device", "cpu")),
        dtype=dtype,
        R1_diag=tuple(meta.get("R1_diag", (0.05, 0.025, 0.025, 0.01))),  # type: ignore[arg-type]
        R2_diag=tuple(meta.get("R2_diag", (0.05, 0.10, 0.10, 0.02))),    # type: ignore[arg-type]
        K1_scale=float(meta.get("K1_scale", 1.0)),
        K2_scale=float(meta.get("K2_scale", 1.0)),
        theta_values=tuple(meta.get("theta_values", (-1.0, 1.0))),       # type: ignore[arg-type]
    )


def _load_json(path: Path) -> Dict[str, object]:
    with open(path, "r") as f:
        return json.load(f)


def load_checkpoint_and_config(
    args: argparse.Namespace,
) -> Tuple[GameConfig, Dict[str, object], Dict[str, object]]:
    """Return (game_config, payload_with_alpha_state, config_meta)."""
    ckpt_path = Path(args.checkpoint)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    adjacent_cfg = ckpt_path.parent / "config.json"

    # Full checkpoint: checkpoint config is canonical.
    if isinstance(payload, dict) and "config" in payload:
        if args.config_json is not None:
            print(
                "[eval] WARNING: --config-json ignored for full checkpoint; "
                "using checkpoint-embedded config."
            )
        cfg_meta = payload["config"]
        cfg = _rebuild_game_config(cfg_meta)
        return cfg, payload, cfg_meta

    # Bare alpha file: require sidecar (explicit or adjacent).
    cfg_path: Optional[Path] = Path(args.config_json) if args.config_json else None
    if cfg_path is None:
        if adjacent_cfg.exists():
            cfg_path = adjacent_cfg
            print(
                f"[eval] WARNING: bare alpha checkpoint; using inferred sidecar "
                f"config at {cfg_path}."
            )
        else:
            raise FileNotFoundError(
                "Bare alpha checkpoint requires config sidecar. "
                "Pass --config-json or place config.json next to checkpoint."
            )

    cfg_meta = _load_json(cfg_path)
    if adjacent_cfg.exists() and cfg_path != adjacent_cfg:
        adjacent_meta = _load_json(adjacent_cfg)
        f1 = compute_config_fingerprint(cfg_meta)
        f2 = compute_config_fingerprint(adjacent_meta)
        if f1 != f2:
            print(
                "[eval] WARNING: provided --config-json differs from adjacent "
                "checkpoint config.json."
            )

    cfg = _rebuild_game_config(cfg_meta)
    return cfg, {"alpha_state_dict": payload}, cfg_meta


def build_action_space(
    *,
    args: argparse.Namespace,
    game: Hexner3DQuadrotorGame,
) -> Optional[BoxActionSpace]:
    if args.no_action_clamp:
        return None

    dtype, device = game.dtype, game.device

    u_lo = torch.full((game.du,), -args.u_max, dtype=dtype, device=device)
    u_hi = torch.full((game.du,), args.u_max, dtype=dtype, device=device)
    v_lo = torch.full((game.dv,), -args.v_max, dtype=dtype, device=device)
    v_hi = torch.full((game.dv,), args.v_max, dtype=dtype, device=device)

    if game.control_cost_mode == "hover_relative":
        u_bias, v_bias = game.control_bias()
        u_lo[0] = -float(u_bias[0].item())
        u_hi[0] = args.u_max - float(u_bias[0].item())
        v_lo[0] = -float(v_bias[0].item())
        v_hi[0] = args.v_max - float(v_bias[0].item())
    else:
        u_lo[0] = 0.0
        v_lo[0] = 0.0

    return BoxActionSpace(u_min=u_lo, u_max=u_hi, v_min=v_lo, v_max=v_hi)


def _solve(
    *,
    game: Hexner3DQuadrotorGame,
    indexer: FullIaryTreeIndexer,
    alpha_module: AlphaParam,
    x0: Tensor,
    p0: Tensor,
    action_space: Optional[BoxActionSpace],
    args: argparse.Namespace,
    u_init: Optional[List[Tensor]] = None,
    v_init: Optional[List[Tensor]] = None,
) -> Tuple[float, Dict[str, object]]:
    with torch.no_grad():
        loss, details = primal_objective_sqp(  # type: ignore
            game=game,
            alpha_module=alpha_module,
            indexer=indexer,
            x0=x0,
            p0=p0,
            action_space=action_space,
            num_sqp_iters=args.sqp_iters,
            sqp_step_size=args.sqp_step_size,
            riccati_reg=args.riccati_reg,
            sqp_line_search=True,
            ls_alpha_min=args.ls_alpha_min,
            ls_backtrack=args.ls_backtrack,
            ls_max_steps=args.ls_max_steps,
            line_search_accept_worse=args.allow_worse_step,
            u_init=u_init,
            v_init=v_init,
            collect_sqp_diagnostics=True,
            return_details=True,
        )
    return float(loss.item()), details


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg, payload, cfg_meta = load_checkpoint_and_config(args)
    if args.device is not None:
        cfg = GameConfig(
            I=cfg.I,
            T=cfg.T,
            K=cfg.K,
            integrator=cfg.integrator,
            control_cost_mode=cfg.control_cost_mode,
            line_search_accept_worse=cfg.line_search_accept_worse,
            device=args.device,
            dtype=cfg.dtype,
            R1_diag=cfg.R1_diag,
            R2_diag=cfg.R2_diag,
            K1_scale=cfg.K1_scale,
            K2_scale=cfg.K2_scale,
            theta_values=cfg.theta_values,
        )

    game = Hexner3DQuadrotorGame(cfg)
    indexer = FullIaryTreeIndexer(I=cfg.I, K=cfg.K)

    alpha_module = AlphaParam(
        indexer=indexer,
        cfg=AlphaParamConfig(),
        dtype=cfg.dtype,
        device=cfg.device,
    )
    alpha_module.load_state_dict(payload["alpha_state_dict"])  # type: ignore[index]
    alpha_module.eval()

    action_space = build_action_space(args=args, game=game)

    x0 = game.default_initial_state()
    p0_list = [args.prior] + [(1.0 - args.prior) / max(cfg.I - 1, 1)] * (cfg.I - 1)
    p0 = torch.tensor(p0_list[:cfg.I], dtype=cfg.dtype, device=cfg.device_resolved)
    p0 = p0 / p0.sum()

    cold_loss, cold_details = _solve(
        game=game,
        indexer=indexer,
        alpha_module=alpha_module,
        x0=x0,
        p0=p0,
        action_space=action_space,
        args=args,
        u_init=None,
        v_init=None,
    )

    solve_mode = "cold_start"
    loss_val = cold_loss
    details = cold_details
    warm_loss: Optional[float] = None
    if args.warm_start:
        cold_sqp = cold_details["sqp_result"]
        warm_loss, warm_details = _solve(
            game=game,
            indexer=indexer,
            alpha_module=alpha_module,
            x0=x0,
            p0=p0,
            action_space=action_space,
            args=args,
            u_init=[u.detach() for u in cold_sqp.u_edges],
            v_init=[v.detach() for v in cold_sqp.v_edges],
        )
        solve_mode = "warm_start"
        loss_val = warm_loss
        details = warm_details

    alpha = details["alpha"]
    sqp_res = details["sqp_result"]

    correctness = rollout_correctness_metrics(
        game=game,
        indexer=indexer,
        belief_tree=sqp_res.belief_tree,
        riccati_sol=sqp_res.riccati_sol,
        alpha=alpha,
        x0=x0,
        action_space=action_space,
        altitude_floor=args.altitude_floor,
    )
    passed, gates = passes_correctness_gates(
        correctness,
        min_separation_fraction=args.gate_min_separation,
        min_terminal_consistency_rate=args.gate_min_terminal_consistency,
    )

    print(f"[eval] Loaded checkpoint: {args.checkpoint}")
    print(f"[eval] Solve mode: {solve_mode}")
    print(f"[eval] Cold-start loss: {cold_loss:.6f}")
    if warm_loss is not None:
        print(f"[eval] Warm-start loss: {warm_loss:.6f}")
    print(
        "[eval] Correctness: "
        f"separation={correctness['alpha']['separation_fraction']:.3f}, "
        f"path_div_step={correctness['alpha']['type_path_divergence_step']}, "
        f"terminal_consistency={correctness['terminal_consistency_rate']:.3f}, "
        f"physical_plausible={correctness['physical_plausible']}, "
        f"passed={passed}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator(device=cfg.device_resolved)
    gen.manual_seed(args.seed)
    rollouts: List[RolloutResult] = []
    for type_idx in range(cfg.I):
        for _ in range(args.num_rollouts_per_type):
            ro = rollout_trajectory(
                game=game,
                indexer=indexer,
                belief_tree=sqp_res.belief_tree,
                riccati_sol=sqp_res.riccati_sol,
                alpha=alpha,
                x0=x0,
                type_index=type_idx,
                action_space=action_space,
                sample_actions=args.sample_actions,
                generator=gen,
            )
            rollouts.append(ro)

    rows: List[Dict[str, object]] = []
    print(f"\n{'─'*60}")
    print(f" Rollout summaries  ({args.num_rollouts_per_type} per type)")
    print(f"{'─'*60}")
    for i, ro in enumerate(rollouts):
        type_idx = i // args.num_rollouts_per_type
        p1_final = ro.x_traj[-1, :3].tolist()
        p2_final = ro.x_traj[-1, 12:15].tolist()
        u_effort = float(torch.sum(ro.u_traj ** 2).item())
        v_effort = float(torch.sum(ro.v_traj ** 2).item())
        rows.append(
            {
                "rollout_idx": i,
                "type_index": type_idx,
                "theta": cfg.theta_values[type_idx],
                "p1_final_pos": p1_final,
                "p2_final_pos": p2_final,
                "p1_control_effort": u_effort,
                "p2_control_effort": v_effort,
                "final_belief": ro.belief_traj[-1].tolist(),
            }
        )
        print(
            f"  type={type_idx} (θ={cfg.theta_values[type_idx]:+.1f})  "
            f"P1 final={[f'{x:.3f}' for x in p1_final]}  "
            f"P2 final={[f'{x:.3f}' for x in p2_final]}  "
            f"||u||²={u_effort:.3f}  ||v||²={v_effort:.3f}"
        )

    summary = {
        "checkpoint": args.checkpoint,
        "solve_mode": solve_mode,
        "cold_start_loss": cold_loss,
        "warm_start_loss": warm_loss,
        "selected_loss": loss_val,
        "correctness": {
            "metrics": correctness,
            "gates": gates,
            "passed": passed,
        },
        "rollouts": rows,
        "config_meta": cfg_meta,
        "config_fingerprint": compute_config_fingerprint(cfg_meta),
    }
    with open(output_dir / "rollout_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[eval] Rollout summary saved to {output_dir / 'rollout_summary.json'}")

    with open(output_dir / "alpha_info.json", "w") as f:
        json.dump(
            {
                "alpha_shape": list(alpha.shape),
                "alpha_min": float(alpha.min().item()),
                "alpha_max": float(alpha.max().item()),
            },
            f,
            indent=2,
        )

    if args.visualize or args.animate:
        try:
            from src.utils.visualization import animate_rollout, plot_trajectories

            for i, ro in enumerate(rollouts):
                type_idx = i // args.num_rollouts_per_type
                title = f"Type {type_idx} (θ={cfg.theta_values[type_idx]:+.1f})"
                run_idx = i % args.num_rollouts_per_type
                if args.visualize:
                    fig = plot_trajectories(
                        ro.x_traj,
                        belief_traj=ro.belief_traj,
                        target_positions=game.type_target_positions(),
                        title=title,
                        save_path=str(output_dir / f"traj_type{type_idx}_run{run_idx}.png"),
                    )
                    fig.clear()
                if args.animate:
                    animate_rollout(
                        ro.x_traj,
                        dt=cfg.tau,
                        belief_traj=ro.belief_traj,
                        target_positions=game.type_target_positions(),
                        title=title,
                        save_path=str(output_dir / f"anim_type{type_idx}_run{run_idx}.mp4"),
                    )
        except ImportError:
            print("[eval] matplotlib not available — skipping visualisation.")

    print(f"\n[eval] All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
