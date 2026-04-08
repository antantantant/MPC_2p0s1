"""Evaluate a trained checkpoint with receding-horizon (MPC-style) SQP.

At each environment step:
  1) Extract the alpha subtree for the current public node/history.
  2) Re-solve the nonlinear game from current state/belief.
  3) Apply only the first control pair (u_t, v_t).
  4) Update state and posterior belief, then repeat.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).parent.parent))  # prefer local repo imports

from src.game import build_game
from src.game.quadrotor_game import Hexner3DQuadrotorGame
from src.optimization.objective_primal_sqp import primal_objective_sqp
from src.rollout.trajectory import RolloutResult
from src.solvers.action_spaces import BoxActionSpace
from src.tree.indexing import FullIaryTreeIndexer
from src.tree.signaling import AlphaParam, AlphaParamConfig
from src.utils.config import GameConfig
from src.utils.correctness import (
    alpha_separation_stats,
    compute_config_fingerprint,
    passes_correctness_gates,
    type_path_divergence_step,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MPC-style evaluation with per-step SQP re-solves."
    )

    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="Config sidecar for bare alpha files (best_alpha.pt/final_alpha.pt).",
    )
    p.add_argument("--output-dir", type=str, default="reports/quad_eval_mpc")
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--num-rollouts-per-type", type=int, default=1)
    p.add_argument("--rollout-steps", type=int, default=0,
                   help="Number of executed MPC environment steps (<=0 uses config K).")
    p.add_argument("--local-horizon", type=int, default=0,
                   help="Local replan horizon for each MPC solve (<=0 uses remaining horizon).")
    p.add_argument("--sample-actions", action="store_true")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--animate", action="store_true")

    p.add_argument("--sqp-iters", type=int, default=10)
    p.add_argument("--sqp-step-size", type=float, default=0.1)
    p.add_argument("--riccati-reg", type=float, default=0.5)
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
        dynamics_model=str(meta.get("dynamics_model", "rigid_body")),
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
        interception_state_weights=tuple(
            meta.get(
                "interception_state_weights",
                (1.0, 1.0, 1.0, 0.25, 0.25, 0.25, 0.1, 0.1, 0.1),
            )
        ),  # type: ignore[arg-type]
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

    if isinstance(payload, dict) and "config" in payload:
        if args.config_json is not None:
            print(
                "[eval_mpc] WARNING: --config-json ignored for full checkpoint; "
                "using checkpoint-embedded config."
            )
        cfg_meta = payload["config"]
        cfg = _rebuild_game_config(cfg_meta)
        return cfg, payload, cfg_meta

    cfg_path: Optional[Path] = Path(args.config_json) if args.config_json else None
    if cfg_path is None:
        if adjacent_cfg.exists():
            cfg_path = adjacent_cfg
            print(
                f"[eval_mpc] WARNING: bare alpha checkpoint; using inferred sidecar "
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
                "[eval_mpc] WARNING: provided --config-json differs from adjacent "
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

    u_lo, u_hi, v_lo, v_hi = game.action_box_bounds(
        u_max=float(args.u_max),
        v_max=float(args.v_max),
    )

    return BoxActionSpace(u_min=u_lo, u_max=u_hi, v_min=v_lo, v_max=v_hi)


class _FixedAlpha(nn.Module):
    """Simple wrapper so primal_objective_sqp can consume a fixed alpha tensor."""

    def __init__(self, alpha: Tensor) -> None:
        super().__init__()
        self.register_buffer("_alpha", alpha)

    def forward(self) -> Tensor:
        return self._alpha


def _extract_alpha_subtree(
    *,
    alpha_full: Tensor,
    full_indexer: FullIaryTreeIndexer,
    start_depth: int,
    start_node: int,
    horizon: int = 0,
) -> Tuple[Tensor, FullIaryTreeIndexer]:
    """Extract alpha tensor for the subtree rooted at (start_depth, start_node)."""
    if not (0 <= start_depth < full_indexer.K):
        raise ValueError(
            f"start_depth must be in [0, {full_indexer.K - 1}], got {start_depth}"
        )

    I = full_indexer.I
    remaining = full_indexer.K - start_depth
    K_rem = remaining if horizon <= 0 else min(horizon, remaining)
    local_indexer = FullIaryTreeIndexer(I=I, K=K_rem)
    max_nodes = local_indexer.max_nodes_per_depth
    alpha_sub = torch.empty(
        (K_rem, max_nodes, I, I),
        dtype=alpha_full.dtype,
        device=alpha_full.device,
    )

    for d in range(K_rem):
        n_local = local_indexer.node_count(d)
        depth = start_depth + d
        scale = I ** d
        for local_node in range(n_local):
            global_node = start_node * scale + local_node
            alpha_sub[d, local_node] = alpha_full[depth, global_node]
        if n_local < max_nodes:
            alpha_sub[d, n_local:] = 1.0 / float(I)

    return alpha_sub, local_indexer


def _solve_local_tree(
    *,
    game: Hexner3DQuadrotorGame,
    indexer: FullIaryTreeIndexer,
    alpha_local: Tensor,
    x_now: Tensor,
    p_now: Tensor,
    action_space: Optional[BoxActionSpace],
    args: argparse.Namespace,
) -> Tuple[float, Dict[str, object]]:
    with torch.no_grad():
        loss, details = primal_objective_sqp(  # type: ignore
            game=game,
            alpha_module=_FixedAlpha(alpha_local),
            indexer=indexer,
            x0=x_now,
            p0=p_now,
            action_space=action_space,
            num_sqp_iters=args.sqp_iters,
            sqp_step_size=args.sqp_step_size,
            riccati_reg=args.riccati_reg,
            sqp_line_search=True,
            ls_alpha_min=args.ls_alpha_min,
            ls_backtrack=args.ls_backtrack,
            ls_max_steps=args.ls_max_steps,
            line_search_accept_worse=args.allow_worse_step,
            u_init=None,
            v_init=None,
            collect_sqp_diagnostics=True,
            return_details=True,
        )
    return float(loss.item()), details


def _update_belief(
    *,
    belief: Tensor,
    alpha_node: Tensor,
    action_index: int,
    eps: float = 1e-10,
) -> Tensor:
    """Bayes update for belief using alpha at the current public node."""
    lam = torch.einsum("i,ia->a", belief, alpha_node)
    denom = lam[action_index].clamp_min(eps)
    b_next = belief * alpha_node[:, action_index] / denom
    b_next = b_next / b_next.sum().clamp_min(eps)
    return b_next


def mpc_rollout_for_type(
    *,
    game: Hexner3DQuadrotorGame,
    full_indexer: FullIaryTreeIndexer,
    alpha_full: Tensor,
    x0: Tensor,
    p0: Tensor,
    type_index: int,
    rollout_steps: int,
    local_horizon: int,
    action_space: Optional[BoxActionSpace],
    args: argparse.Namespace,
    sample_actions: bool,
    generator: torch.Generator,
) -> Tuple[RolloutResult, List[Dict[str, object]]]:
    """Receding-horizon rollout for one realised type."""
    I = full_indexer.I
    K_eval = min(rollout_steps, full_indexer.K)

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

    step_diags: List[Dict[str, object]] = []

    for t in range(K_eval):
        node_before = node_idx
        alpha_local, local_indexer = _extract_alpha_subtree(
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
            args=args,
        )

        sqp_res = details["sqp_result"]
        alpha_eval = details["alpha"]
        alpha_row = alpha_eval[0, 0, type_index]  # (I_actions,)

        if sample_actions:
            action = int(
                torch.multinomial(alpha_row, 1, replacement=True, generator=generator).item()
            )
        else:
            action = int(torch.argmax(alpha_row).item())
        proto_indices[t] = action

        # Use first-step feedback policy from the local SQP solution.
        Ku = sqp_res.riccati_sol.K_u[0][0, action]
        Kv = sqp_res.riccati_sol.K_v[0][0, action]
        kappa_u = sqp_res.riccati_sol.kappa_u[0][0, action]
        kappa_v = sqp_res.riccati_sol.kappa_v[0][0, action]

        u = Ku @ x + kappa_u
        v = Kv @ x + kappa_v
        if action_space is not None:
            u = action_space.clip_u(u)
            v = action_space.clip_v(v)

        u_traj[t] = u
        v_traj[t] = v

        x = game.step(x, u, v)
        x_traj[t + 1] = x

        alpha_node_full = alpha_full[t, node_idx]  # (I_types, I_actions)
        belief = _update_belief(
            belief=belief,
            alpha_node=alpha_node_full,
            action_index=action,
        )
        belief_traj[t + 1] = belief
        node_idx = full_indexer.child_index(t, node_idx, action)

        sqp_diag = sqp_res.diagnostics
        if sqp_diag is not None:
            ric_retries = int(sum(1 for x_retry in sqp_diag.riccati_retry_hist if x_retry))
            keep_prev = int(sum(1 for x_keep in sqp_diag.used_prev_iterate_hist if x_keep))
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
                "local_loss": local_loss,
                "sqp_converged": sqp_converged,
                "sqp_nan": sqp_nan,
                "sqp_riccati_retries": ric_retries,
                "sqp_used_prev_iterates": keep_prev,
                "sqp_last_accepted_step": last_step,
            }
        )

    return (
        RolloutResult(
            x_traj=x_traj,
            u_traj=u_traj,
            v_traj=v_traj,
            belief_traj=belief_traj,
            proto_indices=proto_indices,
        ),
        step_diags,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg, payload, cfg_meta = load_checkpoint_and_config(args)
    if args.device is not None:
        cfg = GameConfig(
            I=cfg.I,
            T=cfg.T,
            K=cfg.K,
            dynamics_model=cfg.dynamics_model,
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
            interception_state_weights=cfg.interception_state_weights,
        )

    game = build_game(cfg)
    indexer = FullIaryTreeIndexer(I=cfg.I, K=cfg.K)

    alpha_module = AlphaParam(
        indexer=indexer,
        cfg=AlphaParamConfig(),
        dtype=cfg.dtype,
        device=cfg.device,
    )
    alpha_module.load_state_dict(payload["alpha_state_dict"])  # type: ignore[index]
    alpha_module.eval()

    with torch.no_grad():
        alpha_full = alpha_module().detach()

    action_space = build_action_space(args=args, game=game)

    x0 = game.default_initial_state()
    p0_list = [args.prior] + [(1.0 - args.prior) / max(cfg.I - 1, 1)] * (cfg.I - 1)
    p0 = torch.tensor(p0_list[:cfg.I], dtype=cfg.dtype, device=cfg.device_resolved)
    p0 = p0 / p0.sum()

    rollout_steps = cfg.K if args.rollout_steps <= 0 else min(args.rollout_steps, cfg.K)
    local_horizon = cfg.K if args.local_horizon <= 0 else min(args.local_horizon, cfg.K)

    print(f"[eval_mpc] Loaded checkpoint: {args.checkpoint}")
    print(
        f"[eval_mpc] Evaluation mode: offline-trained alpha + online MPC, "
        f"rollout_steps={rollout_steps}, local_horizon={local_horizon}, sqp_iters={args.sqp_iters}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator(device=cfg.device_resolved)
    gen.manual_seed(args.seed)

    rollouts: List[RolloutResult] = []
    rollout_step_diags: List[List[Dict[str, object]]] = []
    for type_idx in range(cfg.I):
        for _ in range(args.num_rollouts_per_type):
            ro, step_diags = mpc_rollout_for_type(
                game=game,
                full_indexer=indexer,
                alpha_full=alpha_full,
                x0=x0,
                p0=p0,
                type_index=type_idx,
                rollout_steps=rollout_steps,
                local_horizon=local_horizon,
                action_space=action_space,
                args=args,
                sample_actions=args.sample_actions,
                generator=gen,
            )
            rollouts.append(ro)
            rollout_step_diags.append(step_diags)

    targets = game.type_target_positions()
    rows: List[Dict[str, object]] = []
    term_flags: List[bool] = []
    all_finite = True
    min_alt_p1 = float("inf")
    min_alt_p2 = float("inf")

    print(f"\n{'─'*60}")
    print(f" MPC rollout summaries  ({args.num_rollouts_per_type} per type)")
    print(f"{'─'*60}")

    for i, ro in enumerate(rollouts):
        type_idx = i // args.num_rollouts_per_type
        p1_final = game.player_position(ro.x_traj[-1], 0)
        p2_final = game.player_position(ro.x_traj[-1], 1)
        dists_p1 = torch.norm(targets - p1_final.unsqueeze(0), dim=-1)
        true_d = float(dists_p1[type_idx].item())
        other_d = float(
            torch.min(torch.cat([dists_p1[:type_idx], dists_p1[type_idx + 1:]])).item()
        ) if cfg.I > 1 else float("inf")
        terminal_ok = bool(true_d < other_d)
        term_flags.append(terminal_ok)

        finite = bool(
            torch.isfinite(ro.x_traj).all()
            and torch.isfinite(ro.u_traj).all()
            and torch.isfinite(ro.v_traj).all()
            and torch.isfinite(ro.belief_traj).all()
        )
        all_finite = all_finite and finite
        p1_hist = game.player_position(ro.x_traj, 0)
        p2_hist = game.player_position(ro.x_traj, 1)
        min_alt_p1 = min(min_alt_p1, float(p1_hist[:, 2].min().item()))
        min_alt_p2 = min(min_alt_p2, float(p2_hist[:, 2].min().item()))

        u_effort = float(torch.sum(ro.u_traj ** 2).item())
        v_effort = float(torch.sum(ro.v_traj ** 2).item())
        nan_steps = int(sum(1 for d in rollout_step_diags[i] if d.get("sqp_nan", False)))

        rows.append(
            {
                "rollout_idx": i,
                "type_index": type_idx,
                "theta": cfg.theta_values[type_idx],
                "p1_final_pos": [float(v) for v in p1_final.tolist()],
                "p2_final_pos": [float(v) for v in p2_final.tolist()],
                "dist_true_target_p1": true_d,
                "dist_best_other_target_p1": other_d,
                "terminal_target_consistent": terminal_ok,
                "p1_control_effort": u_effort,
                "p2_control_effort": v_effort,
                "final_belief": [float(v) for v in ro.belief_traj[-1].tolist()],
                "proto_indices": [int(v) for v in ro.proto_indices.tolist()],
                "finite": finite,
                "sqp_nan_steps": nan_steps,
            }
        )
        print(
            f"  type={type_idx} (θ={cfg.theta_values[type_idx]:+.1f})  "
            f"P1 final={[f'{x:.3f}' for x in p1_final.tolist()]}  "
            f"P2 final={[f'{x:.3f}' for x in p2_final.tolist()]}  "
            f"terminal_ok={terminal_ok}  sqp_nan_steps={nan_steps}"
        )

    term_rate = float(sum(term_flags) / len(term_flags)) if term_flags else 0.0
    physical_plausible = bool(
        all_finite and (min_alt_p1 >= args.altitude_floor) and (min_alt_p2 >= args.altitude_floor)
    )
    sep = alpha_separation_stats(alpha=alpha_full, indexer=indexer)
    div_step = type_path_divergence_step(alpha=alpha_full, indexer=indexer)

    correctness_metrics = {
        "alpha": {
            **sep,
            "type_path_divergence_step": div_step,
        },
        "rollouts": rows,
        "terminal_consistency_rate": term_rate,
        "all_finite": all_finite,
        "min_altitude_p1": min_alt_p1,
        "min_altitude_p2": min_alt_p2,
        "altitude_floor": args.altitude_floor,
        "physical_plausible": physical_plausible,
    }
    passed, gates = passes_correctness_gates(
        correctness_metrics,
        min_separation_fraction=args.gate_min_separation,
        min_terminal_consistency_rate=args.gate_min_terminal_consistency,
    )

    print(
        "[eval_mpc] Correctness: "
        f"separation={correctness_metrics['alpha']['separation_fraction']:.3f}, "
        f"path_div_step={correctness_metrics['alpha']['type_path_divergence_step']}, "
        f"terminal_consistency={correctness_metrics['terminal_consistency_rate']:.3f}, "
        f"physical_plausible={correctness_metrics['physical_plausible']}, "
        f"passed={passed}"
    )

    summary = {
        "evaluation_mode": "offline_trained_alpha_with_online_mpc",
        "checkpoint": args.checkpoint,
        "solve_mode": "mpc_receding_horizon",
        "rollout_steps": rollout_steps,
        "local_horizon": local_horizon,
        "sqp_settings": {
            "sqp_iters": args.sqp_iters,
            "sqp_step_size": args.sqp_step_size,
            "riccati_reg": args.riccati_reg,
            "ls_alpha_min": args.ls_alpha_min,
            "ls_backtrack": args.ls_backtrack,
            "ls_max_steps": args.ls_max_steps,
            "allow_worse_step": args.allow_worse_step,
        },
        "correctness": {
            "metrics": correctness_metrics,
            "gates": gates,
            "passed": passed,
        },
        "rollouts": rows,
        "mpc_step_diagnostics": rollout_step_diags,
        "config_meta": cfg_meta,
        "config_fingerprint": compute_config_fingerprint(cfg_meta),
    }
    with open(output_dir / "rollout_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[eval_mpc] Rollout summary saved to {output_dir / 'rollout_summary.json'}")

    with open(output_dir / "alpha_info.json", "w") as f:
        json.dump(
            {
                "alpha_shape": list(alpha_full.shape),
                "alpha_min": float(alpha_full.min().item()),
                "alpha_max": float(alpha_full.max().item()),
            },
            f,
            indent=2,
        )

    if args.visualize or args.animate:
        try:
            from src.utils.visualization import animate_rollout, plot_trajectories

            for i, ro in enumerate(rollouts):
                type_idx = i // args.num_rollouts_per_type
                run_idx = i % args.num_rollouts_per_type
                title = f"MPC type {type_idx} (θ={cfg.theta_values[type_idx]:+.1f})"
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
            print("[eval_mpc] matplotlib not available — skipping visualisation.")

    print(f"\n[eval_mpc] All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
