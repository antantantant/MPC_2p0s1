# scripts/train.py
"""Train α for the 3-D Hexner game with nonlinear quadrotor dynamics (SQP).

Usage
-----
    cd quadrotor-game-solver
    python scripts/train.py                              # defaults
    python scripts/train.py --K 8 --sqp-iters 5 --lr 1e-3

Follows the same pattern as ``scripts/train_hexner_primal.py`` in the parent
LQ project, but uses the SQP inner layer for nonlinear dynamics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

sys.path.append(str(Path(__file__).parent.parent))  # for absolute imports

# ── project imports (absolute, works because pyproject.toml sets pythonpath=["."])
from src.utils.config import GameConfig
from src.game.quadrotor_game import Hexner3DQuadrotorGame
from src.tree.indexing import FullIaryTreeIndexer
from src.tree.signaling import AlphaParam, AlphaParamConfig
from src.solvers.action_spaces import BoxActionSpace
from src.optimization.objective_primal_sqp import primal_objective_sqp
from src.rollout.trajectory import rollout_trajectory, RolloutResult
from src.utils.correctness import (
    compute_config_fingerprint,
    rollout_correctness_metrics,
    passes_correctness_gates,
)


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train the primal SQP solver on the 3-D Hexner quadrotor game. "
            "Uses tree-structured SQP + α-optimisation."
        ),
    )

    # ── Game / tree ─────────────────────────────────────────────────
    p.add_argument("--I", type=int, default=2,
                   help="Number of payoff types (I).")
    p.add_argument("--T", type=float, default=1.0,
                   help="Time horizon T (s).")
    p.add_argument("--K", type=int, default=10,
                   help="Number of discrete steps (depth K).")
    p.add_argument("--integrator", type=str, default="rk4",
                   choices=["euler", "rk4"],
                   help="Integration scheme. rk4 recommended for tau >= 0.1.")
    p.add_argument("--linearized", action="store_true",
                   help="Use linearized dynamics (frozen at hover) for debugging.")
    p.add_argument(
        "--control-cost-mode",
        type=str,
        default="hover_relative",
        choices=["hover_relative", "absolute"],
        help="Control parameterization/cost convention.",
    )
    p.add_argument("--dtype", type=str, default="float64",
                   choices=["float32", "float64"])
    p.add_argument("--device", type=str, default="cpu",
                   help="Device string, e.g. 'cpu', 'cuda:0'.")

    # ── Cost scales ─────────────────────────────────────────────────
    p.add_argument("--theta-values", type=str, default="-1.0,1.0",
                   help="Comma-separated payoff type scalars θ_i.")
    p.add_argument("--R1-diag", type=str, default="0.05,0.025,0.025,0.01",
                   help="Comma-separated running-cost diag for P1.")
    p.add_argument("--R2-diag", type=str, default="0.05,0.10,0.10,0.02",
                   help="Comma-separated running-cost diag for P2.")
    p.add_argument("--K1-scale", type=float, default=10.0,
                   help="Scale for P1 terminal-cost matrix.")
    p.add_argument("--K2-scale", type=float, default=10.0,
                   help="Scale for P2 terminal-cost matrix.")

    # ── SQP ─────────────────────────────────────────────────────────
    # Conservative defaults for nonlinear convergence (tested: converges in ~129 iters)
    p.add_argument("--sqp-iters", type=int, default=50)
    p.add_argument("--sqp-step-size", type=float, default=0.05)
    p.add_argument("--riccati-reg", type=float, default=0.5)
    p.add_argument("--sqp-verbose", action="store_true")
    p.add_argument("--sqp-early-stop", action="store_true")
    p.add_argument("--no-line-search", action="store_true",
                   help="Disable backtracking line search in SQP.")
    p.add_argument("--ls-alpha-min", type=float, default=0.01)
    p.add_argument("--ls-backtrack", type=float, default=0.5)
    p.add_argument("--ls-max-steps", type=int, default=8)
    p.add_argument("--allow-worse-step", action="store_true",
                   help="Allow line search to accept a worse iterate.")

    # ── Optimiser ───────────────────────────────────────────────────
    p.add_argument("--lr", type=float, default=5e-3,
                   help="Learning rate for Adam.")
    p.add_argument("--epochs", type=int, default=100,
                   help="Number of outer-loop iterations.")
    p.add_argument("--alpha-init-scale", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max gradient norm for α logits (0 = no clipping).")

    # ── Action bounds ───────────────────────────────────────────────
    p.add_argument("--no-action-clamp", action="store_true",
                   help="Disable action-space clamping.")
    p.add_argument("--u-max", type=float, default=20.0)
    p.add_argument("--v-max", type=float, default=20.0)

    # ── Prior ───────────────────────────────────────────────────────
    p.add_argument("--prior", type=float, default=0.5,
                   help="Prior probability for type 0 (uniform when 0.5 & I=2).")

    # ── Logging / checkpointing ─────────────────────────────────────
    p.add_argument("--run-dir", type=str, default='../runs/quadrotor_game',
                   help="Directory for checkpoints and logs.")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--eval-cold-start-every", type=int, default=1)
    p.add_argument(
        "--save-best-by",
        type=str,
        default="cold_start_loss",
        choices=["saddle_rank", "cold_start_loss", "pre_step_loss"],
        help=(
            "Deprecated and ignored. Checkpoint selection is now always 'last "
            "checkpoint' (final model state). Kept for backward compatibility."
        ),
    )
    p.add_argument("--gate-min-separation", type=float, default=0.05)
    p.add_argument("--gate-min-terminal-consistency", type=float, default=0.5)
    p.add_argument("--altitude-floor", type=float, default=-5.0)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
#  Config helpers
# ═════════════════════════════════════════════════════════════════════════════

def build_game_config(args: argparse.Namespace) -> GameConfig:
    """Build a ``GameConfig`` from parsed CLI args."""

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    theta_vals = tuple(float(v) for v in args.theta_values.split(","))
    r1 = tuple(float(v) for v in args.R1_diag.split(","))
    r2 = tuple(float(v) for v in args.R2_diag.split(","))

    return GameConfig(
        I=args.I,
        T=args.T,
        K=args.K,
        integrator=args.integrator,
        linearized_mode=args.linearized,
        control_cost_mode=args.control_cost_mode,
        line_search_accept_worse=args.allow_worse_step,
        device=args.device,
        dtype=dtype,
        R1_diag=r1,
        R2_diag=r2,
        K1_scale=args.K1_scale,
        K2_scale=args.K2_scale,
        theta_values=theta_vals,
    )


def build_action_space(
    args: argparse.Namespace,
    game: Hexner3DQuadrotorGame,
) -> Optional[BoxActionSpace]:
    """Build the ``BoxActionSpace`` from CLI args (or None if clamping disabled)."""
    if args.no_action_clamp:
        return None

    dtype, device = game.dtype, game.device

    u_lo = torch.full((game.du,), -args.u_max, dtype=dtype, device=device)
    u_hi = torch.full((game.du,),  args.u_max, dtype=dtype, device=device)

    v_lo = torch.full((game.dv,), -args.v_max, dtype=dtype, device=device)
    v_hi = torch.full((game.dv,),  args.v_max, dtype=dtype, device=device)

    if game.control_cost_mode == "hover_relative":
        # Optimization controls are deltas around hover thrust.
        u_bias, v_bias = game.control_bias()
        u_lo[0] = -float(u_bias[0].item())
        u_hi[0] = args.u_max - float(u_bias[0].item())
        v_lo[0] = -float(v_bias[0].item())
        v_hi[0] = args.v_max - float(v_bias[0].item())
    else:
        # Absolute physical controls.
        u_lo[0] = 0.0
        v_lo[0] = 0.0

    return BoxActionSpace(u_min=u_lo, u_max=u_hi, v_min=v_lo, v_max=v_hi)


def build_prior(args: argparse.Namespace, cfg: GameConfig) -> torch.Tensor:
    """Construct the prior distribution p0 from CLI args."""
    I = cfg.I
    # p0[0] = args.prior, remaining split uniformly
    p0_list = [args.prior] + [(1.0 - args.prior) / max(I - 1, 1)] * (I - 1)
    p0 = torch.tensor(p0_list[:I], dtype=cfg.dtype, device=cfg.device_resolved)
    return p0 / p0.sum()


def run_cold_start_eval(
    *,
    game: Hexner3DQuadrotorGame,
    indexer: FullIaryTreeIndexer,
    alpha_module: AlphaParam,
    action_space: Optional[BoxActionSpace],
    x0: torch.Tensor,
    p0: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[float, Dict[str, object]]:
    """Evaluate current alpha by re-solving SQP from a cold hover start."""
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
            sqp_verbose=args.sqp_verbose,
            sqp_early_stop=args.sqp_early_stop,
            sqp_line_search=not args.no_line_search,
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


# ═════════════════════════════════════════════════════════════════════════════
#  Training loop
# ═════════════════════════════════════════════════════════════════════════════

def run_training(
    *,
    game: Hexner3DQuadrotorGame,
    indexer: FullIaryTreeIndexer,
    alpha_module: AlphaParam,
    action_space: Optional[BoxActionSpace],
    x0: torch.Tensor,
    p0: torch.Tensor,
    args: argparse.Namespace,
    run_dir: Path,
    config_snapshot: Dict[str, object],
    config_fingerprint: str,
) -> None:
    """Execute the outer-loop Adam optimisation over α logits."""

    optimizer = torch.optim.Adam(alpha_module.parameters(), lr=args.lr)
    log_path = run_dir / "train.jsonl"

    print(f"\n{'='*60}")
    print(f" Starting training: {args.epochs} epochs, lr={args.lr}")
    print(f" SQP: {args.sqp_iters} iters, step_size={args.sqp_step_size}")
    print(f"{'='*60}\n")

    # Warm-start: store previous solution to initialize next epoch
    u_prev = None
    v_prev = None
    latest_metrics: Dict[str, object] = {}
    final_loss_val = float("nan")

    for epoch in range(args.epochs):
        t0 = time.time()
        optimizer.zero_grad()

        result = primal_objective_sqp(
            game=game,
            alpha_module=alpha_module,
            indexer=indexer,
            x0=x0,
            p0=p0,
            action_space=action_space,
            num_sqp_iters=args.sqp_iters,
            sqp_step_size=args.sqp_step_size,
            riccati_reg=args.riccati_reg,
            sqp_verbose=args.sqp_verbose,
            sqp_early_stop=args.sqp_early_stop,
            sqp_line_search=not args.no_line_search,
            ls_alpha_min=args.ls_alpha_min,
            ls_backtrack=args.ls_backtrack,
            ls_max_steps=args.ls_max_steps,
            line_search_accept_worse=args.allow_worse_step,
            u_init=u_prev,
            v_init=v_prev,
            collect_sqp_diagnostics=True,
            return_details=True,
        )

        loss, details = result  # type: ignore
        pre_step_loss = float(loss.item())
        final_loss_val = pre_step_loss

        # Backward
        loss.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(alpha_module.parameters(), args.grad_clip)

        # Gradient norm (after clipping)
        grad_norm = 0.0
        for param in alpha_module.parameters():
            if param.grad is not None:
                grad_norm += float(param.grad.data.norm().item()) ** 2
        grad_norm = grad_norm ** 0.5

        optimizer.step()
        dt_epoch = time.time() - t0

        # ── Store solution for warm-start in next epoch ──────────────
        # Detach to avoid keeping gradient graph across epochs
        sqp_result = details["sqp_result"]
        u_prev = [u.detach() for u in sqp_result.u_edges]
        v_prev = [v.detach() for v in sqp_result.v_edges]

        post_step_cold_start_loss: Optional[float] = None
        post_step_metrics: Dict[str, object] = {}
        gate_status: Dict[str, bool] = {}
        passed_gates = False

        if args.eval_cold_start_every > 0 and (epoch % args.eval_cold_start_every == 0):
            post_step_cold_start_loss, cold_details = run_cold_start_eval(
                game=game,
                indexer=indexer,
                alpha_module=alpha_module,
                action_space=action_space,
                x0=x0,
                p0=p0,
                args=args,
            )
            cold_sqp = cold_details["sqp_result"]
            cold_alpha = cold_details["alpha"]

            post_step_metrics = rollout_correctness_metrics(
                game=game,
                indexer=indexer,
                belief_tree=cold_sqp.belief_tree,
                riccati_sol=cold_sqp.riccati_sol,
                alpha=cold_alpha,
                x0=x0,
                action_space=action_space,
                altitude_floor=args.altitude_floor,
            )
            passed_gates, gate_status = passes_correctness_gates(
                post_step_metrics,
                min_separation_fraction=args.gate_min_separation,
                min_terminal_consistency_rate=args.gate_min_terminal_consistency,
            )

            latest_metrics = {
                "post_step_cold_start_loss": post_step_cold_start_loss,
                "correctness_metrics": post_step_metrics,
                "correctness_gates": gate_status,
                "passed_correctness_gates": passed_gates,
            }

        # ── Logging ──────────────────────────────────────────────────
        if epoch % args.log_every == 0:
            alpha_val = details["alpha"]
            alpha_range = (
                float(alpha_val.min().item()),
                float(alpha_val.max().item()),
            )

            sqp_diag = details["sqp_result"].diagnostics
            sqp_info: Dict[str, Any] = {}
            if sqp_diag is not None:
                sqp_info = {
                    "sqp_cost_hist": sqp_diag.cost_hist,
                    "sqp_converged": sqp_diag.converged,
                    "sqp_nan": sqp_diag.nan_or_inf_encountered,
                    "sqp_accepted_step_hist": sqp_diag.accepted_step_hist,
                    "sqp_used_prev_iterate_hist": sqp_diag.used_prev_iterate_hist,
                    "sqp_riccati_retry_hist": sqp_diag.riccati_retry_hist,
                }

            alpha_stats = post_step_metrics.get("alpha", {})

            log_entry = {
                "epoch": epoch,
                "loss": pre_step_loss,
                "pre_step_loss": pre_step_loss,
                "post_step_cold_start_loss": post_step_cold_start_loss,
                "grad_norm": grad_norm,
                "alpha_range": alpha_range,
                "time_s": dt_epoch,
                "checkpoint_selection_policy": "last_checkpoint",
                "passed_correctness_gates": passed_gates,
                "correctness_gates": gate_status,
                "type_path_divergence_step": alpha_stats.get("type_path_divergence_step"),
                "separation_fraction": alpha_stats.get("separation_fraction"),
                "terminal_consistency_rate": post_step_metrics.get("terminal_consistency_rate"),
                "physical_plausible": post_step_metrics.get("physical_plausible"),
                **sqp_info,
            }

            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            conv_str = ""
            if sqp_diag is not None and sqp_diag.converged:
                conv_str = " [SQP converged]"
            nan_str = ""
            if sqp_diag is not None and sqp_diag.nan_or_inf_encountered:
                nan_str = " [NaN!]"
            gate_str = ""
            if post_step_cold_start_loss is not None:
                gate_str = f" post={post_step_cold_start_loss:+.6f} gates={passed_gates}"

            print(
                f"[{epoch:4d}/{args.epochs}]  pre={pre_step_loss:+.6f}{gate_str}  "
                f"|∇|={grad_norm:.3e}  α∈{alpha_range}  "
                f"time_per_epoch={dt_epoch:.2f}s{conv_str}{nan_str}"
            )

        # ── Checkpointing ────────────────────────────────────────────
        if (epoch + 1) % args.save_every == 0:
            ckpt = {
                "epoch": epoch,
                "alpha_state_dict": alpha_module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": pre_step_loss,
                "pre_step_loss": pre_step_loss,
                "post_step_cold_start_loss": post_step_cold_start_loss,
                "config": config_snapshot,
                "config_fingerprint": config_fingerprint,
                "correctness_metrics": post_step_metrics,
                "correctness_gates": gate_status,
            }
            torch.save(ckpt, run_dir / f"checkpoint_{epoch:04d}.pt")

    # ── Final save ───────────────────────────────────────────────────
    torch.save(alpha_module.state_dict(), run_dir / "final_alpha.pt")
    final_checkpoint = {
        "epoch": args.epochs - 1,
        "alpha_state_dict": alpha_module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": final_loss_val,
        "pre_step_loss": final_loss_val,
        "checkpoint_selection_policy": "last_checkpoint",
        "config": config_snapshot,
        "config_fingerprint": config_fingerprint,
        "latest_metrics": latest_metrics,
    }
    torch.save(final_checkpoint, run_dir / "final_checkpoint.pt")
    print(f"\nTraining complete.")
    print(f"Saved to: {run_dir}")


# ═════════════════════════════════════════════════════════════════════════════
#  Post-training evaluation: rollouts + visualisation
# ═════════════════════════════════════════════════════════════════════════════

def run_eval(
    *,
    game: Hexner3DQuadrotorGame,
    indexer: FullIaryTreeIndexer,
    alpha_module: AlphaParam,
    action_space: Optional[BoxActionSpace],
    x0: torch.Tensor,
    p0: torch.Tensor,
    args: argparse.Namespace,
    run_dir: Path,
) -> None:
    """Re-solve SQP with final α, generate rollouts, and save plots/animations."""

    print(f"\n{'─'*60}")
    print(f" Post-training evaluation")
    print(f"{'─'*60}")

    alpha_module.eval()

    with torch.no_grad():
        _, details = primal_objective_sqp(  # type: ignore
            game=game,
            alpha_module=alpha_module,
            indexer=indexer,
            x0=x0,
            p0=p0,
            action_space=action_space,
            num_sqp_iters=args.sqp_iters,
            sqp_step_size=args.sqp_step_size,
            riccati_reg=args.riccati_reg,
            sqp_line_search=not args.no_line_search,
            ls_alpha_min=args.ls_alpha_min,
            ls_backtrack=args.ls_backtrack,
            ls_max_steps=args.ls_max_steps,
            line_search_accept_worse=args.allow_worse_step,
            collect_sqp_diagnostics=True,
            return_details=True,
        )

    alpha = details["alpha"]
    sqp_res = details["sqp_result"]
    target_positions = game.type_target_positions()

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
    with open(run_dir / "correctness_eval.json", "w") as f:
        json.dump(
            {
                "metrics": correctness,
                "gates": gates,
                "passed": passed,
            },
            f,
            indent=2,
        )
    print(
        "  correctness: "
        f"separation={correctness['alpha']['separation_fraction']:.3f}, "
        f"terminal_consistency={correctness['terminal_consistency_rate']:.3f}, "
        f"physical_plausible={correctness['physical_plausible']}, "
        f"passed={passed}"
    )

    gen = torch.Generator(device=game.cfg.device_resolved)
    gen.manual_seed(args.seed)

    for type_idx in range(game.cfg.I):
        theta_i = game.cfg.theta_values[type_idx]
        print(f"  Rollout type {type_idx} (θ={theta_i:+.1f}) ...")

        ro = rollout_trajectory(
            game=game,
            indexer=indexer,
            belief_tree=sqp_res.belief_tree,
            riccati_sol=sqp_res.riccati_sol,
            alpha=alpha,
            x0=x0,
            type_index=type_idx,
            action_space=action_space,
            generator=gen,
        )

        # ── Static trajectory plot ───────────────────────────────────
        try:
            from src.utils.visualization import plot_trajectories, animate_rollout

            fig = plot_trajectories(
                ro.x_traj,
                belief_traj=ro.belief_traj,
                target_positions=target_positions,
                title=f"Type {type_idx} (θ={theta_i:+.1f})",
                save_path=str(run_dir / f"traj_type{type_idx}.png"),
            )
            plt_imported = True
        except ImportError:
            print("    matplotlib not available — skipping plots.")
            plt_imported = False

        # ── Animation ────────────────────────────────────────────────
        if plt_imported:
            try:
                animate_rollout(
                    ro.x_traj,
                    dt=game.cfg.tau,
                    belief_traj=ro.belief_traj,
                    target_positions=target_positions,
                    title=f"Type {type_idx} (θ={theta_i:+.1f})",
                    save_path=str(run_dir / f"anim_type{type_idx}.mp4"),
                )
            except Exception as e:
                print(f"    Animation failed (ffmpeg needed): {e}")

            import matplotlib.pyplot as plt
            plt.close("all")

    print(f"  Eval outputs saved to: {run_dir}")


# ═════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    # ── Build objects ────────────────────────────────────────────────
    cfg = build_game_config(args)
    game = Hexner3DQuadrotorGame(cfg)
    indexer = FullIaryTreeIndexer(I=cfg.I, K=cfg.K)
    action_space = build_action_space(args, game)

    alpha_module = AlphaParam(
        indexer,
        cfg=AlphaParamConfig(init_scale=args.alpha_init_scale),
        dtype=cfg.dtype,
        device=cfg.device,
    )

    x0 = game.default_initial_state()
    p0 = build_prior(args, cfg)

    # ── Run directory ────────────────────────────────────────────────
    if args.run_dir is None:
        run_dir = Path("runs") / f"quad_sqp_K{cfg.K}_I{cfg.I}"
    else:
        run_dir = Path(args.run_dir)

    # Create run directory if it doesn't exist
    run_dir.mkdir(parents=True, exist_ok=True)
    # delete existing log file if present
    log_path = run_dir / "train.jsonl"
    if log_path.exists():
        log_path.unlink()

    # ── Print summary ────────────────────────────────────────────────
    print(f"[train] Game: {cfg.game_name}")
    print(f"  T={cfg.T:.2f}  K={cfg.K}  I={cfg.I}  tau={cfg.tau:.4f}")
    print(f"  dx_joint={game.dx}  du={game.du}  dv={game.dv}")
    print(
        f"  integrator={cfg.integrator}  dtype={cfg.dtype}  "
        f"control_cost_mode={cfg.control_cost_mode}"
    )
    print(f"  tree nodes at depth K: {indexer.node_count(indexer.K)}")
    print(f"  α logits shape: {tuple(alpha_module.logits.shape)}")
    print(f"  x0 (P1 pos): {x0[:3].tolist()}")
    print(f"  x0 (P2 pos): {x0[12:15].tolist()}")
    print(f"  p0: {p0.tolist()}")
    print(f"  targets: {game.type_target_positions().tolist()}")
    if action_space is not None:
        print(f"  u bounds: [{action_space.u_min.tolist()}, {action_space.u_max.tolist()}]")
        print(f"  v bounds: [{action_space.v_min.tolist()}, {action_space.v_max.tolist()}]")
    print(f"  run_dir: {run_dir}")

    # Save config snapshot as JSON
    config_dict = {
        "T": cfg.T, "K": cfg.K, "I": cfg.I,
        "integrator": cfg.integrator,
        "control_cost_mode": cfg.control_cost_mode,
        "line_search_accept_worse": cfg.line_search_accept_worse,
        "dtype": str(cfg.dtype), "device": cfg.device,
        "R1_diag": list(cfg.R1_diag),
        "R2_diag": list(cfg.R2_diag),
        "K1_scale": cfg.K1_scale, "K2_scale": cfg.K2_scale,
        "theta_values": list(cfg.theta_values),
        "lr": args.lr, "epochs": args.epochs,
        "sqp_iters": args.sqp_iters,
        "sqp_step_size": args.sqp_step_size,
        "riccati_reg": args.riccati_reg,
        "ls_alpha_min": args.ls_alpha_min,
        "ls_backtrack": args.ls_backtrack,
        "ls_max_steps": args.ls_max_steps,
        "allow_worse_step": args.allow_worse_step,
        "alpha_init_scale": args.alpha_init_scale,
        "prior": args.prior,
        "u_max": args.u_max, "v_max": args.v_max,
        "eval_cold_start_every": args.eval_cold_start_every,
        "checkpoint_selection_policy": "last_checkpoint",
        "save_best_by_deprecated": args.save_best_by,
        "gate_min_separation": args.gate_min_separation,
        "gate_min_terminal_consistency": args.gate_min_terminal_consistency,
        "altitude_floor": args.altitude_floor,
        "seed": args.seed,
    }
    config_fingerprint = compute_config_fingerprint(config_dict)
    config_dict["config_fingerprint"] = config_fingerprint
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # ── Train ────────────────────────────────────────────────────────
    t0 = time.perf_counter()

    run_training(
        game=game,
        indexer=indexer,
        alpha_module=alpha_module,
        action_space=action_space,
        x0=x0,
        p0=p0,
        args=args,
        run_dir=run_dir,
        config_snapshot=config_dict,
        config_fingerprint=config_fingerprint,
    )

    elapsed = time.perf_counter() - t0
    print(f"\n[train] Total wall time: {elapsed:.2f} s")
    if args.epochs > 0:
        print(f"[train] Average: {1e3 * elapsed / args.epochs:.1f} ms/iter")

    # ── Post-training eval: rollouts + visualisation ─────────────────
    run_eval(
        game=game,
        indexer=indexer,
        alpha_module=alpha_module,
        action_space=action_space,
        x0=x0,
        p0=p0,
        args=args,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    main()
