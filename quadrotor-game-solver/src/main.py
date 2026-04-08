"""Training script for the 3-D Hexner game with nonlinear quadrotor dynamics.

Usage
-----
    python -m src.main                     # defaults
    python -m src.main --K 8 --sqp-iters 5 --lr 1e-3

Mirrors ``scripts/train_hexner_primal.py`` in the parent LQ project
but uses the SQP inner layer.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from .utils.config import GameConfig, make_hexner3d_game_config
from .game import build_game
from .tree.indexing import FullIaryTreeIndexer
from .tree.signaling import AlphaParam, AlphaParamConfig
from .solvers.action_spaces import BoxActionSpace
from .optimization.objective_primal_sqp import primal_objective_sqp
from .rollout.trajectory import rollout_trajectory, RolloutResult


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train α for the 3-D Hexner quadrotor game (SQP primal)."
    )
    # ── Game ─────────────────────────────────────────────────────────
    p.add_argument("--T", type=float, default=2.0, help="Time horizon (s)")
    p.add_argument("--K", type=int, default=5, help="Number of discrete steps")
    p.add_argument("--I", type=int, default=2, help="Number of types")
    p.add_argument(
        "--game-model",
        type=str,
        default="rigid_body",
        choices=["rigid_body", "interception"],
    )
    p.add_argument(
        "--payoff-model",
        type=str,
        default="hexner",
        choices=["hexner", "hexner_mod"],
    )
    p.add_argument("--integrator", type=str, default="euler",
                   choices=["euler", "rk4"])
    p.add_argument("--dtype", type=str, default="float64",
                   choices=["float32", "float64"])
    p.add_argument("--device", type=str, default="cpu")

    # ── SQP ──────────────────────────────────────────────────────────
    p.add_argument("--sqp-iters", type=int, default=3)
    p.add_argument("--sqp-step-size", type=float, default=1.0)
    p.add_argument("--riccati-reg", type=float, default=1e-3)
    p.add_argument("--sqp-verbose", action="store_true")
    p.add_argument("--sqp-early-stop", action="store_true")

    # ── Optimiser ────────────────────────────────────────────────────
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--alpha-init-scale", type=float, default=0.01)

    # ── Action bounds ────────────────────────────────────────────────
    p.add_argument("--no-action-clamp", action="store_true",
                   help="Disable action-space clamping")
    p.add_argument("--u-max", type=float, default=20.0)
    p.add_argument("--v-max", type=float, default=20.0)
    p.add_argument("--u-torque-max", type=float, default=1.5,
                   help="Rigid-body only: per-axis torque bound for P1.")
    p.add_argument("--v-torque-max", type=float, default=1.5,
                   help="Rigid-body only: per-axis torque bound for P2.")

    # ── Logging ──────────────────────────────────────────────────────
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = args.device

    # ── Game setup ───────────────────────────────────────────────────
    if args.game_model == "interception":
        r1_diag = (1.0, 1.0, 1.0)
        r2_diag = (0.4, 0.4, 0.4) if args.payoff_model == "hexner_mod" else (1.0, 1.0, 1.0)
    else:
        r1_diag = (0.05, 0.025, 0.025, 0.01)
        r2_diag = (0.02, 0.04, 0.04, 0.008) if args.payoff_model == "hexner_mod" else (0.05, 0.10, 0.10, 0.02)

    cfg = GameConfig(
        T=args.T,
        K=args.K,
        I=args.I,
        dynamics_model=args.game_model,
        payoff_model=args.payoff_model,
        integrator=args.integrator,
        dtype=dtype,
        device=device,
        R1_diag=r1_diag,
        R2_diag=r2_diag,
    )
    game = build_game(cfg)
    indexer = FullIaryTreeIndexer(I=cfg.I, K=cfg.K)

    print(f"Game: {cfg.game_name}")
    print(f"  T={cfg.T:.2f}  K={cfg.K}  I={cfg.I}  tau={cfg.tau:.4f}")
    print(f"  dx_joint={game.dx}  du={game.du}  dv={game.dv}")
    print(f"  integrator={cfg.integrator}  dtype={dtype}")
    print(f"  tree depth={indexer.K}, leaf nodes={indexer.node_count(indexer.K)}")

    # ── α parameterisation ───────────────────────────────────────────
    alpha_module = AlphaParam(
        indexer,
        cfg=AlphaParamConfig(init_scale=args.alpha_init_scale),
        dtype=dtype,
        device=device,
    )
    print(f"  α logits shape: {tuple(alpha_module.logits.shape)}")

    # ── Action space ─────────────────────────────────────────────────
    action_space = None
    if not args.no_action_clamp:
        u_lo, u_hi, v_lo, v_hi = game.action_box_bounds(
            u_max=float(args.u_max),
            v_max=float(args.v_max),
        )
        if cfg.dynamics_model == "rigid_body":
            u_lo[1:] = -float(args.u_torque_max)
            u_hi[1:] = float(args.u_torque_max)
            v_lo[1:] = -float(args.v_torque_max)
            v_hi[1:] = float(args.v_torque_max)
        action_space = BoxActionSpace(u_min=u_lo, u_max=u_hi,
                                      v_min=v_lo, v_max=v_hi)
        print(f"  action bounds u: [{u_lo.tolist()}, {u_hi.tolist()}]")
        print(f"  action bounds v: [{v_lo.tolist()}, {v_hi.tolist()}]")

    # ── Initial state & prior ────────────────────────────────────────
    x0 = game.default_initial_state()
    p0 = game.default_prior()
    print(f"  x0 (P1 pos): {x0[:3].tolist()}")
    print(f"  x0 (P2 pos): {game.player_position(x0, 1).tolist()}")
    print(f"  p0: {p0.tolist()}")
    print(f"  targets (z·θ_i positions): "
          f"{game.type_target_positions().tolist()}")

    # ── Optimiser ────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(alpha_module.parameters(), lr=args.lr)

    # ── Run directory ────────────────────────────────────────────────
    if args.run_dir is None:
        run_dir = Path("runs") / f"quad_sqp_K{cfg.K}_I{cfg.I}"
    else:
        run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRun directory: {run_dir}")

    # Save config
    config_dict = {
        "T": cfg.T, "K": cfg.K, "I": cfg.I,
        "dynamics_model": cfg.dynamics_model,
        "payoff_model": cfg.payoff_model,
        "integrator": cfg.integrator,
        "dtype": str(dtype), "device": device,
        "lr": args.lr, "epochs": args.epochs,
        "sqp_iters": args.sqp_iters,
        "sqp_step_size": args.sqp_step_size,
        "riccati_reg": args.riccati_reg,
        "alpha_init_scale": args.alpha_init_scale,
        "seed": args.seed,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # ── Training loop ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" Starting training: {args.epochs} epochs, lr={args.lr}")
    print(f" SQP: {args.sqp_iters} iters, step_size={args.sqp_step_size}")
    print(f"{'='*60}\n")

    log_path = run_dir / "train.jsonl"

    best_loss = float("inf")

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
            collect_sqp_diagnostics=True,
            return_details=True,
        )

        loss, details = result  # type: ignore
        loss_val = float(loss.item())

        # Backward
        loss.backward()

        # Check for NaN gradients
        grad_norm = 0.0
        for param in alpha_module.parameters():
            if param.grad is not None:
                grad_norm += float(param.grad.data.norm().item()) ** 2
        grad_norm = grad_norm ** 0.5

        optimizer.step()
        dt_epoch = time.time() - t0

        # ── Logging ──────────────────────────────────────────────────
        if epoch % args.log_every == 0:
            alpha_val = details["alpha"]
            alpha_range = (
                float(alpha_val.min().item()),
                float(alpha_val.max().item()),
            )

            sqp_diag = details["sqp_result"].diagnostics
            sqp_info = {}
            if sqp_diag is not None:
                sqp_info = {
                    "sqp_cost_hist": sqp_diag.cost_hist,
                    "sqp_converged": sqp_diag.converged,
                    "sqp_nan": sqp_diag.nan_or_inf_encountered,
                }

            log_entry = {
                "epoch": epoch,
                "loss": loss_val,
                "grad_norm": grad_norm,
                "alpha_range": alpha_range,
                "time_s": dt_epoch,
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

            print(
                f"[{epoch:4d}/{args.epochs}]  loss={loss_val:+.6f}  "
                f"|∇|={grad_norm:.3e}  α∈{alpha_range}  "
                f"dt={dt_epoch:.2f}s{conv_str}{nan_str}"
            )

        # ── Checkpointing ────────────────────────────────────────────
        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(alpha_module.state_dict(), run_dir / "best_alpha.pt")

        if (epoch + 1) % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "alpha_state_dict": alpha_module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_val,
                },
                run_dir / f"checkpoint_{epoch:04d}.pt",
            )

    # ── Final save ───────────────────────────────────────────────────
    torch.save(alpha_module.state_dict(), run_dir / "final_alpha.pt")
    print(f"\nTraining complete.  Best loss: {best_loss:.6f}")
    print(f"Saved to: {run_dir}")


if __name__ == "__main__":
    main()
