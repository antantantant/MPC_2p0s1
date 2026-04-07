#!/usr/bin/env python3
"""Train Hexner (LQ) with the quadrotor SQP stack via an adapter.

Why this script exists
----------------------
The SQP implementation in ``quadrotor-game-solver`` is intended for nonlinear
quadrotor dynamics. To sanity-check that optimization/plumbing is correct, we
can run the *same* SQP training pipeline on a known linear-quadratic game
(Hexner). If this behaves well, it increases confidence in the SQP stack.

This script:
  1) Builds the standard 2D Hexner LQ game.
  2) Wraps it with an adapter exposing the interface expected by
     ``quadrotor-game-solver/src`` SQP modules.
  3) Trains signaling parameters ``alpha`` with the SQP objective.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


# Ensure both packages are importable when run as:
#   python scripts/train_hexner_sqp_adapter.py
#
# File layout:
#   <repo>/MPC_2p0s1/quadrotor-game-solver/scripts/train_hexner_sqp_adapter.py
QUADROTOR_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = QUADROTOR_ROOT.parent
PACKAGE_PARENT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_PARENT))   # for `import MPC_2p0s1...`
sys.path.insert(0, str(QUADROTOR_ROOT))   # for `import src...`

from MPC_2p0s1.config.base_config import GameConfig as LQGameConfig
from MPC_2p0s1.games.hexner_game import HexnerGame, HexnerParams

from src.optimization.objective_primal_sqp import primal_objective_sqp
from src.solvers.action_spaces import BoxActionSpace
from src.tree.indexing import FullIaryTreeIndexer
from src.tree.signaling import AlphaParam, AlphaParamConfig


class HexnerLQAdapter:
    """Adapter exposing the interface expected by quadrotor SQP code."""

    def __init__(self, lq_game: HexnerGame, line_search_accept_worse: bool = False) -> None:
        self.lq_game = lq_game

        self.device = lq_game.device_resolved
        self.dtype = lq_game.dtype
        self.I = lq_game.I
        self.dx = lq_game.dx
        self.du = lq_game.du
        self.dv = lq_game.dv

        # SQP code accesses game.cfg.tau and game.cfg.line_search_accept_worse.
        self.cfg = SimpleNamespace(
            tau=float(lq_game.cfg.tau),
            line_search_accept_worse=bool(line_search_accept_worse),
        )

    def default_initial_state(self) -> Tensor:
        return self.lq_game.default_initial_state().to(device=self.device, dtype=self.dtype)

    def default_prior(self) -> Tensor:
        return self.lq_game.default_prior().to(device=self.device, dtype=self.dtype)

    def default_hover_control(self) -> Tuple[Tensor, Tensor]:
        # For LQ double-integrator, "hover-equilibrium" corresponds to zero accel.
        u0 = torch.zeros(self.du, device=self.device, dtype=self.dtype)
        v0 = torch.zeros(self.dv, device=self.device, dtype=self.dtype)
        return u0, v0

    def step(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        return self.lq_game.step_dynamics(x, u, v)

    def linearize(self, x: Tensor, u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Linear system: x_next = A x + B1 u + B2 v, so Jacobians are constant.
        if x.ndim != 2:
            raise ValueError(f"Expected x shape (B, dx), got {tuple(x.shape)}")
        batch_size = x.shape[0]
        A = self.lq_game.A.unsqueeze(0).expand(batch_size, -1, -1)
        B1 = self.lq_game.B1.unsqueeze(0).expand(batch_size, -1, -1)
        B2 = self.lq_game.B2.unsqueeze(0).expand(batch_size, -1, -1)
        d = torch.zeros(batch_size, self.dx, device=self.device, dtype=self.dtype)
        return A, B1, B2, d

    def running_cost_mats(self, belief: Tensor) -> Tuple[Tensor, Tensor]:
        return self.lq_game.running_cost_mats(belief)

    def terminal_cost_quad(self, belief: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        Q_bar = torch.einsum("...i, iab -> ...ab", belief, self.lq_game.Q)
        q_bar = torch.einsum("...i, ia -> ...a", belief, self.lq_game.q)
        c_bar = torch.einsum("...i, i -> ...", belief, self.lq_game.c)
        return Q_bar, q_bar, c_bar

    def stage_cost_mats_batch(
        self, beliefs: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Hexner running cost is control-only in this formulation.
        batch_size = beliefs.shape[0]
        Q = torch.zeros(self.dx, self.dx, device=self.device, dtype=self.dtype)
        q = torch.zeros(batch_size, self.dx, device=self.device, dtype=self.dtype)
        c = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
        R_bar, S_bar = self.running_cost_mats(beliefs)
        return Q, q, c, R_bar, S_bar

    def terminal_value_quad_batch(self, beliefs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return self.terminal_cost_quad(beliefs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train Hexner with quadrotor SQP adapter to validate SQP/training "
            "stack on a known LQ game."
        )
    )

    # Game
    p.add_argument("--I", type=int, default=2)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--theta-values", type=str, default="-1.0,1.0")
    p.add_argument("--R1-scale", type=float, default=1.0)
    p.add_argument("--R2-scale", type=float, default=1.0)
    p.add_argument("--K1-scale", type=float, default=1.0)
    p.add_argument("--K2-scale", type=float, default=1.0)
    p.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    p.add_argument("--device", type=str, default="cpu")

    # SQP
    p.add_argument("--sqp-iters", type=int, default=10)
    p.add_argument("--sqp-step-size", type=float, default=1.0)
    p.add_argument("--riccati-reg", type=float, default=1e-3)
    p.add_argument("--max-riccati-reg-tries", type=int, default=5)
    p.add_argument("--riccati-reg-factor", type=float, default=10.0)
    p.add_argument("--sqp-early-stop", action="store_true")
    p.add_argument("--sqp-verbose", action="store_true")
    p.add_argument("--no-line-search", action="store_true")
    p.add_argument("--ls-alpha-min", type=float, default=0.05)
    p.add_argument("--ls-backtrack", type=float, default=0.5)
    p.add_argument("--ls-max-steps", type=int, default=6)
    p.add_argument("--allow-worse-step", action="store_true")
    p.add_argument("--sqp-tol-u", type=float, default=1e-3)
    p.add_argument("--sqp-tol-v", type=float, default=1e-3)
    p.add_argument("--sqp-tol-rel-cost", type=float, default=1e-4)

    # Optimization
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--alpha-init-scale", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--no-warmstart", action="store_true")

    # Action constraints
    p.add_argument("--no-action-clamp", action="store_true")
    p.add_argument("--u-max", type=float, default=4.0)
    p.add_argument("--v-max", type=float, default=4.0)

    # Logging/checkpointing
    p.add_argument("--run-dir", type=str, default="runs/hexner_sqp_adapter")
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _parse_theta_values(theta_values: str, I: int) -> Tuple[float, ...]:
    vals = tuple(float(v.strip()) for v in theta_values.split(",") if v.strip())
    if len(vals) != I:
        raise ValueError(f"--theta-values must have exactly I={I} entries, got {len(vals)}")
    return vals


def _build_action_space(
    *,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    du: int,
    dv: int,
) -> Optional[BoxActionSpace]:
    if args.no_action_clamp:
        return None
    u_lo = torch.full((du,), -args.u_max, device=device, dtype=dtype)
    u_hi = torch.full((du,), args.u_max, device=device, dtype=dtype)
    v_lo = torch.full((dv,), -args.v_max, device=device, dtype=dtype)
    v_hi = torch.full((dv,), args.v_max, device=device, dtype=dtype)
    return BoxActionSpace(u_min=u_lo, u_max=u_hi, v_min=v_lo, v_max=v_hi)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    theta_values = _parse_theta_values(args.theta_values, args.I)

    lq_cfg = LQGameConfig(
        I=args.I,
        dx1=4,
        dx2=4,
        du=2,
        dv=2,
        T=args.T,
        K=args.K,
        dtype=dtype,
        device=args.device,
    )
    lq_params = HexnerParams(
        theta_values=theta_values,
        R1_scale=args.R1_scale,
        R2_scale=args.R2_scale,
        K1_scale=args.K1_scale,
        K2_scale=args.K2_scale,
    )
    lq_game = HexnerGame(cfg=lq_cfg, params=lq_params)
    game = HexnerLQAdapter(lq_game=lq_game, line_search_accept_worse=args.allow_worse_step)

    indexer = FullIaryTreeIndexer(I=args.I, K=args.K)
    alpha_module = AlphaParam(
        indexer=indexer,
        cfg=AlphaParamConfig(init_scale=args.alpha_init_scale),
        dtype=dtype,
        device=lq_cfg.device_resolved,
    )

    action_space = _build_action_space(
        args=args,
        device=lq_cfg.device_resolved,
        dtype=dtype,
        du=game.du,
        dv=game.dv,
    )

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.jsonl"

    config = {
        "I": args.I,
        "T": args.T,
        "K": args.K,
        "theta_values": list(theta_values),
        "R1_scale": args.R1_scale,
        "R2_scale": args.R2_scale,
        "K1_scale": args.K1_scale,
        "K2_scale": args.K2_scale,
        "dtype": args.dtype,
        "device": args.device,
        "sqp_iters": args.sqp_iters,
        "sqp_step_size": args.sqp_step_size,
        "riccati_reg": args.riccati_reg,
        "max_riccati_reg_tries": args.max_riccati_reg_tries,
        "riccati_reg_factor": args.riccati_reg_factor,
        "line_search": not args.no_line_search,
        "ls_alpha_min": args.ls_alpha_min,
        "ls_backtrack": args.ls_backtrack,
        "ls_max_steps": args.ls_max_steps,
        "allow_worse_step": args.allow_worse_step,
        "epochs": args.epochs,
        "lr": args.lr,
        "alpha_init_scale": args.alpha_init_scale,
        "grad_clip": args.grad_clip,
        "warmstart": not args.no_warmstart,
        "u_max": args.u_max,
        "v_max": args.v_max,
        "seed": args.seed,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    optimizer = torch.optim.Adam(alpha_module.parameters(), lr=args.lr)
    x0 = game.default_initial_state()
    p0 = game.default_prior()

    u_prev: Optional[List[Tensor]] = None
    v_prev: Optional[List[Tensor]] = None
    best_loss = float("inf")
    t_start = time.time()

    print(
        f"[train_hexner_sqp_adapter] start: epochs={args.epochs}, "
        f"lr={args.lr}, sqp_iters={args.sqp_iters}, K={args.K}, I={args.I}"
    )

    from tqdm import tqdm # type: ignore[import]
    for epoch in tqdm(range(args.epochs), desc="Training Progress"):
        optimizer.zero_grad()
        collect_diag = (epoch % args.print_every == 0) or (epoch == args.epochs - 1)

        loss, details = primal_objective_sqp(  # type: ignore[misc]
            game=game,
            alpha_module=alpha_module,
            indexer=indexer,
            x0=x0,
            p0=p0,
            action_space=action_space,
            num_sqp_iters=args.sqp_iters,
            sqp_step_size=args.sqp_step_size,
            riccati_reg=args.riccati_reg,
            max_riccati_reg_tries=args.max_riccati_reg_tries,
            riccati_reg_factor=args.riccati_reg_factor,
            collect_sqp_diagnostics=collect_diag,
            sqp_tol_u=args.sqp_tol_u,
            sqp_tol_v=args.sqp_tol_v,
            sqp_tol_rel_cost=args.sqp_tol_rel_cost,
            sqp_verbose=args.sqp_verbose and collect_diag,
            sqp_early_stop=args.sqp_early_stop,
            sqp_line_search=not args.no_line_search,
            ls_alpha_min=args.ls_alpha_min,
            ls_backtrack=args.ls_backtrack,
            ls_max_steps=args.ls_max_steps,
            line_search_accept_worse=args.allow_worse_step,
            u_init=u_prev,
            v_init=v_prev,
            return_details=True,
        )

        loss.backward()
        if args.grad_clip > 0:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(alpha_module.parameters(), args.grad_clip).item()
            )
        else:
            grad_norm = 0.0
        optimizer.step()

        loss_f = float(loss.item())
        sqp_res = details["sqp_result"]
        if not args.no_warmstart:
            u_prev = [u.detach() for u in sqp_res.u_edges]
            v_prev = [v.detach() for v in sqp_res.v_edges]

        if loss_f < best_loss:
            best_loss = loss_f
            torch.save(
                {
                    "epoch": epoch,
                    "alpha_state_dict": alpha_module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_f,
                    "config": config,
                },
                run_dir / "best.pt",
            )

        if (epoch + 1) % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "alpha_state_dict": alpha_module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_f,
                    "config": config,
                },
                run_dir / f"checkpoint_{epoch + 1:04d}.pt",
            )

        if collect_diag:
            elapsed = time.time() - t_start
            sqp_diag = sqp_res.diagnostics
            log_entry: Dict[str, object] = {
                "epoch": epoch,
                "loss": loss_f,
                "grad_norm": grad_norm,
                "elapsed_s": elapsed,
            }
            if sqp_diag is not None:
                log_entry.update(
                    {
                        "sqp_converged": bool(sqp_diag.converged),
                        "sqp_converged_iter": int(sqp_diag.converged_iter),
                        "sqp_nan": bool(sqp_diag.nan_or_inf_encountered),
                        "sqp_cost_last": float(sqp_diag.cost_hist[-1]),
                        "sqp_du_last": float(sqp_diag.du_max_hist[-1]),
                        "sqp_dv_last": float(sqp_diag.dv_max_hist[-1]),
                    }
                )
                print(
                    f"[ep {epoch:04d}] loss={loss_f:+.6f} grad={grad_norm:.2e} "
                    f"sqp_cost={sqp_diag.cost_hist[-1]:+.6f} "
                    f"du={sqp_diag.du_max_hist[-1]:.2e} dv={sqp_diag.dv_max_hist[-1]:.2e} "
                    f"conv={sqp_diag.converged} nan={sqp_diag.nan_or_inf_encountered}"
                )
            else:
                print(f"[ep {epoch:04d}] loss={loss_f:+.6f} grad={grad_norm:.2e}")

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

    torch.save(
        {
            "epoch": args.epochs - 1,
            "alpha_state_dict": alpha_module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": float(loss.item()),
            "config": config,
        },
        run_dir / "latest.pt",
    )
    print(
        f"[train_hexner_sqp_adapter] done: best_loss={best_loss:+.6f}, "
        f"artifacts={run_dir}"
    )


if __name__ == "__main__":
    main()
