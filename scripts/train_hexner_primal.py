# scripts/train_hexner_primal.py
from __future__ import annotations

import argparse
import time

import torch

from MPC_2p0s1.config.base_config import (
    GameConfig,
    TrainingConfig,
    PathsConfig,
    project_relative,
)

from MPC_2p0s1.core.action_spaces import BoxActionSpace
from MPC_2p0s1.games.hexner_game import HexnerGame, HexnerParams
from MPC_2p0s1.outer_opt.alpha_param import AlphaParam, AlphaParamConfig
from MPC_2p0s1.outer_opt.optimizer_loop import run_outer_optimization
from MPC_2p0s1.tree.indexing import FullIaryTreeIndexer


def build_configs_from_args(args: argparse.Namespace):
    """
    Construct GameConfig, TrainingConfig, and PathsConfig objects from CLI args.
    """
    # --- GameConfig ------------------------------------------------------ #
    game_cfg = GameConfig()

    # Hexner: 2D double-integrator per player
    game_cfg.I = args.I
    game_cfg.dx1 = 4
    game_cfg.dx2 = 4
    game_cfg.dx = game_cfg.dx1 + game_cfg.dx2
    game_cfg.du = 2
    game_cfg.dv = 2

    # Time horizon and discretization
    game_cfg.T = float(args.T)
    game_cfg.K = int(args.K)
    game_cfg.tau = game_cfg.T / game_cfg.K

    # Device / dtype / random seed
    game_cfg.device = args.device
    game_cfg.device_resolved = torch.device(args.device)
    game_cfg.dtype = torch.float32 # 32 is the default, but use 64 for better numerical stability 
    game_cfg.seed = int(args.seed)

    # Action bounds
    game_cfg.u_min = float(args.u_min)
    game_cfg.u_max = float(args.u_max)
    game_cfg.v_min = float(args.v_min)
    game_cfg.v_max = float(args.v_max)

    # --- TrainingConfig -------------------------------------------------- #
    train_cfg = TrainingConfig()
    train_cfg.optimizer = args.optimizer.lower()
    train_cfg.lr = float(args.lr)
    train_cfg.weight_decay = float(args.weight_decay)
    train_cfg.max_iters = int(args.max_iters)
    train_cfg.print_every = int(args.print_every)
    train_cfg.checkpoint_every = int(args.checkpoint_every)

    train_cfg.use_mixed_precision = bool(args.mixed_precision)
    train_cfg.grad_clip_norm = (
        float(args.grad_clip_norm) if args.grad_clip_norm is not None else None
    )

    # Staged depth-wise optimization
    train_cfg.staged_depth = not args.no_staged_depth
    train_cfg.initial_unfrozen_depth = int(args.initial_unfrozen_depth)
    train_cfg.depth_increment = int(args.depth_increment)
    train_cfg.depth_increment_every = int(args.depth_increment_every)
    train_cfg.max_unfrozen_depth = (
        None if args.max_unfrozen_depth < 0 else int(args.max_unfrozen_depth)
    )

    # --- PathsConfig ----------------------------------------------------- #
    run_root = project_relative(args.run_dir)
    paths_cfg = PathsConfig(
        root_dir=run_root,
        run_name="",
        create_subdir=False,
    )
    paths_cfg.ensure_exists()

    return game_cfg, train_cfg, paths_cfg


def build_hexner_params_from_args(args: argparse.Namespace) -> HexnerParams:
    """
    Build HexnerParams from CLI args.
    """
    theta_vals = tuple(float(v) for v in args.theta_values.split(","))
    params = HexnerParams(
        theta_values=theta_vals,
        target_z=None,  # default is (0,1,0,0); can be customized later
        R1_scale=float(args.R1_scale),
        R2_scale=float(args.R2_scale),
        K1_scale=float(args.K1_scale),
        K2_scale=float(args.K2_scale),
    )
    return params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the primal 2p0s1 solver (CAMS-MPC style) on Hexner’s game "
            "using the tree-structured Riccati + α-optimization pipeline."
        )
    )

    # Game / tree
    parser.add_argument("--I", type=int, default=2, help="Number of payoff types (Hexner: I=2).")
    parser.add_argument("--T", type=float, default=1.0, help="Time horizon T.")
    parser.add_argument("--K", type=int, default=10, help="Number of time steps (depth K).")

    # Dynamics / control penalties for Hexner
    parser.add_argument(
        "--theta-values",
        type=str,
        default="-1.0,1.0",
        help="Comma-separated payoff type scalars θ_i (e.g., '-1.0,1.0' for I=2).",
    )
    parser.add_argument("--R1-scale", type=float, default=1.0, help="Scale for P1 running-cost matrix R1.")
    parser.add_argument("--R2-scale", type=float, default=1.0, help="Scale for P2 running-cost matrix R2.")
    parser.add_argument("--K1-scale", type=float, default=1.0, help="Scale for P1 terminal-cost matrix K1.")
    parser.add_argument("--K2-scale", type=float, default=1.0, help="Scale for P2 terminal-cost matrix K2.")

    # Action bounds
    parser.add_argument("--u-min", type=float, default=-14.0, help="Lower bound for P1 controls.")
    parser.add_argument("--u-max", type=float, default=14.0, help="Upper bound for P1 controls.")
    parser.add_argument("--v-min", type=float, default=-14.0, help="Lower bound for P2 controls.")
    parser.add_argument("--v-max", type=float, default=14.0, help="Upper bound for P2 controls.")

    # Optimization hyperparameters
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "lbfgs"],
        help="Outer optimizer for α/logits.",
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for Adam / LBFGS.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for optimizer (Adam only).")
    parser.add_argument("--max-iters", type=int, default=2000, help="Maximum number of outer iterations.")
    parser.add_argument("--print-every", type=int, default=50, help="How often to print loss.")
    parser.add_argument("--checkpoint-every", type=int, default=200, help="How often to save checkpoints.")
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help="If set, clip gradient norm of α/logits to this value.",
    )

    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed-precision (AMP) on CUDA devices for speed.",
    )

    # Staged depth-wise schedule
    parser.add_argument(
        "--no-staged-depth",
        action="store_true",
        help="Disable staged depth-wise optimization; update all depths at once.",
    )
    parser.add_argument(
        "--initial-unfrozen-depth",
        type=int,
        default=0,
        help="Initial maximum depth to unfreeze (inclusive) when staged depth is enabled.",
    )
    parser.add_argument(
        "--depth-increment",
        type=int,
        default=1,
        help="Number of additional depths to unfreeze when schedule advances.",
    )
    parser.add_argument(
        "--depth-increment-every",
        type=int,
        default=200,
        help="Iterations between depth increments.",
    )
    parser.add_argument(
        "--max-unfrozen-depth",
        type=int,
        default=-1,
        help="Optional cap on deepest unfrozen depth; negative → no explicit cap.",
    )

    # Device / seed / paths
    parser.add_argument("--device", type=str, default="cpu", help="Device string, e.g., 'cpu' or 'cuda:0'.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="runs/hexner_primal_test",
        help="Directory for checkpoints and logs.",
    )
    parser.add_argument("--prior", type=float, default=0.5, help="Prior probability for type 1 (theta=-1) (type 2 prob = 1 - prior).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Build configs
    game_cfg, train_cfg, paths_cfg = build_configs_from_args(args)
    hexner_params = build_hexner_params_from_args(args)

    # prior tensor
    prior = torch.tensor([args.prior, 1.0 - args.prior], dtype=game_cfg.dtype, device=game_cfg.device_resolved)
    # Instantiate game and structures
    game = HexnerGame(cfg=game_cfg, params=hexner_params, prior=prior)

    indexer = FullIaryTreeIndexer(I=game_cfg.I, K=args.K)
    alpha_cfg = AlphaParamConfig(init_scale=0.1)
    alpha_module = AlphaParam(
        indexer=indexer,
        alpha_cfg=alpha_cfg,
        dtype=game_cfg.dtype,
        device=game_cfg.device_resolved,
    )

    action_space = BoxActionSpace.from_config(game_cfg)

    print(
        f"[train_hexner_primal] I={game_cfg.I}, T={game_cfg.T}, K={args.K}, "
        f"tau={game_cfg.tau:.4f}, device={game_cfg.device_resolved}, "
        f"run_dir='{paths_cfg.run_dir}'"
    )

    # Run outer optimization over α/logits with coarse total timing
    t0 = time.perf_counter()
    run_outer_optimization(
        game=game,
        game_cfg=game_cfg,
        train_cfg=train_cfg,
        paths_cfg=paths_cfg,
        indexer=indexer,
        alpha_module=alpha_module,
        action_space=action_space,
        x0=None,  # use HexnerGame.default_initial_state()
        p0=None,  # use HexnerGame.default_prior()
        depth_schedule=None,  # constructed internally if staged_depth=True
        resume_from=None,
    )
    t1 = time.perf_counter()

    total_time = t1 - t0
    print(
        f"[train_hexner_primal] Total optimization time: {total_time:.2f} s "
        f"(max_iters={train_cfg.max_iters})"
    )
    if train_cfg.max_iters > 0:
        avg_ms = 1e3 * total_time / float(train_cfg.max_iters)
        print(
            f"[train_hexner_primal] Approx. average iteration time: {avg_ms:.1f} ms/iter"
        )


if __name__ == "__main__":
    main()