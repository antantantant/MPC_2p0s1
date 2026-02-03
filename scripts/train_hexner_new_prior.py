# scripts/train_hexner_new_prior.py
"""
Train the signaling policy for a NEW prior, using a pretrained checkpoint as warm-start.

Usage:
    python -m scripts.train_hexner_new_prior \
        --warmstart-checkpoint runs/hexner_primal_test_stable/latest.pt \
        --prior "0.3,0.7" \
        --run-dir runs/hexner_prior_0.3_0.7 \
        --max-iters 500 \
        --lr 1e-3
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import numpy as np
from tqdm import tqdm

from MPC_2p0s1.config.base_config import (
    GameConfig,
    PathsConfig,
    project_relative,
)
from MPC_2p0s1.core.action_spaces import BoxActionSpace
from MPC_2p0s1.core.utils import set_random_seeds
from MPC_2p0s1.games.hexner_game import HexnerGame, HexnerParams
from MPC_2p0s1.outer_opt.alpha_param import AlphaParam, AlphaParamConfig
from MPC_2p0s1.outer_opt.checkpointing import CheckpointMeta
from MPC_2p0s1.outer_opt.objective_primal import primal_objective
from MPC_2p0s1.tree.indexing import FullIaryTreeIndexer


def _save_checkpoint(
    ckpt_path: str,
    alpha_module: AlphaParam,
    optimizer: torch.optim.Optimizer,
    meta: CheckpointMeta,
) -> None:
    """Save checkpoint with alpha and optimizer state."""
    payload = {
        "alpha_state_dict": alpha_module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "meta": {
            "step": meta.step,
            "loss": meta.loss,
            "game_config": meta.game_config,
            "training_config": meta.training_config,
        },
    }
    torch.save(payload, ckpt_path)


def _rebuild_game_config(meta_dict) -> GameConfig:
    """Reconstruct a GameConfig from checkpoint metadata."""
    cfg = GameConfig()
    for k, v in meta_dict.items():
        setattr(cfg, k, v)
    if not hasattr(cfg, "dx"):
        cfg.dx = cfg.dx1 + cfg.dx2
    if not hasattr(cfg, "tau") and hasattr(cfg, "T") and hasattr(cfg, "K"):
        cfg.tau = cfg.T / cfg.K
    cfg.device_resolved = torch.device(getattr(cfg, "device", "cpu"))
    if not hasattr(cfg, "dtype"):
        cfg.dtype = torch.float32
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train signaling policy for a new prior using warm-start from checkpoint."
    )
    
    # Warm-start checkpoint
    parser.add_argument(
        "--warmstart-checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint trained with uniform prior (warm-start).",
    )
    
    # New prior
    parser.add_argument(
        "--prior",
        type=str,
        required=True,
        help="Comma-separated prior probabilities (e.g., '0.3,0.7').",
    )
    
    # Training parameters
    parser.add_argument("--max-iters", type=int, default=500, help="Maximum training iterations.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "lbfgs"])
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    
    # Output
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Output directory. If None, auto-generates from prior.",
    )
    parser.add_argument("--device", type=str, default=None, help="Override device.")
    parser.add_argument("--seed", type=int, default=123)
    
    # Convergence criteria
    parser.add_argument("--convergence-tol", type=float, default=1e-6,
                        help="Stop if loss change < tol for consecutive iterations.")
    parser.add_argument("--convergence-patience", type=int, default=20,
                        help="Number of iterations with small change before stopping.")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Parse prior
    prior_vals = [float(x) for x in args.prior.split(",")]
    prior = torch.tensor(prior_vals, dtype=torch.float32)
    prior = prior / prior.sum()  # Normalize
    
    print("=" * 70)
    print("TRAINING WITH NEW PRIOR (WARM-START)")
    print("=" * 70)
    print(f"Warm-start checkpoint: {args.warmstart_checkpoint}")
    print(f"New prior: {prior.tolist()}")
    
    # Load checkpoint
    print("\n[1] Loading warm-start checkpoint...")
    payload = torch.load(args.warmstart_checkpoint, map_location="cpu")
    meta_dict = payload.get("meta", {})
    
    # Rebuild game config
    game_meta = meta_dict.get("game_config", {})
    game_cfg = _rebuild_game_config(game_meta)
    
    device = torch.device(args.device) if args.device else game_cfg.device_resolved
    game_cfg.device_resolved = device
    print(f"Using device: {device}")
    
    K = game_cfg.K
    I = game_cfg.I
    
    # Set seed
    set_random_seeds(args.seed)
    
    # Create game
    hexner_params = HexnerParams()
    game = HexnerGame(cfg=game_cfg, params=hexner_params, prior=prior)
    action_space = BoxActionSpace.from_config(game_cfg)
    
    # Create alpha module and load warm-start weights
    indexer = FullIaryTreeIndexer(I=I, K=K)
    alpha_cfg = AlphaParamConfig()
    alpha_module = AlphaParam(
        indexer=indexer,
        alpha_cfg=alpha_cfg,
        dtype=game_cfg.dtype,
        device=device,
    )
    
    # Load pretrained alpha weights
    alpha_state = payload.get("alpha_state_dict", {})
    alpha_module.load_state_dict(alpha_state)
    print(f"Loaded alpha weights from checkpoint (step={meta_dict.get('step', '?')})")
    
    # Move prior to device
    prior = prior.to(device=device, dtype=game_cfg.dtype)
    
    # Setup output directory
    if args.run_dir is None:
        prior_str = "_".join(f"{p:.2f}" for p in prior.tolist())
        run_dir = project_relative(f"runs/hexner_prior_{prior_str}")
    else:
        run_dir = project_relative(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Output directory: {run_dir}")
    
    # Create optimizer
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            alpha_module.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.LBFGS(
            alpha_module.parameters(),
            lr=args.lr,
            max_iter=20,
            line_search_fn="strong_wolfe",
        )
    
    # Training loop
    print("\n[2] Starting training...")
    print(f"Max iterations: {args.max_iters}")
    print(f"Learning rate: {args.lr}")
    print(f"Convergence tolerance: {args.convergence_tol}")
    print(f"Convergence patience: {args.convergence_patience}")
    
    # History tracking
    losses = []
    best_loss = float("inf")
    patience_counter = 0
    
    t_start = time.perf_counter()
    
    for step in tqdm(range(args.max_iters)):
        # Compute loss
        optimizer.zero_grad()
        
        loss = primal_objective(
            game=game,
            alpha_module=alpha_module,
            indexer=indexer,
            x0=None,  # Use default initial state
            p0=prior,  # Use the NEW prior
            action_space=action_space,
            return_details=False,
        )
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(alpha_module.parameters(), args.grad_clip_norm)
        
        # Step
        if args.optimizer.lower() == "adam":
            optimizer.step()
        else:
            def closure():
                optimizer.zero_grad()
                l = primal_objective(
                    game=game,
                    alpha_module=alpha_module,
                    indexer=indexer,
                    x0=None,
                    p0=prior,
                    action_space=action_space,
                    return_details=False,
                )
                l.backward()
                return l
            optimizer.step(closure)
        
        loss_val = loss.item()
        losses.append(loss_val)
        
        # Print progress
        if step % args.print_every == 0 or step == args.max_iters - 1:
            elapsed = time.perf_counter() - t_start
            print(f"  Step {step:4d}: loss = {loss_val:.6f}, time = {elapsed:.1f}s")
        
        # Checkpoint
        if step % args.checkpoint_every == 0 or step == args.max_iters - 1:
            ckpt_path = os.path.join(run_dir, f"checkpoint_step_{step:06d}.pt")
            meta = CheckpointMeta(
                step=step,
                loss=loss_val,
                game_config=game_meta,
                training_config={
                    "prior": prior.tolist(),
                    "lr": args.lr,
                    "max_iters": args.max_iters,
                    "warmstart": args.warmstart_checkpoint,
                },
            )
            _save_checkpoint(
                ckpt_path=ckpt_path,
                alpha_module=alpha_module,
                optimizer=optimizer,
                meta=meta,
            )
        
        # Check convergence
        if len(losses) > 1:
            loss_change = abs(losses[-1] - losses[-2])
            if loss_change < args.convergence_tol:
                patience_counter += 1
                if patience_counter >= args.convergence_patience:
                    print(f"\n[CONVERGED] Loss change < {args.convergence_tol} "
                          f"for {args.convergence_patience} iterations.")
                    break
            else:
                patience_counter = 0
        
        # Track best
        if loss_val < best_loss:
            best_loss = loss_val
    
    # Final save
    total_time = time.perf_counter() - t_start
    print(f"\n[3] Training complete!")
    print(f"Total time: {total_time:.1f}s")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Best loss: {best_loss:.6f}")
    
    # Save final checkpoint
    final_path = os.path.join(run_dir, "checkpoint_final.pt")
    meta = CheckpointMeta(
        step=len(losses) - 1,
        loss=losses[-1],
        game_config=game_meta,
        training_config={
            "prior": prior.tolist(),
            "lr": args.lr,
            "max_iters": args.max_iters,
            "warmstart": args.warmstart_checkpoint,
            "converged": patience_counter >= args.convergence_patience,
        },
    )
    _save_checkpoint(
        ckpt_path=final_path,
        alpha_module=alpha_module,
        optimizer=optimizer,
        meta=meta,
    )
    print(f"Saved final checkpoint to: {final_path}")
    
    # Also save as latest.pt
    latest_path = os.path.join(run_dir, "latest.pt")
    _save_checkpoint(
        ckpt_path=latest_path,
        alpha_module=alpha_module,
        optimizer=optimizer,
        meta=meta,
    )
    
    # Save loss history
    np.savez(
        os.path.join(run_dir, "metrics.npz"),
        steps=np.arange(len(losses)),
        losses=np.array(losses),
    )
    print(f"Saved metrics to: {os.path.join(run_dir, 'metrics.npz')}")
    
    # Quick evaluation
    print("\n[4] Quick evaluation of trained policy...")
    from scripts.eval_hexner_diff_priors import (
        solve_ground_truth_with_prior,
        online_mpc_rollout_with_prior,
        compute_ground_truth_trajectory,
        compute_gt_cost,
    )
    from MPC_2p0s1.compact.compact_solution import extract_compact_solution
    
    # Get trained alpha
    alpha_full = alpha_module()
    
    # Solve tree
    belief_tree, riccati_sol, indexer_out, alpha_tree = solve_ground_truth_with_prior(
        game, K, I, alpha_full, prior, indexer=indexer
    )
    
    # Compact solution
    compact = extract_compact_solution(game, K)
    
    # Rollouts
    x0 = game.default_initial_state().to(device)
    x1_init = x0[:4]
    x2_init = x0[4:]
    
    theta_vals = [-1.0, 1.0]

    # prior[0] = P(type 0) = P(θ=-1), prior[1] = P(type 1) = P(θ=+1)
    # GT function expects P(θ=-1), so we pass prior[0]
    p_theta_minus = float(prior[0])
    tr = 0.5 
    
    print(f"\nMPC vs Ground Truth for prior {prior.tolist()}:")
    print(f"Using revelation time t_r = {tr:.3f}")
    print(f"P(θ=-1) = {p_theta_minus:.3f}")
    for type_idx in range(I):
        theta = theta_vals[type_idx]
        
        # MPC rollout
        x_mpc, u_mpc, v_mpc, cost_mpc = online_mpc_rollout_with_prior(
            game, compact, x0, prior, type_idx,
            riccati_sol, indexer_out, alpha_tree, action_space
        )
        
        # GT expects prior = P(θ=-1)
        gt = compute_ground_truth_trajectory(
            T=game_cfg.T, K=K, theta=theta,
            x1_init=x1_init, x2_init=x2_init,
            tr=tr, prior=p_theta_minus, dtype=game_cfg.dtype, device=device
        )
        gt_cost = compute_gt_cost(gt, theta, game_cfg.T, K)
        
        # Position error
        x1_mpc = x_mpc[:, :4]
        x2_mpc = x_mpc[:, 4:]
        pos_err = ((x1_mpc[-1, :2] - gt.x1_traj[-1, :2]).norm().item() +
                   (x2_mpc[-1, :2] - gt.x2_traj[-1, :2]).norm().item()) / 2
        
        print(f"  Type {type_idx} (θ={theta:+.0f}): "
              f"MPC cost = {cost_mpc:.4f}, GT cost = {gt_cost:.4f}, "
              f"pos_err = {pos_err:.4f}")


if __name__ == "__main__":
    main()