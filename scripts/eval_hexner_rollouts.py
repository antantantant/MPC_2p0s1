# scripts/eval_hexner_rollouts.py
from __future__ import annotations

import argparse
import os
from typing import List

import torch

from MPC_2p0s1.config.base_config import GameConfig, TrainingConfig, project_relative
from MPC_2p0s1.core.action_spaces import BoxActionSpace
from MPC_2p0s1.games.hexner_game import HexnerGame, HexnerParams
from MPC_2p0s1.outer_opt.alpha_param import AlphaParam, AlphaParamConfig
from MPC_2p0s1.outer_opt.checkpointing import load_checkpoint
from MPC_2p0s1.outer_opt.objective_primal import primal_objective
from MPC_2p0s1.tree.indexing import FullIaryTreeIndexer
from MPC_2p0s1.tree.rollout import rollout_trajectory, RolloutResult
from MPC_2p0s1.viz.make_report_hexner import make_hexner_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained Hexner α / checkpoint by generating rollouts and "
            "producing basic visualizations (trajectories + beliefs).  [oai_citation:2‡16010_Solving_Football_by_Expl-2.pdf](sediment://file_000000001b0871f58cd1c91b483083b4)"
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a checkpoint file produced by train_hexner_primal.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/hexner_eval",
        help="Directory in which to save figures.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional override for device (e.g., 'cpu', 'cuda:0'). "
        "If omitted, uses device from the checkpoint's GameConfig.",
    )
    parser.add_argument(
        "--num-rollouts-per-type",
        type=int,
        default=1,
        help="Number of rollouts to generate for each payoff type.",
    )
    parser.add_argument(
        "--sample-actions",
        action="store_true",
        help="If set, sample prototypes according to α instead of using argmax.",
    )

    return parser.parse_args()


def _rebuild_game_config(meta_dict) -> GameConfig:
    """
    Reconstruct a GameConfig from a plain-Python dict stored in the checkpoint.

    We start from defaults and then overwrite attributes using the dict keys.
    This approach is robust to config evolution (new fields get defaults).
    """
    cfg = GameConfig()
    for k, v in meta_dict.items():
        setattr(cfg, k, v)
    # Ensure derived fields exist (if not already)
    if not hasattr(cfg, "dx"):
        cfg.dx = cfg.dx1 + cfg.dx2
    if not hasattr(cfg, "tau") and hasattr(cfg, "T") and hasattr(cfg, "K"):
        cfg.tau = cfg.T / cfg.K
    cfg.device_resolved = torch.device(getattr(cfg, "device", "cpu"))
    if not hasattr(cfg, "dtype"):
        cfg.dtype = torch.float32
    return cfg


def _rebuild_training_config(meta_dict) -> TrainingConfig:
    cfg = TrainingConfig()
    for k, v in meta_dict.items():
        setattr(cfg, k, v)
    return cfg


def main() -> None:
    args = parse_args()

    # 1) Load checkpoint payload ONCE just to get metadata, without touching α.
    import torch
    from MPC_2p0s1.outer_opt.checkpointing import CheckpointMeta, load_checkpoint

    payload = torch.load(args.checkpoint, map_location="cpu")
    meta_dict = payload.get("meta", {})

    meta = CheckpointMeta(
        step=int(meta_dict.get("step", 0)),
        loss=float(meta_dict.get("loss", 0.0)),
        game_config=meta_dict.get("game_config", {}),
        training_config=meta_dict.get("training_config", {}),
    )

    # 2) Rebuild configs from stored metadata
    game_cfg = _rebuild_game_config(meta.game_config)
    train_cfg = _rebuild_training_config(meta.training_config)

    # Allow manual device override
    if args.device is not None:
        game_cfg.device = args.device
    game_cfg.device_resolved = torch.device(game_cfg.device)

    # 3) Build indexer and AlphaParam with the CORRECT I and K from game_cfg
    indexer = FullIaryTreeIndexer(I=game_cfg.I, K=game_cfg.K)
    alpha_module = AlphaParam(
        indexer=indexer,
        alpha_cfg=AlphaParamConfig(),
        dtype=game_cfg.dtype,
        device=game_cfg.device_resolved,
    )

    # 4) Now load α/logits state into the correctly-shaped module
    meta, _ = load_checkpoint(
        ckpt_path=args.checkpoint,
        alpha_module=alpha_module,
        optimizer=None,
        map_location=game_cfg.device_resolved,
    )

    # 5) Rebuild game and proceed as before
    hexner_params = HexnerParams()
    game = HexnerGame(cfg=game_cfg, params=hexner_params)

    action_space = BoxActionSpace.from_config(game_cfg)

    loss, details = primal_objective(
        game=game,
        alpha_module=alpha_module,
        indexer=indexer,
        x0=None,
        p0=None,
        action_space=action_space,
        return_details=True,
    )
    print(f"[eval_hexner_rollouts] Loaded checkpoint with loss={float(loss.item()):.6f}")

    alpha = details["alpha"]
    belief_tree = details["belief_tree"]
    riccati_solution = details["riccati_solution"]

    # Generate rollouts for each type
    rollouts: List[RolloutResult] = []
    gen = torch.Generator(device=game_cfg.device_resolved)
    gen.manual_seed(game_cfg.seed)

    for type_index in range(game_cfg.I):
        for _ in range(args.num_rollouts_per_type):
            ro = rollout_trajectory(
                game=game,
                indexer=indexer,
                belief_tree=belief_tree,
                riccati_sol=riccati_solution,
                alpha=alpha,
                x0=game.default_initial_state(),
                type_index=type_index,
                action_space=action_space,
                sample_actions=args.sample_actions,
                generator=gen,
            )
            rollouts.append(ro)

    # Resolve output_dir relative to inner package root
    output_dir = project_relative(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Make a simple visual report (no training losses by default).
    make_hexner_report(
        output_dir=output_dir,
        game=game,
        rollouts=rollouts,
        training_losses=None,
        training_iterations=None,
        add_titles=True,
    )

    print(f"[eval_hexner_rollouts] Saved figures to '{output_dir}'")


if __name__ == "__main__":
    main()