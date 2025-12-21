# outer_opt/optimizer_loop.py
from __future__ import annotations

import os
from typing import Optional

import torch

from ..config.base_config import GameConfig, TrainingConfig, PathsConfig
from ..core.utils import set_random_seeds
from ..core.action_spaces import BoxActionSpace
from ..games.base_lq_game import BaseLQGame
from ..tree.indexing import FullIaryTreeIndexer
from .alpha_param import AlphaParam
from .checkpointing import save_checkpoint
from .depth_schedule import DepthSchedule, make_default_depth_schedule
from .objective_primal import primal_objective


def run_outer_optimization(
    game: BaseLQGame,
    game_cfg: GameConfig,
    train_cfg: TrainingConfig,
    paths_cfg: PathsConfig,
    indexer: FullIaryTreeIndexer,
    alpha_module: AlphaParam,
    action_space: Optional[BoxActionSpace] = None,
    x0: Optional[torch.Tensor] = None,
    p0: Optional[torch.Tensor] = None,
    depth_schedule: Optional[DepthSchedule] = None,
    resume_from: Optional[str] = None,
) -> None:
    """
    Run the outer optimization loop over α/logits for the primal game.

    Parameters
    ----------
    game:
        LQ game instance.
    game_cfg:
        GameConfig used to construct `game`.
    train_cfg:
        TrainingConfig governing optimizer, learning rate, and schedule.
    paths_cfg:
        PathsConfig for logging and checkpointing.
    indexer:
        Tree indexer specifying the I-ary structure of the public game tree.
    alpha_module:
        AlphaParam instance whose parameters are optimized.
    action_space:
        Optional BoxActionSpace to be passed through to the objective
        (no effect on the Riccati recursion itself).
    x0:
        Initial state; if None, uses game.default_initial_state().
    p0:
        Initial prior; if None, uses game.default_prior().
    depth_schedule:
        Optional DepthSchedule. If None and train_cfg.staged_depth is
        True, a default schedule will be constructed.
    resume_from:
        Optional path to a checkpoint from which to resume. If provided,
        alpha_module and optimizer states are restored and iteration
        continues from `meta.step + 1`.
    """
    device = game.device_resolved

    # Seeding and directories
    set_random_seeds(game_cfg.seed)
    paths_cfg.ensure_exists()

    # Optimizer
    optimizer = train_cfg.make_optimizer(list(alpha_module.parameters()))
    start_step = 0

    # Depth schedule
    if depth_schedule is None and train_cfg.staged_depth:
        depth_schedule = make_default_depth_schedule(train_cfg, indexer)

    # Resume from checkpoint if requested
    if resume_from is not None and os.path.exists(resume_from):
        from .checkpointing import load_checkpoint

        meta, optimizer = load_checkpoint(
            resume_from,
            alpha_module=alpha_module,
            optimizer=optimizer,
            map_location=device,
        )
        start_step = meta.step + 1
        print(
            f"[outer_opt] Resumed from checkpoint '{resume_from}' at step={meta.step}, "
            f"loss={meta.loss:.6f}"
        )

    use_amp = train_cfg.use_mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Main optimization loop
    for step in range(start_step, train_cfg.max_iters):
        def closure_lbfgs():
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=False):  # LBFGS + AMP is tricky; keep FP32.
                loss_val = primal_objective(
                    game=game,
                    alpha_module=alpha_module,
                    indexer=indexer,
                    x0=x0,
                    p0=p0,
                    action_space=action_space,
                    return_details=False,
                )
            loss_val.backward()

            # Apply depth mask if any
            if depth_schedule is not None and alpha_module.logits.grad is not None:
                mask = depth_schedule.make_mask(alpha_module.logits, iteration=step)
                alpha_module.logits.grad *= mask

            if train_cfg.grad_clip_norm is not None and train_cfg.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    alpha_module.parameters(),
                    max_norm=train_cfg.grad_clip_norm,
                )
            return loss_val

        optimizer_name = train_cfg.optimizer.lower()

        if optimizer_name == "lbfgs":
            loss = optimizer.step(closure_lbfgs)
        else:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = primal_objective(
                    game=game,
                    alpha_module=alpha_module,
                    indexer=indexer,
                    x0=x0,
                    p0=p0,
                    action_space=action_space,
                    return_details=False,
                )

            scaler.scale(loss).backward()

            # Apply depth mask if any
            if depth_schedule is not None and alpha_module.logits.grad is not None:
                mask = depth_schedule.make_mask(alpha_module.logits, iteration=step)
                alpha_module.logits.grad *= mask

            # Gradient clipping
            if train_cfg.grad_clip_norm is not None and train_cfg.grad_clip_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    alpha_module.parameters(),
                    max_norm=train_cfg.grad_clip_norm,
                )

            scaler.step(optimizer)
            scaler.update()

        loss_val = float(loss.item())

        # Logging to stdout
        if (step % train_cfg.print_every) == 0 or step == train_cfg.max_iters - 1:
            if depth_schedule is not None:
                cur_depth = depth_schedule.current_unfrozen_depth(step)
                depth_info = f", unfrozen_depth≤{cur_depth}"
            else:
                depth_info = ""
            print(
                f"[outer_opt] step={step:06d} loss={loss_val:.6f}"
                f"{depth_info}"
            )

        # Checkpointing
        if (step % train_cfg.checkpoint_every) == 0 or step == train_cfg.max_iters - 1:
            ckpt_path = save_checkpoint(
                paths_cfg=paths_cfg,
                step=step,
                loss=loss_val,
                alpha_module=alpha_module,
                optimizer=optimizer,
                game_cfg=game_cfg,
                train_cfg=train_cfg,
            )
            print(f"[outer_opt] Saved checkpoint to '{ckpt_path}'")