# outer_opt/optimizer_loop.py
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

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
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Metrics history (for saving to metrics.npz)
    steps_hist = []
    losses_hist = []
    fwd_times_hist = []   # forward (LQ + loss assembly)
    bwd_times_hist = []   # backward (grad α + masking + clipping)
    total_times_hist = [] # full outer iteration

    # Main optimization loop
    for step in tqdm(range(start_step, train_cfg.max_iters)):
        step_t0 = time.perf_counter()
        fwd_time = 0.0
        bwd_time = 0.0

        def closure_lbfgs():
            nonlocal fwd_time, bwd_time
            optimizer.zero_grad(set_to_none=True)
            # LBFGS + AMP is tricky; keep FP32 here.
            with torch.cuda.amp.autocast(enabled=False):
                fwd_t0 = time.perf_counter()
                loss_val = primal_objective(
                    game=game,
                    alpha_module=alpha_module,
                    indexer=indexer,
                    x0=x0,
                    p0=p0,
                    action_space=action_space,
                    return_details=False,
                )
                fwd_t1 = time.perf_counter()
            fwd_time += (fwd_t1 - fwd_t0)

            bwd_t0 = time.perf_counter()
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
            bwd_t1 = time.perf_counter()
            bwd_time += (bwd_t1 - bwd_t0)

            return loss_val

        optimizer_name = train_cfg.optimizer.lower()

        if optimizer_name == "lbfgs":
            loss = optimizer.step(closure_lbfgs)
        else:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                fwd_t0 = time.perf_counter()
                loss = primal_objective(
                    game=game,
                    alpha_module=alpha_module,
                    indexer=indexer,
                    x0=x0,
                    p0=p0,
                    action_space=action_space,
                    return_details=False,
                )
                fwd_t1 = time.perf_counter()
            fwd_time = fwd_t1 - fwd_t0

            bwd_t0 = time.perf_counter()
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

            bwd_t1 = time.perf_counter()
            bwd_time = bwd_t1 - bwd_t0

            scaler.step(optimizer)
            scaler.update()

        step_t1 = time.perf_counter()
        total_time = step_t1 - step_t0

        loss_val = float(loss.item())

        # Record metrics
        steps_hist.append(step)
        losses_hist.append(loss_val)
        fwd_times_hist.append(fwd_time)
        bwd_times_hist.append(bwd_time)
        total_times_hist.append(total_time)

        # Logging to stdout
        if (step % train_cfg.print_every) == 0 or step == train_cfg.max_iters - 1:
            if depth_schedule is not None:
                cur_depth = depth_schedule.current_unfrozen_depth(step)
                depth_info = f", unfrozen_depth≤{cur_depth}"
            else:
                depth_info = ""
            print(
                "[outer_opt] step={step:06d} loss={loss:.6f} "
                "fwd={fwd_ms:.1f}ms bwd={bwd_ms:.1f}ms total={tot_ms:.1f}ms{depth_info}".format(
                    step=step,
                    loss=loss_val,
                    fwd_ms=1e3 * fwd_time,
                    bwd_ms=1e3 * bwd_time,
                    tot_ms=1e3 * total_time,
                    depth_info=depth_info,
                )
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

    # Save metrics at the end of the run
    metrics_path = os.path.join(paths_cfg.run_dir, "metrics.npz")
    np.savez(
        metrics_path,
        steps=np.asarray(steps_hist, dtype=np.int64),
        losses=np.asarray(losses_hist, dtype=np.float64),
        fwd_times=np.asarray(fwd_times_hist, dtype=np.float64),
        bwd_times=np.asarray(bwd_times_hist, dtype=np.float64),
        total_times=np.asarray(total_times_hist, dtype=np.float64),
    )
    print(f"[outer_opt] Saved timing/loss metrics to '{metrics_path}'")