# scripts/train_hexner_mod_amortized_ddp.py
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency in some environments
    np = None

from MPC_2p0s1.config.base_config import (
    GameConfig,
    PathsConfig,
    project_relative,
)
from MPC_2p0s1.core.action_spaces import BoxActionSpace
from MPC_2p0s1.core.utils import count_parameters, set_random_seeds
from MPC_2p0s1.games.hexner_mod_game import HexnerModGame, HexnerModParams
from MPC_2p0s1.outer_opt.amortized_alpha import (
    AmortizedAlphaConfig,
    AmortizedAlphaParam,
    amortized_batch_objective,
    primal_objective_from_alpha,
)
from MPC_2p0s1.tree.indexing import FullIaryTreeIndexer


@dataclass
class DistContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str = ""

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _parse_hidden_dims(spec: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("hidden-dims must contain at least one integer, e.g. '256,256'")
    dims = tuple(int(p) for p in parts)
    if any(d <= 0 for d in dims):
        raise ValueError(f"All hidden dims must be positive; got {dims}")
    return dims


def _main_print(ctx: DistContext, msg: str) -> None:
    if ctx.is_main:
        print(msg, flush=True)


def _unwrap_model(model: AmortizedAlphaParam | DDP) -> AmortizedAlphaParam:
    if isinstance(model, DDP):
        return model.module
    return model


def _resolve_ddp_device(device_arg: str, local_rank: int, world_size: int) -> torch.device:
    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"--device={device_arg} requested but CUDA is not available in this environment."
            )
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            return torch.device(f"cuda:{local_rank}")
        if device_arg == "cuda":
            return torch.device("cuda")
        return torch.device(device_arg)
    return torch.device("cpu")


def _setup_distributed(device_arg: str, backend_arg: str) -> tuple[DistContext, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = _resolve_ddp_device(device_arg=device_arg, local_rank=local_rank, world_size=world_size)

    ctx = DistContext(rank=rank, world_size=world_size, local_rank=local_rank, backend="")
    if world_size <= 1:
        return ctx, device

    backend = backend_arg.lower()
    if backend == "auto":
        backend = "nccl" if device.type == "cuda" else "gloo"
    if backend == "nccl" and device.type != "cuda":
        raise ValueError("NCCL backend requires CUDA devices.")

    dist.init_process_group(backend=backend, init_method="env://")
    ctx.backend = backend

    # Keep all ranks synchronized before entering the main loop.
    dist.barrier()
    return ctx, device


def _cleanup_distributed(ctx: DistContext) -> None:
    if ctx.is_distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _reduce_scalar_mean(x: torch.Tensor, ctx: DistContext) -> float:
    value = x.detach()
    if value.ndim != 0:
        value = value.mean()
    if ctx.is_distributed:
        value = value.clone()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / float(ctx.world_size)
    return float(value.cpu().item())


def _save_checkpoint(
    run_dir: str,
    step: int,
    train_loss: float,
    eval_loss: float,
    val_loss: Optional[float],
    model: AmortizedAlphaParam,
    optimizer: torch.optim.Optimizer,
    game_cfg: GameConfig,
    args: argparse.Namespace,
) -> str:
    path = os.path.join(run_dir, f"checkpoint_step_{step:06d}.pt")
    payload = {
        "meta": {
            "step": int(step),
            "train_loss": float(train_loss),
            "eval_loss": float(eval_loss),
            "val_loss": (float(val_loss) if val_loss is not None else None),
            "game_config": game_cfg.as_dict(),
            "args": vars(args),
        },
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, path)

    latest = os.path.join(run_dir, "latest.pt")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(os.path.basename(path), latest)
    except OSError:
        torch.save(payload, latest)

    return path


def _maybe_load_resume(
    resume_from: Optional[str],
    model: AmortizedAlphaParam,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_prefix: str,
) -> int:
    if resume_from is None:
        return 0
    if not os.path.exists(resume_from):
        raise FileNotFoundError(f"resume checkpoint not found: {resume_from}")

    payload = torch.load(resume_from, map_location=device)
    model_state = payload.get("model_state_dict", {})
    model.load_state_dict(model_state)

    opt_state = payload.get("optimizer_state_dict")
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)

    meta = payload.get("meta", {})
    start_step = int(meta.get("step", -1)) + 1
    print(
        f"[{log_prefix}] Resumed from '{resume_from}' "
        f"at step={start_step - 1}.",
        flush=True,
    )
    return max(0, start_step)


def _sample_context_batch(
    game: HexnerModGame,
    batch_size: int,
    prior_min: float,
    p1_pos_jitter: float,
    p2_pos_jitter: float,
    default_context_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a mini-batch of initial states and priors around the fixed game setup.
    """
    device = game.device_resolved
    dtype = game.dtype
    I = game.I
    dx = game.dx

    x_ref = game.default_initial_state().to(device=device, dtype=dtype)
    p_ref = game.default_prior().to(device=device, dtype=dtype)

    x0 = x_ref.view(1, dx).repeat(batch_size, 1)
    if dx != 8:
        raise ValueError(
            f"Hexner-mod amortized sampler expects dx=8 (got dx={dx}). "
            "Please update sampling indices for your game dimensions."
        )

    # P1: [px, py, vx, vy], P2: [px, py, vx, vy]
    x0[:, 0:2] += (2.0 * torch.rand(batch_size, 2, device=device, dtype=dtype) - 1.0) * p1_pos_jitter
    x0[:, 4:6] += (2.0 * torch.rand(batch_size, 2, device=device, dtype=dtype) - 1.0) * p2_pos_jitter
    # Keep initial velocities fixed at default (Hexner defaults are zero velocity).
    x0[:, 2:4] = x_ref[2:4]
    x0[:, 6:8] = x_ref[6:8]

    if I == 2:
        lo = float(prior_min)
        hi = 1.0 - lo
        p_type0 = lo + (hi - lo) * torch.rand(batch_size, device=device, dtype=dtype)
        p0 = torch.stack([p_type0, 1.0 - p_type0], dim=-1)
    else:
        raw = torch.rand(batch_size, I, device=device, dtype=dtype).clamp_min(1e-6)
        p0 = raw / raw.sum(dim=-1, keepdim=True)

    if default_context_prob > 0.0:
        mask = torch.rand(batch_size, device=device) < float(default_context_prob)
        if mask.any():
            x0[mask] = x_ref
            p0[mask] = p_ref

    return x0, p0


def _build_fixed_validation_contexts(
    game: HexnerModGame,
    num_val_states: int,
    p1_pos_jitter: float,
    p2_pos_jitter: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a deterministic validation set with fixed initial states and default prior.
    """
    if num_val_states <= 0:
        raise ValueError(f"num_val_states must be positive, got {num_val_states}.")

    device = game.device_resolved
    dtype = game.dtype
    dx = game.dx

    x_ref = game.default_initial_state().to(device=device, dtype=dtype)
    p_ref = game.default_prior().to(device=device, dtype=dtype)

    x0 = x_ref.view(1, dx).repeat(num_val_states, 1)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    n1 = (2.0 * torch.rand(num_val_states, 2, generator=g) - 1.0) * float(p1_pos_jitter)
    n2 = (2.0 * torch.rand(num_val_states, 2, generator=g) - 1.0) * float(p2_pos_jitter)
    x0[:, 0:2] += n1.to(device=device, dtype=dtype)
    x0[:, 4:6] += n2.to(device=device, dtype=dtype)

    # Keep velocity at default values.
    x0[:, 2:4] = x_ref[2:4]
    x0[:, 6:8] = x_ref[6:8]

    p0 = p_ref.view(1, -1).repeat(num_val_states, 1)
    return x0, p0


def build_configs_from_args(args: argparse.Namespace, device: torch.device):
    game_cfg = GameConfig()
    game_cfg.I = int(args.I)
    game_cfg.dx1 = 4
    game_cfg.dx2 = 4
    game_cfg.dx = game_cfg.dx1 + game_cfg.dx2
    game_cfg.du = 2
    game_cfg.dv = 2
    game_cfg.T = float(args.T)
    game_cfg.K = int(args.K)
    game_cfg.tau = game_cfg.T / game_cfg.K
    game_cfg.device = str(device)
    game_cfg.device_resolved = device
    game_cfg.dtype = torch.float32
    game_cfg.seed = int(args.seed)
    game_cfg.u_min = float(args.u_min)
    game_cfg.u_max = float(args.u_max)
    game_cfg.v_min = float(args.v_min)
    game_cfg.v_max = float(args.v_max)

    run_root = project_relative(args.run_dir)
    paths_cfg = PathsConfig(root_dir=run_root, run_name="", create_subdir=False)
    paths_cfg.ensure_exists()
    return game_cfg, paths_cfg


def build_hexner_mod_params_from_args(args: argparse.Namespace) -> HexnerModParams:
    theta_vals = tuple(float(v) for v in args.theta_values.split(","))
    return HexnerModParams(
        theta_values=theta_vals,
        target_z=None,
        R1_scale=float(args.R1_scale),
        R2_scale=float(args.R2_scale),
        K1_scale=float(args.K1_scale),
        K2_scale=float(args.K2_scale),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DDP training for amortized alpha model f(x0, p0) -> logits for Hexner-mod. "
            "Launch multi-GPU with torchrun."
        )
    )

    # Game
    parser.add_argument("--I", type=int, default=2)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--theta-values", type=str, default="-1.0,1.0")
    parser.add_argument("--R1-scale", type=float, default=1.0)
    parser.add_argument("--R2-scale", type=float, default=0.4)
    parser.add_argument("--K1-scale", type=float, default=1.0)
    parser.add_argument("--K2-scale", type=float, default=1.0)
    parser.add_argument("--u-min", type=float, default=-14.0)
    parser.add_argument("--u-max", type=float, default=14.0)
    parser.add_argument("--v-min", type=float, default=-14.0)
    parser.add_argument("--v-max", type=float, default=14.0)

    # Amortized network
    parser.add_argument("--hidden-dims", type=str, default="256,256")
    parser.add_argument(
        "--activation",
        type=str,
        default="silu",
        choices=["relu", "gelu", "silu", "tanh"],
    )
    parser.add_argument("--dropout", type=float, default=0.0)

    # Optimization
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-rank batch size. Global batch size = batch_size * world_size.",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument(
        "--num-val-states",
        type=int,
        default=16,
        help="Number of fixed validation initial states (evaluated each checkpoint, rank0 only).",
    )
    parser.add_argument(
        "--val-seed",
        type=int,
        default=2026,
        help="Seed used to generate the fixed validation initial-state set.",
    )

    # Context sampling
    parser.add_argument(
        "--prior-min",
        type=float,
        default=0.05,
        help="Minimum probability for each type when I=2 (sample in [prior_min, 1-prior_min]).",
    )
    parser.add_argument("--p1-pos-jitter", type=float, default=0.40)
    parser.add_argument("--p2-pos-jitter", type=float, default=0.40)
    parser.add_argument(
        "--default-context-prob",
        type=float,
        default=0.15,
        help="Fraction of each mini-batch forced to default (x0, p0).",
    )

    # DDP / Misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "nccl", "gloo"],
        help="Distributed backend when WORLD_SIZE > 1.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--run-dir", type=str, default="runs/hexner_mod_amortized_ddp")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dist_ctx, device = _setup_distributed(device_arg=args.device, backend_arg=args.backend)
    try:
        game_cfg, paths_cfg = build_configs_from_args(args=args, device=device)
        params = build_hexner_mod_params_from_args(args)

        if not (0.0 <= args.default_context_prob <= 1.0):
            raise ValueError("--default-context-prob must be in [0, 1].")
        if not (0.0 <= args.prior_min < 0.5):
            raise ValueError("--prior-min must satisfy 0 <= prior-min < 0.5 for I=2.")
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive.")

        # Rank-offset seed avoids identical random batches across processes.
        set_random_seeds(int(args.seed) + int(dist_ctx.rank))
        use_amp = bool(args.mixed_precision) and device.type == "cuda"
        scaler = torch.amp.GradScaler(enabled=use_amp)

        game = HexnerModGame(cfg=game_cfg, params=params, prior=None)
        indexer = FullIaryTreeIndexer(I=game_cfg.I, K=game_cfg.K)
        action_space = BoxActionSpace.from_config(game_cfg)

        model_cfg = AmortizedAlphaConfig(
            hidden_dims=_parse_hidden_dims(args.hidden_dims),
            activation=args.activation,
            dropout=float(args.dropout),
        )
        base_model = AmortizedAlphaParam(
            indexer=indexer,
            dx=game_cfg.dx,
            I=game_cfg.I,
            cfg=model_cfg,
            dtype=game_cfg.dtype,
            device=device,
        )

        if args.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                base_model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
            )
        else:
            optimizer = torch.optim.AdamW(
                base_model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
            )

        start_step = _maybe_load_resume(
            resume_from=args.resume_from,
            model=base_model,
            optimizer=optimizer,
            device=device,
            log_prefix="train_hexner_mod_amortized_ddp",
        )

        if dist_ctx.is_distributed:
            if device.type == "cuda":
                model: AmortizedAlphaParam | DDP = DDP(
                    base_model,
                    device_ids=[device.index],
                    output_device=device.index,
                )
            else:
                model = DDP(base_model)
        else:
            model = base_model

        _main_print(
            dist_ctx,
            (
                "[train_hexner_mod_amortized_ddp] "
                f"K={game_cfg.K}, I={game_cfg.I}, "
                f"batch_size(local)={args.batch_size}, "
                f"batch_size(global)={args.batch_size * dist_ctx.world_size}, "
                f"params={count_parameters(base_model):,}, "
                f"device={device}, world_size={dist_ctx.world_size}, "
                f"run_dir='{paths_cfg.run_dir}'"
            ),
        )

        default_x0 = game.default_initial_state().to(device=device, dtype=game_cfg.dtype)
        default_p0 = game.default_prior().to(device=device, dtype=game_cfg.dtype)

        val_x0 = None
        val_p0 = None
        if dist_ctx.is_main and int(args.num_val_states) > 0:
            val_x0, val_p0 = _build_fixed_validation_contexts(
                game=game,
                num_val_states=int(args.num_val_states),
                p1_pos_jitter=float(args.p1_pos_jitter),
                p2_pos_jitter=float(args.p2_pos_jitter),
                seed=int(args.val_seed),
            )
            _main_print(
                dist_ctx,
                (
                    "[train_hexner_mod_amortized_ddp] "
                    f"Validation set: {int(args.num_val_states)} fixed initial states "
                    "(default prior, rank0 eval)."
                ),
            )

        steps_hist = []
        train_hist = []
        eval_hist = []
        total_times_hist = []
        val_steps_hist = []
        val_hist = []

        t0 = time.perf_counter()
        pbar = tqdm(
            range(start_step, int(args.max_iters)),
            disable=not dist_ctx.is_main,
        )
        for step in pbar:
            iter_t0 = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)

            x0_batch, p0_batch = _sample_context_batch(
                game=game,
                batch_size=int(args.batch_size),
                prior_min=float(args.prior_min),
                p1_pos_jitter=float(args.p1_pos_jitter),
                p2_pos_jitter=float(args.p2_pos_jitter),
                default_context_prob=float(args.default_context_prob),
            )

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                train_loss = amortized_batch_objective(
                    game=game,
                    alpha_model=model,
                    indexer=indexer,
                    x0_batch=x0_batch,
                    p0_batch=p0_batch,
                    action_space=action_space,
                    reduction="mean",
                )

            scaler.scale(train_loss).backward()
            if args.grad_clip_norm is not None and float(args.grad_clip_norm) > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    _unwrap_model(model).parameters(), max_norm=float(args.grad_clip_norm)
                )
            scaler.step(optimizer)
            scaler.update()

            model.eval()
            with torch.no_grad():
                alpha_default = _unwrap_model(model).single_alpha(x0=default_x0, p0=default_p0)
                eval_loss = primal_objective_from_alpha(
                    game=game,
                    alpha=alpha_default,
                    indexer=indexer,
                    x0=default_x0,
                    p0=default_p0,
                    action_space=action_space,
                    return_details=False,
                )

            train_val = _reduce_scalar_mean(train_loss, dist_ctx)
            eval_val = _reduce_scalar_mean(eval_loss, dist_ctx)
            iter_t1 = time.perf_counter()
            iter_ms = 1e3 * (iter_t1 - iter_t0)

            if dist_ctx.is_main:
                steps_hist.append(step)
                train_hist.append(train_val)
                eval_hist.append(eval_val)
                total_times_hist.append(iter_t1 - iter_t0)

                if step % int(args.print_every) == 0 or step == int(args.max_iters) - 1:
                    print(
                        f"[train_hexner_mod_amortized_ddp] step={step:06d} "
                        f"train_loss={train_val:.6f} "
                        f"default_loss={eval_val:.6f} "
                        f"iter={iter_ms:.1f}ms",
                        flush=True,
                    )

            if step % int(args.checkpoint_every) == 0 or step == int(args.max_iters) - 1:
                val_loss_val: Optional[float] = None
                if dist_ctx.is_main and val_x0 is not None and val_p0 is not None:
                    with torch.no_grad():
                        val_loss_t = amortized_batch_objective(
                            game=game,
                            alpha_model=model,
                            indexer=indexer,
                            x0_batch=val_x0,
                            p0_batch=val_p0,
                            action_space=action_space,
                            reduction="mean",
                        )
                    val_loss_val = float(val_loss_t.detach().cpu().item())
                    val_steps_hist.append(step)
                    val_hist.append(val_loss_val)

                if dist_ctx.is_main:
                    ckpt = _save_checkpoint(
                        run_dir=paths_cfg.run_dir,
                        step=step,
                        train_loss=train_val,
                        eval_loss=eval_val,
                        val_loss=val_loss_val,
                        model=_unwrap_model(model),
                        optimizer=optimizer,
                        game_cfg=game_cfg,
                        args=args,
                    )
                    if val_loss_val is None:
                        print(
                            f"[train_hexner_mod_amortized_ddp] Saved checkpoint to '{ckpt}'",
                            flush=True,
                        )
                    else:
                        print(
                            f"[train_hexner_mod_amortized_ddp] Saved checkpoint to '{ckpt}' "
                            f"(val_loss={val_loss_val:.6f})",
                            flush=True,
                        )
                if dist_ctx.is_distributed:
                    dist.barrier()

        t1 = time.perf_counter()
        elapsed = t1 - t0
        _main_print(
            dist_ctx,
            (
                f"[train_hexner_mod_amortized_ddp] Total training time: {elapsed:.2f} s "
                f"(iters={max(0, int(args.max_iters) - start_step)})"
            ),
        )

        if dist_ctx.is_main:
            if np is not None:
                metrics_path = os.path.join(paths_cfg.run_dir, "metrics_amortized.npz")
                np.savez(
                    metrics_path,
                    steps=np.asarray(steps_hist, dtype=np.int64),
                    train_losses=np.asarray(train_hist, dtype=np.float64),
                    default_losses=np.asarray(eval_hist, dtype=np.float64),
                    total_times=np.asarray(total_times_hist, dtype=np.float64),
                    val_steps=np.asarray(val_steps_hist, dtype=np.int64),
                    val_losses=np.asarray(val_hist, dtype=np.float64),
                )
            else:
                metrics_path = os.path.join(paths_cfg.run_dir, "metrics_amortized.pt")
                torch.save(
                    {
                        "steps": steps_hist,
                        "train_losses": train_hist,
                        "default_losses": eval_hist,
                        "total_times": total_times_hist,
                        "val_steps": val_steps_hist,
                        "val_losses": val_hist,
                    },
                    metrics_path,
                )
            print(
                f"[train_hexner_mod_amortized_ddp] Saved metrics to '{metrics_path}'",
                flush=True,
            )
    finally:
        _cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()
