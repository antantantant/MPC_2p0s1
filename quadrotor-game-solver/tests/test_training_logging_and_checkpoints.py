from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.game.quadrotor_game import Hexner3DQuadrotorGame
from src.solvers.action_spaces import BoxActionSpace
from src.tree.indexing import FullIaryTreeIndexer
from src.tree.signaling import AlphaParam, AlphaParamConfig
from src.utils.config import GameConfig
from src.utils.correctness import compute_config_fingerprint


_TRAIN_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
_TRAIN_SPEC = importlib.util.spec_from_file_location("qgs_train_script", _TRAIN_MOD_PATH)
assert _TRAIN_SPEC is not None and _TRAIN_SPEC.loader is not None
_TRAIN_MOD = importlib.util.module_from_spec(_TRAIN_SPEC)
_TRAIN_SPEC.loader.exec_module(_TRAIN_MOD)
run_training = _TRAIN_MOD.run_training


def test_training_logs_pre_post_losses_and_saves_checkpoints(tmp_path: Path) -> None:
    cfg = GameConfig(
        I=2,
        T=0.4,
        K=2,
        integrator="euler",
        dtype=torch.float64,
        device="cpu",
        control_cost_mode="hover_relative",
    )
    game = Hexner3DQuadrotorGame(cfg)
    indexer = FullIaryTreeIndexer(I=cfg.I, K=cfg.K)
    alpha_module = AlphaParam(
        indexer=indexer,
        cfg=AlphaParamConfig(init_scale=0.01),
        dtype=cfg.dtype,
        device=cfg.device,
    )

    u_lo = torch.full((game.du,), -20.0, dtype=cfg.dtype)
    u_hi = torch.full((game.du,), 20.0, dtype=cfg.dtype)
    v_lo = torch.full((game.dv,), -20.0, dtype=cfg.dtype)
    v_hi = torch.full((game.dv,), 20.0, dtype=cfg.dtype)
    action_space = BoxActionSpace(u_min=u_lo, u_max=u_hi, v_min=v_lo, v_max=v_hi)

    args = SimpleNamespace(
        lr=1e-3,
        epochs=2,
        sqp_iters=1,
        sqp_step_size=0.2,
        riccati_reg=0.1,
        sqp_verbose=False,
        sqp_early_stop=False,
        no_line_search=False,
        ls_alpha_min=0.01,
        ls_backtrack=0.5,
        ls_max_steps=2,
        allow_worse_step=False,
        grad_clip=0.0,
        log_every=1,
        save_every=1,
        eval_cold_start_every=1,
        gate_min_separation=0.0,
        gate_min_terminal_consistency=0.0,
        altitude_floor=-10.0,
    )

    config_snapshot = {
        "T": cfg.T,
        "K": cfg.K,
        "I": cfg.I,
        "integrator": cfg.integrator,
        "control_cost_mode": cfg.control_cost_mode,
        "line_search_accept_worse": cfg.line_search_accept_worse,
        "dtype": str(cfg.dtype),
        "device": cfg.device,
        "R1_diag": list(cfg.R1_diag),
        "R2_diag": list(cfg.R2_diag),
        "K1_scale": cfg.K1_scale,
        "K2_scale": cfg.K2_scale,
        "theta_values": list(cfg.theta_values),
    }
    config_fingerprint = compute_config_fingerprint(config_snapshot)
    config_snapshot["config_fingerprint"] = config_fingerprint

    run_dir = tmp_path / "train_smoke"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_training(
        game=game,
        indexer=indexer,
        alpha_module=alpha_module,
        action_space=action_space,
        x0=game.default_initial_state(),
        p0=game.default_prior(),
        args=args,
        run_dir=run_dir,
        config_snapshot=config_snapshot,
        config_fingerprint=config_fingerprint,
    )

    log_file = run_dir / "train.jsonl"
    assert log_file.exists()
    lines = log_file.read_text().strip().splitlines()
    assert len(lines) >= 1
    row = json.loads(lines[0])
    assert "pre_step_loss" in row
    assert "post_step_cold_start_loss" in row

    assert (run_dir / "final_checkpoint.pt").exists()
    assert (run_dir / "final_alpha.pt").exists()
    assert not (run_dir / "best_checkpoint.pt").exists()

    final_ckpt = torch.load(run_dir / "final_checkpoint.pt", map_location="cpu")
    assert final_ckpt["checkpoint_selection_policy"] == "last_checkpoint"
