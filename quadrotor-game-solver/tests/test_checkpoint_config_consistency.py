from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_EVAL_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
_EVAL_SPEC = importlib.util.spec_from_file_location("qgs_eval_script", _EVAL_MOD_PATH)
assert _EVAL_SPEC is not None and _EVAL_SPEC.loader is not None
_EVAL_MOD = importlib.util.module_from_spec(_EVAL_SPEC)
_EVAL_SPEC.loader.exec_module(_EVAL_MOD)
load_checkpoint_and_config = _EVAL_MOD.load_checkpoint_and_config


def test_full_checkpoint_config_is_canonical(tmp_path: Path) -> None:
    ckpt_cfg = {
        "T": 1.0,
        "K": 3,
        "I": 2,
        "integrator": "rk4",
        "control_cost_mode": "hover_relative",
        "line_search_accept_worse": False,
        "dtype": "torch.float64",
        "device": "cpu",
        "R1_diag": [0.05, 0.025, 0.025, 0.01],
        "R2_diag": [0.05, 0.10, 0.10, 0.02],
        "K1_scale": 10.0,
        "K2_scale": 10.0,
        "theta_values": [-1.0, 1.0],
    }
    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "config": ckpt_cfg,
            "alpha_state_dict": {"dummy": torch.tensor([1.0])},
        },
        ckpt_path,
    )

    bogus_cfg = dict(ckpt_cfg)
    bogus_cfg["K"] = 99
    bogus_path = tmp_path / "bogus_config.json"
    with open(bogus_path, "w") as f:
        json.dump(bogus_cfg, f)

    args = SimpleNamespace(
        checkpoint=str(ckpt_path),
        config_json=str(bogus_path),
    )
    cfg, payload, cfg_meta = load_checkpoint_and_config(args)

    assert cfg.K == 3
    assert int(cfg_meta["K"]) == 3
    assert "alpha_state_dict" in payload
