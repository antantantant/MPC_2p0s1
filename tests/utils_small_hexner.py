# test/utils_small_hexner.py
from __future__ import annotations

from dataclasses import dataclass

import torch

from MPC_2p0s1.config.base_config import GameConfig
from MPC_2p0s1.games.hexner_game import HexnerGame, HexnerParams
from MPC_2p0s1.tree.indexing import FullIaryTreeIndexer
from MPC_2p0s1.outer_opt.alpha_param import AlphaParam, AlphaParamConfig
from MPC_2p0s1.core.action_spaces import BoxActionSpace


@dataclass
class SmallHexnerSetup:
    """
    Convenience bundle for a small Hexner test instance.

    Attributes
    ----------
    game:
        HexnerGame instance.
    cfg:
        GameConfig used to construct the game.
    indexer:
        FullIaryTreeIndexer with small (I, K).
    alpha_module:
        AlphaParam with logits initialized to zero (uniform α).
    action_space:
        BoxActionSpace derived from cfg.
    """

    game: HexnerGame
    cfg: GameConfig
    indexer: FullIaryTreeIndexer
    alpha_module: AlphaParam
    action_space: BoxActionSpace


def make_small_hexner_setup(
    I: int = 2,
    K: int = 2,
    T: float = 1.0,
    device: str = "cpu",
) -> SmallHexnerSetup:
    """
    Construct a small Hexner game configuration suitable for unit tests.

    Parameters
    ----------
    I:
        Number of payoff types.
    K:
        Number of time steps (tree depth).
    T:
        Time horizon.
    device:
        Device string, e.g., "cpu" or "cuda:0".

    Returns
    -------
    SmallHexnerSetup
        Bundle with game, config, indexer, α module, and action space.
    """
    dev = torch.device(device)

    # Base game config; we override relevant fields explicitly.
    cfg = GameConfig()
    cfg.I = I
    cfg.dx1 = 4
    cfg.dx2 = 4
    cfg.dx = cfg.dx1 + cfg.dx2
    cfg.du = 2
    cfg.dv = 2

    cfg.T = float(T)
    cfg.K = int(K)
    cfg.tau = cfg.T / cfg.K

    cfg.device = device
    cfg.device_resolved = dev
    cfg.dtype = torch.float32
    cfg.seed = 123

    # Simple symmetric box constraints on actions
    cfg.u_min = -4.0
    cfg.u_max = 4.0
    cfg.v_min = -4.0
    cfg.v_max = 4.0

    # Hexner game (minimal params)
    params = HexnerParams()
    game = HexnerGame(cfg=cfg, params=params)

    # Tree indexer
    indexer = FullIaryTreeIndexer(I=I, K=K)

    # α/logits parameterization – initialize exactly uniform.
    alpha_cfg = AlphaParamConfig(init_scale=0.01)
    alpha_module = AlphaParam(
        indexer=indexer,
        alpha_cfg=alpha_cfg,
        dtype=cfg.dtype,
        device=cfg.device_resolved,
    )
    # with torch.no_grad():
    #     alpha_module.logits.zero_()

    # Optional: for deterministic tests, you can set a seed here:
    torch.manual_seed(cfg.seed)

    # Box-constrained actions
    action_space = BoxActionSpace.from_config(cfg)

    return SmallHexnerSetup(
        game=game,
        cfg=cfg,
        indexer=indexer,
        alpha_module=alpha_module,
        action_space=action_space,
    )