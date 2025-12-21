# config/hexner_config.py
from __future__ import annotations

from dataclasses import dataclass

from .base_config import GameConfig, TrainingConfig, PathsConfig


@dataclass
class HexnerProblemScales:
    """
    Convenience container for Hexner-specific scaling choices.

    These are not strictly required by the solver, but they make it easier to keep
    simulation parameters for the Hexner test case in one place.

    Attributes
    ----------
    T:
        Time horizon in seconds.
    K:
        Number of discrete time steps; τ = T / K.
    max_accel:
        Magnitude bound for accelerations in both x- and y-directions (for both players).
    """

    T: float = 1.0
    K: int = 10
    max_accel: float = 4.0


def make_hexner_game_config(
    scales: HexnerProblemScales | None = None,
    device: str = "auto",
) -> GameConfig:
    """
    Create a `GameConfig` for the 2D Hexner game used throughout the experiments.

    The Hexner test case in the paper models each player as a 2D point mass with
    position and velocity, so each player has a 4-dimensional state (x, y, vx, vy)
    and a 2-dimensional control (ax, ay). The informed player has two payoff types
    corresponding to two possible target locations.  [oai_citation:1‡16010_Solving_Football_by_Expl-2.pdf](sediment://file_000000001b0871f58cd1c91b483083b4)

    Parameters
    ----------
    scales:
        Optional `HexnerProblemScales` specifying horizon, discretization, and
        control bounds. If None, reasonable defaults are used.
    device:
        Desired compute device for tensors associated with this game
        ("cpu", "cuda", "auto", or any torch device string).

    Returns
    -------
    GameConfig
        A game configuration suitable for constructing the Hexner dynamics and costs.
    """
    if scales is None:
        scales = HexnerProblemScales()

    return GameConfig(
        I=2,  # two payoff types / target choices
        dx1=4,
        dx2=4,
        du=2,
        dv=2,
        T=scales.T,
        K=scales.K,
        game_name="hexner",
        device=device,
        # Control bounds are symmetric and moderately loose; they can be overridden
        # if you want to reproduce a specific experimental setup exactly.
        u_min=-scales.max_accel,
        u_max=scales.max_accel,
        v_min=-scales.max_accel,
        v_max=scales.max_accel,
    )


def make_hexner_training_config() -> TrainingConfig:
    """
    Create a `TrainingConfig` with defaults tuned for small Hexner test runs.

    The defaults are conservative and intended for development on a laptop;
    for large-scale experiments on an H100 you can safely increase `max_iters`
    and/or decrease `learning_rate` as needed.
    """
    return TrainingConfig(
        optimizer="adam",
        learning_rate=5e-3,
        weight_decay=0.0,
        max_iters=3_000,
        grad_clip_norm=10.0,
        print_every=50,
        log_every=10,
        checkpoint_every=250,
        staged_depth=True,
        initial_unfrozen_depth=0,
        depth_increment=1,
        depth_increment_every=400,
        max_unfrozen_depth=None,
        use_mixed_precision=False,
    )


def make_hexner_paths_config(run_name: str = "hexner_primal") -> PathsConfig:
    """
    Create a `PathsConfig` for Hexner experiments.

    Parameters
    ----------
    run_name:
        Name of the run; a subdirectory with this name is created under ./runs.

    Returns
    -------
    PathsConfig
        A paths configuration for logging, checkpoints, and figures.
    """
    return PathsConfig(root_dir="./runs_hexner", run_name=run_name, create_subdir=True)