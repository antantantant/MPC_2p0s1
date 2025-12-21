# viz/make_report_hexner.py
from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from ..games.hexner_game import HexnerGame
from ..tree.rollout import RolloutResult
from .plot_training_curves import plot_training_curve
from .plot_trajectories import plot_hexner_trajectory_2d
from .plot_beliefs import plot_belief_trajectory


def make_hexner_report(
    output_dir: str,
    game: HexnerGame,
    rollouts: Sequence[RolloutResult],
    training_losses: Optional[Sequence[float]] = None,
    training_iterations: Optional[Sequence[int]] = None,
    add_titles: bool = True,
) -> None:
    """
    Generate a simple visual report for Hexner’s game.

    The report consists of:
    - A training loss curve (if losses are provided).
    - For each rollout:
        * A 2D state trajectory plot (P1 and P2 positions, targets, belief shading).
        * A belief trajectory plot over time.

    All figures are saved as PNG files in `output_dir`.

    Parameters
    ----------
    output_dir:
        Directory into which figures are written. Created if it does not exist.
    game:
        HexnerGame instance (used for horizon T, targets, etc.).
    rollouts:
        Sequence of RolloutResult objects to visualize.
    training_losses:
        Optional sequence of scalar losses per iteration; if provided, a
        "training_loss.png" figure is saved.
    training_iterations:
        Optional sequence of iteration indices sharing the same length as
        `training_losses`. If None, uses range(len(training_losses)).
    add_titles:
        If True, add simple titles to the generated plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Training loss curve
    if training_losses is not None:
        fig, ax = plt.subplots()
        plot_training_curve(
            values=training_losses,
            iterations=training_iterations,
            ax=ax,
            label="loss",
            ylabel="Loss",
            xlabel="Iteration",
        )
        if add_titles:
            ax.set_title("Training loss over iterations")
        fig.tight_layout()
        fig_path = os.path.join(output_dir, "training_loss.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

    # 2) Per-rollout trajectory and belief plots
    for idx, rollout in enumerate(rollouts):
        # 2a) 2D trajectory
        fig_traj, ax_traj = plt.subplots()
        plot_hexner_trajectory_2d(
            game=game,
            rollout=rollout,
            ax=ax_traj,
            show_targets=True,
            show_belief_colormap=True,
            title=(
                f"Hexner trajectory #{idx}"
                if add_titles
                else None
            ),
        )
        fig_traj.tight_layout()
        traj_path = os.path.join(output_dir, f"hexner_traj_{idx:03d}.png")
        fig_traj.savefig(traj_path, dpi=150)
        plt.close(fig_traj)

        # 2b) Belief trajectory
        fig_bel, ax_bel = plt.subplots()
        plot_belief_trajectory(
            belief_traj=rollout.belief_traj,
            T=game.cfg.T,
            ax=ax_bel,
            labels=None,
            title=(
                f"Belief trajectory #{idx}"
                if add_titles
                else None
            ),
        )
        fig_bel.tight_layout()
        bel_path = os.path.join(output_dir, f"hexner_belief_{idx:03d}.png")
        fig_bel.savefig(bel_path, dpi=150)
        plt.close(fig_bel)