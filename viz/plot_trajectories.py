# viz/plot_trajectories.py
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (imported for side effects)

from ..games.hexner_game import HexnerGame
from ..tree.rollout import RolloutResult


def _tensor_to_numpy(x) -> np.ndarray:
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_hexner_trajectory_2d(
    game: HexnerGame,
    rollout: RolloutResult,
    ax: Optional[plt.Axes] = None,
    show_targets: bool = True,
    show_belief_colormap: bool = True,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot a single Hexner trajectory in 2D (positions of P1 and P2).

    Parameters
    ----------
    game:
        HexnerGame instance (used for layout and targets).
    rollout:
        RolloutResult containing x_traj and belief_traj.
    ax:
        Optional matplotlib Axes. If None, a new figure and axes are created.
    show_targets:
        If True, overlay type-dependent targets zθ.
    show_belief_colormap:
        If True and I=2, color-code time points by public belief Pr[type=1].
    title:
        Optional title for the axes.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    x_traj = _tensor_to_numpy(rollout.x_traj)        # (K+1, dx)
    belief_traj = _tensor_to_numpy(rollout.belief_traj)  # (K+1, I)
    dx1 = game.cfg.dx1

    pos1 = x_traj[:, :2]          # P1 position (x, y)
    pos2 = x_traj[:, dx1 : dx1+2]  # P2 position (x, y)

    if ax is None:
        _, ax = plt.subplots()

    # Plot trajectories for P1 and P2
    ax.plot(pos1[:, 0], pos1[:, 1], marker="o", linestyle="-", label="P1")
    ax.plot(pos2[:, 0], pos2[:, 1], marker="s", linestyle="-", label="P2")

    # Belief shading (for I=2, show Pr[type=1] as color)
    if show_belief_colormap and belief_traj.shape[1] >= 2:
        probs_type1 = belief_traj[:, 1]
        norm = plt.Normalize(vmin=0.0, vmax=1.0)
        cmap = plt.cm.get_cmap("viridis")
        for k in range(len(pos1)):
            color = cmap(norm(probs_type1[k]))
            ax.scatter(pos1[k, 0], pos1[k, 1], color=color, s=50, alpha=0.8)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Pr[type = 1]")

    # Targets
    if show_targets:
        targets = _tensor_to_numpy(game.type_targets())  # (I, dx1)
        for i, target in enumerate(targets):
            ax.scatter(
                target[0],
                target[1],
                marker="*",
                s=120,
                label=f"Target θ={i}",
            )

    ax.set_xlabel("x-position")
    ax.set_ylabel("y-position")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    if title is not None:
        ax.set_title(title)

    return ax


def plot_hexner_trajectory_3d(
    game: HexnerGame,
    rollout: RolloutResult,
    which_player: str = "P1",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot a Hexner trajectory in 3D with time as one axis.

    Parameters
    ----------
    game:
        HexnerGame instance.
    rollout:
        RolloutResult containing x_traj and belief_traj.
    which_player:
        "P1" or "P2" indicating which player's state path to plot.
    ax:
        Optional 3D Axes. If None, a new figure and 3D axes are created.
    title:
        Optional title.

    Returns
    -------
    matplotlib.axes.Axes
        The 3D axes containing the plot.
    """
    x_traj = _tensor_to_numpy(rollout.x_traj)  # (K+1, dx)
    K = x_traj.shape[0] - 1
    dx1 = game.cfg.dx1

    t_grid = np.linspace(0.0, game.cfg.T, K + 1)

    if which_player.upper() == "P1":
        pos = x_traj[:, :2]
        label = "P1"
    elif which_player.upper() == "P2":
        pos = x_traj[:, dx1 : dx1 + 2]
        label = "P2"
    else:
        raise ValueError(f"which_player must be 'P1' or 'P2', got '{which_player}'")

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ax.plot(t_grid, pos[:, 0], pos[:, 1], marker="o", linestyle="-", label=label)
    ax.set_xlabel("time")
    ax.set_ylabel("x-position")
    ax.set_zlabel("y-position")
    ax.legend()

    if title is not None:
        ax.set_title(title)

    return ax