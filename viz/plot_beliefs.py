# viz/plot_beliefs.py
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _to_numpy(x) -> np.ndarray:
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_belief_trajectory(
    belief_traj,
    T: float,
    ax: Optional[plt.Axes] = None,
    labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot a single belief trajectory over time.

    Parameters
    ----------
    belief_traj:
        Tensor or array of shape (K+1, I) with public beliefs p_k at each
        time step.
    T:
        Horizon length; determines the time axis scaling.
    ax:
        Optional matplotlib Axes. If None, a new figure and axes are created.
    labels:
        Optional list of labels for each type; length I. If None, use "type-0",
        "type-1", etc.
    title:
        Optional title for the plot.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    b = _to_numpy(belief_traj)  # (K+1, I)
    K = b.shape[0] - 1
    I = b.shape[1]

    t_grid = np.linspace(0.0, T, K + 1)

    if ax is None:
        _, ax = plt.subplots()

    if labels is None:
        labels = [f"type-{i}" for i in range(I)]

    for i in range(I):
        ax.plot(t_grid, b[:, i], marker="o", linestyle="-", label=labels[i])

    ax.set_xlabel("time")
    ax.set_ylabel("belief p[type]")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    if title is not None:
        ax.set_title(title)

    return ax


def plot_belief_ensemble(
    belief_trajs: Iterable,
    T: float,
    ax: Optional[plt.Axes] = None,
    type_index: int = 0,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot an ensemble of belief trajectories for a fixed type index.

    This is useful for visualizing how quickly and how consistently the
    belief about a particular type converges across multiple rollouts.

    Parameters
    ----------
    belief_trajs:
        Iterable of belief trajectories, each of shape (K+1, I).
    T:
        Horizon length.
    ax:
        Optional matplotlib Axes. If None, a new figure and axes are created.
    type_index:
        Index of the type whose probability is plotted.
    title:
        Optional title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    trajs = [ _to_numpy(b) for b in belief_trajs ]
    if not trajs:
        raise ValueError("plot_belief_ensemble: `belief_trajs` is empty")

    K_plus_1, I = trajs[0].shape
    if not (0 <= type_index < I):
        raise ValueError(
            f"plot_belief_ensemble: type_index={type_index} out of range [0, {I})"
        )

    t_grid = np.linspace(0.0, T, K_plus_1)

    if ax is None:
        _, ax = plt.subplots()

    for b in trajs:
        if b.shape != (K_plus_1, I):
            raise ValueError(
                "plot_belief_ensemble: all trajectories must have the same shape; "
                f"expected {(K_plus_1, I)}, got {b.shape}"
            )
        ax.plot(t_grid, b[:, type_index], alpha=0.3)

    ax.set_xlabel("time")
    ax.set_ylabel(f"belief p[type={type_index}]")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.3)

    if title is not None:
        ax.set_title(title)

    return ax