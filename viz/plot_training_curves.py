# viz/plot_training_curves.py
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _to_numpy(seq: Sequence[float] | np.ndarray) -> np.ndarray:
    if isinstance(seq, np.ndarray):
        return seq
    return np.asarray(list(seq), dtype=float)


def plot_training_curve(
    values: Sequence[float] | np.ndarray,
    iterations: Optional[Sequence[int] | np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    label: str = "loss",
    ylabel: str = "Loss",
    xlabel: str = "Iteration",
) -> plt.Axes:
    """
    Plot a single scalar training curve (e.g., loss vs. iteration).

    Parameters
    ----------
    values:
        Sequence of scalar values (losses, metric values, etc.).
    iterations:
        Optional sequence of iteration indices of the same length as `values`.
        If None, uses range(len(values)).
    ax:
        Optional matplotlib Axes to draw on. If None, a new figure and axes
        are created.
    label:
        Legend label for the curve.
    ylabel:
        Y-axis label (default: "Loss").
    xlabel:
        X-axis label (default: "Iteration").

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    y = _to_numpy(values)
    if iterations is None:
        x = np.arange(len(y))
    else:
        x = _to_numpy(iterations)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(x, y, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)

    if label:
        ax.legend()

    return ax


def plot_multiple_training_curves(
    curves: Mapping[str, Sequence[float] | np.ndarray],
    iterations: Optional[Sequence[int] | np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    ylabel: str = "Loss",
    xlabel: str = "Iteration",
) -> plt.Axes:
    """
    Plot multiple scalar training curves on the same axes.

    Parameters
    ----------
    curves:
        Mapping from curve label to a sequence of values.
    iterations:
        Optional sequence of iteration indices shared across all curves.
        If None, uses range(len(first_curve)).
    ax:
        Optional matplotlib Axes. If None, a new figure and axes are created.
    ylabel:
        Y-axis label (default: "Loss").
    xlabel:
        X-axis label (default: "Iteration").

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    if not curves:
        raise ValueError("plot_multiple_training_curves: `curves` must be non-empty")

    first_key = next(iter(curves))
    first_vals = _to_numpy(curves[first_key])

    if iterations is None:
        x = np.arange(len(first_vals))
    else:
        x = _to_numpy(iterations)

    if ax is None:
        _, ax = plt.subplots()

    for label, vals in curves.items():
        y = _to_numpy(vals)
        if len(y) != len(x):
            raise ValueError(
                f"Curve '{label}' has length {len(y)} but iterations has length {len(x)}"
            )
        ax.plot(x, y, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    return ax