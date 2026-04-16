# core/utils.py
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Optional

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None


def set_random_seeds(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for Python, NumPy (if available), and PyTorch.

    Parameters
    ----------
    seed:
        Seed value to use across libraries.
    deterministic:
        If True, enable more deterministic behavior in PyTorch (may reduce
        performance on GPUs due to disabling certain optimizations).
    """
    random.seed(seed)

    if np is not None:
        np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class Timer:
    """
    Lightweight wall-clock timer for profiling blocks of code.

    Example
    -------
    >>> with Timer("belief_tree"):
    ...     build_belief_tree(...)
    [belief_tree] elapsed: 0.012 s
    """

    label: str = ""
    print_on_exit: bool = True
    _start: float = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = time.perf_counter() - self._start
        if self.print_on_exit:
            prefix = f"[{self.label}] " if self.label else ""
            print(f"{prefix}elapsed: {elapsed:.3f} s")


def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a torch.nn.Module.

    Parameters
    ----------
    module:
        The PyTorch module whose parameters are to be counted.
    trainable_only:
        If True, count only parameters with `requires_grad=True`.

    Returns
    -------
    int
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def tensor_summary(t: torch.Tensor, name: str = "tensor") -> str:
    """
    Produce a short textual summary of a tensor for logging/debugging.

    Parameters
    ----------
    t:
        Tensor to summarize.
    name:
        Human-readable label for the tensor.

    Returns
    -------
    str
        Summary string including name, shape, dtype, device, and basic stats.
    """
    if t.numel() == 0:
        return f"{name}: empty, shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}"

    with torch.no_grad():
        t_flat = t.float().view(-1)
        return (
            f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, "
            f"min={t_flat.min().item():.3e}, max={t_flat.max().item():.3e}, "
            f"mean={t_flat.mean().item():.3e}, std={t_flat.std().item():.3e}"
        )