# config/base_config.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
import os
import torch

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def project_relative(path: str) -> str:
    """
    Interpret `path` as relative to the inner package root unless it is
    already absolute.

    This keeps all generated artifacts (runs, reports, etc.) inside the
    MPC_2p0s1 package tree by default, regardless of current working dir.
    """
    if os.path.isabs(path):
        return path
    return os.path.join(PACKAGE_ROOT, path)

def resolve_device(requested: str | torch.device | None) -> torch.device:
    """
    Resolve a user-specified device string into a concrete torch.device.

    Parameters
    ----------
    requested:
        - "auto": use CUDA if available, otherwise CPU.
        - str accepted by torch.device, e.g. "cpu", "cuda", "cuda:0".
        - torch.device instance.
        - None: defaults to "auto".

    Returns
    -------
    torch.device
        The resolved device.
    """
    if requested is None or (isinstance(requested, str) and requested.lower() == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(requested, torch.device):
        return requested

    return torch.device(requested)


@dataclass
class GameConfig:
    """
    Configuration for a 2p0s1 LQ game instance.

    We give defaults so that `GameConfig()` is legal, and scripts/tests
    can override fields after construction.
    """

    # Core dimensions
    I: int = 2          # number of types
    dx1: int = 4        # P1 state dim (Hexner: 2D pos+vel)
    dx2: int = 4        # P2 state dim
    du: int = 2         # P1 control dim
    dv: int = 2         # P2 control dim

    # Time horizon / discretization
    T: float = 1.0
    K: int = 10

    # Derived dims (filled in __post_init__)
    dx: int = field(init=False)
    tau: float = field(init=False)

    # Device / dtype / seed
    device: str = "cpu"
    device_resolved: torch.device = field(init=False)
    dtype: torch.dtype = torch.float32
    seed: int = 123

    # Simple symmetric box constraints on actions
    u_min: float = -4.0
    u_max: float = 4.0
    v_min: float = -4.0
    v_max: float = 4.0

    def __post_init__(self) -> None:
        # Derive dx and tau if not already set
        self.dx = self.dx1 + self.dx2
        self.tau = self.T / self.K if self.K > 0 else 0.0
        self.device_resolved = torch.device(self.device)

    def as_dict(self) -> Dict[str, Any]:
        """Return a plain-ish dict for checkpoint metadata."""
        # For now, we rely on asdict; torch.device / dtype are torch objects,
        # but torch.save can handle them.
        return asdict(self)


@dataclass
class TrainingConfig:
    """
    Configuration of the outer optimization over belief-splitting parameters (α/logits).

    Attributes
    ----------
    optimizer:
        Name of the optimizer to use ("adam" or "lbfgs" currently).
    learning_rate:
        Base learning rate for the optimizer.
    weight_decay:
        L2 weight decay coefficient (if applicable).
    max_iters:
        Maximum number of outer iterations.
    grad_clip_norm:
        If > 0, gradients are clipped to this global norm; if None or <= 0, no clipping.
    print_every:
        How often (in iterations) to print progress to stdout.
    log_every:
        How often to write metrics to disk.
    checkpoint_every:
        How often to save a checkpoint of the α/logits and optimizer state.
    staged_depth:
        If True, use staged depth-wise optimization (only shallow depths updated
        at early iterations).
    initial_unfrozen_depth:
        Initial maximum depth at which α is trainable (0 for root only).
    depth_increment:
        Depth increment when expanding the trainable region.
    depth_increment_every:
        Number of iterations between depth increments.
    max_unfrozen_depth:
        Optional cap on the maximum depth that will ever be unfrozen. If None,
        the maximum tree depth K is used.
    use_mixed_precision:
        If True, allow autocast/mixed precision during the forward pass (for GPUs).
    """

    optimizer: str = "adam"
    learning_rate: float = 1e-2
    weight_decay: float = 0.0
    max_iters: int = 2_000
    grad_clip_norm: Optional[float] = 10.0

    print_every: int = 50
    log_every: int = 10
    checkpoint_every: int = 200

    staged_depth: bool = True
    initial_unfrozen_depth: int = 0
    depth_increment: int = 1
    depth_increment_every: int = 200
    max_unfrozen_depth: Optional[int] = None

    use_mixed_precision: bool = False

    def make_optimizer(
        self,
        params: list[torch.nn.Parameter] | tuple[torch.nn.Parameter, ...],
    ) -> torch.optim.Optimizer:
        """
        Instantiate a torch optimizer for the given parameters.

        The optimizer choice and hyperparameters are taken from this config.
        """
        name = self.optimizer.lower()
        if name == "adam":
            return torch.optim.Adam(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        if name == "lbfgs":
            # Note: LBFGS performs full-batch updates; caller is responsible for
            # providing a closure compatible with torch.optim.LBFGS.
            return torch.optim.LBFGS(
                params,
                lr=self.learning_rate,
                max_iter=20,
                history_size=100,
            )

        raise ValueError(f"Unsupported optimizer '{self.optimizer}'")

    def as_dict(self) -> dict:
        """Return a plain-Python representation useful for logging."""
        return {
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_iters": self.max_iters,
            "grad_clip_norm": self.grad_clip_norm,
            "print_every": self.print_every,
            "log_every": self.log_every,
            "checkpoint_every": self.checkpoint_every,
            "staged_depth": self.staged_depth,
            "initial_unfrozen_depth": self.initial_unfrozen_depth,
            "depth_increment": self.depth_increment,
            "depth_increment_every": self.depth_increment_every,
            "max_unfrozen_depth": self.max_unfrozen_depth,
            "use_mixed_precision": self.use_mixed_precision,
        }


@dataclass
class PathsConfig:
    """
    Simple configuration for experiment directories.

    Attributes
    ----------
    root_dir:
        Base directory under which all runs for this project are stored.
    run_name:
        Name of the current run; a subdirectory with this name is created under
        root_dir unless `create_subdir` is False.
    create_subdir:
        If True, files are written to `root_dir / run_name`. If False, files are
        written directly to `root_dir`.
    """

    # Default: put runs under MPC_2p0s1/runs
    root_dir: str = os.path.join(PACKAGE_ROOT, "runs")
    run_name: str = "debug"
    create_subdir: bool = True

    @property
    def run_dir(self) -> str:
        """
        Full path to the directory where logs, checkpoints, and figures should be written.
        """
        if self.create_subdir:
            return os.path.join(self.root_dir, self.run_name)
        return self.root_dir

    def ensure_exists(self) -> None:
        """Create the run directory if it does not already exist."""
        os.makedirs(self.run_dir, exist_ok=True)

    def as_dict(self) -> dict:
        """Return a plain-Python representation useful for logging."""
        return {
            "root_dir": self.root_dir,
            "run_name": self.run_name,
            "create_subdir": self.create_subdir,
            "run_dir": self.run_dir,
        }