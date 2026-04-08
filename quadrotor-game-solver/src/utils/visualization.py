"""3-D trajectory visualization for the quadrotor Hexner game.

Supports:
  - static trajectory plot
  - animation of the drones along trajectory
  - optional belief panel (plots P(type=0))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from torch import Tensor

try:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


@dataclass
class AnimationConfig:
    dt_real: float = 0.05
    trail_length: int = 20
    drone_size: float = 80.0
    target_size: float = 120.0
    arm_len_plot: float = 0.15
    figsize: Tuple[int, int] = (10, 8)
    elev: float = 25.0
    azim: float = -45.0
    colors_p1: str = "tab:blue"
    colors_p2: str = "tab:red"
    target_colors: Tuple[str, ...] = ("green", "orange")
    axis_limit: float = 4.0
    auto_view: bool = True
    view_margin: float = 0.20
    min_axis_span: float = 1.0
    azim_step_deg: float = 10.0


def _rotation_matrix_zyx(phi: float, theta: float, psi: float) -> np.ndarray:
    """ZYX Euler rotation matrix (intrinsic)."""
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    return np.array(
        [
            [cpsi * cth, cpsi * sth * sphi - spsi * cphi, cpsi * sth * cphi + spsi * sphi],
            [spsi * cth, spsi * sth * sphi + cpsi * cphi, spsi * sth * cphi - cpsi * sphi],
            [-sth, cth * sphi, cth * cphi],
        ]
    )


def _project_points_camera(points: np.ndarray, elev_deg: float, azim_deg: float) -> np.ndarray:
    """Simple orthographic camera projection proxy for scoring view quality."""
    az = np.deg2rad(azim_deg)
    el = np.deg2rad(elev_deg)

    r_z = np.array(
        [
            [np.cos(az), -np.sin(az), 0.0],
            [np.sin(az), np.cos(az), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    r_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(el), -np.sin(el)],
            [0.0, np.sin(el), np.cos(el)],
        ]
    )
    cam = points @ r_z.T @ r_x.T
    return cam[:, :2]


def _auto_camera_angles(points: np.ndarray, cfg: AnimationConfig) -> Tuple[float, float]:
    """Pick camera angles that maximize 2-D spread of projected 3-D motion."""
    if points.shape[0] < 2:
        return cfg.elev, cfg.azim

    pts = points - points.mean(axis=0, keepdims=True)
    elev_candidates = (15.0, 22.0, 30.0, 38.0, 45.0)
    azim_values = np.arange(-180.0, 180.0, max(1.0, cfg.azim_step_deg))

    best_elev = cfg.elev
    best_azim = cfg.azim
    best_score = -np.inf
    for elev in elev_candidates:
        for azim in azim_values:
            proj = _project_points_camera(pts, elev, azim)
            span = np.ptp(proj, axis=0)
            score = float(span[0] * span[1] + 0.25 * min(span[0], span[1]))
            if score > best_score:
                best_score = score
                best_elev = float(elev)
                best_azim = float(azim)
    return best_elev, best_azim


def _apply_axes_limits(ax: object, points: np.ndarray, cfg: AnimationConfig) -> None:
    """Set data-adaptive axis limits with a margin for readability."""
    if points.shape[0] == 0:
        lim = cfg.axis_limit
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-0.5 * lim, 1.5 * lim)
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = np.maximum(maxs - mins, cfg.min_axis_span)
    half = 0.5 * span * (1.0 + cfg.view_margin)

    ax.set_xlim(center[0] - half[0], center[0] + half[0])
    ax.set_ylim(center[1] - half[1], center[1] + half[1])
    ax.set_zlim(center[2] - half[2], center[2] + half[2])

    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((float(span[0]), float(span[1]), float(span[2])))


def plot_trajectories(
    x_traj: Tensor,
    *,
    belief_traj: Optional[Tensor] = None,
    target_positions: Optional[Tensor] = None,
    title: str = "Quadrotor trajectories",
    cfg: Optional[AnimationConfig] = None,
    save_path: Optional[str] = None,
) -> "Figure":
    """Plot 3-D trajectories of both drones and optional belief panel."""
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for visualization")
    if cfg is None:
        cfg = AnimationConfig()

    x_np = x_traj.detach().cpu().numpy()
    t_steps = x_np.shape[0]

    belief_np: Optional[np.ndarray] = None
    if belief_traj is not None:
        belief_np = belief_traj.detach().cpu().numpy()
        if belief_np.ndim != 2 or belief_np.shape[0] != t_steps:
            raise ValueError("belief_traj must have shape (K+1, I) and match x_traj length")

    if belief_np is None:
        fig = plt.figure(figsize=cfg.figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax_bel = None
    else:
        fig = plt.figure(figsize=(max(cfg.figsize[0] + 2, 12), cfg.figsize[1]))
        gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0])
        ax = fig.add_subplot(gs[0], projection="3d")
        ax_bel = fig.add_subplot(gs[1])

    dx_total = x_np.shape[1]
    dx_single = dx_total // 2
    p1 = x_np[:, :3]
    p2 = x_np[:, dx_single : dx_single + 3]

    ax.plot(p1[:, 0], p1[:, 1], p1[:, 2], "-o", color=cfg.colors_p1, label="P1", markersize=3)
    ax.plot(p2[:, 0], p2[:, 1], p2[:, 2], "-s", color=cfg.colors_p2, label="P2", markersize=3)
    ax.scatter(*p1[0], marker="^", s=cfg.drone_size, color=cfg.colors_p1, zorder=5)
    ax.scatter(*p1[-1], marker="v", s=cfg.drone_size, color=cfg.colors_p1, zorder=5)
    ax.scatter(*p2[0], marker="^", s=cfg.drone_size, color=cfg.colors_p2, zorder=5)
    ax.scatter(*p2[-1], marker="v", s=cfg.drone_size, color=cfg.colors_p2, zorder=5)

    tgt_np: Optional[np.ndarray] = None
    if target_positions is not None:
        tgt_np = target_positions.detach().cpu().numpy()
        for i, c in enumerate(cfg.target_colors[: tgt_np.shape[0]]):
            ax.scatter(
                tgt_np[i, 0],
                tgt_np[i, 1],
                tgt_np[i, 2],
                marker="*",
                s=cfg.target_size,
                color=c,
                label=f"target {i}",
            )

    view_points = np.concatenate([p1, p2] + ([tgt_np] if tgt_np is not None else []), axis=0)
    elev, azim = (cfg.elev, cfg.azim)
    if cfg.auto_view:
        elev, azim = _auto_camera_angles(view_points, cfg)
    ax.view_init(elev=elev, azim=azim)
    _apply_axes_limits(ax, view_points, cfg)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend(loc="upper right")

    if belief_np is not None and ax_bel is not None:
        t = np.arange(t_steps)
        p_type0 = belief_np[:, 0]
        ax_bel.plot(t, p_type0, linewidth=2.0, color="tab:purple", label="P(type=0)")
        ax_bel.set_title("Belief")
        ax_bel.set_xlabel("step")
        ax_bel.set_ylabel("probability")
        ax_bel.set_ylim(0.0, 1.0)
        ax_bel.grid(alpha=0.3)
        ax_bel.legend(loc="best")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    return fig


def animate_rollout(
    x_traj: Tensor,
    dt: float,
    *,
    belief_traj: Optional[Tensor] = None,
    target_positions: Optional[Tensor] = None,
    title: str = "Quadrotor game rollout",
    cfg: Optional[AnimationConfig] = None,
    save_path: Optional[str] = None,
) -> "animation.FuncAnimation":
    """Create a 3-D animation of both drones and optional belief panel."""
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for visualization")
    if cfg is None:
        cfg = AnimationConfig()

    x_np = x_traj.detach().cpu().numpy()
    t_steps = x_np.shape[0]

    belief_np: Optional[np.ndarray] = None
    if belief_traj is not None:
        belief_np = belief_traj.detach().cpu().numpy()
        if belief_np.ndim != 2 or belief_np.shape[0] != t_steps:
            raise ValueError("belief_traj must have shape (K+1, I) and match x_traj length")

    dx_total = x_np.shape[1]
    dx_single = dx_total // 2
    p1 = x_np[:, :3]
    p2 = x_np[:, dx_single : dx_single + 3]
    has_attitude = dx_single >= 12
    if has_attitude:
        euler_p1 = x_np[:, 6:9]
        euler_p2 = x_np[:, dx_single + 6 : dx_single + 9]

    if belief_np is None:
        fig = plt.figure(figsize=cfg.figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax_bel = None
    else:
        fig = plt.figure(figsize=(max(cfg.figsize[0] + 2, 12), cfg.figsize[1]))
        gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0])
        ax = fig.add_subplot(gs[0], projection="3d")
        ax_bel = fig.add_subplot(gs[1])

    tgt_np: Optional[np.ndarray] = None
    if target_positions is not None:
        tgt_np = target_positions.detach().cpu().numpy()
        for i, c in enumerate(cfg.target_colors[: tgt_np.shape[0]]):
            ax.scatter(
                tgt_np[i, 0],
                tgt_np[i, 1],
                tgt_np[i, 2],
                marker="*",
                s=cfg.target_size,
                color=c,
                label=f"target {i}",
            )

    view_points = np.concatenate([p1, p2] + ([tgt_np] if tgt_np is not None else []), axis=0)
    elev, azim = (cfg.elev, cfg.azim)
    if cfg.auto_view:
        elev, azim = _auto_camera_angles(view_points, cfg)
    ax.view_init(elev=elev, azim=azim)
    _apply_axes_limits(ax, view_points, cfg)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)

    trail1, = ax.plot([], [], [], "-", color=cfg.colors_p1, alpha=0.6, linewidth=1.6)
    trail2, = ax.plot([], [], [], "-", color=cfg.colors_p2, alpha=0.6, linewidth=1.6)
    drone1 = ax.scatter([], [], [], s=cfg.drone_size, color=cfg.colors_p1, depthshade=False, label="P1")
    drone2 = ax.scatter([], [], [], s=cfg.drone_size, color=cfg.colors_p2, depthshade=False, label="P2")

    arm = cfg.arm_len_plot
    arm1_lines = [ax.plot([], [], [], color=cfg.colors_p1, lw=2)[0] for _ in range(2)] if has_attitude else []
    arm2_lines = [ax.plot([], [], [], color=cfg.colors_p2, lw=2)[0] for _ in range(2)] if has_attitude else []

    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    belief_text = ax.text2D(0.02, 0.90, "", transform=ax.transAxes)
    ax.legend(loc="upper right")

    p0_marker = None
    belief_time_bar = None
    if belief_np is not None and ax_bel is not None:
        t = np.arange(t_steps)
        p_type0 = belief_np[:, 0]
        ax_bel.plot(t, p_type0, linewidth=2.0, color="tab:purple", label="P(type=0)")
        p0_marker, = ax_bel.plot([0], [p_type0[0]], marker="o", color="tab:purple")
        belief_time_bar = ax_bel.axvline(0, color="gray", linestyle=":", linewidth=1.5)
        ax_bel.set_title("Belief")
        ax_bel.set_xlabel("step")
        ax_bel.set_ylabel("probability")
        ax_bel.set_ylim(0.0, 1.0)
        ax_bel.grid(alpha=0.3)
        ax_bel.legend(loc="best")

    def _draw_arms(lines: list[object], pos: np.ndarray, phi: float, theta: float, psi: float, arm_len: float) -> None:
        r_mat = _rotation_matrix_zyx(phi, theta, psi)
        for idx, body_dir in enumerate([np.array([1, 0, 0]), np.array([0, 1, 0])]):
            world_dir = r_mat @ body_dir
            tip_a = pos + arm_len * world_dir
            tip_b = pos - arm_len * world_dir
            lines[idx].set_data_3d([tip_a[0], tip_b[0]], [tip_a[1], tip_b[1]], [tip_a[2], tip_b[2]])

    def update(frame: int) -> Tuple[object, ...]:
        t_start = max(0, frame - cfg.trail_length)
        trail1.set_data_3d(p1[t_start : frame + 1, 0], p1[t_start : frame + 1, 1], p1[t_start : frame + 1, 2])
        trail2.set_data_3d(p2[t_start : frame + 1, 0], p2[t_start : frame + 1, 1], p2[t_start : frame + 1, 2])

        drone1._offsets3d = (np.array([p1[frame, 0]]), np.array([p1[frame, 1]]), np.array([p1[frame, 2]]))
        drone2._offsets3d = (np.array([p2[frame, 0]]), np.array([p2[frame, 1]]), np.array([p2[frame, 2]]))

        if has_attitude:
            _draw_arms(arm1_lines, p1[frame], *euler_p1[frame], arm)
            _draw_arms(arm2_lines, p2[frame], *euler_p2[frame], arm)

        time_text.set_text(f"t = {frame * dt:.2f} s")
        if belief_np is not None:
            p0_now = float(belief_np[frame, 0])
            belief_text.set_text(f"P(type=0) = {p0_now:.3f}")
            if p0_marker is not None:
                p0_marker.set_data([frame], [p0_now])
            if belief_time_bar is not None:
                belief_time_bar.set_xdata([frame, frame])
        else:
            belief_text.set_text("")

        artists: list[object] = [
            trail1,
            trail2,
            drone1,
            drone2,
            time_text,
            belief_text,
            *arm1_lines,
            *arm2_lines,
        ]
        if p0_marker is not None:
            artists.append(p0_marker)
        if belief_time_bar is not None:
            artists.append(belief_time_bar)
        return tuple(artists)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=t_steps,
        interval=int(cfg.dt_real * 1000),
        blit=False,
    )

    if save_path:
        import os
        import subprocess
        import sys
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            try:
                devnull = open(os.devnull, "w")
                old_stderr = sys.stderr
                sys.stderr = devnull
                anim.save(save_path, writer="ffmpeg", fps=max(1, int(1.0 / cfg.dt_real)))
                sys.stderr = old_stderr
                devnull.close()
                print(f"Saved animation to {save_path}")
            except (subprocess.CalledProcessError, RuntimeError, FileNotFoundError) as exc:
                sys.stderr = old_stderr
                if "devnull" in locals():
                    devnull.close()
                raise RuntimeError("ffmpeg failed to create animation") from exc

    return anim
