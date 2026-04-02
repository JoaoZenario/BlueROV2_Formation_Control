#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from graphs_utils import plot_graph_from_adjacency


class Animation:
    def __init__(self, params):
        self.params = params
        self.N = params.N
        self.dim = params.dim
        self.is_3d = params.dim == 3

        self.colors = [
            "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))
            for r, g, b, _ in plt.cm.hsv(
                [i / max(params.N, 1) for i in range(params.N)]
            )
        ]

        self._build_workspace_figure()
        self._build_telemetry_figure()

    # ============================================================
    # figure builders
    # ============================================================

    def _build_workspace_figure(self):
        self.fig_space = plt.figure("Formation Workspace", figsize=(8, 8))
        if self.is_3d:
            self.ax = self.fig_space.add_subplot(111, projection="3d")
        else:
            self.ax = self.fig_space.add_subplot(111)

        self.ax.set_title(
            f"Formation Workspace - {getattr(self.params, 'simulation_mode', 'simulation')}"
        )
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        if self.is_3d:
            self.ax.set_zlabel("Z (m)")

        self.dots = []
        self.trails = []
        self.edge_lines = []

        for i in range(self.N):
            if self.is_3d:
                dot = self.ax.plot(
                    [],
                    [],
                    [],
                    "o",
                    color=self.colors[i],
                    markersize=5,
                    label=f"Agent {i}",
                )[0]
                trail = self.ax.plot(
                    [], [], [], "-", color=self.colors[i], alpha=0.25, lw=1.0
                )[0]
            else:
                dot = self.ax.plot(
                    [], [], "o", color=self.colors[i], markersize=5, label=f"Agent {i}"
                )[0]
                trail = self.ax.plot(
                    [], [], "-", color=self.colors[i], alpha=0.25, lw=1.0
                )[0]

            self.dots.append(dot)
            self.trails.append(trail)

        self.edge_pairs = []
        for i in range(self.N):
            for j in range(self.N):
                if self.params.A[i, j] == 1:
                    if self.is_3d:
                        line = self.ax.plot(
                            [], [], [], color="green", linestyle="--", lw=1.5, alpha=0.8
                        )[0]
                    else:
                        line = self.ax.plot(
                            [], [], color="green", linestyle="--", lw=1.5, alpha=0.8
                        )[0]
                    self.edge_lines.append(line)
                    self.edge_pairs.append((i, j))

        self.telemetry_text = self.fig_space.text(
            0.02,
            0.02,
            "",
            verticalalignment="bottom",
            family="monospace",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        self.ax.legend(loc="upper right", fontsize="x-small")

    def _build_telemetry_figure(self):
        fov_hist = getattr(self.params, "fov_history", None)
        has_fov = (
            bool(getattr(self.params, "fov", False))
            and fov_hist is not None
            and np.ndim(fov_hist) == 3
            and fov_hist.shape[-1] >= 2
        )
        self.has_fov = has_fov

        self.fig_telemetry = plt.figure("System Telemetry", figsize=(13, 9))
        rows = 3 if has_fov else 2
        gs = self.fig_telemetry.add_gridspec(rows, 2, hspace=0.35, wspace=0.3)

        self.ax_dist = self.fig_telemetry.add_subplot(gs[0, 0])
        self.ax_err = self.fig_telemetry.add_subplot(gs[0, 1])
        self.ax_vel = self.fig_telemetry.add_subplot(gs[1, 0])
        self.ax_acc = self.fig_telemetry.add_subplot(gs[1, 1])

        self.ax_dist.set_title("Inter-Agent Distances")
        self.ax_err.set_title("Formation Error (m)")
        self.ax_vel.set_title("Velocity Norm")
        self.ax_acc.set_title("Acceleration Norm")

        for ax in [self.ax_dist, self.ax_err, self.ax_vel, self.ax_acc]:
            ax.set_xlabel("Time (s)")

        self.ax_dist.set_ylabel("Distance (m)")
        self.ax_err.set_ylabel("Error (m)")
        self.ax_vel.set_ylabel("Speed")
        self.ax_acc.set_ylabel("Acceleration")

        if self.params.conn:
            self.ax_dist.axhline(
                self.params.d_max, color="r", ls="--", alpha=0.8, label="d_max"
            )
        if self.params.coll:
            self.ax_dist.axhline(
                self.params.d_min, color="darkred", ls="--", alpha=0.8, label="d_min"
            )

        self.ax_vel.axhline(
            self.params.v_max, color="k", ls="--", alpha=0.5, label="v_max"
        )
        self.ax_acc.axhline(
            self.params.a_max, color="k", ls="--", alpha=0.5, label="a_max"
        )

        self.dist_lines = []
        self.err_lines = []
        self.vel_lines = []
        self.acc_lines = []

        self.err_mean_line = self.ax_err.plot([], [], "k-", lw=2.0, label="Mean error")[
            0
        ]

        for i in range(self.N):
            self.err_lines.append(
                self.ax_err.plot([], [], color=self.colors[i], label=f"Agent {i}")[0]
            )
            self.vel_lines.append(
                self.ax_vel.plot([], [], color=self.colors[i], label=f"Agent {i}")[0]
            )
            self.acc_lines.append(
                self.ax_acc.plot([], [], color=self.colors[i], label=f"Agent {i}")[0]
            )

        for i, j in self.edge_pairs:
            self.dist_lines.append(
                self.ax_dist.plot([], [], lw=1.2, label=f"{i}->{j}")[0]
            )

        if self.has_fov:
            self.ax_fov_x = self.fig_telemetry.add_subplot(gs[2, 0])
            self.ax_fov_y = self.fig_telemetry.add_subplot(gs[2, 1])

            self.ax_fov_x.set_title("Horizontal FOV Error (deg)")
            self.ax_fov_y.set_title("Vertical FOV Error (deg)")
            self.ax_fov_x.set_xlabel("Time (s)")
            self.ax_fov_y.set_xlabel("Time (s)")

            self.ax_fov_x.axhline(
                np.degrees(self.params.theta_max_x), color="r", ls="--", alpha=0.7
            )
            self.ax_fov_x.axhline(
                -np.degrees(self.params.theta_max_x), color="r", ls="--", alpha=0.7
            )
            self.ax_fov_y.axhline(
                np.degrees(self.params.theta_max_y), color="r", ls="--", alpha=0.7
            )
            self.ax_fov_y.axhline(
                -np.degrees(self.params.theta_max_y), color="r", ls="--", alpha=0.7
            )

            self.fov_x_lines = []
            self.fov_y_lines = []
            for i in range(self.N):
                self.fov_x_lines.append(
                    self.ax_fov_x.plot(
                        [], [], color=self.colors[i], label=f"Agent {i}"
                    )[0]
                )
                self.fov_y_lines.append(
                    self.ax_fov_y.plot(
                        [], [], color=self.colors[i], label=f"Agent {i}"
                    )[0]
                )

        self.ax_dist.legend(loc="upper right", fontsize="xx-small", ncol=2)
        self.ax_vel.legend(loc="upper right", fontsize="xx-small", ncol=2)
        self.ax_acc.legend(loc="upper right", fontsize="xx-small", ncol=2)
        self.ax_err.legend(loc="upper right", fontsize="xx-small", ncol=2)

    # ============================================================
    # helpers
    # ============================================================

    @staticmethod
    def _safe_set_data(line, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.size == 0 or y.size == 0:
            line.set_data([], [])
            return
        n = min(x.size, y.size)
        line.set_data(x[:n], y[:n])

    def _get_frame_state(self, frame):
        x = self.params.history[frame]
        v = (
            self.params.v_history_full[frame]
            if len(self.params.v_history_full) > frame
            else None
        )
        q = (
            self.params.q_history[frame]
            if hasattr(self.params, "q_history") and len(self.params.q_history) > frame
            else None
        )
        return x, v, q

    def _workspace_limits(self, frame):
        hist = self.params.history[: frame + 1]
        pts = hist.reshape(-1, self.dim)
        mins = np.min(pts, axis=0)
        maxs = np.max(pts, axis=0)
        pad = 1.0

        self.ax.set_xlim(mins[0] - pad, maxs[0] + pad)
        self.ax.set_ylim(mins[1] - pad, maxs[1] + pad)
        if self.is_3d:
            self.ax.set_zlim(mins[2] - pad, maxs[2] + pad)

    def _barrier_scalar(self, r):
        r = np.asarray(r, dtype=float)
        r2 = r**2
        eps = 1e-9
        out = np.zeros_like(r, dtype=float)

        if self.params.conn:
            out += np.log(
                self.params.d_max**2 / np.maximum(self.params.d_max**2 - r2, eps)
            )
        if self.params.coll:
            out += np.log(
                self.params.d_min**2 / np.maximum(r2 - self.params.d_min**2, eps)
            )
        return out

    def _barrier_vector(self, d):
        """
        Full vector barrier B(d), with d shape (..., 2).
        """
        d = np.asarray(d, dtype=float)
        r = np.linalg.norm(d, axis=-1)
        return self._barrier_scalar(r)

    def _grad_barrier(self, d):
        """
        Full vector gradient of the barrier for 2D visualization.
        """
        d = np.asarray(d, dtype=float)
        r2 = np.sum(d * d, axis=-1, keepdims=True)
        eps = 1e-9

        grad = np.zeros_like(d)

        if self.params.conn:
            grad += (2.0 / np.maximum(self.params.d_max**2 - r2, eps)) * d
        if self.params.coll:
            grad += -(2.0 / np.maximum(r2 - self.params.d_min**2, eps)) * d

        return grad

    # ============================================================
    # update plots
    # ============================================================

    def plot_handle(self, frame, u_history):
        x, v, q = self._get_frame_state(frame)
        t = np.arange(frame + 1) * self.params.dt

        # Workspace agents and trails
        for i in range(self.N):
            self.dots[i].set_data([x[i, 0]], [x[i, 1]])
            if self.is_3d:
                self.dots[i].set_3d_properties([x[i, 2]])

            self._safe_set_data(
                self.trails[i],
                self.params.history[: frame + 1, i, 0],
                self.params.history[: frame + 1, i, 1],
            )
            if self.is_3d:
                self.trails[i].set_3d_properties(self.params.history[: frame + 1, i, 2])

        # Workspace edges + telemetry distances
        cmap = plt.get_cmap("RdYlGn_r")
        current_max_dist = 0.0

        for idx, (i, j) in enumerate(self.edge_pairs):
            rel_hist = (
                self.params.history[: frame + 1, i, :]
                - self.params.history[: frame + 1, j, :]
            )
            dist_hist = np.linalg.norm(rel_hist, axis=1)
            self._safe_set_data(self.dist_lines[idx], t, dist_hist)

            pi, pj = x[i], x[j]
            self.edge_lines[idx].set_data([pi[0], pj[0]], [pi[1], pj[1]])
            if self.is_3d:
                self.edge_lines[idx].set_3d_properties([pi[2], pj[2]])

            d = dist_hist[-1] if dist_hist.size else 0.0
            current_max_dist = max(current_max_dist, d)

            target = getattr(
                self.params, "R", 0.5 * (self.params.d_min + self.params.d_max)
            )
            if d > target:
                denom = max(self.params.d_max - target, 1e-9)
                cval = (d - target) / denom
            else:
                denom = max(target - self.params.d_min, 1e-9)
                cval = (target - d) / denom

            self.edge_lines[idx].set_color(cmap(np.clip(cval, 0.0, 1.0)))

        # Telemetry per agent
        for i in range(self.N):
            self._safe_set_data(
                self.err_lines[i], t, self.params.err_history[: frame + 1, i]
            )
            self._safe_set_data(
                self.vel_lines[i], t, self.params.vel_history[: frame + 1, i]
            )
            self._safe_set_data(
                self.acc_lines[i], t, self.params.accel_history[: frame + 1, i]
            )

        # Mean formation error in meters
        if self.N > 1:
            mean_err = np.mean(self.params.err_history[: frame + 1, 1:], axis=1)
        else:
            mean_err = self.params.err_history[: frame + 1, 0]
        self._safe_set_data(self.err_mean_line, t, mean_err)

        # FOV telemetry
        if self.has_fov:
            for i in range(self.N):
                self._safe_set_data(
                    self.fov_x_lines[i], t, self.params.fov_history[: frame + 1, i, 0]
                )
                self._safe_set_data(
                    self.fov_y_lines[i], t, self.params.fov_history[: frame + 1, i, 1]
                )

        # Axes limits
        T = self.params.steps * self.params.dt
        self.ax_dist.set_xlim(0, T)
        self.ax_vel.set_xlim(0, T)
        self.ax_acc.set_xlim(0, T)
        self.ax_err.set_xlim(0, T)

        if frame >= 0:
            self.ax_dist.set_ylim(
                0, max(self.params.d_max * 1.1, current_max_dist * 1.1)
            )
            self.ax_vel.set_ylim(
                0,
                max(
                    self.params.v_max * 1.1,
                    np.max(self.params.vel_history[: frame + 1]) * 1.1,
                ),
            )
            self.ax_acc.set_ylim(
                0,
                max(
                    self.params.a_max * 1.1,
                    np.max(self.params.accel_history[: frame + 1]) * 1.1,
                ),
            )
            self.ax_err.set_ylim(
                0,
                max(
                    0.1,
                    np.max(self.params.err_history[: frame + 1]) * 1.1,
                ),
            )

        if self.has_fov:
            self.ax_fov_x.set_xlim(0, T)
            self.ax_fov_y.set_xlim(0, T)
            max_x = max(
                np.degrees(self.params.theta_max_x) + 5.0,
                np.max(np.abs(self.params.fov_history[: frame + 1, :, 0])) * 1.1,
            )
            max_y = max(
                np.degrees(self.params.theta_max_y) + 5.0,
                np.max(np.abs(self.params.fov_history[: frame + 1, :, 1])) * 1.1,
            )
            self.ax_fov_x.set_ylim(-max_x, max_x)
            self.ax_fov_y.set_ylim(-max_y, max_y)

        self._workspace_limits(frame)

        # Telemetry text
        stats = []
        stats.append(f"time = {frame * self.params.dt:6.2f} s")
        stats.append(f"mode = {getattr(self.params, 'simulation_mode', 'simulation')}")
        stats.append("-" * 24)
        current_mean_err = (
            np.mean(self.params.err_history[frame, 1:]) if self.N > 1 else 0.0
        )
        stats.append(f"mean err = {current_mean_err:6.3f} m")
        for i in range(self.N):
            stats.append(f"A{i}: |v| = {self.params.vel_history[frame, i]:5.2f}")
        self.telemetry_text.set_text("\n".join(stats))

    # ============================================================
    # topology / barriers
    # ============================================================

    def plot_topology(self, save_path=None, show=False):
        positions = (
            self.params.history[0]
            if hasattr(self.params, "history") and len(self.params.history) > 0
            else self.params.x
        )
        return plot_graph_from_adjacency(
            positions=positions,
            adjacency=self.params.A,
            directed=bool(self.params.mode_graph),
            title="Graph Topology",
            show=show,
            save_path=save_path,
        )

    def plot_barrier(self, show=True, save_prefix=None):
        """
        Plot both:
        1) scalar radial barrier slice (intuition)
        2) full vector recentered barrier surface in 2D
        """
        grid_limit = 1.5 * self.params.d_max

        # ---------- scalar radial slice ----------
        d_range = np.linspace(-grid_limit, grid_limit, 1200)
        B_vals = self._barrier_scalar(np.abs(d_range))

        d_des_norm = getattr(self.params, "R", 1.0)
        B_des = self._barrier_scalar(np.array([d_des_norm]))[0]
        grad_approx = (
            self._barrier_scalar(np.array([d_des_norm + 1e-3]))[0]
            - self._barrier_scalar(np.array([d_des_norm - 1e-3]))[0]
        ) / (2e-3)
        B_rec_slice = B_vals - B_des - grad_approx * (np.abs(d_range) - d_des_norm)

        fig1 = plt.figure("Barrier Slice", figsize=(8, 5))
        plt.plot(d_range, B_vals, "--", color="gray", alpha=0.7, label="Barrier")
        plt.plot(d_range, B_rec_slice, "b", lw=2, label="Recentered slice")
        plt.axvline(d_des_norm, color="g", ls=":", label="Target norm")
        plt.axvline(-d_des_norm, color="g", ls=":")
        plt.axvline(self.params.d_max, color="r", ls=":", label="d_max")
        plt.axvline(-self.params.d_max, color="r", ls=":")
        plt.axvline(self.params.d_min, color="darkred", ls=":", label="d_min")
        plt.axvline(-self.params.d_min, color="darkred", ls=":")
        plt.title("Scalar Barrier Slice")
        plt.xlabel("Relative coordinate")
        plt.ylabel("Potential")
        plt.grid(alpha=0.3)
        plt.legend()

        # ---------- full vector recentered barrier in 2D ----------
        res = 150
        x = np.linspace(-grid_limit, grid_limit, res)
        y = np.linspace(-grid_limit, grid_limit, res)
        X, Y = np.meshgrid(x, y)
        D = np.stack([X, Y], axis=-1)

        d_des = np.array([d_des_norm, 0.0])
        B = self._barrier_vector(D)
        B_des_full = self._barrier_vector(d_des[None, None, :])[0, 0]
        grad_B_des = self._grad_barrier(d_des[None, None, :])[0, 0]

        Delta = D - d_des[None, None, :]
        B_rec = (
            B
            - B_des_full
            - (grad_B_des[0] * Delta[..., 0] + grad_B_des[1] * Delta[..., 1])
        )

        fig2 = plt.figure("Vector Recentered Barrier", figsize=(8, 7))
        ax2 = fig2.add_subplot(111, projection="3d")
        ax2.plot_surface(X, Y, B_rec, cmap="plasma", edgecolor="none", alpha=0.9)
        ax2.scatter(
            [d_des[0]],
            [d_des[1]],
            [0.0],
            color="red",
            s=80,
            label="Desired equilibrium",
        )
        ax2.set_title("Full Vector Recentered Barrier Surface")
        ax2.set_xlabel("d_x")
        ax2.set_ylabel("d_y")
        ax2.set_zlabel("B_rec(d)")
        ax2.view_init(elev=25, azim=-55)
        ax2.legend()

        if save_prefix is not None:
            fig1.savefig(
                f"{save_prefix}_barrier_slice.png", dpi=200, bbox_inches="tight"
            )
            fig2.savefig(
                f"{save_prefix}_barrier_vector.png", dpi=200, bbox_inches="tight"
            )

        if show:
            plt.show()

        return fig1, fig2

    # ============================================================
    # animation / static / save
    # ============================================================

    def replay_animation(
        self,
        params,
        u_history,
        show_topology=True,
        show_barrier=False,
        save_path=None,
        fps=30,
    ):
        def update_frame(frame):
            self.plot_handle(frame, u_history)
            artists = (
                self.dots
                + self.trails
                + self.edge_lines
                + self.dist_lines
                + self.err_lines
                + [self.err_mean_line]
                + self.vel_lines
                + self.acc_lines
                + [self.telemetry_text]
            )
            if self.has_fov:
                artists += self.fov_x_lines + self.fov_y_lines
            return artists

        self.animation = FuncAnimation(
            self.fig_space,
            update_frame,
            frames=min(params.steps, len(params.history)),
            interval=1000 / fps,
            blit=False,
            repeat=False,
        )

        if save_path is not None:
            if save_path.endswith(".gif"):
                self.animation.save(save_path, writer=PillowWriter(fps=fps))
            else:
                self.animation.save(save_path, fps=fps, dpi=150)

        if show_topology:
            self.plot_topology(show=False)
        if show_barrier:
            self.plot_barrier(show=False)

        plt.show()

    def show_static_plots(
        self,
        params,
        u_history,
        show_topology=True,
        show_barrier=False,
        save_prefix=None,
    ):
        if len(params.history) == 0:
            plt.show()
            return

        frame = min(params.steps - 1, len(params.history) - 1)
        self.plot_handle(frame, u_history)

        if save_prefix is not None:
            self.fig_space.savefig(
                f"{save_prefix}_workspace.png", dpi=200, bbox_inches="tight"
            )
            self.fig_telemetry.savefig(
                f"{save_prefix}_telemetry.png", dpi=200, bbox_inches="tight"
            )

        if show_topology:
            self.plot_topology(
                save_path=(
                    None if save_prefix is None else f"{save_prefix}_topology.png"
                ),
                show=False,
            )
        if show_barrier:
            self.plot_barrier(show=False, save_prefix=save_prefix)

        plt.show()
