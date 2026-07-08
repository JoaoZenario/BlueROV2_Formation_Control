#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from bluerov2_control.graphs_utils import plot_graph_from_adjacency
from bluerov2_control.rot_utils import rotation_matrix_from_quaternion


class Animation:
    def __init__(self, params):
        self.params = params
        self.cfg = params.config
        self.scenario = params.scenario
        self.data = params.data
        self.N = self.cfg.N
        self.dim = self.cfg.dim
        self.is_3d = self.cfg.dim == 3
        self.debug_history = getattr(self.data, "debug_history", [])
        self._R_cb = (
            np.eye(3)
            if getattr(self.cfg, "fov_R_cb", None) is None
            else np.asarray(self.cfg.fov_R_cb, dtype=float).reshape(3, 3)
        )
        self._p_cb = (
            np.zeros(3)
            if getattr(self.cfg, "fov_p_cb", None) is None
            else np.asarray(self.cfg.fov_p_cb, dtype=float).reshape(3)
        )

        self.colors = [
            "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))
            for r, g, b, _ in plt.cm.hsv(
                [i / max(self.cfg.N, 1) for i in range(self.cfg.N)]
            )
        ]

        self._build_workspace_figure()
        self._build_telemetry_figure()
        self._build_force_torque_figure()

    # ============================================================
    # figure builders
    # ============================================================

    def _build_workspace_figure(self):
        self.fig_space = plt.figure("Formation Workspace", figsize=(8, 8))
        if self.is_3d:
            self.ax = self.fig_space.add_subplot(111, projection="3d")
        else:
            self.ax = self.fig_space.add_subplot(111)

        self.ax.set_title(f"Formation Workspace - {self.cfg.simulation_mode}")
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
                if self.scenario.A[i, j] == 1:
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

        self.is_bluerov_mode = str(self.cfg.simulation_mode).startswith("bluerov")
        self.rov_body_lines = []
        if self.is_bluerov_mode:
            # Approximate BlueROV2 Heavy envelope [m] used for visualization only.
            self.rov_len = 0.457
            self.rov_wid = 0.338
            self.rov_hei = 0.254

            if self.is_3d:
                hx = 0.5 * self.rov_len
                hy = 0.5 * self.rov_wid
                hz = 0.5 * self.rov_hei
                self._rov_body_corners_b = np.array(
                    [
                        [+hx, +hy, +hz],
                        [+hx, +hy, -hz],
                        [+hx, -hy, +hz],
                        [+hx, -hy, -hz],
                        [-hx, +hy, +hz],
                        [-hx, +hy, -hz],
                        [-hx, -hy, +hz],
                        [-hx, -hy, -hz],
                    ],
                    dtype=float,
                )
                self._rov_body_edges = [
                    (0, 1),
                    (2, 3),
                    (4, 5),
                    (6, 7),
                    (0, 2),
                    (4, 6),
                    (1, 3),
                    (5, 7),
                    (0, 4),
                    (2, 6),
                    (1, 5),
                    (3, 7),
                ]
                for i in range(self.N):
                    edges_i = []
                    for _ in self._rov_body_edges:
                        edges_i.append(
                            self.ax.plot(
                                [], [], [], color=self.colors[i], lw=1.0, alpha=0.8
                            )[0]
                        )
                    self.rov_body_lines.append(edges_i)
            else:
                self._rov_body_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
                for i in range(self.N):
                    edges_i = []
                    for _ in self._rov_body_edges:
                        edges_i.append(
                            self.ax.plot(
                                [], [], color=self.colors[i], lw=1.2, alpha=0.8
                            )[0]
                        )
                    self.rov_body_lines.append(edges_i)

        # Body-attached frame visualization
        self.frame_axis_len = 0.2 * max(getattr(self.cfg, "R", 1.0), 0.5)
        self.frame_x_lines = []
        self.frame_y_lines = []
        self.frame_z_lines = []
        for _i in range(self.N):
            if self.is_3d:
                self.frame_x_lines.append(
                    self.ax.plot([], [], [], color="tab:red", lw=1.4, alpha=0.9)[0]
                )
                self.frame_y_lines.append(
                    self.ax.plot([], [], [], color="tab:green", lw=1.4, alpha=0.9)[0]
                )
                self.frame_z_lines.append(
                    self.ax.plot([], [], [], color="tab:blue", lw=1.4, alpha=0.9)[0]
                )
            else:
                self.frame_x_lines.append(
                    self.ax.plot([], [], color="tab:red", lw=1.4, alpha=0.9)[0]
                )
                self.frame_y_lines.append(
                    self.ax.plot([], [], color="tab:green", lw=1.4, alpha=0.9)[0]
                )

        # FoV boundary visualization (draw only when FoV is active)
        # Use a compact local pyramid instead of long rays.
        self.fov_pyramid_len = 0.8 * self.frame_axis_len
        self.fov_rays = []
        if bool(getattr(self.cfg, "fov", False)):
            n_rays = 8 if self.is_3d else 3
            for _i in range(self.N):
                rays_i = []
                for _ in range(n_rays):
                    if self.is_3d:
                        rays_i.append(
                            self.ax.plot(
                                [],
                                [],
                                [],
                                color="tab:orange",
                                lw=1.0,
                                alpha=0.7,
                                ls="-.",
                            )[0]
                        )
                    else:
                        rays_i.append(
                            self.ax.plot(
                                [],
                                [],
                                color="tab:orange",
                                lw=1.0,
                                alpha=0.7,
                                ls="-.",
                            )[0]
                        )
                self.fov_rays.append(rays_i)

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
        fov_hist = self.data.fov_history
        has_fov = (
            bool(self.cfg.fov)
            and fov_hist is not None
            and np.ndim(fov_hist) == 3
            and fov_hist.shape[-1] >= 2
        )
        self.has_fov = has_fov

        content_rows = 3 if has_fov else 2
        self.fig_telemetry = plt.figure(
            "System Telemetry", figsize=(13, 4 * content_rows)
        )
        gs = self.fig_telemetry.add_gridspec(content_rows, 2, hspace=0.4, wspace=0.3)

        self.ax_dist = self.fig_telemetry.add_subplot(gs[0, 0])
        self.ax_err = self.fig_telemetry.add_subplot(gs[0, 1])
        self.ax_vel = self.fig_telemetry.add_subplot(gs[1, 0])
        self.ax_acc = self.fig_telemetry.add_subplot(gs[1, 1])

        self.ax_dist.grid(True, ls="--", alpha=0.5)
        self.ax_err.grid(True, ls="--", alpha=0.5)
        self.ax_vel.grid(True, ls="--", alpha=0.5)
        self.ax_acc.grid(True, ls="--", alpha=0.5)

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

        self.dist_dmax_line = None
        self.dist_dmin_line = None
        if self.cfg.conn:
            self.dist_dmax_line = self.ax_dist.plot(
                [], [], color="r", ls="--", alpha=0.8, label="d_max"
            )[0]
        if self.cfg.coll:
            self.dist_dmin_line = self.ax_dist.plot(
                [], [], color="darkred", ls="--", alpha=0.8, label="d_min"
            )[0]

        self.ax_vel.axhline(
            self.cfg.v_max, color="k", ls="--", alpha=0.5, label="v_max"
        )
        self.ax_acc.axhline(
            self.cfg.a_max, color="k", ls="--", alpha=0.5, label="a_max"
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

            self.ax_fov_x.grid(True, ls="--", alpha=0.5)
            self.ax_fov_y.grid(True, ls="--", alpha=0.5)

            self.ax_fov_x.set_title("Horizontal FOV Error (deg)")
            self.ax_fov_y.set_title("Vertical FOV Error (deg)")
            self.ax_fov_x.set_xlabel("Time (s)")
            self.ax_fov_y.set_xlabel("Time (s)")

            self.ax_fov_x.axhline(
                np.degrees(self.cfg.theta_max_x), color="r", ls="--", alpha=0.7
            )
            self.ax_fov_x.axhline(
                -np.degrees(self.cfg.theta_max_x), color="r", ls="--", alpha=0.7
            )
            self.ax_fov_y.axhline(
                np.degrees(self.cfg.theta_max_y), color="r", ls="--", alpha=0.7
            )
            self.ax_fov_y.axhline(
                -np.degrees(self.cfg.theta_max_y), color="r", ls="--", alpha=0.7
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

    def _build_force_torque_figure(self):
        """Separate figure for applied forces and torques (or thruster forces / cmd)."""
        mode = self.cfg.simulation_mode
        is_bluerov = mode.startswith("bluerov")
        is_thruster = mode == "bluerov_clf_qp_thrusters"

        if is_thruster:
            n_rows, n_cols = 1, 1
            fig_h = 4
        elif is_bluerov:
            n_rows, n_cols = 2, 3  # row 0: forces, row 1: torques
            fig_h = 8
        else:
            n_rows, n_cols = 1, 3
            fig_h = 4

        self.fig_force_torque = plt.figure("Forces & Torques", figsize=(14, fig_h))
        gs_ft = self.fig_force_torque.add_gridspec(
            n_rows, n_cols, hspace=0.4, wspace=0.3
        )

        # ---- Force panels (row 0) ----
        if is_thruster:
            force_limit = self.cfg.thruster_limit
            force_limit_label = "±thruster_limit"
            force_titles = ["Per-Thruster Forces (N)"]
            force_ylabel = "Force (N)"
            self._force_comp_slices = None  # thruster mode sentinel
            n_force_comp = 8
        elif is_bluerov:
            force_limit = self.cfg.tau_v_max
            force_limit_label = "±tau_v_max" if force_limit is not None else None
            force_titles = ["$f_x$ (N)", "$f_y$ (N)", "$f_z$ (N)"]
            force_ylabel = "Force (N)"
            self._force_comp_slices = [0, 1, 2]
            n_force_comp = 3
        elif mode == "single_integrator":
            force_limit = self.cfg.v_max
            force_limit_label = "v_max"
            force_titles = ["$v_x$ (m/s)", "$v_y$ (m/s)", "$v_z$ (m/s)"]
            force_ylabel = "m/s"
            self._force_comp_slices = [0, 1, 2]
            n_force_comp = 3
        else:  # double_integrator
            force_limit = self.cfg.a_max
            force_limit_label = "a_max"
            force_titles = ["$a_x$ (m/s²)", "$a_y$ (m/s²)", "$a_z$ (m/s²)"]
            force_ylabel = "m/s²"
            self._force_comp_slices = [0, 1, 2]
            n_force_comp = 3

        self._force_limit = force_limit

        self.ax_force_panels = []
        for k in range(n_cols if not is_thruster else 1):
            ax = self.fig_force_torque.add_subplot(gs_ft[0, k])
            ax.grid(True, ls="--", alpha=0.5)
            ax.set_title(force_titles[k] if not is_thruster else force_titles[0])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(force_ylabel)
            self.ax_force_panels.append(ax)

        self.force_limit_pos_lines = []
        self.force_limit_neg_lines = []
        for k, ax in enumerate(self.ax_force_panels):
            if force_limit is not None:
                self.force_limit_pos_lines.append(
                    ax.plot(
                        [],
                        [],
                        color="r",
                        ls="--",
                        alpha=0.8,
                        label=force_limit_label if k == 0 else None,
                    )[0]
                )
                self.force_limit_neg_lines.append(
                    ax.plot([], [], color="r", ls="--", alpha=0.8)[0]
                )
            else:
                self.force_limit_pos_lines.append(None)
                self.force_limit_neg_lines.append(None)

        self.force_lines = []
        linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
        if self._force_comp_slices is None:
            # Thruster mode
            for k in range(n_force_comp):
                for i in range(self.N):
                    ls = linestyles[i % len(linestyles)]
                    self.force_lines.append(
                        self.ax_force_panels[0].plot(
                            [],
                            [],
                            ls=ls,
                            color=self.colors[i],
                            lw=1.0,
                            alpha=0.8,
                            label=f"A{i} f{k+1}" if i == 0 else None,
                        )[0]
                    )
        else:
            for k in range(3):
                for i in range(self.N):
                    self.force_lines.append(
                        self.ax_force_panels[k].plot(
                            [],
                            [],
                            color=self.colors[i],
                            lw=1.2,
                            label=f"Agent {i}",
                        )[0]
                    )

        for ax in self.ax_force_panels:
            ax.legend(loc="upper right", fontsize="xx-small", ncol=2)

        # ---- Torque panels (row 1, bluerov non-thruster only) ----
        self.ax_torque_panels = []
        self.torque_lines = []
        if is_bluerov and not is_thruster:
            torque_titles = ["$\\tau_x$ (N·m)", "$\\tau_y$ (N·m)", "$\\tau_z$ (N·m)"]
            for k in range(3):
                ax = self.fig_force_torque.add_subplot(gs_ft[1, k])
                ax.grid(True, ls="--", alpha=0.5)
                ax.set_title(torque_titles[k])
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Torque (N·m)")
                self.ax_torque_panels.append(ax)
                for i in range(self.N):
                    self.torque_lines.append(
                        ax.plot(
                            [], [], color=self.colors[i], lw=1.2, label=f"Agent {i}"
                        )[0]
                    )
            for ax in self.ax_torque_panels:
                ax.legend(loc="upper right", fontsize="xx-small", ncol=2)

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

    def _extract_distance_limits(self, frame: int):
        n = frame + 1
        t = np.arange(n) * self.cfg.dt
        d_min = np.full(n, self.cfg.d_min, dtype=float)
        d_max = np.full(n, self.cfg.d_max, dtype=float)

        if not self.debug_history:
            return t, d_min, d_max

        for k in range(min(n, len(self.debug_history))):
            step_debug = self.debug_history[k]
            dmin_vals = []
            dmax_vals = []
            if isinstance(step_debug, list):
                for dbg in step_debug:
                    if not isinstance(dbg, dict):
                        continue
                    if "d_min_eff" in dbg:
                        dmin_vals.extend(
                            np.asarray(dbg["d_min_eff"], dtype=float).reshape(-1)
                        )
                    if "d_max_eff" in dbg:
                        dmax_vals.extend(
                            np.asarray(dbg["d_max_eff"], dtype=float).reshape(-1)
                        )
            if dmin_vals:
                d_min[k] = float(np.max(dmin_vals))
            if dmax_vals:
                d_max[k] = float(np.min(dmax_vals))

        return t, d_min, d_max

    def _extract_force_limits(self, frame: int, panel_idx: int):
        n = frame + 1
        t = np.arange(n) * self.cfg.dt
        if self._force_limit is None:
            return t, None

        lim = np.full(n, float(self._force_limit), dtype=float)
        if not self.debug_history:
            return t, lim

        for k in range(min(n, len(self.debug_history))):
            step_debug = self.debug_history[k]
            cand = []
            if not isinstance(step_debug, list):
                continue
            for dbg in step_debug:
                if not isinstance(dbg, dict):
                    continue
                if "tau_v_max_eff" in dbg:
                    cand.extend(
                        np.asarray(dbg["tau_v_max_eff"], dtype=float).reshape(-1)
                    )
                if "tau_v_max" in dbg:
                    cand.extend(np.asarray(dbg["tau_v_max"], dtype=float).reshape(-1))
                if "f_max" in dbg:
                    fmax = np.asarray(dbg["f_max"], dtype=float).reshape(-1)
                    if fmax.size:
                        if self._force_comp_slices is None:
                            cand.append(float(np.max(np.abs(fmax))))
                        elif panel_idx < fmax.size:
                            cand.append(float(abs(fmax[panel_idx])))
                if "f_min" in dbg:
                    fmin = np.asarray(dbg["f_min"], dtype=float).reshape(-1)
                    if fmin.size:
                        if self._force_comp_slices is None:
                            cand.append(float(np.max(np.abs(fmin))))
                        elif panel_idx < fmin.size:
                            cand.append(float(abs(fmin[panel_idx])))
            if cand:
                lim[k] = float(np.max(cand))

        return t, lim

    def _get_frame_state(self, frame):
        x = self.data.history[frame]
        v = (
            self.data.v_history_full[frame]
            if len(self.data.v_history_full) > frame
            else None
        )
        q = self.data.q_history[frame] if len(self.data.q_history) > frame else None
        return x, v, q

    def _workspace_limits(self, frame):
        hist = self.data.history[: frame + 1]
        pts = hist.reshape(-1, self.dim)
        mins = np.min(pts, axis=0)
        maxs = np.max(pts, axis=0)
        pad = 1.0

        self.ax.set_xlim(mins[0] - pad, maxs[0] + pad)
        self.ax.set_ylim(mins[1] - pad, maxs[1] + pad)
        if self.is_3d:
            self.ax.set_zlim(mins[2] - pad, maxs[2] + pad)

    def _rotation_from_history(self, q_frame, i):
        if q_frame is None:
            return np.eye(3)
        qi = np.asarray(q_frame[i], dtype=float).reshape(4)
        return rotation_matrix_from_quaternion(qi)

    def _set_2d_line(self, line, p0, p1):
        self._safe_set_data(line, [p0[0], p1[0]], [p0[1], p1[1]])

    def _set_3d_line(self, line, p0, p1):
        line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        line.set_3d_properties([p0[2], p1[2]])

    def _clear_line(self, line):
        if self.is_3d:
            line.set_data([], [])
            line.set_3d_properties([])
        else:
            line.set_data([], [])

    def _update_frame_and_fov_overlays(self, frame, x, q):
        for i in range(self.N):
            p_i = np.asarray(x[i], dtype=float).reshape(3)
            R_b2I = self._rotation_from_history(q, i)

            if self.rov_body_lines:
                if self.is_3d:
                    corners_I = p_i + (R_b2I @ self._rov_body_corners_b.T).T
                    for k, (a, b) in enumerate(self._rov_body_edges):
                        self._set_3d_line(
                            self.rov_body_lines[i][k], corners_I[a], corners_I[b]
                        )
                else:
                    ex2 = R_b2I[:2, 0]
                    ey2 = R_b2I[:2, 1]
                    ex2 /= max(np.linalg.norm(ex2), 1e-9)
                    ey2 /= max(np.linalg.norm(ey2), 1e-9)
                    c = p_i[:2]
                    hx = 0.5 * self.rov_len
                    hy = 0.5 * self.rov_wid
                    corners = [
                        c + hx * ex2 + hy * ey2,
                        c + hx * ex2 - hy * ey2,
                        c - hx * ex2 - hy * ey2,
                        c - hx * ex2 + hy * ey2,
                    ]
                    for k, (a, b) in enumerate(self._rov_body_edges):
                        self._safe_set_data(
                            self.rov_body_lines[i][k],
                            [corners[a][0], corners[b][0]],
                            [corners[a][1], corners[b][1]],
                        )

            ex = R_b2I @ np.array([1.0, 0.0, 0.0])
            ey = R_b2I @ np.array([0.0, 1.0, 0.0])
            ez = R_b2I @ np.array([0.0, 0.0, 1.0])

            p_ex = p_i + self.frame_axis_len * ex
            p_ey = p_i + self.frame_axis_len * ey
            if self.is_3d:
                p_ez = p_i + self.frame_axis_len * ez
                self._set_3d_line(self.frame_x_lines[i], p_i, p_ex)
                self._set_3d_line(self.frame_y_lines[i], p_i, p_ey)
                self._set_3d_line(self.frame_z_lines[i], p_i, p_ez)
            else:
                self._set_2d_line(self.frame_x_lines[i], p_i, p_ex)
                self._set_2d_line(self.frame_y_lines[i], p_i, p_ey)

            if not self.fov_rays:
                continue

            dbg_i = None
            if frame < len(self.debug_history):
                step_dbg = self.debug_history[frame]
                if isinstance(step_dbg, list) and i < len(step_dbg):
                    dbg_i = step_dbg[i]

            if not isinstance(dbg_i, dict) or not dbg_i.get("fov_active", False):
                for ray in self.fov_rays[i]:
                    self._clear_line(ray)
                continue

            alpha_h_lim = float(dbg_i.get("alpha_h_lim", 1.0))
            alpha_v_lim = float(dbg_i.get("alpha_v_lim", 1.0))
            tan_h = np.tan(self.cfg.theta_max_x) * alpha_h_lim
            tan_v = np.tan(self.cfg.theta_max_y) * alpha_v_lim

            p_cam = p_i + R_b2I @ self._p_cb
            if self.is_3d:
                corners_c = [
                    np.array([1.0, +tan_h, +tan_v]),
                    np.array([1.0, +tan_h, -tan_v]),
                    np.array([1.0, -tan_h, -tan_v]),
                    np.array([1.0, -tan_h, +tan_v]),
                ]
                corners_I = []
                for c_c in corners_c:
                    c_b = self._R_cb.T @ c_c
                    c_I = p_cam + self.fov_pyramid_len * (R_b2I @ c_b)
                    corners_I.append(c_I)

                # Apex-to-corner edges
                for k in range(4):
                    self._set_3d_line(self.fov_rays[i][k], p_cam, corners_I[k])

                # Base perimeter edges
                base_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
                for k, (a, b) in enumerate(base_pairs, start=4):
                    self._set_3d_line(self.fov_rays[i][k], corners_I[a], corners_I[b])
            else:
                c0_c = np.array([1.0, +tan_h, 0.0])
                c1_c = np.array([1.0, -tan_h, 0.0])
                c0_I = p_cam + self.fov_pyramid_len * (R_b2I @ (self._R_cb.T @ c0_c))
                c1_I = p_cam + self.fov_pyramid_len * (R_b2I @ (self._R_cb.T @ c1_c))

                # Small local FoV triangle in 2D.
                self._set_2d_line(self.fov_rays[i][0], p_cam, c0_I)
                self._set_2d_line(self.fov_rays[i][1], p_cam, c1_I)
                self._set_2d_line(self.fov_rays[i][2], c0_I, c1_I)

    def _barrier_scalar(self, r):
        r = np.asarray(r, dtype=float)
        r2 = r**2
        eps = self.cfg.lgv_epsilon
        out = np.zeros_like(r, dtype=float)

        if self.cfg.conn:
            out += np.log(self.cfg.d_max**2 / np.maximum(self.cfg.d_max**2 - r2, eps))
        if self.cfg.coll:
            out += np.log(self.cfg.d_min**2 / np.maximum(r2 - self.cfg.d_min**2, eps))
        return out

    def _connectivity_barrier_scalar(self, r):
        r = np.asarray(r, dtype=float)
        r2 = r**2
        eps = self.cfg.lgv_epsilon
        if not self.cfg.conn:
            return np.zeros_like(r, dtype=float)
        return np.log(self.cfg.d_max**2 / np.maximum(self.cfg.d_max**2 - r2, eps))

    def _collision_barrier_scalar(self, r):
        r = np.asarray(r, dtype=float)
        r2 = r**2
        eps = self.cfg.lgv_epsilon
        if not self.cfg.coll:
            return np.zeros_like(r, dtype=float)
        return np.log(self.cfg.d_min**2 / np.maximum(r2 - self.cfg.d_min**2, eps))

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
        eps = self.cfg.lgv_epsilon

        grad = np.zeros_like(d)

        if self.cfg.conn:
            grad += (2.0 / np.maximum(self.cfg.d_max**2 - r2, eps)) * d
        if self.cfg.coll:
            grad += -(2.0 / np.maximum(r2 - self.cfg.d_min**2, eps)) * d

        return grad

    def _barrier_edge_reference_2d(self):
        """
        Pick a desired relative vector from an actual graph edge for plotting.

        Returns
        -------
        (i, j), d_des_2d
            Directed edge (i, j) and desired relative vector projected to XY.
        """
        if len(self.edge_pairs) == 0:
            return None, np.array([self.cfg.R, 0.0], dtype=float)

        i, j = self.edge_pairs[0]
        d_des = np.asarray(self.scenario.relative_pos_desired[i, j], dtype=float)

        if d_des.size >= 2:
            d_des_2d = d_des[:2]
        else:
            d_des_2d = np.array([self.cfg.R, 0.0], dtype=float)

        if np.linalg.norm(d_des_2d) < 1e-12:
            d_des_2d = np.array([self.cfg.R, 0.0], dtype=float)

        return (i, j), d_des_2d

    # ============================================================
    # update plots
    # ============================================================

    def plot_handle(self, frame, u_history):
        x, v, q = self._get_frame_state(frame)
        t = np.arange(frame + 1) * self.cfg.dt

        # Workspace agents and trails
        for i in range(self.N):
            self.dots[i].set_data([x[i, 0]], [x[i, 1]])
            if self.is_3d:
                self.dots[i].set_3d_properties([x[i, 2]])

            self._safe_set_data(
                self.trails[i],
                self.data.history[: frame + 1, i, 0],
                self.data.history[: frame + 1, i, 1],
            )
            if self.is_3d:
                self.trails[i].set_3d_properties(self.data.history[: frame + 1, i, 2])

        # Workspace edges + telemetry distances
        cmap = plt.get_cmap("RdYlGn_r")
        current_max_dist = 0.0

        t_lim, d_min_lim, d_max_lim = self._extract_distance_limits(frame)
        if self.dist_dmax_line is not None:
            self._safe_set_data(self.dist_dmax_line, t_lim, d_max_lim)
        if self.dist_dmin_line is not None:
            self._safe_set_data(self.dist_dmin_line, t_lim, d_min_lim)

        for idx, (i, j) in enumerate(self.edge_pairs):
            rel_hist = (
                self.data.history[: frame + 1, i, :]
                - self.data.history[: frame + 1, j, :]
            )
            dist_hist = np.linalg.norm(rel_hist, axis=1)
            self._safe_set_data(self.dist_lines[idx], t, dist_hist)

            pi, pj = x[i], x[j]
            self.edge_lines[idx].set_data([pi[0], pj[0]], [pi[1], pj[1]])
            if self.is_3d:
                self.edge_lines[idx].set_3d_properties([pi[2], pj[2]])

            d = dist_hist[-1] if dist_hist.size else 0.0
            current_max_dist = max(current_max_dist, d)

            target = getattr(self.cfg, "R", 0.5 * (self.cfg.d_min + self.cfg.d_max))
            if d > target:
                denom = max(self.cfg.d_max - target, 1e-9)
                cval = (d - target) / denom
            else:
                denom = max(target - self.cfg.d_min, 1e-9)
                cval = (target - d) / denom

            self.edge_lines[idx].set_color(cmap(np.clip(cval, 0.0, 1.0)))

        # Workspace body frames + FoV overlays
        self._update_frame_and_fov_overlays(frame, x, q)

        # Telemetry per agent
        for i in range(self.N):
            self._safe_set_data(
                self.err_lines[i], t, self.data.err_history[: frame + 1, i]
            )
            self._safe_set_data(
                self.vel_lines[i], t, self.data.vel_history[: frame + 1, i]
            )
            self._safe_set_data(
                self.acc_lines[i], t, self.data.accel_history[: frame + 1, i]
            )

        # Applied force / command components
        if u_history is not None and len(u_history) > 0:
            u_hist = np.asarray(u_history[: frame + 1], dtype=float)
            # u_hist shape: (frame+1, N, input_dim)
            if self._force_comp_slices is None:
                # Thruster mode: 8×N lines on panel 0
                for k in range(8):
                    for i in range(self.N):
                        line = self.force_lines[k * self.N + i]
                        if u_hist.shape[-1] > k:
                            self._safe_set_data(line, t, u_hist[:, i, k])
            else:
                # 3-component mode: panel k shows component k for all agents
                for k in range(3):
                    for i in range(self.N):
                        line = self.force_lines[k * self.N + i]
                        if u_hist.shape[-1] > k:
                            self._safe_set_data(line, t, u_hist[:, i, k])
                # Torque panels (components 3,4,5 — BlueROV2 non-thruster)
                for k in range(len(self.ax_torque_panels)):
                    comp = 3 + k
                    for i in range(self.N):
                        line = self.torque_lines[k * self.N + i]
                        if u_hist.shape[-1] > comp:
                            self._safe_set_data(line, t, u_hist[:, i, comp])

        for k, _ax in enumerate(self.ax_force_panels):
            line_p = self.force_limit_pos_lines[k]
            line_n = self.force_limit_neg_lines[k]
            if line_p is None or line_n is None:
                continue
            t_f, lim_f = self._extract_force_limits(frame, k)
            if lim_f is None:
                continue
            self._safe_set_data(line_p, t_f, lim_f)
            self._safe_set_data(line_n, t_f, -lim_f)

        # Mean formation error in meters
        if self.N > 1:
            mean_err = np.mean(self.data.err_history[: frame + 1, 1:], axis=1)
        else:
            mean_err = self.data.err_history[: frame + 1, 0]
        self._safe_set_data(self.err_mean_line, t, mean_err)

        # FOV telemetry
        if self.has_fov:
            for i in range(self.N):
                self._safe_set_data(
                    self.fov_x_lines[i], t, self.data.fov_history[: frame + 1, i, 0]
                )
                self._safe_set_data(
                    self.fov_y_lines[i], t, self.data.fov_history[: frame + 1, i, 1]
                )

        # Axes limits
        T = self.cfg.steps * self.cfg.dt
        self.ax_dist.set_xlim(0, T)
        self.ax_vel.set_xlim(0, T)
        self.ax_acc.set_xlim(0, T)
        self.ax_err.set_xlim(0, T)
        for ax in self.ax_force_panels:
            ax.set_xlim(0, T)
        for ax in self.ax_torque_panels:
            ax.set_xlim(0, T)

        if frame >= 0:
            self.ax_dist.set_ylim(
                0,
                max(
                    np.max(d_max_lim) * 1.1 if d_max_lim.size else self.cfg.d_max * 1.1,
                    current_max_dist * 1.1,
                ),
            )
            self.ax_vel.set_ylim(
                0,
                max(
                    self.cfg.v_max * 1.1,
                    np.max(self.data.vel_history[: frame + 1]) * 1.1,
                ),
            )
            self.ax_acc.set_ylim(
                0,
                max(
                    self.cfg.a_max * 1.1,
                    np.max(self.data.accel_history[: frame + 1]) * 1.1,
                ),
            )
            self.ax_err.set_ylim(
                0,
                max(
                    0.1,
                    np.max(self.data.err_history[: frame + 1]) * 1.1,
                ),
            )
            # Force panel ylim: based on actual data + limit reference
            if u_history is not None and len(u_history) > 0:
                u_hist = np.asarray(u_history[: frame + 1], dtype=float)
                force_ref = self._force_limit
                if self._force_comp_slices is None:
                    # Thruster mode: all 8 components
                    n_comp = min(8, u_hist.shape[-1])
                    fdata = u_hist[:, :, :n_comp]
                    _, lim_series = self._extract_force_limits(frame, 0)
                    lim_ref = (
                        np.max(np.abs(lim_series))
                        if lim_series is not None
                        else force_ref
                    )
                    y_top = max(
                        np.max(np.abs(fdata)) * 1.1 if fdata.size else 0.1,
                        (lim_ref * 1.2 if lim_ref is not None else 0.1),
                        0.1,
                    )
                    self.ax_force_panels[0].set_ylim(-y_top, y_top)
                else:
                    for k, ax in enumerate(self.ax_force_panels):
                        if u_hist.shape[-1] > k:
                            fvals = u_hist[:, :, k]
                            _, lim_series = self._extract_force_limits(frame, k)
                            lim_ref = (
                                np.max(np.abs(lim_series))
                                if lim_series is not None
                                else force_ref
                            )
                            y_top = max(
                                np.max(np.abs(fvals)) * 1.1 if fvals.size else 0.1,
                                (lim_ref * 1.2 if lim_ref is not None else 0.1),
                                0.1,
                            )
                            ax.set_ylim(-y_top, y_top)
                    # Torque panels auto-scale
                    for k, ax in enumerate(self.ax_torque_panels):
                        comp = 3 + k
                        if u_hist.shape[-1] > comp:
                            tvals = u_hist[:, :, comp]
                            y_top = max(
                                np.max(np.abs(tvals)) * 1.1 if tvals.size else 0.1, 0.1
                            )
                            ax.set_ylim(-y_top, y_top)

        if self.has_fov:
            self.ax_fov_x.set_xlim(0, T)
            self.ax_fov_y.set_xlim(0, T)
            max_x = max(
                np.degrees(self.cfg.theta_max_x) + 5.0,
                np.max(np.abs(self.data.fov_history[: frame + 1, :, 0])) * 1.1,
            )
            max_y = max(
                np.degrees(self.cfg.theta_max_y) + 5.0,
                np.max(np.abs(self.data.fov_history[: frame + 1, :, 1])) * 1.1,
            )
            self.ax_fov_x.set_ylim(-max_x, max_x)
            self.ax_fov_y.set_ylim(-max_y, max_y)

        self._workspace_limits(frame)

        # Telemetry text
        stats = []
        stats.append(f"time = {frame * self.cfg.dt:6.2f} s")
        stats.append(f"mode = {self.cfg.simulation_mode}")
        stats.append("-" * 24)
        current_mean_err = (
            np.mean(self.data.err_history[frame, 1:]) if self.N > 1 else 0.0
        )
        stats.append(f"mean err = {current_mean_err:6.3f} m")
        for i in range(self.N):
            stats.append(f"A{i}: |v| = {self.data.vel_history[frame, i]:5.2f}")
        self.telemetry_text.set_text("\n".join(stats))

    # ============================================================
    # topology / barriers
    # ============================================================

    def plot_topology(self, save_path=None, show=False):
        positions = (
            self.data.history[0] if len(self.data.history) > 0 else self.scenario.x
        )
        return plot_graph_from_adjacency(
            positions,
            self.scenario.A,
            self.cfg.mode_graph,
            title="Graph Topology",
            show=show,
        )

    def plot_barrier(self, show=True, save_prefix=None):
        """
        Plot both:
        1) scalar radial barrier slice (intuition)
        2) full vector recentered barrier surface in 2D
        """
        grid_limit = 1.5 * self.cfg.d_max
        edge_ref, d_des = self._barrier_edge_reference_2d()
        d_des_norm = float(np.linalg.norm(d_des))
        unit_des = d_des / max(d_des_norm, 1e-12)

        # ---------- scalar slice along selected edge direction ----------
        s_range = np.linspace(-grid_limit, grid_limit, 1200)
        D_line = s_range[:, None] * unit_des[None, :]
        r_line = np.linalg.norm(D_line, axis=1)
        B_conn_vals = self._connectivity_barrier_scalar(r_line)
        B_coll_vals = self._collision_barrier_scalar(r_line)
        B_vals = self._barrier_vector(D_line)
        B_des = self._barrier_vector(d_des[None, :])[0]
        grad_B_des = self._grad_barrier(d_des[None, :])[0]
        B_rec_slice = B_vals - B_des - ((D_line - d_des[None, :]) @ grad_B_des)

        fig1 = plt.figure("Barrier Slice", figsize=(8, 5))
        plt.plot(s_range, B_vals, "--", color="gray", alpha=0.7, label="Barrier")
        plt.plot(s_range, B_rec_slice, "b", lw=2, label="Recentered slice")
        plt.axvline(d_des_norm, color="g", ls=":", label="Target norm")
        plt.axvline(-d_des_norm, color="g", ls=":")
        plt.axvline(self.cfg.d_max, color="r", ls=":", label="d_max")
        plt.axvline(-self.cfg.d_max, color="r", ls=":")
        plt.axvline(self.cfg.d_min, color="darkred", ls=":", label="d_min")
        plt.axvline(-self.cfg.d_min, color="darkred", ls=":")
        if edge_ref is None:
            plt.title("Scalar Barrier Slice (fallback desired vector)")
        else:
            plt.title(f"Scalar Barrier Slice Along Edge {edge_ref[0]}->{edge_ref[1]}")
        plt.xlabel("Coordinate along selected desired direction")
        plt.ylabel("Potential")
        plt.grid(alpha=0.3)
        plt.legend()

        # ---------- separate barrier components ----------
        fig3, (axc, axk) = plt.subplots(1, 2, figsize=(12, 4.5))

        axc.plot(s_range, B_conn_vals, color="tab:green", lw=2, label="B_conn")
        axc.axvline(self.cfg.d_max, color="r", ls=":", label="d_max")
        axc.axvline(-self.cfg.d_max, color="r", ls=":")
        axc.set_title("Connectivity Barrier")
        axc.set_xlabel("Coordinate along selected desired direction")
        axc.set_ylabel("Potential")
        axc.grid(alpha=0.3)
        axc.legend()

        axk.plot(s_range, B_coll_vals, color="tab:orange", lw=2, label="B_coll")
        axk.axvline(self.cfg.d_min, color="darkred", ls=":", label="d_min")
        axk.axvline(-self.cfg.d_min, color="darkred", ls=":")
        axk.set_title("Collision-Avoidance Barrier")
        axk.set_xlabel("Coordinate along selected desired direction")
        axk.set_ylabel("Potential")
        axk.grid(alpha=0.3)
        axk.legend()

        if edge_ref is None:
            fig3.suptitle("Barrier Components (fallback desired vector)")
        else:
            fig3.suptitle(f"Barrier Components Along Edge {edge_ref[0]}->{edge_ref[1]}")
        fig3.tight_layout()

        # ---------- full vector recentered barrier in 2D ----------
        res = 150
        x = np.linspace(-grid_limit, grid_limit, res)
        y = np.linspace(-grid_limit, grid_limit, res)
        X, Y = np.meshgrid(x, y)
        D = np.stack([X, Y], axis=-1)

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
            label="Desired equilibrium (selected edge)",
        )
        if edge_ref is None:
            ax2.set_title("Full Vector Recentered Barrier Surface (fallback desired)")
        else:
            ax2.set_title(
                f"Full Vector Recentered Barrier Surface (edge {edge_ref[0]}->{edge_ref[1]})"
            )
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
            fig3.savefig(
                f"{save_prefix}_barrier_components.png", dpi=200, bbox_inches="tight"
            )

        if show:
            plt.show()

        return fig1, fig2, fig3

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
        n_frames_total = min(params.config.steps, len(params.data.history))
        stride = max(1, int(getattr(self.cfg, "animation_frame_stride", 1)))
        frame_indices = range(0, n_frames_total, stride)

        def update_frame(frame):
            self.plot_handle(frame, u_history)
            artists = (
                self.dots
                + self.trails
                + self.edge_lines
                + self.frame_x_lines
                + self.frame_y_lines
                + self.dist_lines
                + self.err_lines
                + [self.err_mean_line]
                + self.vel_lines
                + self.acc_lines
                + self.force_lines
                + [self.telemetry_text]
            )
            if self.rov_body_lines:
                for edges_i in self.rov_body_lines:
                    artists += edges_i
            if self.is_3d:
                artists += self.frame_z_lines
            if self.fov_rays:
                for rays_i in self.fov_rays:
                    artists += rays_i
            if self.has_fov:
                artists += self.fov_x_lines + self.fov_y_lines
            return artists

        self.animation = FuncAnimation(
            self.fig_space,
            update_frame,
            frames=frame_indices,
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
        if len(params.data.history) == 0:
            plt.show()
            return

        frame = min(params.config.steps - 1, len(params.data.history) - 1)
        self.plot_handle(frame, u_history)

        if save_prefix is not None:
            self.fig_space.savefig(
                f"{save_prefix}_workspace.png", dpi=200, bbox_inches="tight"
            )
            self.fig_telemetry.savefig(
                f"{save_prefix}_telemetry.png", dpi=200, bbox_inches="tight"
            )
            self.fig_force_torque.savefig(
                f"{save_prefix}_force_torque.png", dpi=200, bbox_inches="tight"
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
