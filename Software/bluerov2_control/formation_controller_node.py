#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
formation_controller_node.py
════════════════════════════

ARCHITECTURE
──────────────────────────────────────────────────────────
  PX4 VehicleOdometry  (self + neighbours)
      │  convert NED→NWU
      ▼
  BlueROV2Controller.do_control(p_target_I = leader_position_nwu)
      │  CLF-QP + FoV barrier (active only when target is in front of camera)
      │  outputs tau [Fx,Fy,Fz,Mx,My,Mz] NWU body frame
      ▼
  Pool boundary barrier (optional, additive force in body frame)
      ▼
  Rotate NWU→NED, normalise, clamp
      ▼
  VehicleThrustSetpoint + VehicleTorqueSetpoint → PX4

FoV DESIGN
───────────
The FoV barrier keeps p_target_I (the leader's CURRENT position, updated
every tick from its odometry) within the follower's camera frustum.

CRITICAL: The FoV model assumes the camera points along body x-axis and
the target is in FRONT (r_x > 0 in camera frame).  If the target is behind
or directly to the side, alpha_h/alpha_v blow up and OSQP fails.
Guard: FoV is only passed to do_control when the target vector has a
positive x-component in camera frame.  Otherwise the robot rotates freely
until the target comes back into view.

POOL BOUNDARY BARRIER
──────────────────────
Optional additive CBF repulsive force from tank walls/floor/surface.
Coordinates in NWU from kth_marinarium.sdf.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
from typing import Dict, Optional

from geometry_msgs.msg import Wrench
from px4_msgs.msg import (
    VehicleOdometry,
    VehicleThrustSetpoint,
    VehicleTorqueSetpoint,
    VehicleControlMode,
)

from geometry_msgs.msg import Vector3

from bluerov2_control.controller_refactored import BlueROV2Controller, BlueROV2Config
from bluerov2_control.blue_rov2_refactored import BlueROV2


# ──────────────────────────────────────────────────────────────────────────────
# Frame helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ned_quat_to_nwu(q_ned: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q_ned
    return np.array([qx, qy, qz, qw], dtype=float)

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _rotation_matrix_from_quaternion_nwu(q: np.ndarray) -> np.ndarray:
    """3×3 R_b2I from NWU [qx,qy,qz,qw]."""
    qx, qy, qz, qw = q
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [  2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qw*qx)],
        [  2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)],
    ], dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# FoV feasibility guard
# ──────────────────────────────────────────────────────────────────────────────

def _target_in_front_of_camera(
    p_i_nwu: np.ndarray,
    q_i_nwu: np.ndarray,
    p_target_nwu: np.ndarray,
    R_cb: np.ndarray,
    p_cb: np.ndarray,
    fov_eps: float = 1e-3,
) -> bool:
    """
    Returns True only if the target vector has a positive x-component in
    the camera frame (i.e. the target is in front of the camera).

    The FoV normalised coordinates alpha_h = r_y / (r_x tan_h) blow up
    when r_x ≤ 0, causing OSQP constraint bound overflow.
    We use a margin of fov_eps = 1e-3 m (not the tiny 1e-6 used inside
    the controller) to give a comfortable buffer before activating FoV.
    """
    rel_I = p_target_nwu - p_i_nwu          # inertial relative vector NWU
    R_b2I = _rotation_matrix_from_quaternion_nwu(q_i_nwu)
    r_b   = R_b2I.T @ rel_I                  # body frame NWU
    r_c   = R_cb @ (r_b - p_cb)              # camera frame
    return float(r_c[0]) > fov_eps


# ──────────────────────────────────────────────────────────────────────────────
# Pool boundary barrier
# ──────────────────────────────────────────────────────────────────────────────

class PoolBoundaryBarrier:
    """
    Reciprocal-quadratic CBF repulsive force from tank walls/floor/surface.
    All coordinates in NWU inertial frame.

    Pool geometry from kth_marinarium.sdf (tank at ENU z = -97.0):
      NWU x ∈ [-2.5, +2.5]    North–South, 5 m
      NWU y ∈ [-1.75, +1.75]  West–East,   3.5 m  (NWU y = ENU y)
      NWU z ∈ [-97.0, -94.4]  z = ENU z;  floor=-97, surface=-94.4
    """

    def __init__(
        self,
        x_min=-2.5, x_max=2.5,
        y_min=-1.75, y_max=1.75,
        z_min=-97.0, z_max=-94.4,
        margin=0.3, h_max=0.5, k_barrier=2.0,
    ):
        self._walls = [
            (0, +1.0, x_min + margin),
            (0, -1.0, x_max - margin),
            (1, +1.0, y_min + margin),
            (1, -1.0, y_max - margin),
            (2, +1.0, z_min - margin),
            (2, -1.0, 0.0),  # No upper z barrier (surface) since we only care about floor collision
        ]
        self._h_max     = max(h_max, 1e-6)
        self._k_barrier = k_barrier

    def compute(self, p_nwu: np.ndarray) -> np.ndarray:
        f = np.zeros(3, dtype=float)
        for axis, sign, limit in self._walls:
            h = sign * (p_nwu[axis] - limit)
            if h <= 0.0:
                h = 1e-4
            if h >= self._h_max:
                continue
            f_mag = self._k_barrier * (1.0/h - 1.0/self._h_max) / (h * h)
            grad = np.zeros(3)
            grad[axis] = sign
            f += f_mag * grad
        return f


# ──────────────────────────────────────────────────────────────────────────────
# Node
# ──────────────────────────────────────────────────────────────────────────────

class FormationControllerNode(Node):

    def __init__(self):
        super().__init__("formation_controller_node")

        # ── 1. Parameters ─────────────────────────────────────────────────────
        self.declare_parameter("agent_id",        0)
        self.declare_parameter("neighbors",       "")
        self.declare_parameter("desired_rel_x",   "")
        self.declare_parameter("desired_rel_y",   "")
        self.declare_parameter("desired_rel_z",   "")
        self.declare_parameter("agent_ns_prefix", "agent_")
        self.declare_parameter("mode",            "clf_qp_tau")
        self.declare_parameter("dt",              0.02)

        # CLF-QP
        self.declare_parameter("d_min",             1.0)
        self.declare_parameter("d_max",            10.0)
        self.declare_parameter("use_connectivity", True)
        self.declare_parameter("use_collision",    True)
        self.declare_parameter("use_recentered",   True)
        self.declare_parameter("k_form",            1.0)
        self.declare_parameter("clf_alpha",         2.0)
        self.declare_parameter("k_att",             5.0)
        self.declare_parameter("k_att_d",           2.0)

        # Normalisation
        self.declare_parameter("Fx_max_N",   88.0)
        self.declare_parameter("Fy_max_N",   88.0)
        self.declare_parameter("Fz_max_N",  137.0)
        self.declare_parameter("Mx_max_Nm",  30.0)
        self.declare_parameter("My_max_Nm",  16.5)
        self.declare_parameter("Mz_max_Nm",  21.0)
        self.declare_parameter("thrust_sat", 0.1)
        self.declare_parameter("torque_sat", 0.1)

        # FoV
        self.declare_parameter("use_fov",           False)
        self.declare_parameter("use_fov_torque",    True)
        # fov_target_id: which agent's position to track as FoV target.
        # For a follower this should be the leader ID.
        # -1 means use no FoV (same as use_fov=False).
        self.declare_parameter("fov_target_id",     -1)
        self.declare_parameter("theta_h_max_deg",   45.0)
        self.declare_parameter("theta_v_max_deg",   30.0)
        self.declare_parameter("mu_fov_pos",         1.0)
        self.declare_parameter("mu_fov_att",         1.0)
        self.declare_parameter("k_fov_att",          2.0)
        self.declare_parameter("k_fov_omega",        2.0)
        self.declare_parameter("k_fov_backstep",     3.0)
        self.declare_parameter("omega_fov_max",      1.0)
        self.declare_parameter("alpha_h_des",        0.0)
        self.declare_parameter("alpha_v_des",        0.0)
        self.declare_parameter("fov_eps",            1e-6)
        self.declare_parameter("use_adaptive_fov_barrier", False)
        self.declare_parameter("lambda_sigma_fov_h", 2.0)
        self.declare_parameter("gamma_sigma_fov_h",  1.0)
        self.declare_parameter("sigma_fov_h_max",    0.0)
        self.declare_parameter("lambda_sigma_fov_v", 2.0)
        self.declare_parameter("gamma_sigma_fov_v",  1.0)
        self.declare_parameter("sigma_fov_v_max",    0.0)
        # Guard margin: FoV only active when target is this many metres in front
        self.declare_parameter("fov_rx_min",         0.1)

        # Pool boundary barrier
        self.declare_parameter("use_pool_barrier",  False)
        self.declare_parameter("pool_x_min",  -3.3)
        self.declare_parameter("pool_x_max",   1.0)
        self.declare_parameter("pool_y_min",  -5.7)
        self.declare_parameter("pool_y_max",   1.3)
        self.declare_parameter("pool_z_min",  97.0) # There is no upper z barrier since we only care about floor collision, but we still need to set this to a valid number below the initial depth.
        self.declare_parameter("pool_z_max",  94.5)
        self.declare_parameter("pool_margin",  0.3)
        self.declare_parameter("pool_h_max",   0.5)
        self.declare_parameter("pool_k",       2.0)

        # ── 2. Read parameters ────────────────────────────────────────────────
        gv = lambda name: self.get_parameter(name).value

        self.agent_id: int = int(gv("agent_id"))
        self.mode: str     = gv("mode")
        dt: float          = float(gv("dt"))

        self.Fx_max  = float(gv("Fx_max_N"))
        self.Fy_max  = float(gv("Fy_max_N"))
        self.Fz_max  = float(gv("Fz_max_N"))
        self.Mx_max  = float(gv("Mx_max_Nm"))
        self.My_max  = float(gv("My_max_Nm"))
        self.Mz_max  = float(gv("Mz_max_Nm"))
        self.thr_sat = float(gv("thrust_sat"))
        self.tor_sat = float(gv("torque_sat"))

        # Neighbour graph
        nbr_str = str(gv("neighbors"))
        self.neighbor_ids: list[int] = (
            [int(v.strip()) for v in nbr_str.split(",") if v.strip()]
            if nbr_str.strip() else []
        )
        n_nbr = len(self.neighbor_ids)

        def _csv(s): return [float(v.strip()) for v in s.split(",") if v.strip()] if s.strip() else []
        def _pad(lst, n): return (list(lst) + [0.0] * n)[:n]

        rel_x = _pad(_csv(str(gv("desired_rel_x"))), n_nbr)
        rel_y = _pad(_csv(str(gv("desired_rel_y"))), n_nbr)
        rel_z = _pad(_csv(str(gv("desired_rel_z"))), n_nbr)

        self.desired_rel: np.ndarray = (
            np.array([[rel_x[k], rel_y[k], rel_z[k]] for k in range(n_nbr)], dtype=float)
            if n_nbr > 0 else np.empty((0, 3))
        )
        self.incidence_signs = np.ones(n_nbr, dtype=float)

        # FoV configuration
        self.use_fov: bool       = bool(gv("use_fov"))
        self.use_fov_torque: bool = bool(gv("use_fov_torque"))
        self.fov_target_id: int  = int(gv("fov_target_id"))
        self.fov_rx_min: float   = float(gv("fov_rx_min"))

        # ── 3. BlueROV2Config ─────────────────────────────────────────────────
        self.cfg = BlueROV2Config(
            dim=3,
            d_min=float(gv("d_min")),
            d_max=float(gv("d_max")),
            use_connectivity=bool(gv("use_connectivity")),
            use_collision=bool(gv("use_collision")),
            use_recentered=bool(gv("use_recentered")),
            k_form=float(gv("k_form")),
            clf_alpha=float(gv("clf_alpha")),
            k_att=float(gv("k_att")),
            k_att_d=float(gv("k_att_d")),
            dt=dt,
            use_fov=self.use_fov,
            theta_h_max=np.deg2rad(float(gv("theta_h_max_deg"))),
            theta_v_max=np.deg2rad(float(gv("theta_v_max_deg"))),
            mu_fov_pos=float(gv("mu_fov_pos")),
            mu_fov_att=float(gv("mu_fov_att")),
            k_fov_att=float(gv("k_fov_att")),
            k_fov_omega=float(gv("k_fov_omega")),
            k_fov_backstep=float(gv("k_fov_backstep")),
            omega_fov_max=float(gv("omega_fov_max")),
            alpha_h_des=float(gv("alpha_h_des")),
            alpha_v_des=float(gv("alpha_v_des")),
            fov_eps=float(gv("fov_eps")),
            use_adaptive_fov_barrier=bool(gv("use_adaptive_fov_barrier")),
            lambda_sigma_fov_h=float(gv("lambda_sigma_fov_h")),
            gamma_sigma_fov_h=float(gv("gamma_sigma_fov_h")),
            sigma_fov_h_max=float(gv("sigma_fov_h_max")),
            lambda_sigma_fov_v=float(gv("lambda_sigma_fov_v")),
            gamma_sigma_fov_v=float(gv("gamma_sigma_fov_v")),
            sigma_fov_v_max=float(gv("sigma_fov_v_max")),
        )
        self.rov_model  = BlueROV2()
        self.controller = BlueROV2Controller(self.agent_id, self.cfg, self.rov_model)

        # ── 4. Pool barrier ───────────────────────────────────────────────────
        self.use_pool_barrier: bool = bool(gv("use_pool_barrier"))
        self.pool_barrier = PoolBoundaryBarrier(
            x_min=float(gv("pool_x_min")), x_max=float(gv("pool_x_max")),
            y_min=float(gv("pool_y_min")), y_max=float(gv("pool_y_max")),
            z_min=float(gv("pool_z_min")), z_max=float(gv("pool_z_max")),
            margin=float(gv("pool_margin")),
            h_max=float(gv("pool_h_max")),
            k_barrier=float(gv("pool_k")),
        )

        # ── 5. State ──────────────────────────────────────────────────────────
        self.p_i: Optional[np.ndarray] = None
        self.q_i   = np.array([0.0, 0.0, 0.0, 1.0])
        self.nu_i  = np.zeros(6)
        self.enabled = False

        # All agent positions in NWU — includes neighbours AND the FoV target.
        # Initialised to None so we wait for real data before using them.
        self.agent_positions: Dict[int, Optional[np.ndarray]] = {}
        all_tracked_ids = set(self.neighbor_ids)
        if self.use_fov and self.fov_target_id >= 0:
            all_tracked_ids.add(self.fov_target_id)
        for aid in all_tracked_ids:
            self.agent_positions[aid] = None

        # ── 6. QoS ───────────────────────────────────────────────────────────
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── 7. Publishers ─────────────────────────────────────────────────────
        ns_prefix = str(gv("agent_ns_prefix"))
        ns = f"/{ns_prefix}{self.agent_id}"

        self.thrust_pub = self.create_publisher(
            VehicleThrustSetpoint, f"{ns}/fmu/in/vehicle_thrust_setpoint", px4_qos)
        self.torque_pub = self.create_publisher(
            VehicleTorqueSetpoint, f"{ns}/fmu/in/vehicle_torque_setpoint", px4_qos)
        self.wrench_pub = self.create_publisher(Wrench, f"{ns}/debug_wrench", 10)
        self.barrier_pub = self.create_publisher(Wrench, f"{ns}/debug_barrier", 10)
        self.fov_pub = self.create_publisher(Vector3, f"{ns}/fov_error", 10)

        # ── 8. Subscribers ────────────────────────────────────────────────────
        self.create_subscription(
            VehicleOdometry, f"{ns}/fmu/out/vehicle_odometry",
            self._self_odom_cb, px4_qos)
        self.create_subscription(
            VehicleControlMode, f"{ns}/fmu/out/vehicle_control_mode",
            self._control_mode_cb, px4_qos)

        # Subscribe to every agent we track (neighbours + FoV target)
        for aid in all_tracked_ids:
            topic = f"/{ns_prefix}{aid}/fmu/out/vehicle_odometry"
            self.create_subscription(
                VehicleOdometry, topic, self._make_agent_cb(aid), px4_qos)
            self.get_logger().info(
                f"[agent_{self.agent_id}] Subscribing to agent {aid}: {topic}")

        # ── 9. Timer ──────────────────────────────────────────────────────────
        self.create_timer(dt, self._control_step)

        self.get_logger().info(
            f"\n{'='*55}\n"
            f"  agent_id       : {self.agent_id}\n"
            f"  neighbor_ids   : {self.neighbor_ids}\n"
            f"  desired_rel    :\n{self.desired_rel}\n"
            f"  use_fov        : {self.use_fov}\n"
            f"  fov_target_id  : {self.fov_target_id}\n"
            f"  use_pool_barrier: {self.use_pool_barrier}\n"
            f"  thrust_sat     : {self.thr_sat}  torque_sat: {self.tor_sat}\n"
            f"{'='*55}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def _control_mode_cb(self, msg: VehicleControlMode) -> None:
        gate = bool(msg.flag_armed) and bool(msg.flag_control_offboard_enabled)
        if gate and not self.enabled:
            self.enabled = True
            self.get_logger().info(f"[agent_{self.agent_id}] Enabled.")
        elif (not gate) and self.enabled:
            self.enabled = False
            self.get_logger().info(f"[agent_{self.agent_id}] Disabled.")
            self._publish_zero()

    def _self_odom_cb(self, msg: VehicleOdometry) -> None:
        self.p_i  = np.array(msg.position,         dtype=float)
        self.q_i  = _ned_quat_to_nwu(np.array(msg.q,               dtype=float))
        self.nu_i[:3] = np.array(msg.velocity,         dtype=float)
        self.nu_i[3:] = np.array(msg.angular_velocity, dtype=float)

    def _make_agent_cb(self, aid: int):
        """Single callback factory for all tracked agents (neighbours + FoV target)."""
        def _cb(msg: VehicleOdometry) -> None:
            self.agent_positions[aid] = np.array(msg.position, dtype=float)
        return _cb

    # ──────────────────────────────────────────────────────────────────────────
    # Control step
    # ──────────────────────────────────────────────────────────────────────────

    def _control_step(self) -> None:

        if not self.enabled:
            return

        if self.p_i is None:
            self.get_logger().warn(
                f"[agent_{self.agent_id}] Waiting for self odom...",
                throttle_duration_sec=5.0)
            self._publish_zero()
            return

        # ── Pool barrier force (inertial NWU) ─────────────────────────────────
        f_barrier_I = np.zeros(3)
        if self.use_pool_barrier:
            f_barrier_I = self.pool_barrier.compute(self.p_i)
            bw = Wrench()
            bw.force.x, bw.force.y, bw.force.z = (
                float(f_barrier_I[0]), float(f_barrier_I[1]), float(f_barrier_I[2]))
            self.barrier_pub.publish(bw)

        # ── Determine FoV target (dynamic — current leader position) ───────────
        p_target_I: Optional[np.ndarray] = None
        tau_extra = np.zeros(3)

        if self.use_fov and self.fov_target_id >= 0:
            p_target_raw = self.agent_positions.get(self.fov_target_id)

            if p_target_raw is not None:
                rel_I = p_target_raw - self.p_i
                R_b2I = _rotation_matrix_from_quaternion_nwu(self.q_i)
                r_b = R_b2I.T @ rel_I

                fov_feasible = _target_in_front_of_camera(
                    p_i_nwu=self.p_i,
                    q_i_nwu=self.q_i,
                    p_target_nwu=p_target_raw,
                    R_cb=self.cfg.R_cb,
                    p_cb=self.cfg.p_cb,
                    fov_eps=self.fov_rx_min,
                )

                if fov_feasible:
                    p_target_I = p_target_raw
                else:
                    p_target_I = None

                    # --- safe rotation control ---
                    rx = np.clip(r_b[0], 0.05, None)

                    yaw_err   = np.arctan2(r_b[1], rx)
                    pitch_err = np.arctan2(-r_b[2], rx)

                    k_rot = 1.0

                    tau_extra = np.array([
                        0.0,
                        np.clip(k_rot * pitch_err, -1.0, 1.0),
                        np.clip(k_rot * yaw_err,   -1.0, 1.0),
                    ])
                    
        # ── Leader: publish barrier force only (no formation task) ────────────
        if len(self.neighbor_ids) == 0:
            self._publish_zero()
            return

        # ── Wait for all neighbour odoms ──────────────────────────────────────
        for nid in self.neighbor_ids:
            if self.agent_positions.get(nid) is None:
                self.get_logger().warn(
                    f"[agent_{self.agent_id}] Waiting for agent {nid} odom...",
                    throttle_duration_sec=5.0)
                self._publish_zero()
                return

        # ── Relative states in NWU ────────────────────────────────────────────
        n_nbr = len(self.neighbor_ids)
        relative_states = np.zeros((n_nbr, 3), dtype=float)
        for k, nid in enumerate(self.neighbor_ids):
            relative_states[k] = self.p_i - self.agent_positions[nid]

        # ── CLF-QP ────────────────────────────────────────────────────────────
        try:
            tau_nwu, debug = self.controller.do_control(
                p_i=self.p_i,
                q_i=self.q_i,
                nu_i=self.nu_i,
                relative_states=relative_states,
                desired_relative_states=self.desired_rel,
                incidence_signs=self.incidence_signs,
                mode=self.mode,
                p_target_I=p_target_I,        # None when target is behind camera
                use_fov_torque=self.use_fov_torque,
            )
        except Exception as exc:
            self.get_logger().error(
                f"[agent_{self.agent_id}] do_control error: {exc}",
                throttle_duration_sec=2.0)
            self._publish_zero()
            return

        tau_nwu = np.asarray(tau_nwu, dtype=float).reshape(6)
        if isinstance(debug, dict):
            self._publish_fov_error(debug)

        # If FoV is enabled but the target is NOT feasible (p_target_I is None),
        # we apply the search torques calculated earlier.
        if self.use_fov and p_target_I is None and 'tau_extra' in locals():
            # tau_extra[1] is Pitch, tau_extra[2] is Yaw
            tau_nwu[4] += tau_extra[1] 
            tau_nwu[5] += tau_extra[2]

        if debug.get("mode") == "fallback":
            self.get_logger().warn(
                f"[agent_{self.agent_id}] QP fallback: {debug.get('reason')}",
                throttle_duration_sec=5.0)

        # ── Add pool barrier in body frame ────────────────────────────────────
        if self.use_pool_barrier and np.any(f_barrier_I != 0.0):
            R_b2I = _rotation_matrix_from_quaternion_nwu(self.q_i)
            tau_nwu[:3] += R_b2I.T @ f_barrier_I

        self._publish_tau(tau_nwu)

    # ──────────────────────────────────────────────────────────────────────────
    # Publishing
    # ──────────────────────────────────────────────────────────────────────────

    def _publish_tau(self, tau_nwu: np.ndarray) -> None:
        now_us = int(self.get_clock().now().nanoseconds / 1000)

        force_ned  = tau_nwu[:3]
        torque_ned = tau_nwu[3:]

        thr_x = _clamp(force_ned[0]  / max(self.Fx_max, 1e-6), -self.thr_sat, self.thr_sat)
        thr_y = _clamp(force_ned[1]  / max(self.Fy_max, 1e-6), -self.thr_sat, self.thr_sat)
        thr_z = _clamp(force_ned[2]  / max(self.Fz_max, 1e-6), -self.thr_sat, self.thr_sat)
        tor_x = _clamp(torque_ned[0] / max(self.Mx_max, 1e-6), -self.tor_sat, self.tor_sat)
        tor_y = _clamp(torque_ned[1] / max(self.My_max, 1e-6), -self.tor_sat, self.tor_sat)
        tor_z = _clamp(torque_ned[2] / max(self.Mz_max, 1e-6), -self.tor_sat, self.tor_sat)

        thr = VehicleThrustSetpoint()
        thr.timestamp = now_us; thr.timestamp_sample = 0
        thr.xyz = [float(thr_x), float(thr_y), float(thr_z)]
        self.thrust_pub.publish(thr)

        tor = VehicleTorqueSetpoint()
        tor.timestamp = now_us; tor.timestamp_sample = 0
        tor.xyz = [float(tor_x), float(tor_y), float(tor_z)]
        self.torque_pub.publish(tor)

        w = Wrench()
        w.force.x,  w.force.y,  w.force.z  = float(tau_nwu[0]), float(tau_nwu[1]), float(tau_nwu[2])
        w.torque.x, w.torque.y, w.torque.z = float(tau_nwu[3]), float(tau_nwu[4]), float(tau_nwu[5])
        self.wrench_pub.publish(w)
    
    def _publish_fov_error(self, debug: dict) -> None:
        """Publish raw FoV angular errors (same values shown in logs)"""
        msg = Vector3()

        if debug.get("fov_active", False):
            theta_h = float(debug.get("theta_h", 0.0))
            theta_v = float(debug.get("theta_v", 0.0))

            msg.x = theta_h   # horizontal error [rad]
            msg.y = theta_v   # vertical error [rad]
            msg.z = 0.0
        else:
            msg.x = msg.y = msg.z = 0.0

        self.fov_pub.publish(msg)

    def _publish_zero(self) -> None:
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        thr = VehicleThrustSetpoint()
        thr.timestamp = now_us; thr.timestamp_sample = 0; thr.xyz = [0.,0.,0.]
        self.thrust_pub.publish(thr)
        tor = VehicleTorqueSetpoint()
        tor.timestamp = now_us; tor.timestamp_sample = 0; tor.xyz = [0.,0.,0.]
        self.torque_pub.publish(tor)


def main(args=None):
    rclpy.init(args=args)
    node = FormationControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
