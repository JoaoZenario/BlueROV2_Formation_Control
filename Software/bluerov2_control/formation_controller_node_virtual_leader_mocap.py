#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import rclpy
import numpy as np

from typing import Dict, Optional

from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from geometry_msgs.msg import Wrench, Vector3, Point
from nav_msgs.msg import Odometry

from px4_msgs.msg import (
    VehicleThrustSetpoint,
    VehicleTorqueSetpoint,
    VehicleControlMode,
)

from bluerov2_control.controller_refactored import (
    BlueROV2Controller,
    BlueROV2Config,
)

from bluerov2_control.blue_rov2_refactored import BlueROV2


# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _ned_quat_to_nwu(q_ned: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q_ned
    return np.array([qx, qy, qz, qw], dtype=float)


def _rotation_matrix_from_quaternion_nwu(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q

    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz),     1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy),     2*(qy*qz+qw*qx),   1-2*(qx*qx+qy*qy)],
    ], dtype=float)


# ─────────────────────────────────────────────────────
# Pool barrier
# ─────────────────────────────────────────────────────

class PoolBoundaryBarrier:

    def __init__(
        self,
        x_min=-3.3,
        x_max=1.0,
        y_min=-5.7,
        y_max=1.3,
        z_min=97.0,
        z_max=94.5,
        margin=0.3,
        h_max=0.5,
        k_barrier=2.0,
    ):

        self._walls = [
            (0, +1.0, x_min + margin),
            (0, -1.0, x_max - margin),
            (1, +1.0, y_min + margin),
            (1, -1.0, y_max - margin),
            (2, +1.0, z_min - margin/2),
            (2, -1.0, 0.0),
        ]

        self._h_max = max(h_max, 1e-6)
        self._k_barrier = k_barrier

    def compute(self, p_nwu: np.ndarray) -> np.ndarray:

        f = np.zeros(3)

        for axis, sign, limit in self._walls:

            h = sign * (p_nwu[axis] - limit)

            if h <= 0.0:
                h = 1e-4

            if h >= self._h_max:
                continue

            f_mag = self._k_barrier * (
                (1.0 / h) - (1.0 / self._h_max)
            ) / (h * h)

            grad = np.zeros(3)
            grad[axis] = sign

            f += f_mag * grad

        return f


# ─────────────────────────────────────────────────────
# Main node
# ─────────────────────────────────────────────────────

class FormationControllerNode(Node):

    def __init__(self):

        super().__init__("formation_controller_node")

        # =====================================================
        # Parameters
        # =====================================================

        self.declare_parameter("agent_id", 0)
        self.declare_parameter("robot_name", "Bubble")

        self.declare_parameter("neighbors", "")
        self.declare_parameter("desired_rel_x", "0.0")
        self.declare_parameter("desired_rel_y", "-1.5")
        self.declare_parameter("desired_rel_z", "0.0")

        self.declare_parameter("mode", "clf_qp_tau")
        self.declare_parameter("dt", 0.02)

        # =====================================================
        # !Virtual target modes
        # =====================================================

        # 0 = real leader
        # 1 = fixed goal
        # 2 = moving trajectory

        self.declare_parameter("target_mode", 1)

        self.declare_parameter("virtual_target_x", -1.25)
        self.declare_parameter("virtual_target_y", -2.0)
        self.declare_parameter("virtual_target_z", 95.0)

        # =====================================================
        # Controller
        # =====================================================

        self.declare_parameter("d_min", 0.6)
        self.declare_parameter("d_max", 4.0)

        self.declare_parameter("use_connectivity", True)
        self.declare_parameter("use_collision", True)
        self.declare_parameter("use_recentered", True)

        self.declare_parameter("k_form", 8.0)
        self.declare_parameter("clf_alpha", 0.08)

        self.declare_parameter("k_att", 4.0)
        self.declare_parameter("k_att_d", 2.0)

        # =====================================================
        # Saturations
        # =====================================================

        self.declare_parameter("Fx_max_N", 88.0)
        self.declare_parameter("Fy_max_N", 88.0)
        self.declare_parameter("Fz_max_N", 137.0)

        self.declare_parameter("Mx_max_Nm", 30.0)
        self.declare_parameter("My_max_Nm", 16.5)
        self.declare_parameter("Mz_max_Nm", 21.0)

        self.declare_parameter("thrust_sat", 0.1)
        self.declare_parameter("torque_sat", 0.1)

        # =====================================================
        # FoV
        # =====================================================

        self.declare_parameter("use_fov", False)
        self.declare_parameter("use_fov_torque", True)
        self.declare_parameter("fov_target_id", -1)

        self.declare_parameter("theta_h_max_deg", 45.0)
        self.declare_parameter("theta_v_max_deg", 30.0)

        # =====================================================
        # Pool barrier
        # =====================================================

        self.declare_parameter("use_pool_barrier", True)

        # =====================================================
        # Read parameters
        # =====================================================

        gv = lambda name: self.get_parameter(name).value

        self.agent_id = int(gv("agent_id"))
        self.robot_name = str(gv("robot_name"))

        self.target_mode = int(gv("target_mode"))

        self.virtual_target = np.array([
            float(gv("virtual_target_x")),
            float(gv("virtual_target_y")),
            float(gv("virtual_target_z")),
        ])

        self.goal_position = self.virtual_target.copy()

        self.mode = str(gv("mode"))

        self.dt = float(gv("dt"))

        self.use_fov = bool(gv("use_fov"))
        self.use_fov_torque = bool(gv("use_fov_torque"))

        self.fov_target_id = int(gv("fov_target_id"))

        self.thr_sat = float(gv("thrust_sat"))
        self.tor_sat = float(gv("torque_sat"))

        self.Fx_max = float(gv("Fx_max_N"))
        self.Fy_max = float(gv("Fy_max_N"))
        self.Fz_max = float(gv("Fz_max_N"))

        self.Mx_max = float(gv("Mx_max_Nm"))
        self.My_max = float(gv("My_max_Nm"))
        self.Mz_max = float(gv("Mz_max_Nm"))

        # =====================================================
        # Neighbours
        # =====================================================

        nbr_str = str(gv("neighbors"))

        self.neighbor_ids = (
            [int(v.strip()) for v in nbr_str.split(",") if v.strip()]
            if nbr_str.strip()
            else []
        )

        n_nbr = len(self.neighbor_ids)

        def _csv(s):
            return [float(v.strip()) for v in s.split(",") if v.strip()]

        def _pad(lst, n):
            return (list(lst) + [0.0] * n)[:n]

        rel_x = _pad(_csv(str(gv("desired_rel_x"))), n_nbr)
        rel_y = _pad(_csv(str(gv("desired_rel_y"))), n_nbr)
        rel_z = _pad(_csv(str(gv("desired_rel_z"))), n_nbr)

        self.desired_rel = np.array([
            [rel_x[k], rel_y[k], rel_z[k]]
            for k in range(n_nbr)
        ])

        self.incidence_signs = np.ones(n_nbr)

        # =====================================================
        # Controller config
        # =====================================================

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
            dt=self.dt,
            use_fov=self.use_fov,
            theta_h_max=np.deg2rad(float(gv("theta_h_max_deg"))),
            theta_v_max=np.deg2rad(float(gv("theta_v_max_deg"))),
        )

        self.rov_model = BlueROV2()

        self.controller = BlueROV2Controller(
            self.agent_id,
            self.cfg,
            self.rov_model,
        )

        # =====================================================
        # Pool barrier
        # =====================================================

        self.use_pool_barrier = bool(gv("use_pool_barrier"))

        self.pool_barrier = PoolBoundaryBarrier()

        # =====================================================
        # State
        # =====================================================

        self.p_i = None

        self.q_i = np.array([0.0, 0.0, 0.0, 1.0])

        self.nu_i = np.zeros(6)

        self.enabled = False

        self.agent_positions: Dict[int, Optional[np.ndarray]] = {}

        # =====================================================
        # QoS
        # =====================================================

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # =====================================================
        # Publishers
        # =====================================================

        ns = f"/{self.robot_name}"

        self.thrust_pub = self.create_publisher(
            VehicleThrustSetpoint,
            f"{ns}/fmu/in/vehicle_thrust_setpoint",
            px4_qos,
        )

        self.torque_pub = self.create_publisher(
            VehicleTorqueSetpoint,
            f"{ns}/fmu/in/vehicle_torque_setpoint",
            px4_qos,
        )

        self.debug_pub = self.create_publisher(
            Wrench,
            f"{ns}/debug_wrench",
            10,
        )

        # =====================================================
        # Subscribers
        # =====================================================
    
        self.create_subscription(
            Odometry,
            f"/mocap/{self.robot_name.lower()}/odom",
            self._self_odom_cb,
            10,
        )

        self.create_subscription(
            VehicleControlMode,
            f"{ns}/fmu/out/vehicle_control_mode",
            self._control_mode_cb,
            px4_qos,
        )

        #! Virtual leader
        self.create_subscription(
            Point,
            "goal",
            self._goal_cb,
            10
        )

        self.start_mission_time = None

        # =====================================================
        # Timer
        # =====================================================

        self.create_timer(self.dt, self._control_step)

        self.get_logger().info(
            f"\n"
            f"target_mode={self.target_mode}\n"
            f"virtual_target={self.virtual_target}\n"
        )

    # ─────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────

    def _control_mode_cb(self, msg):
        gate = bool(msg.flag_armed) and bool(msg.flag_control_offboard_enabled)

        if gate and not self.enabled:
            self.enabled = True
            # Capture the "T-Zero" for our trajectory
            self.start_mission_time = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().info(f"Enabled. Mission clock started at {self.start_mission_time:.2f}s")

        elif (not gate) and self.enabled:
            self.enabled = False
            self.start_mission_time = None # Reset for next run
            self._publish_zero()

    def _self_odom_cb(self, msg: Odometry) -> None:
        # Fix: Extract x, y, z from the position object
        self.p_i = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ], dtype=float)

        # Fix: Extract x, y, z, w from the orientation object
        # Note: _ned_quat_to_nwu expects [qw, qx, qy, qz] based on your helper
        q_raw = np.array([
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z
        ], dtype=float)
        
        self.q_i = _ned_quat_to_nwu(q_raw)

        # Fix: Extract linear and angular velocities from the twist
        self.nu_i[:3] = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ], dtype=float)

        self.nu_i[3:] = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z
        ], dtype=float)

    def _make_agent_cb(self, aid):

        def _cb(msg):

            self.agent_positions[aid] = np.array(
                msg.pose.pose.position,
                dtype=float
            )

        return _cb

    def _goal_cb(self, msg: Point):

        self.goal_position = np.array([
            msg.x,
            msg.y,
            msg.z,
        ], dtype=float)

        self.get_logger().info(
            f"New virtual goal: {self.goal_position}"
        )

    # ─────────────────────────────────────────────────────
    # Target generation
    # ─────────────────────────────────────────────────────

    def get_target_position(self):

        # =================================================
        # MODE 0 -> REAL LEADER
        # =================================================

        if self.target_mode == 0:

            target = self.agent_positions.get(
                self.fov_target_id
            )

            if target is None:
                return None

            return target.copy()

        # =================================================
        # MODE 1 -> FIXED GOAL
        # =================================================

        elif self.target_mode == 1:

            return self.goal_position.copy()

        # =================================================
        # MODE 2 -> MOVING TRAJECTORY
        # =================================================

        elif self.target_mode == 2:
            if self.start_mission_time is None:
                # If armed but for some reason time wasn't set, use current pos
                return self.p_i.copy() if self.p_i is not None else self.virtual_target

            # Calculate "Mission Time" (starts at 0.0)
            now = self.get_clock().now().nanoseconds * 1e-9
            t_mission = now - self.start_mission_time

            # Use t_mission for the math
            x = 1.0 * math.sin(0.1 * t_mission) - 1.25
            y = -2.0
            z = 95.0

            return np.array([x, y, z])

        return None

    # ─────────────────────────────────────────────────────
    # Control step
    # ─────────────────────────────────────────────────────

    def _control_step(self):

        if not self.enabled:
            return

        if self.p_i is None:
            self._publish_zero()
            return

        # =================================================
        # Pool barrier
        # =================================================

        f_barrier_I = np.zeros(3)

        if self.use_pool_barrier:
            f_barrier_I = self.pool_barrier.compute(self.p_i)

        # =================================================
        # Target
        # =================================================

        target_pos = self.get_target_position()

        if target_pos is None and self.target_mode == 0:
            self._publish_zero()
            return

        distance_to_goal = np.linalg.norm(self.p_i - target_pos)
        
        self.get_logger().info(f"Distance to goal: {distance_to_goal:.2f}m", 
                                   throttle_duration_sec=2.0)

        if distance_to_goal <= 0.1:
            # Using throttle_duration to avoid flooding the terminal
            self.get_logger().info("Reached goal!")

        # =================================================
        # Relative states
        # =================================================

        if self.target_mode == 0:

            # Normal formation mode
            n_nbr = len(self.neighbor_ids)

            relative_states = np.zeros((n_nbr, 3))

            for k, nid in enumerate(self.neighbor_ids):

                relative_states[k] = (
                    self.p_i - self.agent_positions[nid]
                )

            desired_rel = self.desired_rel

        else:

            # Virtual leader mode:
            # treat target directly as formation anchor

            n_nbr = 1

            relative_states = np.array([
                self.p_i - target_pos
            ])

            desired_rel = self.desired_rel

        # =================================================
        # CLF-QP
        # =================================================

        try:

            tau_nwu, debug = self.controller.do_control(
                p_i=self.p_i,
                q_i=self.q_i,
                nu_i=self.nu_i,
                relative_states=relative_states,
                desired_relative_states=desired_rel,
                incidence_signs=np.ones(n_nbr),
                mode=self.mode,
                p_target_I=target_pos,
                use_fov_torque=self.use_fov_torque,
            )

        except Exception as exc:

            self.get_logger().error(str(exc))

            self._publish_zero()

            return

        tau_nwu = np.asarray(tau_nwu).reshape(6)

        # =================================================
        # Add pool barrier
        # =================================================

        if np.any(f_barrier_I != 0.0):

            R_b2I = _rotation_matrix_from_quaternion_nwu(
                self.q_i
            )

            tau_nwu[:3] += R_b2I.T @ f_barrier_I

        # =================================================
        # Publish
        # =================================================

        self._publish_tau(tau_nwu)

    # ─────────────────────────────────────────────────────
    # Publishing
    # ─────────────────────────────────────────────────────

    def _publish_tau(self, tau_nwu):

        now_us = int(
            self.get_clock().now().nanoseconds / 1000
        )

        force = tau_nwu[:3]
        torque = tau_nwu[3:]

        thr = VehicleThrustSetpoint()
        thr.timestamp = now_us
        thr.timestamp_sample = now_us
        thr.xyz = [0.0, 0.0, 0.0]

        thr.xyz = [
            float(_clamp(force[0] / self.Fx_max,
                         -self.thr_sat,
                         self.thr_sat)),

            float(_clamp(-force[1] / self.Fy_max,
                         -self.thr_sat,
                         self.thr_sat)),

            float(_clamp(-force[2] / self.Fz_max,
                         -self.thr_sat,
                         self.thr_sat)),
        ]

        tor = VehicleTorqueSetpoint()
        tor.timestamp = now_us
        tor.timestamp_sample = now_us
        tor.xyz = [0.0, 0.0, 0.0]

        tor.xyz = [
            float(_clamp(torque[0] / self.Mx_max,
                         -self.tor_sat,
                         self.tor_sat)),

            float(_clamp(-torque[1] / self.My_max,
                         -self.tor_sat,
                         self.tor_sat)),

            float(_clamp(-torque[2] / self.Mz_max,
                         -self.tor_sat,
                         self.tor_sat)),
        ]

        self.thrust_pub.publish(thr)
        self.torque_pub.publish(tor)

    def _publish_zero(self):

        now_us = int(
            self.get_clock().now().nanoseconds / 1000
        )

        thr = VehicleThrustSetpoint()
        thr.timestamp = now_us
        thr.timestamp_sample = now_us
        thr.xyz = [0.0, 0.0, 0.0]

        tor = VehicleTorqueSetpoint()
        tor.timestamp = now_us
        tor.timestamp_sample = now_us
        tor.xyz = [0.0, 0.0, 0.0]

        self.thrust_pub.publish(thr)
        self.torque_pub.publish(tor)


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

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
