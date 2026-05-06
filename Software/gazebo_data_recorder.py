#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gazebo_data_recorder.py
═══════════════════════
Records full 6D wrench + placeholder for FoV so that plot_from_gazebo.py works nicely.
"""

from __future__ import annotations

import signal
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Wrench
from geometry_msgs.msg import Vector3
from px4_msgs.msg import VehicleOdometry


class GazeboDataRecorder(Node):

    def __init__(self):
        super().__init__("gazebo_data_recorder")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("agent_ids",          [0, 1, 2])
        self.declare_parameter("agent_ns_prefix",    "itrl_rov_")
        self.declare_parameter("desired_offsets_x",  [0.0, 1.25, -1.25])
        self.declare_parameter("desired_offsets_y",  [0.0, 0.0, 0.0])
        self.declare_parameter("desired_offsets_z",  [0.0, 0.0, 0.0])
        self.declare_parameter("output_file",        "/tmp/gazebo_run.npz")
        self.declare_parameter("record_hz",          10.0)

        gv = lambda n: self.get_parameter(n).value

        self.agent_ids: List[int] = list(gv("agent_ids"))
        self.ns_prefix: str       = str(gv("agent_ns_prefix"))
        self.output_file: str     = str(gv("output_file"))
        record_hz: float          = float(gv("record_hz"))
        self.record_dt: float     = 1.0 / record_hz

        n = len(self.agent_ids)
        ox = list(gv("desired_offsets_x"))
        oy = list(gv("desired_offsets_y"))
        oz = list(gv("desired_offsets_z"))
        self.desired_offsets = np.array([[ox[k], oy[k], oz[k]] for k in range(n)], dtype=float)

        # ── State storage ─────────────────────────────────────────────────────
        self.positions:   Dict[int, Optional[np.ndarray]] = {aid: None for aid in self.agent_ids}
        self.velocities:  Dict[int, Optional[np.ndarray]] = {aid: None for aid in self.agent_ids}
        self.quaternions: Dict[int, Optional[np.ndarray]] = {aid: None for aid in self.agent_ids}
        
        # Full 6D wrench: [fx, fy, fz, tx, ty, tz]
        self.wrenches: Dict[int, np.ndarray] = {aid: np.zeros(6, dtype=float) for aid in self.agent_ids}
        # FoV errors: [alpha_h_error, alpha_v_error] in radians
        self.fov_errors: Dict[int, np.ndarray] = {aid: np.zeros(2, dtype=float) for aid in self.agent_ids}
        # Storage for the latest barrier force
        self.barrier_forces: Dict[int, np.ndarray] = {aid: np.zeros(3, dtype=float) for aid in self.agent_ids}
        # History list
        self.barrier_history: List[np.ndarray] = []

        # ── Recorded history ──────────────────────────────────────────────────
        self.history:        List[np.ndarray] = []   # (N, 3)
        self.vel_history:    List[List[float]] = []  # speed magnitudes
        self.v_history_full: List[np.ndarray] = []   # (N, 3)
        self.accel_history:  List[List[float]] = []
        self.err_history:    List[List[float]] = []
        self.tau_history:    List[np.ndarray] = []   # <<< NOW (N, 6) full wrench
        self.q_history:      List[np.ndarray] = []   # (N, 4)
        self.fov_history:    List[np.ndarray] = []   # (N, 2) placeholder
        self.timestamps:     List[float] = []

        self._prev_velocities: Optional[np.ndarray] = None
        self._start_time: Optional[float] = None

        # ── QoS ───────────────────────────────────────────────────────────────
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── Subscribers ───────────────────────────────────────────────────────
                # ── Subscribers ───────────────────────────────────────────────────────
        for aid in self.agent_ids:
            ns = f"/{self.ns_prefix}{aid}"
            self.get_logger().info(f"Subscribing to agent {aid} under {ns}")

            # Odometry
            self.create_subscription(
                VehicleOdometry,
                f"{ns}/fmu/out/vehicle_odometry",
                self._make_odom_cb(aid),
                px4_qos,
            )

            # Wrench - try both possible topic names
            self.create_subscription(
                Wrench,
                f"{ns}/cmd_wrench",
                self._make_wrench_cb(aid),
                10,
            )
            self.create_subscription(
                Wrench,
                f"{ns}/debug_wrench",      # fallback
                self._make_wrench_cb(aid),
                10,
            )

            # FoV
            self.create_subscription(
                Vector3,
                f"{ns}/fov_error",
                self._make_fov_cb(aid),
                10,
            )

            # Pool barrier force
            self.create_subscription(
                Wrench,
                f"{ns}/debug_barrier",
                self._make_barrier_cb(aid),
                10,
            )

        self.create_timer(self.record_dt, self._record_step)

        self.get_logger().info(
            f"\n{'='*60}\n"
            f"  Gazebo Data Recorder Started\n"
            f"  Agents     : {self.agent_ids}\n"
            f"  Record @   : {record_hz:.1f} Hz\n"
            f"  Output     : {self.output_file}\n"
            f"{'='*60}"
        )

    def _make_odom_cb(self, aid: int):
        def _cb(msg: VehicleOdometry):
            self.positions[aid]   = np.array(msg.position, dtype=float)
            self.velocities[aid]  = np.array(msg.velocity, dtype=float)

            # PX4 [qw, qx, qy, qz] → NWU scalar-last [qx, -qy, -qz, qw]
            qw, qx, qy, qz = msg.q
            self.quaternions[aid] = np.array([qx, -qy, -qz, qw], dtype=float)
        return _cb

    def _make_wrench_cb(self, aid: int):
        def _cb(msg: Wrench):
            f = np.array([msg.force.x, msg.force.y, msg.force.z], dtype=float)
            t = np.array([msg.torque.x, msg.torque.y, msg.torque.z], dtype=float)
            self.wrenches[aid] = np.concatenate([f, t])
            
            # Debug logging (throttled)
            if np.any(np.abs(f) > 0.05) or np.any(np.abs(t) > 0.05):
                self.get_logger().info(
                    f"Agent {aid} | Force: {f.round(3)} N | Torque: {t.round(3)} Nm",
                    throttle_duration_sec=1.0
                )
        return _cb

    def _make_fov_cb(self, aid: int):
        def _cb(msg: Vector3):
            h_err = float(msg.x)
            v_err = float(msg.y)
            self.fov_errors[aid] = np.array([h_err, v_err], dtype=float)

            if abs(h_err) > 0.01 or abs(v_err) > 0.01:
                self.get_logger().info(
                    f"FoV Agent {aid} | h_err={h_err:.4f} rad ({np.rad2deg(h_err):.2f}°) | "
                    f"v_err={v_err:.4f} rad",
                    throttle_duration_sec=2.0
                )
        return _cb

    def _make_barrier_cb(self, aid: int):
        def _cb(msg: Wrench):
            self.barrier_forces[aid] = np.array([msg.force.x, msg.force.y, msg.force.z])
        return _cb

    def _record_step(self) -> None:
        if any(self.positions[aid] is None for aid in self.agent_ids):
            return

        now = time.monotonic()
        if self._start_time is None:
            self._start_time = now
        self.timestamps.append(now - self._start_time)

        n = len(self.agent_ids)
        pos_step = np.array([self.positions[aid] for aid in self.agent_ids])

        # Velocity
        vel_step = np.array([self.velocities[aid] if self.velocities[aid] is not None else np.zeros(3)
                            for aid in self.agent_ids])
        speeds = np.linalg.norm(vel_step, axis=1)

        # Acceleration
        if self._prev_velocities is not None:
            accel = np.linalg.norm((vel_step - self._prev_velocities) / self.record_dt, axis=1)
        else:
            accel = np.zeros(n)
        self._prev_velocities = vel_step.copy()

        # Formation error (relative to leader)
        leader_pos = pos_step[0]
        errors = [float(np.linalg.norm(pos_step[k] - (leader_pos + self.desired_offsets[k])))
                  for k in range(n)]

        fov_step = np.array([self.fov_errors[aid].copy() for aid in self.agent_ids])

        # Full wrench (N, 6)
        tau_step = np.array([self.wrenches[aid] for aid in self.agent_ids])  # (N, 6)

        # Quaternions
        q_step = np.array([self.quaternions[aid] if self.quaternions[aid] is not None
                          else np.array([0.,0.,0.,1.]) for aid in self.agent_ids])

        barrier_step = np.array([self.barrier_forces[aid].copy() for aid in self.agent_ids])

        # === DEBUG: Check what we're actually saving ===
        mean_fov_h = np.mean([abs(self.fov_errors[aid][0]) for aid in self.agent_ids])
        self.get_logger().info(
            f"DEBUG SAVE | mean_fov_h being saved = {mean_fov_h:.4f} rad",
            throttle_duration_sec=1.0
        )

        # Store
        self.history.append(pos_step.copy())
        self.vel_history.append(speeds.tolist())
        self.v_history_full.append(vel_step.copy())
        self.accel_history.append(accel.tolist())
        self.err_history.append(errors)
        self.tau_history.append(tau_step.copy())
        self.q_history.append(q_step.copy())
        self.fov_history.append(fov_step.copy())
        self.barrier_history.append(barrier_step)

        # Progress
        if len(self.timestamps) % max(1, int(5.0 / self.record_dt)) == 0:
            self.get_logger().info(
                f"t={self.timestamps[-1]:.1f}s | err={np.mean(errors):.3f}m"
            )

    def save(self) -> None:
        steps = len(self.history)
        if steps == 0:
            self.get_logger().warn("No data recorded.")
            return

        np.savez(
            self.output_file,
            history=np.array(self.history, dtype=float),
            vel_history=np.array(self.vel_history, dtype=float),
            v_history_full=np.array(self.v_history_full, dtype=float),
            accel_history=np.array(self.accel_history, dtype=float),
            err_history=np.array(self.err_history, dtype=float),
            tau_history=np.array(self.tau_history, dtype=float),   # (steps, N, 6)
            q_history=np.array(self.q_history, dtype=float),
            fov_history=np.array(self.fov_history, dtype=float),   # (steps, N, 2)
            barrier_history=np.array(self.barrier_history, dtype=float),
            timestamps=np.array(self.timestamps, dtype=float),
            agent_ids=np.array(self.agent_ids),
            desired_offsets=self.desired_offsets,
            dt_mean=np.float64(np.mean(np.diff(self.timestamps))) if steps > 1 else self.record_dt,
            n_agents=np.int32(len(self.agent_ids)),
            steps=np.int32(steps),
        )

        self.get_logger().info(f"\n✅ Saved {steps} steps to {self.output_file}\n")


def main(args=None):
    rclpy.init(args=args)
    node = GazeboDataRecorder()

    def _shutdown(sig, frame):
        node.get_logger().info("Ctrl+C → saving data...")
        node.save()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        rclpy.spin(node)
    finally:
        node.save()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()