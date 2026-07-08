#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_from_gazebo.py
═══════════════════
Updated to handle full 6D tau_history from the new recorder.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import matplotlib.pyplot as plt

# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Plot Gazebo simulation data")
parser.add_argument("--file",    default="/tmp/gazebo_run.npz",
                    help="Path to .npz file from gazebo_data_recorder")
parser.add_argument("--animate", action="store_true",
                    help="Show live animation instead of static final frame")
parser.add_argument("--save",    default=None, metavar="PREFIX",
                    help="Save figures with this prefix (e.g. /tmp/run1)")
parser.add_argument("--no-rov",  action="store_true",
                    help="Skip BlueROV2 body box overlay")
args = parser.parse_args()

# ── load data ─────────────────────────────────────────────────────────────────
if not os.path.exists(args.file):
    print(f"ERROR: File not found: {args.file}")
    sys.exit(1)

print(f"Loading: {args.file}")
d = np.load(args.file, allow_pickle=False)

history         = d["history"]          
vel_history     = d["vel_history"]      
v_history_full  = d["v_history_full"]   
accel_history   = d["accel_history"]    
err_history     = d["err_history"]      
tau_history     = d["tau_history"]      # Now (steps, N, 6)
q_history       = d["q_history"]        
timestamps      = d["timestamps"]       
agent_ids       = d["agent_ids"].tolist()
desired_offsets = d["desired_offsets"] 
dt_mean         = float(d["dt_mean"])
n_agents        = int(d["n_agents"])
steps           = int(d["steps"])

print(f"  Steps    : {steps}")
print(f"  Agents   : {agent_ids}")
print(f"  Duration : {timestamps[-1]:.1f} s")
print(f"  Mean dt  : {dt_mean*1000:.1f} ms  ({1/dt_mean:.1f} Hz)\n")

# ── add plot_animator_refactored.py to path ───────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
for _candidate in [
    _here,
    os.path.join(_here, "bluerov2_control"),
    os.path.expanduser("~/px4_ws/install/bluerov2_control/lib/python3.10/site-packages/bluerov2_control"),
    os.path.expanduser("~/px4_ws/src/bluerov2_control/bluerov2_control"),
]:
    if os.path.exists(os.path.join(_candidate, "plot_animator_refactored.py")):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        print(f"Found plot_animator_refactored.py in: {_candidate}")
        break
else:
    print("WARNING: Could not find plot_animator_refactored.py automatically.")

# ── build params ─────────────────────────────────────────────────────────────
from dataclasses import dataclass, field
from typing import Any

@dataclass
class GazeboConfig:
    N:               int   = 3
    dim:             int   = 3
    dt:              float = 0.02
    steps:           int   = 0

    d_min:           float = 0.3
    d_max:           float = 2.0

    conn:            bool  = True
    coll:            bool  = True
    fov:             bool  = True

    v_max:           float = 1.0
    a_max:           float = 2.0
    tau_v_max:       float = 88.0
    thruster_limit:  float = 1.0
    lgv_epsilon:     float = 1e-6

    simulation_mode: str   = "bluerov2_gazebo"
    mode_graph:      str   = "formation"
    R:               float = 5.0

    fov_R_cb:        Any   = None
    fov_p_cb:        Any   = None
    theta_max_x:     float = 0.785
    theta_max_y:     float = 0.524


@dataclass
class GazeboScenario:
    x:                  np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    A:                  np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    relative_pos_desired: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))


@dataclass
class GazeboData:
    history:        np.ndarray = field(default_factory=lambda: np.empty((0, 3, 3)))
    vel_history:    np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    accel_history:  np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    err_history:    np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    fov_history:    np.ndarray = field(default_factory=lambda: np.empty((0, 3, 2)))
    barrier_history: np.ndarray = field(default_factory=lambda: np.empty((0, 3, 3)))
    v_history_full: np.ndarray = field(default_factory=lambda: np.empty((0, 3, 3)))
    q_history:      np.ndarray = field(default_factory=lambda: np.empty((0, 3, 4)))
    debug_history:  list       = field(default_factory=list)


@dataclass
class GazeboParams:
    config:   GazeboConfig
    scenario: GazeboScenario
    data:     GazeboData


# ── Populate ─────────────────────────────────────────────────────────────────
cfg = GazeboConfig(
    N=n_agents,
    dim=3,
    dt=dt_mean,
    steps=steps,
    d_min=0.6,
    d_max=3.0,
    conn=True,
    coll=True,
    fov=True,
    v_max=1.0,
    simulation_mode="bluerov2_gazebo",
    mode_graph="formation",
    R=max(5.0, float(np.max(np.abs(history[:, :, :2]))) + 1.0),
    tau_v_max=88.0,
)

x0 = history[0].copy()

A = np.zeros((n_agents, n_agents), dtype=float)
for k in range(1, n_agents):
    A[0, k] = 1.0

rel_des = desired_offsets.copy()

scenario = GazeboScenario(x=x0, A=A, relative_pos_desired=rel_des)

# ── u_history: now use full 6D wrench directly ───────────────────────────────
# tau_history is already (steps, N, 6) → perfect for Animation
u_history = [tau_history[t] for t in range(steps)]

fov_raw = d.get("fov_history", np.zeros((steps, n_agents, 2)))
fov_viz = fov_raw.copy()

barrier_history = d.get("barrier_history", np.zeros((steps, n_agents, 3)))

# Horizontal: Multiply by theta_max_x (0.785 rad ≈ 45°) and convert to deg
fov_viz[:, :, 0] *= (0.785 * 180 / np.pi) 

# Vertical: Multiply by theta_max_y (0.524 rad ≈ 30°) and convert to deg
fov_viz[:, :, 1] *= (0.524 * 180 / np.pi)

data = GazeboData(
    history        = history,
    vel_history    = vel_history,
    accel_history  = accel_history,
    err_history    = err_history,
    fov_history    = fov_viz,
    barrier_history = barrier_history,
    v_history_full = v_history_full,
    q_history      = q_history,
    debug_history  = [],
)

params = GazeboParams(config=cfg, scenario=scenario, data=data)

# ── Run Animation ────────────────────────────────────────────────────────────
try:
    from plot_animator_refactored import Animation
    print("Successfully imported Animation\n")
except ImportError as e:
    print(f"ERROR importing Animation: {e}")
    sys.exit(1)

try:
    anim = Animation(params)
except Exception as e:
    print(f"ERROR creating Animation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── Visualization ────────────────────────────────────────────────────────────
if args.animate:
    print("Showing live animation…")
    anim.replay_animation(
        params,
        u_history,
        show_topology=True,
        show_barrier=False,
        save_path=None if args.save is None else f"{args.save}_animation.gif",
    )
else:
    print("Showing static final-frame plots…")
    anim.show_static_plots(
        params,
        u_history,
        show_topology=True,
        show_barrier=False,
        save_prefix=args.save,
    )

def plot_boundary_distances(params, timestamps):
    """
    Plots the distance from each agent to the pool boundaries (Walls and Floor).
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.subplots_adjust(hspace=0.3)

    # Pool limits (matching your PoolBoundaryBarrier in the node)
    X_MIN, X_MAX = -3.3, 1.0
    Y_MIN, Y_MAX = -5.7, 1.3
    Z_MIN = 97.0 # Floor

    pos_history = params.data.history  # Shape: (steps, N, 3)
    N = params.config.N
    
    for i in range(N):
        label = f"Agent {agent_ids[i]}"
        
        # Extract X, Y, Z for this agent across all time
        x_coords = pos_history[:, i, 0]
        y_coords = pos_history[:, i, 1]
        z_coords = pos_history[:, i, 2]

        # 1. Distance to Side Walls (X-axis)
        # Distance to the closer of the two walls
        dist_x = np.minimum(np.abs(x_coords - X_MIN), np.abs(x_coords - X_MAX))
        ax1.plot(timestamps, dist_x, label=label)
        
        # 2. Distance to Front/Back Walls (Y-axis)
        dist_y = np.minimum(np.abs(y_coords - Y_MIN), np.abs(y_coords - Y_MAX))
        ax2.plot(timestamps, dist_y, label=label)
        
        # 3. Distance to Floor (Z-axis)
        dist_z = np.abs(z_coords - Z_MIN)
        ax3.plot(timestamps, dist_z, label=label)

    # Formatting
    ax1.set_title("Distance to Side Walls (X-axis)")
    ax1.set_ylabel("Distance [m]")
    ax1.axhline(y=0.3, color='r', linestyle='--', label='Barrier Margin') #[cite: 1]
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', fontsize='small')

    ax2.set_title("Distance to Front/Back Walls (Y-axis)")
    ax2.set_ylabel("Distance [m]")
    ax2.axhline(y=0.3, color='r', linestyle='--')
    ax2.grid(True, linestyle='--', alpha=0.6)

    ax3.set_title("Distance to Floor (Z-axis)")
    ax3.set_ylabel("Distance [m]")
    ax3.axhline(y=0.3, color='r', linestyle='--')
    ax3.set_xlabel("Time [s]")
    ax3.grid(True, linestyle='--', alpha=0.6)

    if args.save:
        plt.savefig(f"{args.save}_boundary_distances.png")
        print(f"Saved distance plots to {args.save}_boundary_distances.png")
    
    plt.show()

def plot_pool_boundaries(params, timestamps):
    """
    Plots agent trajectories against physical pool walls and barrier margins.
    Z-axis is inverted to show 'Up' as the surface and 'Down' as the floor.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
    fig.subplots_adjust(hspace=0.4)

    # Define the physical pool limits
    X_LIMITS = (-3.3, 1.0)
    Y_LIMITS = (-5.7, 1.3)
    Z_FLOOR  = 97.0
    Z_SURFACE = 99.0 # Approximate surface if depth is ~2m
    MARGIN   = 0.3  

    pos_history = params.data.history  # Shape: (steps, N, 3)[cite: 2]
    N = params.config.N
    wall_color = 'black'
    margin_color = 'red'

    for i in range(N):
        label = f"Agent {agent_ids[i]}"
        ax1.plot(timestamps, pos_history[:, i, 0], label=label)
        ax2.plot(timestamps, pos_history[:, i, 1], label=label)
        ax3.plot(timestamps, pos_history[:, i, 2], label=label)

    # --- X and Y Axes (Standard) ---
    for wall in X_LIMITS:
        ax1.axhline(y=wall, color=wall_color, linestyle='-', linewidth=1.5)
    ax1.axhline(y=X_LIMITS[0] + MARGIN, color=margin_color, linestyle='--', alpha=0.7)
    ax1.axhline(y=X_LIMITS[1] - MARGIN, color=margin_color, linestyle='--', alpha=0.7)
    ax1.set_title("X-Axis Trajectory (Side Walls)")

    for wall in Y_LIMITS:
        ax2.axhline(y=wall, color=wall_color, linestyle='-', linewidth=1.5)
    ax2.axhline(y=Y_LIMITS[0] + MARGIN, color=margin_color, linestyle='--', alpha=0.7)
    ax2.axhline(y=Y_LIMITS[1] - MARGIN, color=margin_color, linestyle='--', alpha=0.7)
    ax2.set_title("Y-Axis Trajectory (Front/Back Walls)")

    # --- Z-Axis (Inverted for Depth Intuition) ---
    ax3.axhline(y=Z_FLOOR, color=wall_color, linestyle='-', linewidth=1.5, label='Pool Floor')
    ax3.axhline(y=Z_FLOOR - MARGIN, color=margin_color, linestyle='--', alpha=0.7, label='Floor Margin')
    ax3.set_title("Z-Axis Trajectory (Depth)")
    ax3.set_ylabel("Altitude [m]")
    ax3.set_xlabel("Time [s]")
    
    # This command flips the axis so higher values (surface) are at the top
    # and lower values (floor) are at the bottom.
    ax3.invert_yaxis() 

    for ax in [ax1, ax2, ax3]:
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper right', fontsize='x-small', ncol=2)

    plt.show()

# ── Execution ──────────────────────────────────────────────────────────────
if not args.animate:
    # This ensures the plot pops up after the static plots are closed
    plot_boundary_distances(params, timestamps)
    plot_pool_boundaries(params, timestamps)
