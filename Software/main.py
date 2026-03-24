#!/usr/bin/env python3

##########################################################
#                                                        #
# Main - Formation Control Simulation                    #
# Code Author: João Zenário                              #
#                                                        #     
##########################################################

"""
Clean simulation loop for multi-robot formation control.

Architecture:
- RobotSimulator: One per robot, handles dynamics integration
- Controllers: One per robot, computes control input
- GraphManager: Manages communication topology
- History is collected (no plotting in loop), then visualized separately
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List, Dict, Tuple

from robot_simulator import RobotSimulator
from graphs_utils import get_neighbor_lists_from_adjacency
from controller import (
    SingleIntegratorController,
    DoubleIntegratorController,
    FieldOfViewController,
)
from robots import SingleIntegrator, DoubleIntegrator
from systems import RK4Integrator
from parameters import Parameters
from plot_animator import Animation



class FormationControlSimulation:
    """
    Main simulation orchestrator for multi-robot formation control.
    
    Manages initialization of all components and runs the main simulation loop.
    """
    
    def __init__(self, params: Parameters):
        """
        Initialize the simulation with all components.
        
        Args:
            params: Parameters object with all configuration
        """
        self.params = params
        self.N = self.params.N
        self.dim = self.params.dim
        
        print("\n" + "="*60)
        print("        BLUE ROV2 FORMATION CONTROL SIMULATION")
        print("="*60)
        self._print_configuration()
        print("="*60 + "\n")
        
        # Step 1: Initialize graph
        print("[1/5] Initializing graph topology...")
        self.neighbors_dict = get_neighbor_lists_from_adjacency(params.A)
        print(f"      Graph initialized with {len(self.neighbors_dict)} agents")
        
        # Step 2: Initialize robot simulators
        print("[2/5] Initializing robot simulators...")
        self.simulators = self._initialize_simulators()
        print(f"      Created {len(self.simulators)} simulators")
        
        # Step 3: Initialize controllers
        print("[3/5] Initializing controllers...")
        self.controllers = self._initialize_controllers()
        print(f"      Created {len(self.controllers)} controllers")
        
        # Step 4: Initialize FOV controllers (if enabled)
        self.fov_controllers = None
        if params.fov:
            print("[4/5] Initializing FOV controllers...")
            self.fov_controllers = self._initialize_fov_controllers()
            print(f"      Created {len(self.fov_controllers)} FOV controllers")
        else:
            print("[4/5] FOV disabled - skipping FOV controller initialization")
        
        # Step 5: Initialize state tracking
        print("[5/5] Initializing state tracking...")
        self._initialize_state_tracking()
        print("      State tracking ready\n")
    
    def _print_configuration(self):
        """Print simulation configuration dashboard."""
        dashboard = (
            f"--- CONFIGURATION ---\n"
            f"Agents (N):              {self.params.N}\n"
            f"Dimension:               {self.params.dim}D\n"
            f"Dynamics:                {'Double' if self.params.mode_dynamics else 'Single'} Integrator\n"
            f"Graph Type:              {'Directed' if self.params.mode_graph else 'Undirected'}\n"
            f"\n--- CONSTRAINTS ---\n"
            f"Connectivity:            {['Off', 'On'][self.params.conn]}\n"
            f"Collision Avoidance:     {['Off', 'On'][self.params.coll]}\n"
            f"Field of View:           {['Off', 'On'][self.params.fov]}\n"
            f"Recentered Barrier:      {['Off', 'On'][self.params.rec]}\n"
            f"\n--- PHYSICS & CONTROL ---\n"
            f"Time Step (dt):          {self.params.dt}s ({self.params.steps} steps)\n"
            f"Gains:                   c1={self.params.c1}, c2={self.params.c2}, "
            f"c3={self.params.c3}\n"
            f"FOV Gains:               c4_h={self.params.c4_h}, c4_v={self.params.c4_v}, "
            f"kp_yaw={self.params.kp_yaw}\n"
            f"Filter:                  ωn={self.params.wn if self.params.mode_dynamics else 'N/A'}\n"
            f"Limits:                  v_max={self.params.v_max}m/s, "
            f"a_max={self.params.a_max}m/s²\n"
            f"Thresholds:              d_max={self.params.d_max}m, d_min={self.params.d_min}m\n"
            f"Formation Radius:        {self.params.R}m"
        )
        print(dashboard)
    
    def _initialize_simulators(self) -> List[RobotSimulator]:
        """Create a RobotSimulator for each robot."""
        simulators = []
        
        # Choose dynamics model
        if self.params.mode_dynamics:
            system_class = DoubleIntegrator
        else:
            system_class = SingleIntegrator
        
        for i in range(self.N):
            # Create system dynamics for this robot
            system = system_class(dim=self.dim)
            
            # Get initial state from parameters
            x0 = self.params.x[i].copy()
            
            # Create simulator
            sim = RobotSimulator(
                robot_id=i,
                system=system,
                x0=x0,
                dt=self.params.dt,
                integrator=RK4Integrator(),
                u_min=getattr(system, 'u_min', None),
                u_max=getattr(system, 'u_max', None),
                u_norm_max=getattr(system, 'u_norm_max', None),
            )
            simulators.append(sim)
        
        return simulators
    
    def _initialize_controllers(self) -> List:
        """Create a Controller for each robot."""
        controllers = []
        
        # Prepare control gains
        control_gains = {
            'c1': self.params.c1,
            'c2': self.params.c2,
            'c3': self.params.c3,
            'wn': self.params.wn,
        }
        
        # Prepare constraints
        constraints = {
            'conn': bool(self.params.conn),
            'coll': bool(self.params.coll),
            'rec': bool(self.params.rec),
            'fov': bool(self.params.fov),
        }
        
        # Prepare limits
        limits = {
            'v_max': self.params.v_max,
            'a_max': self.params.a_max,
            'd_max': self.params.d_max,
            'd_min': self.params.d_min,
            'R': self.params.R,
            'dt': self.params.dt,
        }
        
        # Create appropriate controller type
        for i in range(self.N):
            if self.params.mode_dynamics:
                controller = DoubleIntegratorController(
                    robot_id=i,
                    dim=self.dim,
                    control_gains=control_gains,
                    constraints=constraints,
                    limits=limits,
                )
            else:
                controller = SingleIntegratorController(
                    robot_id=i,
                    dim=self.dim,
                    control_gains=control_gains,
                    constraints=constraints,
                    limits=limits,
                )
            controllers.append(controller)
        
        return controllers
    
    def _initialize_fov_controllers(self) -> List[FieldOfViewController]:
        """Create a FieldOfViewController for each robot (if FOV enabled)."""
        fov_controllers = []
        
        fov_gains = {
            'kp_yaw': self.params.kp_yaw,
            'kp_pitch': self.params.kp_pitch,
            'c4_h': self.params.c4_h,
            'c4_v': self.params.c4_v,
        }
        
        fov_limits = {
            'theta_max_x': self.params.theta_max_x,
            'theta_max_y': self.params.theta_max_y,
            'dt': self.params.dt,
        }
        
        for i in range(self.N):
            fov_ctrl = FieldOfViewController(
                robot_id=i,
                dim=self.dim,
                fov_gains=fov_gains,
                fov_limits=fov_limits,
            )
            fov_controllers.append(fov_ctrl)
        
        return fov_controllers
    
    def _initialize_state_tracking(self):
        """Initialize arrays to track state throughout simulation."""
        # Positions
        self.x = np.array([sim.state for sim in self.simulators])
        
        # Velocities (for double integrator)
        if self.params.mode_dynamics:
            self.v = self.params.v.copy()
            # Filter states for double integrator controllers
            self.x_lf = self.params.x_lf.copy()
            self.v_lf = self.params.ẋ_lf.copy()
        else:
            self.v = None
            self.x_lf = None
            self.v_lf = None
        
        # Quaternions (for FOV)
        self.q = self.params.q.copy()
        
        # History for plotting/analysis
        self.history = {
            'time': [],
            'positions': [],
            'velocities': [],
            'accelerations': [] if self.params.mode_dynamics else None,
            'quaternions': [] if self.params.fov else None,
            'formation_errors': [],
            'fov_errors': [] if self.params.fov else None,
        }
    
    def run(self) -> Dict:
        """
        Run the main simulation loop.
        
        Returns:
            Dictionary containing collected history for post-processing
        """
        print("Starting simulation...\n")
        
        u_history = []  # Track control inputs for animation
        
        for step in range(self.params.steps):
            # ========================================
            # Step 1: Compute control for each robot
            # ========================================
            u = []
            
            for i in range(self.N):
                # Get current state
                state_i = self.x[i]
                
                # Get neighbors
                neighbors_i = self.neighbors_dict[i]
                
                # Extract neighbor states
                if len(neighbors_i) > 0:
                    neighbors_state = self.x[neighbors_i]  # Get positions of neighbors
                    # Compute relative states
                    rel_state = self.x[i] - neighbors_state  # z_ij = x_i - x_j
                else:
                    rel_state = np.array([]).reshape(0, self.dim)
                
                # Get desired relative states
                desired_rel_state = self.params.relative_pos_desired[i, neighbors_i]
                if len(neighbors_i) == 0:
                    desired_rel_state = np.array([]).reshape(0, self.dim)
                
                # Compute control
                is_leader = (i == 0)
                
                # Leader follows a predefined trajectory
                if is_leader:
                    u_i = self.params.get_leader_acceleration(step)
                elif self.params.mode_dynamics:
                    # Double integrator: pass velocity and filter states
                    u_i = self.controllers[i].doControl(
                        state_i=state_i,
                        relative_states=rel_state,
                        desired_relative_states=desired_rel_state,
                        velocity_i=self.v[i],
                        x_lf=self.x_lf[i],
                        v_lf=self.v_lf[i],
                        is_leader=is_leader,
                    )
                    # Update filter states
                    self.x_lf[i] = self.controllers[i].x_lf
                    self.v_lf[i] = self.controllers[i].v_lf
                else:
                    # Single integrator: velocity command
                    u_i = self.controllers[i].doControl(
                        state_i=state_i,
                        relative_states=rel_state,
                        desired_relative_states=desired_rel_state,
                        is_leader=is_leader,
                    )
                
                u.append(u_i)
            
            u_history.append(np.array(u).copy())
            
            # ========================================
            # Step 2: Integrate dynamics for each robot
            # ========================================
            for i in range(self.N):
                if self.params.mode_dynamics:
                    # Double integrator: x = [position, velocity]
                    # We need to manually update velocity
                    self.v[i] += u[i] * self.params.dt
                    
                    # Apply velocity saturation for double integrator
                    v_norm = np.linalg.norm(self.v[i])
                    if v_norm > self.params.v_max:
                        self.v[i] *= self.params.v_max / v_norm
                    
                    self.x[i] += self.v[i] * self.params.dt
                else:
                    # Single integrator: position update via simulator
                    # For single integrator, u is the velocity command
                    self.x[i] = self.simulators[i].integrate_dynamics(u[i])
            
            # ========================================
            # Step 3: Handle FOV orientation (if enabled)
            # ========================================
            fov_errors_step = []
            if self.params.fov:
                fov_errors_step = [[0.0, 0.0]]  # Leader has no FOV error
                for i in range(1, self.N):
                    # Find target (first neighbor)
                    neighbors_i = self.neighbors_dict[i]
                    if len(neighbors_i) > 0:
                        target_idx = neighbors_i[0]
                        omega_cmd, alpha_h, alpha_v, q_new = (
                            self.fov_controllers[i].compute_orientation_input(
                                position_i=self.x[i],
                                position_target=self.x[target_idx],
                                quaternion_i=self.q[i],
                            )
                        )
                        # Store updated quaternion returned by FOV controller
                        self.q[i] = q_new
                        fov_errors_step.append([np.degrees(alpha_h), np.degrees(alpha_v)])
                    else:
                        fov_errors_step.append([0.0, 0.0])
            
            # ========================================
            # Step 4: Update params for original animator
            # ========================================
            # Populate params object as the original code did
            self.params.x = self.x.copy()
            self.params.v = self.v.copy() if self.params.mode_dynamics else np.array(u)
            self.params.q = self.q.copy()
            
            # Store position history
            self.params.history = np.vstack([self.params.history, self.x[np.newaxis, :, :]])
            
            # Store full velocity history (velocity vectors, not just magnitudes)
            if not hasattr(self.params, 'v_history_full') or not isinstance(self.params.v_history_full, np.ndarray):
                self.params.v_history_full = np.empty((0, self.N, self.dim))
            self.params.v_history_full = np.vstack([self.params.v_history_full, self.params.v[np.newaxis, :, :]])
            
            # Store velocity history (magnitudes only, for telemetry plot)
            if isinstance(self.params.vel_history, np.ndarray):
                self.params.vel_history = self.params.vel_history.tolist()
            if self.params.mode_dynamics:
                current_speeds = [np.linalg.norm(v) for v in self.v]
            else:
                current_speeds = [np.linalg.norm(v) for v in u]
            self.params.vel_history.append(current_speeds)
            self.params.vel_history = np.array(self.params.vel_history)
            
            # Store acceleration history 
            if self.params.mode_dynamics:
                if isinstance(self.params.accel_history, np.ndarray):
                    self.params.accel_history = self.params.accel_history.tolist()
                current_accels = [np.linalg.norm(a) for a in u]
                self.params.accel_history.append(current_accels)
                self.params.accel_history = np.array(self.params.accel_history)
            
            # Store formation error
            formation_errors = self._compute_formation_errors()
            if isinstance(self.params.err_history, np.ndarray):
                self.params.err_history = self.params.err_history.tolist()
            self.params.err_history.append(formation_errors)
            self.params.err_history = np.array(self.params.err_history)
            
            # Store FOV errors
            if self.params.fov and len(fov_errors_step) > 0:
                if isinstance(self.params.fov_history, np.ndarray):
                    self.params.fov_history = self.params.fov_history.tolist()
                self.params.fov_history.append(fov_errors_step)
                self.params.fov_history = np.array(self.params.fov_history)
            
            # Store quaternion history (for FOV visualization during animation replay)
            if self.params.fov:
                if not hasattr(self.params, 'q_history') or not isinstance(self.params.q_history, np.ndarray):
                    self.params.q_history = np.empty((0, self.N, 4))
                self.params.q_history = np.vstack([self.params.q_history, self.q[np.newaxis, :, :]])

            
            # ========================================
            # Step 5: Log progress
            # ========================================
            current_time = step * self.params.dt
            
            # Print progress
            if (step + 1) % max(1, self.params.steps // 10) == 0:
                mean_error = np.mean(formation_errors[1:])
                print(f"Step {step+1}/{self.params.steps} - "
                      f"Time: {current_time:.2f}s - "
                      f"Avg Formation Error: {mean_error:.2f}%")
        
        print("\nSimulation complete!\n")
        
        # Store control history for animation
        self.u_history = u_history
        
        return self.history
    
    def _compute_formation_errors(self) -> np.ndarray:
        """
        Compute formation tracking error for each robot.
        
        Returns:
            Array of formation errors (in percentage)
        """
        errors = np.zeros(self.N)
        
        for i in range(self.N):
            if i == 0:
                errors[i] = 0.0  # Leader has no error
            else:
                # Target position: leader + desired offset
                target_pos = self.x[0] + self.params.relative_pos_desired[i, 0]
                error_norm = np.linalg.norm(self.x[i] - target_pos)
                # Normalize by formation radius
                errors[i] = (error_norm / self.params.R) * 100.0
        
        return errors


def main():
    """Main entry point."""
    # Initialize parameters from command line
    params = Parameters.function_arguments()
    
    # Create and run simulation
    sim = FormationControlSimulation(params)
    history = sim.run()
    
    # Play animation from collected history
    from plot_animator import Animation
    animator = Animation(params)
    animator.replay_animation(params, sim.u_history)
    
    return history


if __name__ == "__main__":
    main()