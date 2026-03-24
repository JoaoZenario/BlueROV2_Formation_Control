#!/usr/bin/env python3

##########################################################
#                                                        #
# Robot Controllers                                      #
# Code Author: João Zenário                              #
# Equations and concepts based on: Esteban's thesis      #
#                                                        #     
##########################################################

"""
This module defines controller classes for different integrator types.
Each controller takes the robot state and neighbor information, and
computes the control input.
"""

import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation as R


class ControllerBase(ABC):
    """
    Abstract base class for all robot controllers.
    
    Attributes:
        robot_id (int): Robot identifier
        dim (int): State dimension
        control_gains (dict): Dictionary of control gains (c1, c2, c3, c4_h, c4_v, wn, etc.)
        constraints (dict): Constraint flags (conn, coll, rec, fov)
        limits (dict): Physical limits (v_max, a_max, theta_max_x, theta_max_y)
    """
    
    def __init__(
        self,
        robot_id: int,
        dim: int,
        control_gains: dict,
        constraints: dict,
        limits: dict,
    ):
        """
        Initialize a controller.
        
        Args:
            robot_id: Robot ID (0-indexed)
            dim: State dimension (2 or 3)
            control_gains: Dict with keys {c1, c2, c3, c4_h, c4_v, wn, kp_yaw, kp_pitch}
            constraints: Dict with keys {conn, coll, rec, fov}
            limits: Dict with keys {v_max, a_max, d_max, d_min, R, dt}
        """
        self.robot_id = robot_id
        self.dim = dim
        self.control_gains = control_gains.copy()
        self.constraints = constraints.copy()
        self.limits = limits.copy()
    
    # ============================
    # --- Constraint Methods ---
    # ============================
    
    def connectivity_maintenance(self, relative_state: np.ndarray) -> float:
        """
        Connectivity maintenance barrier function.
        
        Args:
            relative_state: z_k = x_i - x_j (relative position)
        
        Returns:
            Barrier function value W_conn
        """
        W_conn = 1 / max(self.limits["d_max"]**2 - np.sum(relative_state**2), 1e-6)
        return W_conn
    
    def collision_avoidance(self, relative_state: np.ndarray) -> float:
        """
        Collision avoidance barrier function.
        
        Args:
            relative_state: z_k = x_i - x_j (relative position)
        
        Returns:
            Barrier function value W_coll
        """
        W_coll = 1 / max(np.sum(relative_state**2) - self.limits["d_min"]**2, 1e-6)
        return W_coll

    def recentered_barrier(
        self,
        relative_state: np.ndarray,
        desired_relative_state: np.ndarray,
    ) -> np.ndarray:
        """
        Recentered Lyapunov barrier function.
        
        Args:
            relative_state: z_k = x_i - x_j (current relative position)
            desired_relative_state: z_target = o_i - o_j (desired relative position)
        
        Returns:
            Gradient of recentered barrier function
        """
        conn_enabled = self.constraints.get("conn", True)
        coll_enabled = self.constraints.get("coll", True)
        
        # Evaluate constraints at current position
        if conn_enabled:
            W_conn_current = self.connectivity_maintenance(relative_state)
            W_conn_target = self.connectivity_maintenance(desired_relative_state)
        else:
            W_conn_current = 0.0
            W_conn_target = 0.0
        
        if coll_enabled:
            W_coll_current = self.collision_avoidance(relative_state)
            W_coll_target = self.collision_avoidance(desired_relative_state)
        else:
            W_coll_current = 0.0
            W_coll_target = 0.0
        
        W_grad_current = W_conn_current - W_coll_current
        W_grad_target = W_conn_target - W_coll_target
        
        # Recentered barrier: ∇B(z) = ∇B(z_current) - ∇B(z_target)
        return (1.0 + W_grad_current - W_grad_target) * (relative_state - desired_relative_state)
    
    def compute_barrier_gradient(
        self,
        relative_state: np.ndarray,
        desired_relative_state: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the barrier function gradient (for non-recentered case).
        
        Returns:
            Gradient vector
        """
        if self.constraints.get("rec", False):
            # Recentered barrier
            return self.recentered_barrier(relative_state, desired_relative_state)
        else:
            # Regular barriers
            conn_term = (
                self.connectivity_maintenance(relative_state) 
                if self.constraints.get("conn", False) 
                else 0.0
            )
            coll_term = (
                self.collision_avoidance(relative_state) 
                if self.constraints.get("coll", False) 
                else 0.0
            )
            W_grad = conn_term - coll_term
            return W_grad * relative_state
    
    @abstractmethod
    def doControl(self, *args, **kwargs) -> np.ndarray:
        """Compute control input. Implemented by subclasses."""
        pass


class SingleIntegratorController(ControllerBase):
    """
    Controller for single-integrator dynamics: ẋ = u (velocity control).
    
    The control law directly computes desired velocity based on:
    - Formation consensus term: v_tilde
    - Barrier function term: v_star
    """
    
    def doControl(
        self,
        state_i: np.ndarray,
        relative_states: np.ndarray,
        desired_relative_states: np.ndarray,
        is_leader: bool = False,
    ) -> np.ndarray:
        """
        Compute velocity command for robot i.
        
        Args:
            state_i: Current state of robot i
            relative_states: MxD matrix of relative states [z_ij = x_i - x_j]
            desired_relative_states: MxD matrix of desired relative states
            is_leader: Whether this robot is the leader
        
        Returns:
            Control input u_i (velocity command)
        """
        # Leader has no control law (or can have a predefined trajectory)
        if is_leader:
            return np.zeros(self.dim)
        
        # Initialize control contributions
        v_tilde = np.zeros(self.dim)
        v_star = np.zeros(self.dim)
        
        # Sum contributions from all neighbors
        for k in range(len(relative_states)):
            z_k = relative_states[k]
            z_target_k = desired_relative_states[k]
            
            # Consensus term: difference between current and desired relative position
            v_tilde -= (z_k - z_target_k)
            
            # Barrier term
            W_grad = self.compute_barrier_gradient(z_k, z_target_k)
            v_star -= self.control_gains["c1"] * W_grad
        
        # Compute control input
        u = self.control_gains["c2"] * v_tilde + v_star
        
        # Velocity saturation
        u_norm = np.linalg.norm(u)
        if u_norm > self.limits["v_max"]:
            u *= self.limits["v_max"] / u_norm
        
        return u


class DoubleIntegratorController(ControllerBase):
    """
    Controller for double-integrator dynamics: v̇ = u (acceleration control).
    
    Uses command filtering (backstepping) and includes velocity barrier functions
    to keep velocity within limits.
    
    Attributes:
        x_lf: Command-filtered position (per robot)
        v_lf: Derivative of filtered position
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # State for command filter (will be initialized per robot in simulation)
        # These are typically stored in the simulator or outside this class
        self.x_lf = np.zeros(self.dim)
        self.v_lf = np.zeros(self.dim)
    
    def command_filter(
        self,
        x_lf_current: np.ndarray,
        v_lf_current: np.ndarray,
        x_star: np.ndarray,
        dt: float,
    ) -> tuple:
        """
        Apply command filter (2nd-order low-pass filter).
        
        Implements: ẍ = -2*ωn*ẋ - ωn²*(x - x*)
        
        Args:
            x_lf_current: Current filtered position
            v_lf_current: Current filtered velocity
            x_star: Desired (unfiltered) position
            dt: Time step
        
        Returns:
            (x_lf_new, v_lf_new)
        """
        wn = self.control_gains["wn"]
        
        # Acceleration of the filter
        x_acc = -2 * wn * v_lf_current - wn ** 2 * (x_lf_current - x_star)
        
        # Update velocity and position
        v_lf_new = v_lf_current + x_acc * dt
        x_lf_new = x_lf_current + v_lf_new * dt
        
        return x_lf_new, v_lf_new
    
    def velocity_saturation_barrier(self, velocity_i: np.ndarray) -> np.ndarray:
        """
        Barrier function for velocity saturation.
        
        Prevents velocity from exceeding v_max without hard saturation.
        """
        v_sq = np.sum(velocity_i ** 2)
        denominator = max(self.limits["v_max"] ** 2 - v_sq, 1e-6)
        return (1.0 / denominator) * velocity_i
    
    def doControl(
        self,
        state_i: np.ndarray,
        relative_states: np.ndarray,
        desired_relative_states: np.ndarray,
        velocity_i: np.ndarray,
        x_lf: np.ndarray = None,
        v_lf: np.ndarray = None,
        is_leader: bool = False,
    ) -> np.ndarray:
        """
        Compute acceleration command for robot i.
        
        Args:
            state_i: Current state of robot i (position only)
            relative_states: MxD matrix of relative states
            desired_relative_states: MxD matrix of desired relative states
            velocity_i: Current velocity of robot i
            x_lf: Current filtered position (optional, for tracking filter state)
            v_lf: Current filtered velocity (optional, for tracking filter state)
            is_leader: Whether this robot is the leader
        
        Returns:
            Control input u_i (acceleration command)
        """
        if is_leader:
            return np.zeros(self.dim)
        
        # Use internal state if not provided
        if x_lf is None:
            x_lf = self.x_lf
        if v_lf is None:
            v_lf = self.v_lf

        # Compute virtual control (same as single integrator)
        z1 = np.zeros(self.dim)
        for k in range(len(relative_states)):
            z_k = relative_states[k]
            z_target_k = desired_relative_states[k]
            nominal_error = z_k - z_target_k
            W_grad = self.compute_barrier_gradient(z_k, z_target_k)
            z1 -= (nominal_error + W_grad)  # Sum contributions from all neighbors
        
        # Virtual control
        alpha_i = self.control_gains["c1"] * z1
        
        # Apply command filter
        dt = self.limits["dt"]
        x_lf_new, v_lf_new = self.command_filter(x_lf, v_lf, alpha_i, dt)
        
        # Update internal filter state
        self.x_lf = x_lf_new
        self.v_lf = v_lf_new
        
        # Tracking error
        z2 = velocity_i - x_lf_new
        
        # Acceleration command (Equation 3.82 from thesis)
        u = v_lf_new - self.control_gains["c2"] * z2 + z1

        # Velocity barrier term
        u_v_barrier = self.velocity_saturation_barrier(velocity_i)
        u -= self.control_gains["c3"] * u_v_barrier
        
        # Acceleration saturation
        u_norm = np.linalg.norm(u)
        if u_norm > self.limits["a_max"]:
            u *= self.limits["a_max"] / u_norm

        return u
    
    def reset_filter(self):
        """Reset the command filter state."""
        self.x_lf = np.zeros(self.dim)
        self.v_lf = np.zeros(self.dim)


class FieldOfViewController:
    """
    Separate controller for Field-of-View orientation constraints.
    
    Computes angular velocity commands to keep the target in view.
    """
    
    def __init__(
        self,
        robot_id: int,
        dim: int,
        fov_gains: dict,
        fov_limits: dict,
    ):
        """
        Args:
            robot_id: Robot ID
            dim: State dimension (typically 3 for 3D with quaternions)
            fov_gains: Dict with keys {kp_yaw, kp_pitch, c4_h, c4_v}
            fov_limits: Dict with keys {theta_max_x, theta_max_y}
        """
        self.robot_id = robot_id
        self.dim = dim
        self.fov_gains = fov_gains.copy()
        self.fov_limits = fov_limits.copy()
    
    def compute_orientation_input(
        self,
        position_i: np.ndarray,
        position_target: np.ndarray,
        quaternion_i: np.ndarray,
    ) -> tuple:
        """
        Compute angular velocity command to keep target in field of view.
        
        Args:
            position_i: Position of this robot
            position_target: Position of target robot
            quaternion_i: Quaternion orientation [x, y, z, w]
        
        Returns:
            (omega_cmd, alpha_h, alpha_v) - angular velocity and errors
        """
        # Relative bearing
        p_diff = position_target - position_i
        d_ij = np.linalg.norm(p_diff)
        
        if d_ij < 1e-6:
            return np.zeros(3), 0.0, 0.0
        
        # Transform to body frame
        rot_i = R.from_quat(quaternion_i).as_matrix()
        beta_ij = rot_i.T @ (p_diff / d_ij)
        
        # Extract angular errors
        alpha_h = np.arctan2(beta_ij[1], beta_ij[0])  # Horizontal (yaw)
        alpha_v = np.arctan2(beta_ij[2], beta_ij[0])  # Vertical (pitch)
        
        # Barrier functions       
        W_h = 1.0 / max(self.fov_limits["theta_max_x"]**2 - alpha_h**2, 1e-6)
        W_v = 1.0 / max(self.fov_limits["theta_max_y"]**2 - alpha_v**2, 1e-6)
        
        # Angular velocity command
        omega_cmd = np.array([
            0.0,
            -(self.fov_gains["kp_pitch"] * alpha_v) - (self.fov_gains["c4_v"] * W_v * alpha_v),
            (self.fov_gains["kp_yaw"] * alpha_h) + (self.fov_gains["c4_h"] * W_h * alpha_h),
        ])
        
        # Saturation
        omega_max = 1.0
        omega_norm = np.linalg.norm(omega_cmd)
        if omega_norm > omega_max:
            omega_cmd *= omega_max / omega_norm

        # Update quaternion (compute + apply in controller scope)
        q_updated = self.update_quaternion(
            quaternion_i,
            omega_cmd,
            self.fov_limits["dt"],
        )

        return omega_cmd, alpha_h, alpha_v, q_updated
        
    def update_quaternion(
        self,
        quaternion_i: np.ndarray,
        omega: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Update quaternion orientation based on body-frame angular velocity.

        Args:
            quaternion_i: Current quaternion [x, y, z, w]
            omega: Angular velocity [p, q, r] in body frame
            dt: Timestep

        Returns:
            Updated quaternion as np.ndarray
        """
        q_current = quaternion_i

        rotation_vec = omega * dt
        angle = np.linalg.norm(rotation_vec)

        if angle > 1e-8:
            delta_rot = R.from_rotvec(rotation_vec)
            q_obj_current = R.from_quat(q_current)
            q_new_obj = q_obj_current * delta_rot
            q_new = q_new_obj.as_quat()
        else:
            q_new = q_current.copy()

        # Normalize to prevent drift
        q_new /= np.linalg.norm(q_new)

        return q_new