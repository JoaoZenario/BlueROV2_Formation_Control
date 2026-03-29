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
from BlueROV2_thrust import rotation_matrix
from qpsolvers import solve_qp
from scipy import sparse


class ControllerBase(ABC):
    """
    Abstract base class for all robot controllers.
    
    Attributes:
        robot_id (int): Robot identifier
        params (Parameters): Simulation parameters/information
    """
    
    def __init__(
        self,
        robot_id: int,
        params
    ):
        """
        Initialize a controller.
        
        Args:
            robot_id: Robot ID (0-indexed)
            params: Simulation parameters (contains dimension, control gains, constraints, limits)
        """
        self.robot_id = robot_id
        self.params = params
    
    # ============================
    # --- Constraint Methods ---
    # ============================
    
    def connectivity_maintenance(
        self,
        relative_state: np.ndarray
    ) -> float:
        """
        Connectivity maintenance barrier function.
        
        Args:
            relative_state: z_k = x_i - x_j (relative position)
        
        Returns:
            Barrier function value W_conn
        """
        W_conn = 1 / max(self.params.d_max**2 - np.sum(relative_state**2), 1e-6)
        return W_conn
    
    def collision_avoidance(
        self,
        relative_state: np.ndarray
    ) -> float:
        """
        Collision avoidance barrier function.
        
        Args:
            relative_state: z_k = x_i - x_j (relative position)
        
        Returns:
            Barrier function value W_coll
        """
        W_coll = 1 / max(np.sum(relative_state**2) - self.params.d_min**2, 1e-6)
        return W_coll

    def recentered_barrier(
        self,
        relative_state: np.ndarray,
        desired_relative_state: np.ndarray
    ) -> np.ndarray:
        """
        Recentered Lyapunov barrier function.
        
        Args:
            relative_state: z_k = x_i - x_j (current relative position)
            desired_relative_state: z_target = o_i - o_j (desired relative position)
        
        Returns:
            Gradient of recentered barrier function
        """
        
        # Evaluate constraints at current position
        if self.params.conn:
            W_conn_current = self.connectivity_maintenance(relative_state)
            W_conn_target = self.connectivity_maintenance(desired_relative_state)
        else:
            W_conn_current = 0.0
            W_conn_target = 0.0
        
        if self.params.coll:
            W_coll_current = self.collision_avoidance(relative_state)
            W_coll_target = self.collision_avoidance(desired_relative_state)
        else:
            W_coll_current = 0.0
            W_coll_target = 0.0
        
        W_grad_current = W_conn_current - W_coll_current
        W_grad_target = W_conn_target - W_coll_target
        
        # Recentered barrier: ∇B(z) = ∇B(z_current) - ∇B(z_target)
        return (1.0 + W_grad_current - W_grad_target) * (relative_state - desired_relative_state)
    
    def velocity_saturation_barrier(
        self,
        velocity_i: np.ndarray
    ) -> np.ndarray:
        """
        Barrier function for velocity saturation.
        
        Prevents velocity from exceeding v_max without hard saturation.
        """
        v_sat = (1.0 / (max(self.params.v_max ** 2 - np.sum(velocity_i ** 2), 1e-6))) * velocity_i
        return v_sat

    def compute_barrier_gradient(
        self,
        relative_state: np.ndarray,
        desired_relative_state: np.ndarray
    ) -> np.ndarray:
        """
        Compute the barrier function gradient (for non-recentered case).
        
        Returns:
            Gradient vector
        """
        if self.params.rec:
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
    
    def command_filter(
        self,
        x_lf_current: np.ndarray,
        v_lf_current: np.ndarray,
        x_star: np.ndarray
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

        x_acc = -2 * self.params.wn * v_lf_current - self.params.wn ** 2 * (x_lf_current - x_star) # Acceleration of the filter
        
        # Update velocity and position
        v_lf_new = v_lf_current + x_acc * self.params.dt
        x_lf_new = x_lf_current + v_lf_new * self.params.dt
        
        return x_lf_new, v_lf_new

    @abstractmethod
    def doControl(
        self,
        *args,
        **kwargs
    ) -> np.ndarray:
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
        relative_states: np.ndarray,
        desired_relative_states: np.ndarray,
        is_leader: bool = False
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
            return np.zeros(self.params.dim)
        
        # Initialize control contributions
        v_tilde = np.zeros(self.params.dim)
        v_star = np.zeros(self.params.dim)
        
        # Sum contributions from all neighbors
        for k in range(len(relative_states)):
            z_k = relative_states[k]
            z_target_k = desired_relative_states[k]
            
            # Consensus term: difference between current and desired relative position
            v_tilde -= (z_k - z_target_k)
            
            # Barrier term
            W_grad = self.compute_barrier_gradient(z_k, z_target_k)
            v_star -= self.params.c1 * W_grad
        
        # Compute control input
        u = self.params.c2 * v_tilde + v_star
        
        # Velocity saturation
        u_norm = np.linalg.norm(u)
        if u_norm > self.params.v_max:
            u *= self.params.v_max / u_norm
        
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
    
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # State for command filter (will be initialized per robot in simulation)
        # These are typically stored in the simulator or outside this class
        self.x_lf = np.zeros(self.params.dim)
        self.v_lf = np.zeros(self.params.dim)
    
    def doControl(
        self,
        relative_states: np.ndarray,
        desired_relative_states: np.ndarray,
        velocity_i: np.ndarray,
        x_lf: np.ndarray = None,
        v_lf: np.ndarray = None,
        is_leader: bool = False
    ) -> np.ndarray:
        """
        Compute acceleration command for robot i.
        
        Args:
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
            return np.zeros(self.params.dim)
        
        # Use internal state if not provided
        if x_lf is None:
            x_lf = self.x_lf
        if v_lf is None:
            v_lf = self.v_lf

        # Compute virtual control (same as single integrator)
        z1 = np.zeros(self.params.dim)
        for k in range(len(relative_states)):
            z_k = relative_states[k]
            z_target_k = desired_relative_states[k]
            nominal_error = z_k - z_target_k
            W_grad = self.compute_barrier_gradient(z_k, z_target_k)
            z1 -= (nominal_error + W_grad)  # Sum contributions from all neighbors
        
        # Virtual control
        alpha_i = self.params.c1 * z1
        
        # Apply command filter
        x_lf_new, v_lf_new = self.command_filter(
            x_lf, # Current filtered position.
            v_lf, # Current filtered velocity.
            alpha_i # Desired position from the virtual control (alpha_i is the "jerk" target).
        )
        
        # Update internal filter state
        self.x_lf = x_lf_new
        self.v_lf = v_lf_new
        
        # Tracking error
        z2 = velocity_i - x_lf_new
        
        # Acceleration command (Equation 3.82 from thesis)
        u = v_lf_new - self.params.c2 * z2 + z1

        # Velocity barrier term
        u_v_barrier = self.velocity_saturation_barrier(velocity_i)
        u -= self.params.c3 * u_v_barrier
        
        # Acceleration saturation
        u_norm = np.linalg.norm(u)
        if u_norm > self.params.a_max:
            u *= self.params.a_max / u_norm

        return u

class FieldOfViewController(ControllerBase):
    """
    Separate controller for Field-of-View orientation constraints.
    
    Computes angular velocity commands to keep the target in view.
    """
    
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def compute_orientation_input(
        self,
        position_i: np.ndarray,
        position_target: np.ndarray,
        quaternion_i: np.ndarray
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
        W_h = 1.0 / max(self.params.theta_max_x**2 - alpha_h**2, 1e-6)
        W_v = 1.0 / max(self.params.theta_max_y**2 - alpha_v**2, 1e-6)
        
        # Angular velocity command
        omega_cmd = np.array([
            0.0,
            -(self.params.kp_pitch * alpha_v) - (self.params.c4_v * W_v * alpha_v),
            (self.params.kp_yaw * alpha_h) + (self.params.c4_h * W_h * alpha_h),
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
        )

        return omega_cmd, alpha_h, alpha_v, q_updated
        
    def update_quaternion(
        self,
        quaternion_i: np.ndarray,
        omega: np.ndarray
    ) -> np.ndarray:
        """
        Update quaternion orientation based on body-frame angular velocity.

        Args:
            quaternion_i: Current quaternion [x, y, z, w]
            omega: Angular velocity [p, q, r] in body frame

        Returns:
            Updated quaternion as np.ndarray
        """
        q_current = quaternion_i

        rotation_vec = omega * self.params.dt
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

    def doControl(
        self,
        *args,
        **kwargs
    ) -> np.ndarray:
        """
        This method is not used directly for control, but could be implemented
        to return the latest computed angular velocity command if needed.
        """
        return np.zeros(3)

class BlueROV2Controller(ControllerBase):
    def __init__(self, robot_id, params, rov_model):
        # Always call the parent so self.params and self.robot_id are available
        super().__init__(robot_id, params)
        
        self.rov = rov_model
        self.v_f = np.zeros(6)
        self.dot_v_f = np.zeros(6)

    def _get_dynamics_matrices(
        self,
        state
    ) -> tuple :
        p = state[0:3] # Position [x, y, z]
        q = state[3:6] # Orientation (Euler angles) [phi, theta, psi]
        eta = np.hstack((p, q)) # Pose vector
        nu = state[6:12] # [u, v, w, p, q, r]
        phi, theta, psi = eta[3:6]
        
        M = self.rov.M # Total inertia matrix (rigid body + added mass)        
        C = self.rov._coriolis(nu) # Coriolis-centripetal matrix
        D = self.rov._damping(nu) # Hydrodynamic damping matrix
        G = self.rov._restoring(phi, theta, psi) # Hydrostatic restoring forces and moments
        
        return M, C, D, G, nu
    
    def get_actuator_matrix(
        self
    ) -> np.ndarray:
        # Distances from center of gravity (m) - Standard BlueROV2 Heavy
        lx = 0.150  # longitudinal
        ly = 0.150  # lateral
        lz = 0.010  # vertical offset for vectors
        
        # Angle for vectored thrusters (45 degrees)
        c45 = np.cos(np.pi/4)
        s45 = np.sin(np.pi/4)

        # B matrix: Each column is the 6D contribution of one thruster
        # Thrusters 1-4: Vectored (Horizontal plane)
        # Thrusters 5-8: Vertical
        B = np.array([
            # T1    T2     T3     T4     T5    T6     T7    T8
            [ s45,  s45,  -s45,  -s45,   0,    0,     0,    0], # Fx
            [ c45, -c45,   c45, -c45,   0,    0,     0,    0], # Fy
            [ 0,    0,     0,     0,    -1,   -1,    -1,   -1], # Fz
            [ 0,    0,     0,     0,   -ly,   ly,   -ly,   ly], # Tx (Roll)
            [ 0,    0,     0,     0,    lx,   lx,   -lx,  -lx], # Ty (Pitch)
            [ lz,  -lz,    lz,   -lz,    0,    0,     0,    0]  # Tz (Yaw - simplified)
        ])
        return B

    def controller1(
        self,
        state,
        v_star_inertial
    ) -> np.ndarray:
        """
        Command-filtered backstepping controller enhanced through feedback linearization
        Translates Inertial Formation Command to Body Wrench.
        v_star_inertial: 6D desired velocity [vx, vy, vz, p_d, q_d, r_d]
        """
        M, C, D, G, nu = self._get_dynamics_matrices(state)
        eta = state[0:6]
        
        R_b2n = rotation_matrix(eta[3], eta[4], eta[5]) # Rotation matrix from body frame to inertial frame (The robot's perspective)
        R_n2b = R_b2n.T # Rotation matrix from inertial frame to body frame (The world's perspective)
        
        v_star_body = np.zeros(6) # Initialize body-frame command
        v_star_body[:3] = R_n2b @ v_star_inertial[:3] # Transform world's command velocity to robot's body frame
        v_star_body[3:] = v_star_inertial[3:] # Rotational commands are the same in both frames

        # Apply command filter to get the filtered velocity command (v_f) and its derivative (dot_v_f)
        self.v_f, self.dot_v_f = self.command_filter(
            self.v_f,      # Current state of filter
            self.dot_v_f,  # Current rate of filter
            v_star_body,   # The "jerk" target from the leader
        )
        s_v = nu - self.v_f # Filtered inner-loop error (s_v = v - v_f)

        Kv = np.diag([40.0, 40.0, 40.0, 10.0, 10.0, 10.0]) # Gain matrix for velocity tracking (Positive Definite)
        d = Kv @ s_v # Disturbances and ummodeled dynamics compensation
        h = C @ nu + D @ nu + G
        tau = M @ self.dot_v_f + h - d
        return tau

    def controller2(
        self,
        state,
        v_star_body
    ) -> np.ndarray:
        """
        CLF-QP based controller (control Lyapunov function inside a quadratic program)
        """
        M, C, D, G, nu = self._get_dynamics_matrices(state)
        
        e_v = nu - v_star_body # Velocity error
        V = 0.5 * e_v.T @ M @ e_v # Quadratic CLF based on kinetic energy of the error
        grad_V = M @ e_v # Gradient of the CLF with respect to velocity (∇V = M * e_v)
        
        f_sys = -np.linalg.inv(M) @ (C @ nu + D @ nu + G) # System dynamics without control input (f(x) in CLF-QP formulation) (With the damping term the systems converges to the desired position in a faster way but without it the system converges with slighly more accuracy, although it takes longer)
        LfV = grad_V.T @ f_sys # Lie derivative of V along the system dynamics (LfV = ∇V * f(x))
        LgV = grad_V.T @ np.linalg.inv(M) # Lie derivative of V along the control input (LgV = ∇V * g(x), where g(x) = M^-1 for fully actuated systems)
        
        # Setup QP Matrices: min 0.5 * x.T @ P @ x + q.T @ x
        # x = [tau_0, ..., tau_5, delta] (7 variables)
        P = np.eye(7)
        P[6, 6] = 1000.0 # High penalty (p) for the slack variable delta
        q = np.zeros(7)
        
        # Inequality Constraint: LfV + LgV * tau <= -alpha * V + delta
        # Rewritten as: [LgV, -1] @ [tau, delta].T <= -alpha * V - LfV
        alpha = 2.0
        A = np.hstack([LgV, [-1.0]])
        b = -alpha * V - LfV
        
        # Bounds (Thruster Saturation)
        tau_limit = 40.0 # Newtons
        lb = np.array([-tau_limit]*6 + [0.0]) # delta must be >= 0
        ub = np.array([tau_limit]*6 + [np.inf])
        
        P_sparse = sparse.csc_matrix(P)
        A_sparse = sparse.csc_matrix(A)
        sol = solve_qp(P_sparse, q, A_sparse, b, None, None, lb, ub, solver="osqp")
        
        if sol is not None:
            return sol[:6] # Return the optimized wrench
        else:
            return np.zeros(6) # Fallback

    def controller3(
        self,
        state,
        v_star_inertial
    ) -> np.ndarray:
        """
        CLF-QP Controller with Direct Thruster Allocation.
        Translates Inertial Command to 8 Individual Thruster Forces.
        """
        # 1. Get Physics Matrices and Current Velocity
        M, C, D, G, nu = self._get_dynamics_matrices(state)
        eta = state[0:6]
        B = self.get_actuator_matrix()
        
        R_b2n = rotation_matrix(eta[3], eta[4], eta[5]) # Rotation matrix from body frame to inertial frame (The robot's perspective)
        R_n2b = R_b2n.T # Rotation matrix from inertial frame to body frame (The world's perspective)
        
        v_star_body = np.zeros(6) # Initialize body-frame command
        v_star_body[:3] = R_n2b @ v_star_inertial[:3] # Transform world's command velocity to robot's body frame
        v_star_body[3:] = v_star_inertial[3:] # Rotational commands are the same in both frames

        e_v = nu - v_star_body # Velocity error
        V = 0.5 * e_v.T @ M @ e_v # Quadratic CLF based on kinetic energy of the error
        grad_V = M @ e_v # Gradient of the CLF with respect to velocity (∇V = M * e_v)
        
        f_sys = -np.linalg.inv(M) @ (C @ nu + D @ nu + G) # System dynamics without control input (f(x) in CLF-QP formulation) (With the damping term the systems converges to the desired position in a faster way but without it the system converges with slighly more accuracy, although it takes longer)
        LfV = grad_V.T @ f_sys # Lie derivative of V along the system dynamics (LfV = ∇V * f(x))
        LgV_B = (grad_V.T @ np.linalg.inv(M)) @ B # LgV_B represents how the 8 thrusters affect the Energy V

        # Setup QP Matrices: min 0.5 * x.T @ P @ x + q.T @ x
        # Variables: [f1, f2, f3, f4, f5, f6, f7, f8, delta] (9 total)
        P = np.eye(9)
        P[8, 8] = 1000.0  # Penalty for the slack variable delta
        q = np.zeros(9)
        
        # Stability Constraint: LfV + LgV_B * f <= -alpha * V + delta
        alpha = 2.0
        # Inequality is: [LgV_B, -1] @ [f, delta].T <= -alpha * V - LfV
        A = np.hstack([LgV_B, [-1.0]])
        b = -alpha * V - LfV
        
        # Physical Bounds (Thruster Limits in Newtons)
        f_limit = 40.0 
        lb = np.array([-f_limit]*8 + [0.0])      # Each motor >= -40N, delta >= 0
        ub = np.array([f_limit]*8 + [np.inf])    # Each motor <= 40N
                
        P_sparse = sparse.csc_matrix(P)
        A_sparse = sparse.csc_matrix(A)
        sol = solve_qp(P_sparse, q, A_sparse, b, None, None, lb, ub, solver="osqp")
        
        if sol is not None:
            # Return only the first 8 values (the thruster forces)
            return sol[:8] 
        else:
            # Fallback if optimization fails
            return np.zeros(8)

    def doControl(
        self,
        **kwargs
    ) -> np.ndarray:
        """
        Implementation of abstract method from ControllerBase.
        Not used for BlueROV2 low-level wrench calculation.
        """
        return np.zeros(6)