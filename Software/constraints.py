#!/usr/bin/env python3

##########################################################
#                                                        #
# Constraints for the simulation                         #
# Code Author: João Zenário                              #
# Equations and concepts based on: Esteban's thesis      #
#                                                        #     
##########################################################

import numpy as np
from parameters import Parameters
from scipy.spatial.transform import Rotation as R

class Constraints:
    def __init__(self):
        pass
    
    #* Connectivity Maintenance Constraint.
    def connectivity_maintenance(self, params, d):
        # Lyapunov barrier functions throughout the whole trajectory.
        # Connectivity Maintenance - Equation 3.81.
        # max() is used to prevent division by zero, if robots touch the limit, and also to prevent the opposite behvior when agents are initially out of the raidus threshold.
        W_conn = 1 / (max(params.d_max**2 - np.sum(d**2), 1e-2))
        return W_conn

    #* Collision Avoidance Constraint.
    def collision_avoidance(self, params, d):
        # Lyapunov barrier functions throughout the whole trajectory.
        # Collision Avoidance - Equation 5.8 and 5.9.
        # max() is used to prevent division by zero, if robots touch the limit, and also to prevent the opposite behvior when agents are initially out of the raidus threshold.
        W_coll = 1 / (max(np.sum(d**2) - params.d_min**2, 1e-2))
        return W_coll

    #* Recentered Lyapunov Barrier Function.
    def recentered_barrier(self, params, z_k, i, j):
        target_vector = params.relative_pos_desired[i, j]

        # Connectivity Maintenance Constraint.
        if params.conn:
            W_conn_current = self.connectivity_maintenance(params, z_k)
            W_conn_target = self.connectivity_maintenance(params, target_vector)
        else:
            W_conn_current = 0
            W_conn_target = 0
        # Collision Avoidance Constraint.
        if params.coll:
            W_coll_current = self.collision_avoidance(params, z_k)
            W_coll_target = self.collision_avoidance(params, target_vector)
        else:
            W_coll_current = 0
            W_coll_target = 0
        W_grad_current = W_conn_current - W_coll_current # Gradient at current position.
        W_grad_target = W_conn_target - W_coll_target # Gradient at target position
        
        # For recentered barrier function: ∇B(z) = ∇B̄(z) - ∇B̄(0)
        return (W_grad_current - W_grad_target) * z_k

    #* Field of View Constraint (Rectangular/Pyramidal 3D).
    def field_of_view(self, params):
        fov_errors = [[0.0, 0.0]]  # Leader placeholder (has no error).
        
        for i in range(1, params.N):
            # Find the target, which is the lowest ID agent i is connected to. This is no longer needed since each agent is connected to only one other agent.
            connected_indices = np.where(params.A[i] == 1)[0]
            target_idx = connected_indices[0]

            # Relative bearing in Body Frame.
            p_diff = params.x[target_idx] - params.x[i]
            d_ij = np.linalg.norm(p_diff)
            rot_i = R.from_quat(params.q[i]).as_matrix()
            beta_ij = rot_i.T @ (p_diff / d_ij)
            
            # Extract angular offsets.
            alpha_h = np.arctan2(beta_ij[1], beta_ij[0]) # Horizontal.
            alpha_v = np.arctan2(beta_ij[2], beta_ij[0]) # Vertical.
            
            # Barrier functions.
            W_h = 1 / (max(params.theta_max_x**2 - alpha_h**2, 1e-2)) # Horizontal.
            W_v = 1 / (max(params.theta_max_y**2 - alpha_v**2, 1e-2)) # Veritcal.

            # Control Commands.
            omega_cmd = np.array([
                0.0,                           
                -(params.kp_pitch * alpha_v) - (params.c4_v * W_v * alpha_v), 
                (params.kp_yaw * alpha_h) + (params.c4_h * W_h * alpha_h)  
            ])

            # Saturate rotation speed/rate.
            omega_max = 1
            mag = np.linalg.norm(omega_cmd)
            if mag > omega_max:
                omega_cmd *= omega_max / mag

            self.update_quaternion(i, omega_cmd, params.dt, params)
            fov_errors.append([np.degrees(alpha_h), np.degrees(alpha_v)])

        return fov_errors