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
    def connectivity_maintenance(self, params, z_k):
        W_conn = 1 / (max(params.d_max**2 - np.sum(z_k**2), 1e-2))
        return W_conn
    #* Collision Avoidance Constraint.
    def collision_avoidance(self, params, z_k):
        W_coll = 1 / (max(np.sum(z_k**2) - params.d_min**2, 1e-2))
        return W_coll

    #* Field of View Constraint (Rectangular/Pyramidal 3D)
    def field_of_view(self, params):
        fov_percent_errors = [0.0]  # Leader placeholder
        
        for i in range(1, params.N):
            # 1. Relative bearing in Body Frame
            p_diff = params.x[0] - params.x[i]
            d_ij = np.linalg.norm(p_diff)
            rot_i = R.from_quat(params.q[i]).as_matrix()
            beta_ij = rot_i.T @ (p_diff / d_ij)
            
            # 2. Extract angular offsets
            alpha_h = np.arctan2(beta_ij[1], beta_ij[0]) # Horizontal
            alpha_v = np.arctan2(beta_ij[2], beta_ij[0]) # Vertical
            
            # 3. SQUARED Barrier Functions
            W_h = 1 / (max(params.theta_max_x**2 - alpha_h**2, 1e-2))
            W_v = 1 / (max(params.theta_max_y**2 - alpha_v**2, 1e-2))

            # 4. Control Commands
            omega_cmd = np.array([
                0.0,                           
                -(params.kp_pitch * alpha_v) - (params.c3_v * W_v * alpha_v), 
                (params.kp_yaw * alpha_h) + (params.c3_h * W_h * alpha_h)  
            ])

            omega_max = 1
            mag = np.linalg.norm(omega_cmd)
            if mag > omega_max:
                omega_cmd *= omega_max / mag

            self.update_quaternion(i, omega_cmd, params.dt, params)

            # 5. Calculate Normalized Error for the Plot
            # We check which axis is closer to its limit
            err_h = np.abs(alpha_h) / params.theta_max_x
            err_v = np.abs(alpha_v) / params.theta_max_y
            
            # Total error is the maximum of the two, converted to percentage
            total_err_percent = max(err_h, err_v) * 100
            fov_percent_errors.append(total_err_percent)

        return fov_percent_errors