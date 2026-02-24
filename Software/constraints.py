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

    #* Field of View Constraint (assuming agents are only looking at the leader).
    def field_of_view(self, params):
        fov_err_deg = [0.0] # Leader has no error.

        for i in range(1, params.N):
            z_k = params.x[0] - params.x[i] # Vector from agent i to leader (0). 
            target_yaw = np.arctan2(z_k[1], z_k[0]) # Angle agents have to face the leader.
            alpha_k = (params.yaw[i] - target_yaw + np.pi) % (2 * np.pi) - np.pi # Current angular error alpha_k (normalized -pi to pi)
    
            # Barrier Function (Equation 4.80).
            # Denominator goes to zero as error approaches the delta_alpha limit.
            W_fov = 1 / max(params.delta_alpha**2 - alpha_k**2, 1e-2)
            
            # Control Law (Kinematic Yaw Rate).
            # Negative sign ensures agents are moving opposite to the error.
            yaw_rate = -(params.kp_yaw * alpha_k + params.c3 * W_fov * alpha_k)
            
            # Saturation. Set max rotation rate to become closer to reality.
            if abs(yaw_rate) > params.r_max:
                yaw_rate = np.sign(yaw_rate) * params.r_max

            params.yaw[i] += yaw_rate * params.dt # Update the yaw state.
            params.yaw[i] = (params.yaw[i] + np.pi) % (2 * np.pi) - np.pi # Wrap yaw to [-pi, pi] to prevent the variable from growing to infinity.
            fov_err_deg.append(np.degrees(alpha_k)) # Store in degrees for the plot.
            
        return fov_err_deg