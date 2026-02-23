#!/usr/bin/env python3

##########################################################
#                                                        #
# Double Integrator                                      #
# Code Author: João Zenário                              #
# Equations and concepts based on: Esteban's thesis      #
#                                                        #     
##########################################################

import numpy as np

class double_integrator:
    def __init__(self):
        pass

    #* Function to set leader's acceleration throughout the whole simulation.
    def leader_acceleration(self, params, u):
        """
        if params.dim == 2:
            if (params.counter < 0.5 * params.steps):
                target_v = np.array([10, 10])
            else:
                target_v = np.array([-7, -13])
            u[0] = target_v - params.v[0]
        if params.dim == 3:
            if (params.counter < 0.5 * params.steps):
                target_v = np.array([10, 10, -5]) 
            else:
                target_v = np.array([-9, 5, 20]) 
            u[0] = target_v - params.v[0]
        """
        u[0] = [0, 0, 0]
        return u

    #* Function that applies the command filter as a form of backstepping control (command-filtered backstepping).
    def command_filter(self, params, i, x_star_l, dt):
        # Calculate the second derivative (acceleration of the filter).
        # Equation: x_lf = [ωn² / (s² + 2*ωn*s + ωn²)] * x*_l.
        ẍ_lf  = -2 * params.wn * params.ẋ_lf[i] - params.wn**2 * (params.x_lf[i] - x_star_l)

        # Update the first derivative (velocity of the filter).
        params.ẋ_lf[i] += ẍ_lf * dt

        # Update the filtered position.
        params.x_lf[i] += params.ẋ_lf[i] * dt

        return params.x_lf[i], params.ẋ_lf[i]

    #* Function to compute each agent's acceleration in the exact moment without constraints.
    # The agents are in an initial pose (with a certain relative pose to each others)
    # and the goal is the agents moving to a pose where the relative pose to each others is the desired one (relative_pose_desired). 
    def acceleration_no_constraints(self, params, u):
        # Generalized.
        for i in range(1, params.N):
            # Initialize position error vector.
            pos_error = np.zeros(params.dim)
            for j in range(params.N):
                if params.A[i, j] == 1:
                    pos_error += (params.x[i] - params.x[j] - params.relative_pos_desired[i, j]) # Compute the position error.

            alpha_i = -params.c1 * pos_error # Compute the virtual control.
            alpha_f, alpha_dot_f = self.command_filter(params, i, alpha_i, params.dt) # Apply command filter. 
            vel_error = params.v[i] - alpha_f # Compute the velocity error.
            u[i] = alpha_dot_f - params.c2 * vel_error - pos_error # Acceleration of agent i.

            u_i = np.linalg.norm(u[i]) # Magnitude.
            # Set a max acceleration to become closer to reality.
            if u_i > params.a_max:
                u[i] *= params.a_max / u_i
        return u

    #* Function to compute each agent's acceleration in the exact moment with connectivity maintenance.
    def acceleration_connectivity_maintenance(self, params, u):
        # Generalized.
        for i in range(1, params.N):
            # Initialize sum vector.
            z1 = np.zeros(params.dim)
            for j in range(params.N):
                if params.A[i, j] == 1:
                    z_k = params.x[i] - params.x[j]
                    pos_error = (z_k - params.relative_pos_desired[i, j]) # Compute the position error.
                    # Lyapunov barrier functions for connectivity maintenance throughout the whole trajectory. Equation 3.81.
                    # max() is used to prevent division by zero, if robots touch the limit, and also to prevent the opposite behvior when agents are initially out of the raidus threshold.
                    W_grad = (1 / (max(params.d_max**2 - np.sum(z_k**2), 1e-2))) * z_k
                    z1 += (pos_error + W_grad) # Sum everything (contribution agent j has on agent i).

            alpha_i = -params.c1 * z1
            alpha_f, alpha_dot_f = self.command_filter(params, i, alpha_i, params.dt) # Apply command filter.
            z2 = params.v[i] - alpha_f # Compute velocity error.
            u[i] = alpha_dot_f - params.c2 * z2 - z1 # Agent i acceleration. Equation 3.82.

            u_i = np.linalg.norm(u[i]) # Magnitude.
            # Set a max acceleration in order to don't explode with the simulation.
            if u_i > params.a_max:
                u[i] *= params.a_max / u_i
        return u
    
    #* Function to compute each agent's acceleration in the exact moment with connectivity maintenance and collision avoidance.
    def acceleration_connectivity_maintenance_and_collision_avoidance(self, params, u):
        # Generalized.
        for i in range(1, params.N):
            # Initialize velocity vectors.
            z1 = np.zeros(params.dim)
            for j in range(params.N):
                if params.A[i, j] == 1:
                    z_k = params.x[i] - params.x[j]
                    pos_error = (z_k - params.relative_pos_desired[i, j]) # Compute the position error.
                    # Lyapunov barrier functions for connectivity maintenance and collision avoidance throughout the whole trajectory.
                    # max() is used to prevent division by zero, if robots touch the limit, and also to prevent the opposite behvior when agents are initially out of the raidus threshold.
                    # Connectivity Maintenance - Equation 3.81 and Collision Avoidance - Equation 5.8 and 5.9
                    W_conn = 1 / (max(params.d_max**2 - np.sum(z_k**2), 1e-2))
                    W_coll = 1 / (max(np.sum(z_k**2) - params.d_min**2, 1e-2))
                    W_grad = (W_conn - W_coll) * z_k
                    z1 += (pos_error + W_grad) # Sum everything (contribution agent j has on agent i).

            alpha_i = -params.c1 * z1 # Compute the virtual control.
            alpha_f, alpha_dot_f = self.command_filter(params, i, alpha_i, params.dt) # Apply command filter.
            z2 = params.v[i] - alpha_f
            u[i] = alpha_dot_f - params.c2 * z2 - z1 # Agent i acceleration. Equation 3.82.

            u_i = np.linalg.norm(u[i]) # Magnitude.
            # Set a max acceleration in order to don't explode with the simulation.
            if u_i > params.a_max:
                u[i] *= params.a_max / u_i
        return u

    #* Function to compute each agent's orientation in order to keep their "leader" in the field-of-view.
    def compute_yaw_fov(self, params):
        current_fov_errors = [0.0] # Leader has no FOV error

        for i in range(1, params.N):
            # 1. Calculate direction to leader (Target)
            vec_to_leader = params.x[0] - params.x[i]
            target_yaw = np.arctan2(vec_to_leader[1], vec_to_leader[0])
            
            # 2. Calculate current angular error (alpha_k)
            # Normalizes error between -pi and pi
            alpha_k = (target_yaw - params.yaw[i] + np.pi) % (2 * np.pi) - np.pi
            
            # 3. Compute the Barrier Term
            # As alpha_k -> delta_alpha, the denominator -> 0, and the gain -> infinity
            # We use a safety epsilon (1e-3) to prevent division by zero
            W_fov = 1 / (max(params.delta_alpha**2 - alpha_k**2, 1e-2))
            
            # 4. Control Law: P-control + Barrier-Push
            # c_fov is a weight you should add to your params (e.g., params.c_fov = 0.5)
            yaw_rate = params.kp_yaw * alpha_k + (params.c3 * W_fov * alpha_k)
            
            # 5. Apply update
            params.yaw[i] += yaw_rate * params.dt
            
            # Store in degrees for the animation/plot
            current_fov_errors.append(np.degrees(alpha_k))
            
        return current_fov_errors