#!/usr/bin/env python3

##########################################################
#                                                        #
# Double Integrator                                      #
# Code Author: João Zenário                              #
# Equations and concepts based on: Esteban's thesis      #
#                                                        #     
##########################################################

import numpy as np
from constraints import Constraints

class Double_integrator(Constraints):
    def __init__(self):
        pass

    #* Function to set leader's acceleration throughout the whole simulation.
    def leader_acceleration(self, params, u):
        if params.dim == 2:
            if (params.counter < 0.5 * params.steps):
                target_v = np.array([0.5, 0.7])
            else:
                target_v = np.array([-0.3, -0.9])
            u[0] = target_v - params.v[0]
        if params.dim == 3:
            if (params.counter < 0.5 * params.steps):
                target_v = np.array([0.5, 0.5, -0.3]) 
            else:
                target_v = np.array([-0.1, 0.2, 0.3]) 
            u[0] = target_v - params.v[0]

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
    
    #* Function to compute each agent's acceleration in the exact moment.
    def compute_acceleration(self, params, u):
        for i in range(1, params.N):
            # Initialize velocity vector.
            z1 = np.zeros(params.dim)
            for j in range(params.N):
                if params.A[i, j] == 1:
                    z_k = params.x[i] - params.x[j]
                    nominal_error = (z_k - params.relative_pos_desired[i, j])

                    if params.rec: # Recentered Lyapunov barrier function.
                        W_grad = self.recentered_barrier(params, z_k, i, j)
                    else: # "Regular" Lyapunov barrier function.
                        W_conn = self.connectivity_maintenance(params, z_k) if params.conn else 0 # Connectivity Maintenance Constraint.
                        W_coll = self.collision_avoidance(params, z_k) if params.coll else 0 # Collision Avoidance Constraint.
                        W_grad = (W_conn - W_coll) * z_k

                    z1 -= (nominal_error + W_grad) # Sum everything (contribution agent j has on agent i).

            alpha_i = params.c1 * z1 # Compute the virtual control.
            alpha_f, alpha_dot_f = self.command_filter(params, i, alpha_i, params.dt) # Apply command filter.
            z2 = params.v[i] - alpha_f
            u[i] = alpha_dot_f - params.c2 * z2 + z1 # Agent i acceleration. Equation 3.82.
            
            u_v_barrier = self.velocity_saturation(params, i)
            u[i] -= params.c3 * u_v_barrier

            # Saturation. Set a max acceleration in order to don't explode with the simulation.
            u_norm = np.linalg.norm(u[i])
            if u_norm > params.a_max:
                u[i] *= params.a_max / u_norm
                
        return u

    #* Function to apply barrier function in order to control/limit the velocity in double agent systems, since we can't to saturate it directly given that the acceleration is the input.
    def velocity_saturation(self, params, i):
        u_v_barrier = (1 / (max(params.v_max**2 - np.sum(params.v[i]**2), 1e-6))) * params.v[i]
        return u_v_barrier