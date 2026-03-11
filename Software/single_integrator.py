#!/usr/bin/env python3

##########################################################
#                                                        #
# Single Integrator                                      #
# Code Author: João Zenário                              #
# Equations and concepts based on: Esteban's thesis      #
#                                                        #     
##########################################################

import numpy as np
from constraints import Constraints

class Single_integrator(Constraints):
    def __init__(self):
        Constraints.__init__(self)
        pass
    
    #* Function to set leader's velocity throughout the whole simulation.
    def leader_velocity(self, params, u):
        if params.dim == 2:
            if (params.counter < 0.5 * params.steps):
                u[0, 0] = 1  # Leader's x speed.
                u[0, 1] = 4 # Leader's y speed.
            else:
                u[0, 0] = -1.0 # Leader's x speed.
                u[0, 1] = -1.0 # Leader's y speed.
        if params.dim == 3:
            if (params.counter < 0.5 * params.steps):
                u[0, 0] = 5  # Leader's x speed.
                u[0, 1] = 20 # Leader's y speed.
                u[0, 2] = 5  # Leader's z speed.
            else:
                u[0, 0] = -10.0 # Leader's x speed.
                u[0, 1] = -10.0 # Leader's y speed.
                u[0, 2] = 5.0   # Leader's z speed.
        u[0] = [0, 0, 0]
        u_norm = np.linalg.norm(u[0]) # Magnitude.

        # Set a max velocity to become closer to reality.
        if u_norm > params.v_max:
            u[0] *= params.v_max / u_norm

        return u

    #* Function to compute each agent's velocity in the exact moment.
    def compute_velocity(self, params, u):
        for i in range(1, params.N):
            # Initialize velocity vectors.
            v_tilde = np.zeros(params.dim)
            v_star = np.zeros(params.dim)
            u_head = np.zeros(params.dim)
            for j in range(params.N):
                if params.A[i, j] == 1:
                    z_k = params.x[i] - params.x[j] # Computes the real relative pose.
                    v_tilde -= (z_k - params.relative_pos_desired[i, j]) # Add the contribution of neighbor j to agent i's velocity.
                    u_head = u[j].copy() # Store agent's j (edge's head) velocity vector.

                    if params.rec: # Recentered Lyapunov barrier function.
                        W_grad = self.recentered_barrier(params, z_k, i, j)
                    else: # "Regular" Lyapunov barrier function.
                        W_conn = self.connectivity_maintenance(params, z_k) if params.conn else 0 # Connectivity Maintenance Constraint.
                        W_coll = self.collision_avoidance(params, z_k) if params.coll else 0 # Collision Avoidance Constraint.
                        W_grad = (W_conn - W_coll) * z_k
                    v_star -= params.c1 * W_grad # Virtual control. The Incidence Matrix (Eik) is already being take into account in the if statement (Adjacency matrix).

            u[i] = params.c2 * v_tilde + v_star + u_head
            u_i = np.linalg.norm(u[i]) # Magnitude.

            # Saturation. Set a max velocity to become closer to reality.
            if u_i > params.v_max:
                u[i] *= params.v_max / u_i

        return u