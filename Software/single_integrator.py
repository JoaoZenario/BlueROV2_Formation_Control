#!/usr/bin/env python3

##########################################################
#                                                        #
# Single Integrator                                      #
# Code Author: João Zenário                              #
# Equations and concepts based on: Esteban's thesis      #
#                                                        #     
##########################################################

import numpy as np

class single_integrator:
    def __init__(self):
        pass
    
    #* Function to set leader's velocity throughout the whole simulation.
    def leader_velocity(self, params, u):
        """
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
        """
        u[0] = [0, 0, 0]
        return u


    #* Function to compute each agent's velocity in the exact moment without constraints.
    # The agents are in an initial pose (with a certain relative pose to each others)
    # and the goal is the agents moving to a pose where the relative pose to each others is the desired one (relative_pose_desired). 
    def velocity_no_constraints(self, params, u):
        # Generalized.
        for i in range(1, params.N):
            # Initialize velocity vectors.
            agent_i_velocity = np.zeros(params.dim)
            u_head = np.zeros(params.dim)
            for j in range(params.N):
                if params.A[i, j] == 1:
                    relative_pos_real = params.x[i] - params.x[j] # Computes the real relative pose.
                    agent_i_velocity -= (relative_pos_real - params.relative_pos_desired[i, j]) # Add the contribution of neighbor j to agent i's velocity.
                    u_position = params.c2 * agent_i_velocity # Velocity needed for agent i to reach desired position.
                    u_head = u[j].copy() # Store agent's j (edge's head) velocity vector.
            # It's crucial to add agent's j velocity to the velocity that agent i would initially need to achieve the desired pose,
            # because agent j is in movement and that must be taken into account.
            u[i] = u_position + u_head # Agent's i velocity.
            u_i = np.linalg.norm(u[i]) # Magnitude.

            # Set a max velocity to become closer to reality.
            if u_i > params.v_max:
                u[i] *= params.v_max / u_i

        return u

    #* Function to compute each agent's velocity in the exact moment with connectivity maintenance.
    def velocity_connectivity_maintenance(self, params, u):
        delta_k = params.d_max**2
        # Generalized.
        for i in range(1, params.N):
            # Initialize velocity vectors.
            agent_i_velocity = np.zeros(params.dim)
            v_tilde = np.zeros(params.dim)
            v_star = np.zeros(params.dim)
            u_head = np.zeros(params.dim)
            for j in range(params.N):
                if params.A[i, j] == 1:
                    z_k = params.x[i] - params.x[j] # Computes the real relative pose.
                    # Velocity needed for agent i to reach desired position, by adding the contribution of neighbor j to agent i's velocity.
                    v_tilde += (z_k - params.relative_pos_desired[i, j])
                    u_head = u[j].copy() # Store agent's j (edge's head) velocity vector.

                    # Lyapunov barrier functions for connectivity maintenance throughout the whole trajectory.
                    W_grad = (1 / (max(params.d_max**2 - np.sum(z_k**2), 1e-2))) * z_k # max() is used to prevent division by zero if robots touch the limit. Equation 3.81.
                    v_star -= params.c1 * W_grad # Virtual control. The Incidence Matrix (Eik) is already being take into account in the if statement (Adjacency matrix).

            # It's crucial to add adjent's j velocity to the velocity that agent i would initially need to achieve the desired pose,
            # because agent j is in movement and that must be taken into account.
            u[i] = -(params.c2 * v_tilde) + v_star + u_head # Agent's i velocity. Equation 3.82.
            u_i = np.linalg.norm(u[i]) # Magnitude.

            # Set a max velocity in order to don't explode with the simulation.
            if u_i > params.v_max:
                u[i] *= params.v_max / u_i
        return u

    #* Function to compute each agent's velocity in the exact moment with connectivity maintenance and collision avoidance.
    def velocity_connectivity_maintenance_and_collision_avoidance(self, params, u):
        # Generalized.
        for i in range(1, params.N):
            # Initialize velocity vectors.
            agent_i_velocity = np.zeros(params.dim)
            v_tilde = np.zeros(params.dim)
            v_star = np.zeros(params.dim)
            u_head = np.zeros(params.dim)
            for j in range(params.N):
                if params.A[i, j] == 1:
                    z_k = params.x[i] - params.x[j] # Computes the real relative pose.
                    # Velocity needed for agent i to reach desired position, by adding the contribution of neighbor j to agent i's velocity.
                    v_tilde += (z_k - params.relative_pos_desired[i, j])
                    u_head = u[j].copy() # Store agent's j (edge's head) velocity vector.
                    # Lyapunov barrier functions for connectivity maintenance and collision avoidance throughout the whole trajectory.
                    # max() is used to prevent division by zero if robots touch the limit.
                    # Connectivity Maintenance - Equation 3.81 and Collision Avoidance - Equation 5.8 and 5.9
                    W_grad = ((1 / (max(params.d_max**2 - np.sum(z_k**2), 1e-2))) - (1 / (max(np.sum(z_k**2) - params.d_min**2, 1e-2))))* z_k
                    v_star -= params.c1 * W_grad # Virtual control. The Incidence Matrix (Eik) is already being take into account in the if statement (Adjacency matrix).

            # It's crucial to add adjent's j velocity to the velocity that agent i would initially need to achieve the desired pose,
            # because agent j is in movement and that must be taken into account.
            u[i] = -(params.c2 * v_tilde) + v_star + u_head # Agent's i velocity. Equation 3.82.
            u_i = np.linalg.norm(u[i]) # Magnitude.

            # Set a max velocity in order to don't explode with the simulation.
            if u_i > params.v_max:
                u[i] *= params.v_max / u_i
        return u