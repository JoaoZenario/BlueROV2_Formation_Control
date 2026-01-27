#!/usr/bin/env python3

##########################################################
#                                                        #
# Formation Controller of a directed graph               #
# Code Author: João Zenário                              #
# Equations and concepts based on: Esteban's thesis      #
#                                                        #     
##########################################################

import numpy as np

class BROV_Formation:
    def __init__(self):
        pass  

    #* Function to update the pose of each agent after the step (moving at u velocity for dt time).
    def update_step(self, frame, params, animator):
        t = params.steps * params.dt
        params.counter += 1
        # To compute the velocity we'll use equation 2.11 and 2.13 (with a slight change):
        # ̇ẋ_i = u_i = -c * ∑_j [(x_i - x_j) - (o_i - o_j)] for all j connected to i.
        u = np.zeros_like(params.x) # u will store each agent's velocity.
        u = self.leader_velocity(params, u)

        match params.mode_constraint:
            case 0:
                u = self.velocity_no_constraints(params, u)
            case 1:
                u = self.velocity_connectivity_maintenance(params, u)
            case 2:
                u = self.velocity_connectivity_maintenance_and_collision_avoidance(params, u)

        params.x += u * params.dt # Update positions using Euler's Method (x_i+1 = x_i + u * dt).
        params.history.append(params.x.copy()) # Add agent's latest pose to history.
        history_np = np.array(params.history)

        current_time = frame * params.dt
        stats = f"Time: {current_time:5.2f}s\n"
        stats += "-" * 15 + "\n"
        
        for i in range(params.N):
            # Calculate speed (magnitude of velocity vector).
            speed = np.linalg.norm(u[i]) # Compute the speed using the Euclidean norm: |v| = √(u_x² + u_y² + u_z²).  
            stats += f"ROV {i+1} Vel: {speed:5.2f} m/s\n"

            if i: # Skip if i=0.
                # Calculate actual distance to leader using the same method (Euclidean distance): d = √[(u_x_i - u_x_0)² + (u_y_i - u_y_0)² + (u_z_i - u_z_0)²].
                dist_to_leader = np.linalg.norm(params.x[i] - params.x[0])
                stats += f"Distance to Leader: {dist_to_leader:5.2f}m (Target: {params.target_dist[i]:5.2f}m)\n"
        
        animator.telemetry_text.set_text(stats) # Store the statistics for the animation.

        for i in range(params.N):
            # Update the dots on the plot.
            animator.dots[i].set_data([params.x[i, 0]], [params.x[i, 1]]) # For x and y.
            # Stores all time steps of each agent until this moment.
            animator.trails[i].set_data(history_np[:, i, 0], history_np[:, i, 1]) # For x and y.

            if params.dim == 3:
                animator.dots[i].set_3d_properties([params.x[i, 2]]) # # Update the dots on the plot for z.
                animator.trails[i].set_3d_properties(history_np[:, i, 2]) # # Stores all time steps of each agent until this moment for z.

        return animator.dots + animator.trails + [animator.telemetry_text] # For efficiency, returning these variables is crucial since they were the only ones that changed.

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