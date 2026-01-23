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

    #* Function to compute each agent's velocity in the exact moment.
    # The agents are in an initial pose (with a certain relative pose to each others)
    # and the goal is the agents moving to a pose where the relative pose to each others is the desired one (relative_pose_desired).   
    def consensus_control(self, params):
        t = params.steps * params.dt
        params.counter += 1
        # To compute the velocity we'll use equation 2.11 and 2.13 (with a slight change):
        # ̇ẋ_i = u_i = -c * ∑_j [(x_i - x_j) - (o_i - o_j)] for all j connected to i.
        u = np.zeros_like(params.x) # u will store each agent's velocity.

        for i in range(params.N):
            if i == 0:
                if params.dim == 2:
                    if (params.counter < 0.5 * params.steps):
                        u[i, 0] = 5  # Leader's x speed.
                        u[i, 1] = 20 # Leader's y speed.
                    else:
                        u[i, 0] = -10.0 # Leader's x speed.
                        u[i, 1] = -10.0 # Leader's y speed.
                if params.dim == 3:
                    if (params.counter < 0.5 * params.steps):
                        u[i, 0] = 5  # Leader's x speed.
                        u[i, 1] = 20 # Leader's y speed.
                        u[i, 2] = 5  # Leader's z speed.
                    else:
                        u[i, 0] = -10.0 # Leader's x speed.
                        u[i, 1] = -10.0 # Leader's y speed.
                        u[i, 2] = 5.0   # Leader's z speed.

            else:
                agent_i_velocity = np.zeros(params.dim) # Initialize velocity vector.
                #j = 0 # Since the leader is the only one taken into account.
                # Generalized.
                for j in range(params.N):
                    if params.A[i, j] == 1:
                        relative_pos_real = params.x[i] - params.x[j] # Computes the real relative pose.
                        agent_i_velocity -= (relative_pos_real - params.relative_pos_desired[i, j]) # Add the contribution of neighbor j to agent i's velocity.
                        u_position = params.c * agent_i_velocity # Velocity needed for agent i to reach desired position.
                        u_leader = u[j].copy() # Store leader's velocity vector.
                # It's crucial to add leader's velocity to the velocity that agent i would initially need to achieve the desired pose,
                # because leader is in movement and that must be taken into account.
                u[i] = u_position + u_leader # Agent's i velocity.
        return u # Used in the update_step function.

    #* Function to update the pose of each agent after the step (moving at u velocity for dt time).
    def update_step(self, frame, params, animator):
        u = self.consensus_control(params)
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
                stats += f"Distance to Leader: {dist_to_leader:5.2f}m (Target: {params.target_dist:5.2f}m)\n"
        
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