#!/usr/bin/env python3

##########################################################
#                                                        #
# Formation Controller                                   #
# Code Author: João Zenário                              #
# Equations and concepts based on: Esteban's thesis      #
#                                                        #     
##########################################################

import numpy as np

class BROV_Formation:
    def __init__(self):
        pass  

    #* Function to update the pose of each agent after the step (moving at u acceleration for dt time).
    def update_step(self, frame, params, animator):
        edge_count = 0
        current_max_dist = 0
        t = params.steps * params.dt
        params.counter += 1
        u = np.zeros_like(params.x) # u will store each agent's acceleration.

        if params.mode_dynamics:
            u = self.leader_acceleration(params, u)

            match params.mode_constraint:
                case 0:
                    u = self.acceleration_no_constraints(params, u)
                case 1:
                    u = self.acceleration_connectivity_maintenance(params, u)
                case 2:
                    u = self.acceleration_connectivity_maintenance_and_collision_avoidance(params, u)

            params.v += u * params.dt # Update velocities (v̇ = u <=> v = u * dt).
            
            for i in range(params.N):
                v_i = np.linalg.norm(params.v[i]) # Magnitude.
                # Set a max velocity to become closer to reality.
                if v_i > params.v_max:
                    params.v[i] *= params.v_max / v_i

            params.x += params.v * params.dt # Update positions (ẋ = v <=> x = v * dt) .

            # Store velocity of each agent over time for the animation.
            current_speeds = [np.linalg.norm(v) for v in params.v]
            params.vel_history.append(current_speeds)
            vel_hist_np = np.array(params.vel_history)

            # Store acceleration of each agent over time for the animation.
            current_accels = [np.linalg.norm(a) for a in u]
            params.accel_history.append(current_accels)
            accel_hist_np = np.array(params.accel_history)
        else:
            u = self.leader_velocity(params, u)

            match params.mode_constraint:
                case 0:
                    u = self.velocity_no_constraints(params, u)
                case 1:
                    u = self.velocity_connectivity_maintenance(params, u)
                case 2:
                    u = self.velocity_connectivity_maintenance_and_collision_avoidance(params, u)
            
            params.x += u * params.dt # Update positions (ẋ = u <=> x = u * dt).

            # Store velocity of each agent over time for the animation.
            current_speeds = [np.linalg.norm(v) for v in u]
            params.vel_history.append(current_speeds)
            vel_hist_np = np.array(params.vel_history)

        params.history.append(params.x.copy()) # Add agent's latest pose to history.
        history_np = np.array(params.history)

        current_time = frame * params.dt
        stats = f"Time: {current_time:5.2f}s\n"
        stats += "-" * 15 + "\n"
        time_axis = np.linspace(0, current_time, len(params.vel_history))

        for i in range(params.N):
            # Calculate speed (magnitude of velocity vector).
            if params.mode_dynamics:
                speed = np.linalg.norm(params.v[i]) # Compute the speed using the Euclidean norm: |v| = √(u_x² + u_y² + u_z²).  
            else:
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
            # Velocity.
            animator.vel_lines[i].set_data(time_axis, vel_hist_np[:, i]) # Update velocity animation.
            
            if params.mode_dynamics:
                # Acceleration.
                animator.accel_lines[i].set_data(time_axis, accel_hist_np[:, i])

            if params.dim == 3:
                animator.dots[i].set_3d_properties([params.x[i, 2]]) # # Update the dots on the plot for z.
                animator.trails[i].set_3d_properties(history_np[:, i, 2]) # # Stores all time steps of each agent until this moment for z.

            # For loop in order to compute the distance of each edge for the final plot.
            for j in range(params.N):
                if params.A[i, j] == 1:
                    z_k_hist = history_np[:, i, :] - history_np[:, j, :]
                    dist_hist = np.linalg.norm(z_k_hist, axis=1)
                    animator.dist_lines[edge_count].set_data(time_axis, dist_hist[-len(time_axis):])
                    edge_count += 1

                    # Compute the biggest edge for each step in order to update the animation.
                    d = np.linalg.norm(params.x[i] - params.x[j])
                    if d > current_max_dist:
                        current_max_dist = d
                        
        # Axis limits.
        # Distance.
        animator.ax_dist.set_xlim(0, max(2, current_time))
        animator.ax_dist.set_ylim(0, max(current_max_dist, params.d_max) * 1.05)
        # Velocity.
        animator.ax_vel.set_xlim(0, max(2, current_time))
        animator.ax_vel.set_ylim(0, params.v_max * 1.05)
        if params.mode_dynamics:
            # Acceleration.
            animator.ax_accel.set_xlim(0, max(2, current_time))
            animator.ax_accel.set_ylim(0, params.a_max * 1.05)

        return animator.dots + animator.trails + animator.dist_lines + [animator.telemetry_text] # For efficiency, returning these variables is crucial since they were the only ones that changed.

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

    #* Function to set leader's acceleration throughout the whole simulation.
    def leader_acceleration(self, params, u):
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
        return u

    #* Function to compute each agent's acceleration in the exact moment without constraints.
    # The agents are in an initial pose (with a certain relative pose to each others)
    # and the goal is the agents moving to a pose where the relative pose to each others is the desired one (relative_pose_desired). 
    def acceleration_no_constraints(self, params, u):
        # Generalized.
        for i in range(1, params.N):
            # Initialize sum vector.
            u_sum = np.zeros(params.dim)
            for j in range(params.N):
                if params.A[i, j] == 1:
                    z_k = params.x[i] - params.x[j] # Computes the real relative pose.
                    pos_error = z_k - params.relative_pos_desired[i, j] # Compute the position error.
                    vel_error = params.gamma * (params.v[i] - params.v[j]) # Compute the velocity error.
                    u_sum += (pos_error + vel_error) # Sum of both position and velocity errors.
            u[i] = -params.c2 * u_sum # Acceleration of agent i.

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
            u_sum = np.zeros(params.dim)
            for j in range(params.N):
                if params.A[i, j] == 1:
                    z_k = params.x[i] - params.x[j] # Computes the real relative pose.
                    pos_error = (z_k - params.relative_pos_desired[i, j]) # Compute the position error.
                    vel_error = params.gamma * (params.v[i] - params.v[j]) # Compute the velocity error.
                    # Lyapunov barrier functions for connectivity maintenance throughout the whole trajectory. Equation 3.81.
                    # max() is used to prevent division by zero, if robots touch the limit, and also to prevent the opposite behvior when agents are initially out of the raidus threshold.
                    W_grad = (1 / (max(params.d_max**2 - np.sum(z_k**2), 1e-2))) * z_k
                    u_sum += (pos_error + vel_error + params.c1 * W_grad) # Sum everything (contribution agent j has on agent i). Equation 4 of Wei Ren's paper.

            u[i] = -params.c2 * u_sum # Agent i acceleration. Equation 3.82.

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
            u_sum = np.zeros(params.dim)
            for j in range(params.N):
                if params.A[i, j] == 1:
                    z_k = params.x[i] - params.x[j] # Computes the real relative pose.
                    pos_error = (z_k - params.relative_pos_desired[i, j]) # Compute the position error.
                    vel_error = params.gamma * (params.v[i] - params.v[j]) # Compute the velocity error.
                    # Lyapunov barrier functions for connectivity maintenance and collision avoidance throughout the whole trajectory.
                    # max() is used to prevent division by zero, if robots touch the limit, and also to prevent the opposite behvior when agents are initially out of the raidus threshold.
                    # Connectivity Maintenance - Equation 3.81 and Collision Avoidance - Equation 5.8 and 5.9
                    W_grad = ((1 / (max(params.d_max**2 - np.sum(z_k**2), 1e-2))) - (1 / (max(np.sum(z_k**2) - params.d_min**2, 1e-2))))* z_k
                    u_sum += (pos_error + vel_error + params.c1 * W_grad) # Sum everything (contribution agent j has on agent i). Equation 4 of Wei Ren's paper.

            u[i] = -params.c2 * u_sum # Agent i acceleration. Equation 3.82.

            u_i = np.linalg.norm(u[i]) # Magnitude.
            # Set a max acceleration in order to don't explode with the simulation.
            if u_i > params.a_max:
                u[i] *= params.a_max / u_i
        return u