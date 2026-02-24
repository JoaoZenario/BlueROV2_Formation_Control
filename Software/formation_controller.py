#!/usr/bin/env python3

##########################################################
#                                                        #
# Formation Controller                                   #
# Code Author: João Zenário                              #
# Equations and concepts based on: Esteban's thesis      #
#                                                        #     
##########################################################

import numpy as np
from single_integrator import Single_integrator
from double_integrator import Double_integrator
from constraints import Constraints
import mpl_toolkits.mplot3d.art3d as art3d

class BROV_Formation(Single_integrator, Double_integrator, Constraints):
    def __init__(self):
        Single_integrator.__init__(self)
        Double_integrator.__init__(self)
        Constraints.__init__(self)

    #* Function to update the pose of each agent after the step (moving at u acceleration for dt time).
    def update_step(self, frame, params, animator):
        edge_count = 0
        current_max_dist = 0
        t = params.steps * params.dt
        params.counter += 1
        u = np.zeros_like(params.x) # u will store each agent's acceleration.
        current_errors = []
        current_fov_errors = [0.0] * params.N
        L = 0.7 # Length of the FOV line.

        for i in range(params.N):
            if i == 0:
                current_errors.append(0.0)
            else:
                # Target = Leader Position + Desired Relative Offset from Leader.
                target_pos = params.x[0] + params.relative_pos_desired[i, 0]
                dist_err = np.linalg.norm(params.x[i] - target_pos)
                # Normalize by formation radius R.
                percent_err = (dist_err / max(params.R, 0.1)) * 100
                current_errors.append(percent_err)

        if params.mode_dynamics:
            u = self.leader_acceleration(params, u) # Leader acceleration.
            u = self.compute_acceleration(params, u) # Every other agent acceleration.
            
            if params.fov: current_fov_errors = self.field_of_view(params) # Apply Field-of-View constraint.

            params.v += u * params.dt # Update velocities (v̇ = u <=> v = u * dt).
            
            for i in range(params.N):
                v_i = np.linalg.norm(params.v[i]) # Magnitude.
                # Set a max velocity to become closer to reality.
                if v_i > params.v_max:
                    params.v[i] *= params.v_max / v_i

            params.x += params.v * params.dt # Update positions (ẋ = v <=> x = v * dt) .

            # Store velocity of each agent over time for the animation.
            current_speeds = [np.linalg.norm(v) for v in params.v]

            # Store acceleration of each agent over time for the animation.
            current_accels = [np.linalg.norm(a) for a in u]
            params.accel_history.append(current_accels)
            accel_hist_np = np.array(params.accel_history)
        else:
            u = self.leader_velocity(params, u) # Leader velocity.
            u = self.compute_velocity(params, u) # Every other agent velocity.
            
            if params.fov: current_fov_errors = self.field_of_view(params) # Apply Field-of-View constraint.

            params.x += u * params.dt # Update positions (ẋ = u <=> x = u * dt).

            # Store velocity of each agent over time for the animation.
            current_speeds = [np.linalg.norm(v) for v in u]

        # Store error of each agent over time for the animation.
        params.err_history.append(current_errors)
        err_hist_np = np.array(params.err_history)
        # Store velocity of each agent over time for the animation.
        params.vel_history.append(current_speeds)
        vel_hist_np = np.array(params.vel_history)
        if params.fov:
            # Store FOV error for each agent over time for the simulation.
            params.fov_history.append(current_fov_errors)
            fov_hist_np = np.array(params.fov_history)   


        params.history.append(params.x.copy()) # Add agent's latest pose to history.
        history_np = np.array(params.history)

        current_time = frame * params.dt
        stats = f"Time: {current_time:5.2f}s\n"
        stats += "-" * 15 + "\n"
        time_axis = np.linspace(0, current_time, len(params.vel_history))

        for i in range(params.N):
            # Error.
            animator.err_lines[i].set_data(time_axis, err_hist_np[:, i])
            # Update the dots on the plot.
            animator.dots[i].set_data([params.x[i, 0]], [params.x[i, 1]]) # For x and y.
            # Stores all time steps of each agent until this moment.
            animator.trails[i].set_data(history_np[:, i, 0], history_np[:, i, 1]) # For x and y.
            # Velocity.
            animator.vel_lines[i].set_data(time_axis, vel_hist_np[:, i]) # Update velocity animation.

            # Field of view shadow:
            if params.fov:
                animator.fov_err_lines[i].set_data(time_axis, fov_hist_np[:, i])

                for l in range(animator.shadow):
                    # Calculate intermediate angles from 0 to delta_alpha
                    fraction_l = (l + 1) / animator.shadow
                    fraction_r = l / (animator.shadow - 1)

                    alpha_l = params.yaw[i] + (params.delta_alpha * fraction_l)
                    alpha_r = params.yaw[i] - (params.delta_alpha * fraction_r)

                    p_l = params.x[i] + np.array([L * np.cos(alpha_l), L * np.sin(alpha_l), 0])
                    p_r = params.x[i] + np.array([L * np.cos(alpha_r), L * np.sin(alpha_r), 0])

                    # Left lines
                    animator.fov_lines[i][l].set_data([params.x[i, 0], p_l[0]], [params.x[i, 1], p_l[1]])
                    animator.fov_lines[i][l].set_3d_properties([params.x[i, 2], p_l[2]])

                    # Right lines
                    animator.fov_lines[i][l + animator.shadow].set_data([params.x[i, 0], p_r[0]], [params.x[i, 1], p_r[1]])
                    animator.fov_lines[i][l + animator.shadow].set_3d_properties([params.x[i, 2], p_r[2]])

            if params.mode_dynamics:
                speed = np.linalg.norm(params.v[i]) # Compute the speed using the Euclidean norm: |v| = √(u_x² + u_y² + u_z²).  
                animator.accel_lines[i].set_data(time_axis, accel_hist_np[:, i]) # Acceleration.
            else:
                speed = np.linalg.norm(u[i]) # Compute the speed using the Euclidean norm: |v| = √(u_x² + u_y² + u_z²).
            stats += f"ROV {i+1} Vel: {speed:5.2f} m/s\n"

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

            if i: # Skip if i=0.
                # Calculate actual distance to leader using the same method (Euclidean distance): d = √[(u_x_i - u_x_0)² + (u_y_i - u_y_0)² + (u_z_i - u_z_0)²].
                dist_to_leader = np.linalg.norm(params.x[i] - params.x[0])
                stats += f"Distance to Leader: {dist_to_leader:5.2f}m (Target: {params.target_dist[i]:5.2f}m)\n"

        animator.telemetry_text.set_text(stats) # Store the statistics for the animation.
                        
        # Axis limits.
        # Error.
        animator.ax_err.set_xlim(0, params.steps * params.dt)
        animator.ax_err.set_ylim(0, max(20, np.max(err_hist_np) if len(err_hist_np) > 0 else 100))
        # Distance.
        animator.ax_dist.set_xlim(0, params.steps * params.dt)
        animator.ax_dist.set_ylim(0, max(current_max_dist, params.d_max) * 1.05)
        # Velocity.
        animator.ax_vel.set_xlim(0, params.steps * params.dt)
        animator.ax_vel.set_ylim(0, params.v_max * 1.05)
        # Acceleration.
        if params.mode_dynamics:
            animator.ax_accel.set_xlim(0, params.steps * params.dt)
            animator.ax_accel.set_ylim(0, params.a_max * 1.05)
        # FOV.
        if params.fov:
            if len(params.fov_history) > 0:
                # Look only at the current frame's errors for all agents.
                current_frame_errors = fov_hist_np[-1, :]
                
                # Find the spread of the current errors.
                frame_min = np.min(current_frame_errors)
                frame_max = np.max(current_frame_errors)
                
                # Define the boundary we want to see (the constraint limits).
                limit = np.degrees(params.delta_alpha)
                
                # The y-axis will now tightly fit the current errors OR the limits, whichever is larger at this specific moment.
                y_bottom = min(frame_min - 2, -limit - 2)
                y_top = max(frame_max + 2, limit + 2)
            
            animator.ax_fov.set_xlim(0, params.steps * params.dt)
            animator.ax_fov.set_ylim(y_bottom, y_top)

        animator.fig_telemetry.canvas.draw_idle()

        return animator.dots + animator.trails + animator.dist_lines + animator.err_lines + [animator.telemetry_text] # For efficiency, returning these variables is crucial since they were the only ones that changed.