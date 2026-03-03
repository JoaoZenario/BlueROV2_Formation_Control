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
from scipy.spatial.transform import Rotation as R

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
            animator.dots[i].set_3d_properties([params.x[i, 2]])
            # Stores all time steps of each agent until this moment.
            animator.trails[i].set_data(history_np[:, i, 0], history_np[:, i, 1]) # For x and y.
            animator.trails[i].set_3d_properties(history_np[:, i, 2])
            # Velocity.
            animator.vel_lines[i].set_data(time_axis, vel_hist_np[:, i]) # Update velocity animation.

            if params.fov:
                animator.fov_err_lines[i].set_data(time_axis, fov_hist_np[:, i])

                # Get the rotation matrix
                rot_matrix = R.from_quat(params.q[i]).as_matrix()
                fwd, side, up = rot_matrix[:, 0], rot_matrix[:, 1], rot_matrix[:, 2]

                # We will draw "N" vertical fans to fill the volume
                # Divide the shadow lines between horizontal and vertical
                num_rays = animator.shadow 
                
                for l in range(num_rays):
                    # 1. Horizontal spread factor (-1 to 1)
                    s_frac = (l / (num_rays - 1)) * 2 - 1
                    ang_h = params.theta_max_x * s_frac
                    
                    # 2. Create the "Left" and "Right" boundary rays of the rectangle
                    # These go from the top edge to the bottom edge
                    dx = L * np.tan(ang_h)
                    dy_max = L * np.tan(params.theta_max_y)

                    # Point at the Top of this vertical strip
                    p_top = params.x[i] + (fwd * L + side * dx + up * dy_max)
                    # Point at the Bottom of this vertical strip
                    p_bottom = params.x[i] + (fwd * L + side * dx - up * dy_max)

                    # Set the ray: Line from ROV to Top, then to Bottom, then back to ROV
                    # This creates a triangle slice for every step in 'l'
                    animator.fov_lines[i][l].set_data(
                        [params.x[i, 0], p_top[0], p_bottom[0], params.x[i, 0]], 
                        [params.x[i, 1], p_top[1], p_bottom[1], params.x[i, 1]]
                    )
                    animator.fov_lines[i][l].set_3d_properties(
                        [params.x[i, 2], p_top[2], p_bottom[2], params.x[i, 2]]
                    )

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
        # FOV Axis Scaling (Percentage Mode)
        if params.fov and len(params.fov_history) > 0:
            animator.ax_fov.set_xlim(0, params.steps * params.dt)
            
            # 1. Get the max error of the CURRENT frame only (across all N agents)
            # fov_hist_np[-1, :] accesses the very last entry in the history
            current_max_err = np.max(fov_hist_np[-1, :])
            
            # 2. Set the top limit to either 120% OR the current max + buffer
            # This ensures the axis "shrinks" back down if the error improves
            y_top_inst = max(110, current_max_err + 10)
            
            # 3. Apply the limit
            animator.ax_fov.set_ylim(0, y_top_inst) 
            animator.ax_fov.set_ylabel("FOV Usage (%)")

        animator.fig_telemetry.canvas.draw_idle()

        return animator.dots + animator.trails + animator.dist_lines + animator.err_lines + [animator.telemetry_text] # For efficiency, returning these variables is crucial since they were the only ones that changed.

    def update_quaternion(self, i, omega, dt, params):
        """
        Updates the quaternion orientation for agent i.
        omega: [p, q, r] angular velocity vector in body frame
        """
        # 1. Get current quaternion [x, y, z, w]
        q_current = params.q[i]
        
        # 2. Convert angular velocity to a quaternion rate of change
        # We use the standard kinematic formula: dq/dt = 0.5 * q * [0, omega]
        # In Scipy, we can do this by creating a small rotation vector
        rotation_vec = omega * dt
        angle = np.linalg.norm(rotation_vec)
        
        if angle > 1e-8:
            # Create a rotation object from the rotation vector (exponential map)
            delta_rot = R.from_rotvec(rotation_vec)
            
            # Current orientation as a Scipy Rotation object
            q_obj_current = R.from_quat(q_current)
            
            # 3. Apply the rotation: q_new = q_current * delta_q
            # Note: Order matters! Body-frame rotation is applied on the right.
            q_new_obj = q_obj_current * delta_rot
            
            # 4. Extract back to array and ensure normalization
            params.q[i] = q_new_obj.as_quat()
        
        # Normalize to prevent numerical drift (crucial for long simulations)
        params.q[i] /= np.linalg.norm(params.q[i])