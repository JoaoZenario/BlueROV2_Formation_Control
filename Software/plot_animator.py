#!/usr/bin/env python3

##########################################################
#                                                        #
# Animation Plot for the Formation Controller            #
# Code Author: João Zenário                              #
#                                                        #     
##########################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

class Animation():
    def __init__(self, params):
        # Formation Workspace.
        self.fig_space = plt.figure("Formation Workspace", figsize=(8, 8))
        proj = '3d' if params.dim == 3 else None
        self.ax = self.fig_space.add_subplot(111, projection=proj)

        # Setup 3D or 2D limits.
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        if params.dim == 3:
            self.ax.set_zlim(-10, 10)
            self.ax.set_zlabel('Z (m)')
        self.ax.set_title("ROV Physical Formation")

        # Performance Telemetry.
        self.fig_telemetry = plt.figure("System Telemetry", figsize=(12, 10))
        
        rows = 2
        if params.fov:
            rows = 3
        
        gs = self.fig_telemetry.add_gridspec(rows, 2, hspace=0.3, wspace=0.3)

        # Distances plot.
        self.ax_dist = self.fig_telemetry.add_subplot(gs[0, 0])
        self.ax_dist.set_title("Inter-Agent Distances")
        if params.conn: self.ax_dist.axhline(y=params.d_max, color='r', ls='--')
        if params.coll: self.ax_dist.axhline(y=params.d_min, color='darkred', ls='--')
        self.dist_lines = []

        # Formation Error plot.
        self.ax_err = self.fig_telemetry.add_subplot(gs[0, 1])
        self.ax_err.set_title("Formation Offset Error (%)")
        self.err_lines = []

        # Velocity plot.
        self.ax_vel = self.fig_telemetry.add_subplot(gs[1, 0])
        self.ax_vel.set_title("Velocities (m/s)")
        if not params.mode_dynamics:
            self.ax_vel.axhline(y=params.v_max, color='k', ls='--', alpha=0.5)
        self.vel_lines = []

        # Acceleration plot.
        if params.mode_dynamics:
            self.ax_accel = self.fig_telemetry.add_subplot(gs[1, 1])
            self.ax_accel.set_title("Accelerations (m/s²)")
            self.ax_accel.axhline(y=params.a_max, color='k', ls='--', alpha=0.5)
            self.accel_lines = []

        # 5. Field of View plot.
        if params.fov:
            self.ax_fov_x = self.fig_telemetry.add_subplot(gs[2, 0])
            self.ax_fov_x.set_title("Horizontal FOV Error (deg)")
            
            self.ax_fov_y = self.fig_telemetry.add_subplot(gs[2, 1])
            self.ax_fov_y.set_title("Vertical FOV Error (deg)")
            
            # Add the limit lines here as well
            self.ax_fov_x.axhline(y=np.degrees(params.theta_max_x), color='r', ls='--')
            self.ax_fov_x.axhline(y=-np.degrees(params.theta_max_x), color='r', ls='--')
            
            self.ax_fov_y.axhline(y=np.degrees(params.theta_max_y), color='r', ls='--')
            self.ax_fov_y.axhline(y=-np.degrees(params.theta_max_y), color='r', ls='--')
            
            self.fov_x_lines = []
            self.fov_y_lines = []

        # Agents and their path representation.
        self.dots = [self.ax.plot([], [], 'o', color=params.colors[i], markersize=5, label=f"ROV {i+1}")[0] for i in range(params.N)] # Create N dots, one for each agent.
        self.trails = [self.ax.plot([], [], '-', color=params.colors[i], alpha=0.2)[0] for i in range(params.N)] # Create the path trail for each agent.

        # Create the legend on the 3D axis
        self.ax.legend(loc='upper right', shadow=True, fontsize='x-small')

        if params.fov:
            self.shadow = 50 # number of lines for each side (to simulate shadow).
            self.fov_lines = []

        for i in range(params.N):
            self.err_lines.append(self.ax_err.plot([], [], color=params.colors[i], label=f'ROV {i+1}')[0])
            self.vel_lines.append(self.ax_vel.plot([], [], color=params.colors[i], label=f'ROV {i+1}')[0])
            if params.fov:
                self.fov_x_lines.append(self.ax_fov_x.plot([], [], color=params.colors[i], label=f'ROV {i+1}')[0])
                self.fov_y_lines.append(self.ax_fov_y.plot([], [], color=params.colors[i], label=f'ROV {i+1}')[0])
            if params.mode_dynamics:
                self.accel_lines.append(self.ax_accel.plot([], [], color=params.colors[i], label=f'ROV {i+1}')[0])
            
            # Distance lines based on adjacency.
            for j in range(params.N):
                if params.A[i, j] == 1:
                    self.dist_lines.append(self.ax_dist.plot([], [], label=f'{i+1}->{j+1}')[0])

            # FOV Shadow lines (cone).
            if params.fov:
                agent_fov_shadow = [self.ax.plot([], [], [], color='crimson', alpha=0.1, lw=0.5)[0] for _ in range(self.shadow * 2)]
                self.fov_lines.append(agent_fov_shadow)
        
        self.telemetry_text = self.fig_space.text(0.02, 0.02, '', 
                                    verticalalignment='bottom', 
                                    family='monospace',
                                    fontsize=7,
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Standardize legends across all telemetry plots.  
        self.ax_dist.legend(loc='upper right', fontsize='xx-small', ncol=2)
        self.ax_err.legend(loc='upper right', fontsize='xx-small', ncol=2)
        self.ax_vel.legend(loc='upper right', fontsize='xx-small', ncol=2)
        self.ax_dist.legend(loc='upper right', fontsize='xx-small', ncol=2)
        if params.fov:
            self.ax_fov_x.legend(loc='upper right', fontsize='xx-small', ncol=2)
            self.ax_fov_y.legend(loc='upper right', fontsize='xx-small', ncol=2)
        if params.mode_dynamics:
            self.ax_accel.legend(loc='upper right', fontsize='xx-small', ncol=2)

    def plot_data(self, frame, params):
        edge_count = 0
        current_max_dist = 0
        current_time = frame * params.dt
        stats = f"Time: {current_time:5.2f}s\n"
        stats += "-" * 15 + "\n"
        time_axis = np.linspace(0, current_time, len(params.vel_history))

        for i in range(params.N):
            # Update the dots on the plot.
            self.dots[i].set_data([params.x[i, 0]], [params.x[i, 1]]) # x and y.
            if params.dim == 3:
                self.dots[i].set_3d_properties([params.x[i, 2]]) # z.
            # Update the trails on the plot.
            self.trails[i].set_data(params.history_np[:, i, 0], params.history_np[:, i, 1]) # x and y.
            if params.dim == 3:
                self.trails[i].set_3d_properties(params.history_np[:, i, 2]) # z.
            # Distance (to leader).
            if i: # Skip if i=0 (leader).
                # Calculate actual distance to leader using the same method (Euclidean distance): d = √[(u_x_i - u_x_0)² + (u_y_i - u_y_0)² + (u_z_i - u_z_0)²].
                dist_to_leader = np.linalg.norm(params.x[i] - params.x[0])
                stats += f"Distance to Leader: {dist_to_leader:5.2f}m (Target: {params.target_dist[i]:5.2f}m)\n"

            # Distance (edge length).
            for j in range(params.N):
                if params.A[i, j] == 1:
                    z_k_hist = params.history_np[:, i, :] - params.history_np[:, j, :]
                    dist_hist = np.linalg.norm(z_k_hist, axis=1)
                    self.dist_lines[edge_count].set_data(time_axis, dist_hist[-len(time_axis):])
                    edge_count += 1

                    # Compute the biggest edge for each step in order to update the animation.
                    d = np.linalg.norm(params.x[i] - params.x[j])
                    if d > current_max_dist:
                        current_max_dist = d
            # Velocity.
            self.vel_lines[i].set_data(time_axis, params.vel_hist_np[:, i])
            if params.mode_dynamics:
                speed = np.linalg.norm(params.v[i]) # Compute the speed using the Euclidean norm: |v| = √(u_x² + u_y² + u_z²).  
            else:
                speed = np.linalg.norm(u[i]) # Compute the speed using the Euclidean norm: |v| = √(u_x² + u_y² + u_z²).
            stats += f"ROV {i+1} Vel: {speed:5.2f} m/s\n"
            # Acceleration.
            if params.mode_dynamics:
                self.accel_lines[i].set_data(time_axis, params.accel_hist_np[:, i])
            # Formation error.
            self.err_lines[i].set_data(time_axis, params.err_hist_np[:, i])
            # Field of View error.
            if params.fov:
                err_x = params.fov_hist_np[:, i, 0] 
                err_y = params.fov_hist_np[:, i, 1]
                
                self.fov_x_lines[i].set_data(time_axis, err_x)
                self.fov_y_lines[i].set_data(time_axis, err_y)
                self.fov_shadow(params, i)

        self.telemetry_text.set_text(stats) # Store the statistics for the animation.
        
        return current_max_dist

    def fov_shadow(self, params, i):
        L = 0.7 # Length of the FOV line.
        # Get the rotation matrix
        rot_matrix = R.from_quat(params.q[i]).as_matrix()
        fwd, side, up = rot_matrix[:, 0], rot_matrix[:, 1], rot_matrix[:, 2]

        # We will draw "N" vertical fans to fill the volume
        # Divide the shadow lines between horizontal and vertical
        num_rays = self.shadow 
        
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
            self.fov_lines[i][l].set_data(
                [params.x[i, 0], p_top[0], p_bottom[0], params.x[i, 0]], 
                [params.x[i, 1], p_top[1], p_bottom[1], params.x[i, 1]]
            )
            self.fov_lines[i][l].set_3d_properties(
                [params.x[i, 2], p_top[2], p_bottom[2], params.x[i, 2]]
            )

    def plot_size(self, params, current_max_dist):
        # Axis limits.
        # Distance.
        self.ax_dist.set_xlim(0, params.steps * params.dt)
        self.ax_dist.set_ylim(0, max(current_max_dist, params.d_max) * 1.05)
        # Velocity.
        self.ax_vel.set_xlim(0, params.steps * params.dt)
        if params.vel_hist_np.size > 0:
            # Look at the maximum velocity in the CURRENT iteration across all agents
            current_max_vel = np.max(params.vel_hist_np[-1, :])
            # Set Y-limit: floor of 1.0 to prevent a "squashed" look at zero, or 1.2x current max
            self.ax_vel.set_ylim(0, max(1.0, current_max_vel * 1.2))
        # Acceleration.
        if params.mode_dynamics:
            self.ax_accel.set_xlim(0, params.steps * params.dt)
            self.ax_accel.set_ylim(0, params.a_max * 1.05)
        # Error.
        if params.err_hist_np.size > 0:
            self.ax_err.set_xlim(0, params.steps * params.dt)
            # Look at the maximum error in the CURRENT iteration
            current_max_err = np.max(params.err_hist_np[-1, :])
            # Set Y-limit: floor of 10% error so the plot doesn't look empty when settled
            self.ax_err.set_ylim(0, max(10.0, current_max_err * 1.2))
        # FOV Axis Scaling (Percentage Mode)
        # FOV Axis Scaling (Horizontal and Vertical)
        if params.fov and params.fov_hist_np.size > 0:
            self.ax_fov_x.set_xlim(0, params.steps * params.dt)
            self.ax_fov_y.set_xlim(0, params.steps * params.dt)
            
            limit_deg_x = np.degrees(params.theta_max_x)
            limit_deg_y = np.degrees(params.theta_max_y)

            # Current max error (already in degrees from history)
            current_max_abs_x = np.max(np.abs(params.fov_hist_np[-1, :, 0]))
            current_max_abs_y = np.max(np.abs(params.fov_hist_np[-1, :, 1]))

            # Dynamic Scaling: Use the larger of (Limit + 5) or (Current Error * 1.1)
            win_x = max(limit_deg_x + 5, current_max_abs_x * 1.1)
            self.ax_fov_x.set_ylim(-win_x, win_x)

            win_y = max(limit_deg_y + 5, current_max_abs_y * 1.1)
            self.ax_fov_y.set_ylim(-win_y, win_y)

        self.fig_telemetry.canvas.draw_idle()

    def plot_handle(self, frame, params):
        current_max_dist = self.plot_data(frame, params)
        self.plot_size(params, current_max_dist)

    def plot_animation(self, params, controller):
        self.animation = FuncAnimation(self.fig_space, controller.update_step, 
                                       frames=params.steps, fargs=(params, self), 
                                       interval=10, blit=False, repeat=False)

        #self.animation.save('rov_formation_3d.mp4', writer='ffmpeg', fps=30, dpi=150)

        plt.show()