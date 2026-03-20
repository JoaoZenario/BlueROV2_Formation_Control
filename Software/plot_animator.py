#!/usr/bin/env python3

##########################################################
#                                                        #
# Animation Plot for the Formation Controller            #
# Code Author: João Zenário                              #
#                                                        #     
##########################################################

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import matplotlib.colors as mcolors

class Animation():
    #* Function to define and itialize the simulation and all plots.
    def __init__(self, params):
        #self.plot_barrier(params)
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

        # Edge dynamic representation in simulation.
        self.line_edges = []
        for i in range(params.N):
            for j in range(params.N):
                if params.A[i, j] == 1:
                    line, = self.ax.plot([], [], [], lw=1.5, alpha=0.8) # Create a line for each edge.
                    self.line_edges.append(line) # Store the indices and the line object.

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
        self.ax_vel.axhline(y=params.v_max, color='k', ls='--', alpha=0.5)
        self.vel_lines = []

        # Acceleration plot.
        if params.mode_dynamics:
            self.ax_accel = self.fig_telemetry.add_subplot(gs[1, 1])
            self.ax_accel.set_title("Accelerations (m/s²)")
            self.ax_accel.axhline(y=params.a_max, color='k', ls='--', alpha=0.5)
            self.accel_lines = []

        # Field of View plot.
        if params.fov:
            self.ax_fov_x = self.fig_telemetry.add_subplot(gs[2, 0])
            self.ax_fov_x.set_title("Horizontal FOV Error (deg)")
            
            self.ax_fov_y = self.fig_telemetry.add_subplot(gs[2, 1])
            self.ax_fov_y.set_title("Vertical FOV Error (deg)")
            
            self.ax_fov_x.axhline(y=np.degrees(params.theta_max_x), color='r', ls='--')
            self.ax_fov_x.axhline(y=-np.degrees(params.theta_max_x), color='r', ls='--')
            
            self.ax_fov_y.axhline(y=np.degrees(params.theta_max_y), color='r', ls='--')
            self.ax_fov_y.axhline(y=-np.degrees(params.theta_max_y), color='r', ls='--')
            
            self.fov_x_lines = []
            self.fov_y_lines = []

        # Agents and their path representation.
        self.dots = [self.ax.plot([], [], 'o', color=params.colors[i], markersize=5, label=f"ROV {i+1}")[0] for i in range(params.N)] # Create N dots, one for each agent.
        self.trails = [self.ax.plot([], [], '-', color=params.colors[i], alpha=0.1)[0] for i in range(params.N)] # Create the path trail for each agent.

        # Create the legend on the 3D axis.
        self.ax.legend(loc='upper right', shadow=True, fontsize='x-small')

        if params.fov:
            self.shadow = 50 # number of lines for each side (to simulate field of view).
            self.fov_lines = []
        
        # Plot label.
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
        #self.ax_err.legend(loc='upper right', fontsize='xx-small', ncol=2)
        self.ax_vel.legend(loc='upper right', fontsize='xx-small', ncol=2)
        self.ax_dist.legend(loc='upper right', fontsize='xx-small', ncol=2)
        if params.fov:
            self.ax_fov_x.legend(loc='upper right', fontsize='xx-small', ncol=2)
            self.ax_fov_y.legend(loc='upper right', fontsize='xx-small', ncol=2)
        if params.mode_dynamics:
            self.ax_accel.legend(loc='upper right', fontsize='xx-small', ncol=2)

    #* Function to gather and store the data in order to update the simulation and plots. 
    def plot_data(self, frame, params, u):
        edge_count = 0
        current_max_dist = 0
        current_time = frame * params.dt
        stats = f"Time: {current_time:5.2f}s\n"
        stats += "-" * 15 + "\n"
        time_axis = np.linspace(0, current_time, len(params.vel_history))

        cmap = plt.get_cmap('RdYlGn_r') # Get colormap for the edge representation (smooth transition from red to yellow to green, based on distance).

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

                    d = dist_hist[-1]

                    # Update workspace line position.
                    pos_i = params.x[i]
                    pos_j = params.x[j]
                    self.line_edges[edge_count].set_data([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]])
                    if params.dim == 3:
                        self.line_edges[edge_count].set_3d_properties([pos_i[2], pos_j[2]])

                    # Smooth color transition.
                    if d > params.R:
                        color = (d - params.R) / (params.d_max - params.R)
                    else:
                        color = (params.R - d) / (params.R - params.d_min)
                    
                    color = np.clip(color, 0, 1)
                    self.line_edges[edge_count].set_color(cmap(color))

                    edge_count += 1

                    # Save lenght od biggest edge on that moment to update the edge distance plot. 
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

    #* Function to simulate simulate the field of view of each agent throughout the whole simulation.
    def fov_shadow(self, params, i):
        L = 0.7 # Length of the FOV line.
        # Get the rotation matrix
        rot_matrix = R.from_quat(params.q[i]).as_matrix()
        fwd, side, up = rot_matrix[:, 0], rot_matrix[:, 1], rot_matrix[:, 2]
        
        for l in range(self.shadow):
            # Evenly spread all rays across the whole horizontal lenght.
            s_frac = (l / (self.shadow - 1)) * 2 - 1
            ang_h = params.theta_max_x * s_frac
            
            # Create the "Left" and "Right" boundary rays of the rectangle, from the top edge to the bottom edge.
            dx = L * np.tan(ang_h)
            dy_max = L * np.tan(params.theta_max_y)

            # Point at the Top of this vertical strip.
            p_top = params.x[i] + (fwd * L + side * dx + up * dy_max)
            # Point at the Bottom of this vertical strip.
            p_bottom = params.x[i] + (fwd * L + side * dx - up * dy_max)

            # Set the ray.
            self.fov_lines[i][l].set_data(
                [params.x[i, 0], p_top[0], p_bottom[0], params.x[i, 0]], 
                [params.x[i, 1], p_top[1], p_bottom[1], params.x[i, 1]]
            )
            self.fov_lines[i][l].set_3d_properties(
                [params.x[i, 2], p_top[2], p_bottom[2], params.x[i, 2]]
            )

    #* Function to set plot size, according to variable limits and/or variable values during the simulation. 
    def plot_size(self, params, current_max_dist):
        # Axis limits.
        # Distance.
        self.ax_dist.set_xlim(0, params.steps * params.dt)
        self.ax_dist.set_ylim(0, max(current_max_dist, params.d_max) * 1.05)
        # Velocity.
        self.ax_vel.set_xlim(0, params.steps * params.dt)
        if params.vel_hist_np.size > 0:
            # Look at the maximum velocity in the current iteration across all agents (for velocity plot).
            current_max_vel = np.max(params.vel_hist_np[-1, :])
            self.ax_vel.set_ylim(0, max(params.v_max * 1.05, current_max_vel * 1.2))
        # Acceleration.
        if params.mode_dynamics:
            self.ax_accel.set_xlim(0, params.steps * params.dt)
            self.ax_accel.set_ylim(0, params.a_max * 1.05)
        # Error.
        if params.err_hist_np.size > 0:
            self.ax_err.set_xlim(0, params.steps * params.dt)
            # Look at the maximum formation error in the current iteration across all agents (for error plot).
            current_max_err = np.max(params.err_hist_np[-1, :])
            self.ax_err.set_ylim(0, max(10.0, current_max_err * 1.2))
        # FOV (Horizontal and Vertical).
        if params.fov and params.fov_hist_np.size > 0:
            self.ax_fov_x.set_xlim(0, params.steps * params.dt)
            self.ax_fov_y.set_xlim(0, params.steps * params.dt)
            
            limit_deg_x = np.degrees(params.theta_max_x)
            limit_deg_y = np.degrees(params.theta_max_y)

            # Current max angle error.
            current_max_abs_x = np.max(np.abs(params.fov_hist_np[-1, :, 0]))
            current_max_abs_y = np.max(np.abs(params.fov_hist_np[-1, :, 1]))

            win_x = max(limit_deg_x + 5, current_max_abs_x * 1.1)
            self.ax_fov_x.set_ylim(-win_x, win_x)

            win_y = max(limit_deg_y + 5, current_max_abs_y * 1.1)
            self.ax_fov_y.set_ylim(-win_y, win_y)

        self.fig_telemetry.canvas.draw_idle()

    #* Function to call other functions that handle the simulation and plots.
    def plot_handle(self, frame, params, u):
        current_max_dist = self.plot_data(frame, params, u)
        self.plot_size(params, current_max_dist)


    def get_barrier(self, params, d):
        d_sq = d**2
        epsilon = 1e-6
        den_conn = np.maximum(params.d_max**2 - d_sq, epsilon)
        den_coll = np.maximum(d_sq - params.d_min**2, epsilon)
        b_conn = np.log(params.d_max**2 / den_conn)
        b_coll = np.log(params.d_min**2 / den_coll)
        
        return 0.5 * (b_conn + b_coll)

    def plot_barrier(self, params):
        # --- 1. Common Data Setup ---
        grid_limit = 1.5 * params.d_max
        res = 100
        
        d_range = np.linspace(-grid_limit, grid_limit, 1000)
        B_vals_2d = self.get_barrier(params, np.abs(d_range))
        
        V_target = self.get_barrier(params, params.R)
        grad_mag = (self.get_barrier(params, params.R + 0.01) - 
                    self.get_barrier(params, params.R - 0.01)) / 0.02
        
        # --- FIGURE 1: 2D Potential Slice ---
        plt.figure("2D Slice", figsize=(8, 6))
        B_rec_2d = B_vals_2d - V_target - grad_mag * (np.abs(d_range) - params.R)
        
        plt.plot(d_range, B_vals_2d, '--', color='gray', label='Regular Barrier', alpha=0.6)
        plt.plot(d_range, B_rec_2d, 'b', label='Recentered Barrier', linewidth=2)
        plt.axvline(params.R, color='g', linestyle=':', label='Target Distance')
        plt.axvline(-params.R, color='g', linestyle=':')
        plt.axvline(params.d_max, color='r', linestyle=':', label='Δk')
        plt.axvline(-params.d_max, color='r', linestyle=':')
        plt.axvline(params.d_min, color='y', linestyle=':', label='δk')
        plt.axvline(-params.d_min, color='y', linestyle=':')

        plt.ylim(-2, 15)
        plt.title("2D Potential Cross-section ($x_2 = 0$)")
        plt.xlabel("Relative Coordinate $x_1$ (m)")
        plt.ylabel("Potential Value")
        plt.legend()
        plt.grid(alpha=0.3)
        # REMOVED plt.show() from here

        # --- Meshgrid for 3D Plots ---
        x = np.linspace(-grid_limit, grid_limit, res)
        y = np.linspace(-grid_limit, grid_limit, res)
        X, Y = np.meshgrid(x, y)
        Dist = np.sqrt(X**2 + Y**2)
        V_3d = self.get_barrier(params, Dist)

        # --- FIGURE 2: 3D Normal Barrier ---
        fig2 = plt.figure("3D Regular", figsize=(8, 7))
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.plot_surface(X, Y, V_3d, cmap='viridis', edgecolor='none', alpha=0.9)
        ax2.set_title("3D Regular Barrier (Symmetric)")
        ax2.set_zlim(-2, 20)
        ax2.view_init(elev=20, azim=-45)
        # REMOVED plt.show() from here

        # --- FIGURE 3: 3D Recentered Barrier ---
        term_linear = grad_mag * (X - params.R)
        V_rec_3d = V_3d - V_target - term_linear

        fig3 = plt.figure("3D Recentered", figsize=(8, 7))
        ax3 = fig3.add_subplot(111, projection='3d')
        ax3.plot_surface(X, Y, V_rec_3d, cmap='plasma', edgecolor='none', alpha=0.9)
        
        # TARGET DOT LOGIC
        ax3.scatter([params.R], [0], [0], color='red', s=150, 
                   label='Desired Equilibrium', edgecolors='black', zorder=5)
        
        ax3.set_title("3D Recentered Barrier (Tilted)")
        ax3.set_zlim(-5, 25)
        ax3.view_init(elev=20, azim=-60)
        ax3.legend()

        # --- FINAL SHOW ---
        # Calling this once at the very end opens all active figures at the same time
        plt.show()

    #* Function to show, and save (if desired), the whole animation.
    def plot_animation(self, params, controller):
        #"""
        self.animation = FuncAnimation(self.fig_space, controller.update_step, 
                                       frames=params.steps, fargs=(params, self), 
                                       interval=10, blit=False, repeat=False)
        #self.animation.save('rov_formation_3d.mp4', writer='ffmpeg', fps=30, dpi=150)
        plt.show()
        #"""