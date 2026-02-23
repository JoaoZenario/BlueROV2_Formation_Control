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

class Animation:
    #* Function to setup the animation.
    def __init__(self, params):
        # Animation setup.
        # Create a figure with two subplots, left for physical space, right for distance vs time.
        self.fig = plt.figure(figsize=(12, 10))
        
        # Physical Space Subplot.
        proj = '3d' if params.dim == 3 else None
        self.ax = self.fig.add_subplot(121, projection=proj)

        if params.dim == 3:
            self.ax.set_zlim(-10, 10) # Plot limits based on the agent's z pose.
            self.ax.set_zlabel('Z (m)')

        self.ax.set_xlim(-10, 10) # Plot limits based on the agent's x pose.
        self.ax.set_ylim(-10, 10) # Plot limits based on the agent's y pose.
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.grid(True)
        self.ax.set_title("ROV Formation Over Time")

        self.dots = [self.ax.plot([], [], 'o', color=params.colors[i], markersize=5, label=f"ROV {i+1}")[0] for i in range(params.N)] # Create N dots, one for each agent.
        self.trails = [self.ax.plot([], [], '-', color=params.colors[i], alpha=0.2)[0] for i in range(params.N)] # Create the path trail for each agent.
        self.ax.legend(loc='upper right', shadow=True, fontsize='x-small')

        if params.mode_dynamics:
            size = 520
        else:
            size = 420

        # Error Subplot
        self.ax_err = self.fig.add_subplot(size + 1) 
        self.ax_err.set_title("Formation Offset Error")
        self.ax_err.set_xlabel("Time (s)")
        self.ax_err.set_ylabel("Error (%)")
        self.ax_err.grid(True, linestyle=':', alpha=0.6)
        self.ax_err.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        self.err_lines = []

        # Distance Subplot.
        self.ax_dist = self.fig.add_subplot(size + 2)
        self.ax_dist.set_title("Inter-Agent Distances Over Time")
        self.ax_dist.set_ylabel(r"$|z_k|$ (m)")
        # Plot static dashed lines for constraints.
        if params.mode_constraint == 1:
            self.ax_dist.axhline(y=params.d_max, color='red', linestyle='--', linewidth=1.5, label=r'Limit $\Delta_k$')
        if params.mode_constraint > 1:
            self.ax_dist.axhline(y=params.d_max, color='red', linestyle='--', linewidth=1.5, label=r'Limit $\Delta_k$')
            self.ax_dist.axhline(y=params.d_min, color='darkred', linestyle='--', linewidth=1.5, label=r'Limit $\delta_k$')
        self.dist_lines = [] # Initialize distance lines for each edge in the adjacency matrix.

        # Velocity Subplot.
        self.ax_vel = self.fig.add_subplot(size + 4)
        self.ax_vel.set_title("Agent's Velocities Over Time")
        self.ax_vel.set_xlabel("Time (s)")
        self.ax_vel.set_ylabel(r"$|v_i|$ (m/s)")
        self.ax_dist.grid(True, linestyle=':', alpha=0.6)
        self.ax_vel.axhline(y=params.v_max, color='black', linestyle='--', alpha=0.5, label='Max Speed')
        self.vel_lines = [] # Initialize velocity line for each agent.

        # Add the FOV Evolution Subplot
        self.ax_fov = self.fig.add_subplot(size + 6) # Adjust 'size' based on your layout
        self.ax_fov.set_title("FOV Constraint Evolution")
        self.ax_fov.set_xlabel("Time (s)")
        self.ax_fov.set_ylabel(r"$\alpha_{error}$ (deg)")
        self.ax_fov.grid(True, linestyle=':', alpha=0.6)
        
        # Plot the upper and lower FOV boundaries
        fov_limit = np.degrees(params.delta_alpha)
        self.ax_fov.axhline(y=fov_limit, color='red', linestyle='--', alpha=0.5, label=r'+$\Delta\alpha$')
        self.ax_fov.axhline(y=-fov_limit, color='darkred', linestyle='--', alpha=0.5, label=r'-$\Delta\alpha$')

        if params.mode_dynamics:
            # Acceleration Subplot.                
            self.ax_accel = self.fig.add_subplot(size + 8)
            self.ax_accel.set_title("Agent's Acceleration Over Time")
            self.ax_accel.set_xlabel("Time (s)")
            self.ax_accel.set_ylabel(r"$|a_i|$ (m/s²)")
            self.ax_accel.grid(True, linestyle=':', alpha=0.6)
            self.ax_accel.axhline(y=params.a_max, color='black', linestyle='--', alpha=0.5, label='Max Acceleration')
            self.accel_lines = [] # Initialize acceleration line form each agent.

        self.shadow = 20 # Number of lines per side
        self.fov_lines = []
        self.fov_err_lines = []

        for i in range(params.N):
            # Error line
            e_line, = self.ax_err.plot([], [], lw=1.5, color=params.colors[i], label=f'ROV {i+1}')
            self.err_lines.append(e_line)

            # Velocity line.
            v_line, = self.ax_vel.plot([], [], lw=1.5, color=params.colors[i], label=f'ROV {i+1}')
            self.vel_lines.append(v_line)

            # FOV line.
            fov_line, = self.ax_fov.plot([], [], lw=1.5, color=params.colors[i], label=f'ROV {i+1}')
            self.fov_err_lines.append(fov_line)

            if params.mode_dynamics:
                # Acceleration line.
                a_line, = self.ax_accel.plot([], [], lw=1.5, color=params.colors[i], label=f"ROV {i+1}")
                self.accel_lines.append(a_line)

            for j in range(params.N):
                if params.A[i, j] == 1:
                    # Distance line.
                    d_line, = self.ax_dist.plot([], [], lw=1.5, label=f'$e_{{{i+1} \\to {j+1}}}$')
                    self.dist_lines.append(d_line)

            # Complete shadow.
            agent_lines = []
            for _ in range(self.shadow * 2):
                line, = self.ax.plot([], [], [], color='crimson', 
                                     alpha=0.1, lw=0.5) # Very low alpha for the shadow effect
                agent_lines.append(line)
            self.fov_lines.append(agent_lines)

        # Textbox for legend.
        self.ax_err.legend(loc='upper right', fontsize='xx-small', ncol=2)
        #self.ax_dist.legend(loc='upper right', fontsize='xx-small', ncol=2)
        self.ax_vel.legend(loc='upper right', fontsize='xx-small', ncol=2)
        self.ax_fov.legend(loc='upper right', fontsize='xx-small', ncol=2)

        if params.mode_dynamics:
            self.ax_accel.legend(loc='upper right', fontsize='xx-small', ncol=2)

        # Telemetry Text.
        # Create a Text Box to show the time and agent's velocity.
        self.telemetry_text = self.fig.text(0.02, 0.02, '', 
                                    verticalalignment='bottom', 
                                    family='monospace',
                                    fontsize=6,
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    #* Function to run and plot the animation that shows the robot's movement and formation process throughout time.
    def plot_animation(self, params, controller):
        self.animation = FuncAnimation(self.fig, controller.update_step, frames=params.steps, fargs=(params, self), interval=10, blit=False, repeat = False) # Run animation.
        #self.animation.save('rov_formation.mp4', writer='ffmpeg', fps=30, dpi=100) # Save animation.
        # Plot animation.
        plt.tight_layout()
        plt.show()
        #return self.animation # Uncomment when running on jupyter notebook.