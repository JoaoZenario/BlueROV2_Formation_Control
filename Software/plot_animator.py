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
            self.ax_fov = self.fig_telemetry.add_subplot(gs[2, :])
            self.ax_fov.set_title("FOV Angular Error (deg)")
            fov_lim = np.degrees(params.delta_alpha)
            self.ax_fov.axhline(y=fov_lim, color='r', ls='--')
            self.ax_fov.axhline(y=-fov_lim, color='darkred', ls='--')
            self.fov_err_lines = []

        # Agents and their path representation.
        self.dots = [self.ax.plot([], [], 'o', color=params.colors[i], markersize=5, label=f"ROV {i+1}")[0] for i in range(params.N)] # Create N dots, one for each agent.
        self.trails = [self.ax.plot([], [], '-', color=params.colors[i], alpha=0.2)[0] for i in range(params.N)] # Create the path trail for each agent.

        # Create the legend on the 3D axis
        self.ax.legend(loc='upper right', shadow=True, fontsize='x-small')

        if params.fov:
            self.shadow = 25 # number of lines for each side (to simulate shadow).
            self.fov_lines = []

        for i in range(params.N):
            self.err_lines.append(self.ax_err.plot([], [], color=params.colors[i], label=f'ROV {i+1}')[0])
            self.vel_lines.append(self.ax_vel.plot([], [], color=params.colors[i], label=f'ROV {i+1}')[0])
            if params.fov: self.fov_err_lines.append(self.ax_fov.plot([], [], color=params.colors[i], label=f'ROV {i+1}')[0])
            if params.mode_dynamics: self.accel_lines.append(self.ax_accel.plot([], [], color=params.colors[i], label=f'ROV {i+1}')[0])
            
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
        if params.fov: self.ax_fov.legend(loc='upper right', fontsize='xx-small', ncol=2)
        if params.mode_dynamics: self.ax_accel.legend(loc='upper right', fontsize='xx-small', ncol=2)

    def plot_animation(self, params, controller):
        self.animation = FuncAnimation(self.fig_space, controller.update_step, 
                                       frames=params.steps, fargs=(params, self), 
                                       interval=10, blit=False, repeat=False)

        #self.animation.save('rov_formation_3d.mp4', writer='ffmpeg', fps=30, dpi=150)

        plt.show()