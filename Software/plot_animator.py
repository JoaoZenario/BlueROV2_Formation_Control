#!/usr/bin/env python3

##########################################################
#                                                        #
# Animation Plot for the Formation Controller            #
# Code Author: João Zenário                              #
#                                                        #     
##########################################################

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

class Animation:
    #* Function to setup the animation.
    def __init__(self, params):
        # Animation setup.
        # Create a figure with two subplots, left for physical space, right for distance vs time.
        self.fig = plt.figure(figsize=(16, 8))
        
        # Physical Space Subplot.
        proj = '3d' if params.dim == 3 else None
        self.ax = self.fig.add_subplot(121, projection=proj)

        if params.dim == 3:
            self.ax.set_zlim(-200, 200) # Plot limits based on the agent's z pose.
            self.ax.set_zlabel('Z (m)')

        self.ax.set_xlim(-200, 200) # Plot limits based on the agent's x pose.
        self.ax.set_ylim(-200, 200) # Plot limits based on the agent's y pose.
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.grid(True)
        self.ax.set_title("ROV Formation Evolution Over Time")

        self.dots = [self.ax.plot([], [], 'o', color=params.colors[i], markersize=5, label=f"ROV {i+1}")[0] for i in range(params.N)] # Create N dots, one for each agent.
        self.trails = [self.ax.plot([], [], '-', color=params.colors[i], alpha=0.2)[0] for i in range(params.N)] # Create the path trail for each agent.
        self.ax.legend(loc='upper right', shadow=True, fontsize='medium')

        # Distance Analysis Subplot.
        self.ax_dist = self.fig.add_subplot(122)
        self.ax_dist.set_title("Inter-Agent Distances vs Constraints")
        self.ax_dist.set_xlabel("Time (s)")
        self.ax_dist.set_ylabel(r"$|z_k|$ (m)")
        self.ax_dist.grid(True, linestyle=':', alpha=0.6)

        # Plot static dashed lines for constraints.
        self.ax_dist.axhline(y=params.d_max, color='red', linestyle='--', linewidth=1.5, label=r'Limit $\Delta_k$')
        self.ax_dist.axhline(y=params.d_min, color='darkred', linestyle='--', linewidth=1.5, label=r'Limit $\delta_k$')

        # Initialize distance lines for each edge in the adjacency matrix.
        self.dist_lines = []
        for i in range(params.N):
            for j in range(params.N):
                if params.A[i, j] == 1:
                    # Each line gets a unique label for the edge.
                    line, = self.ax_dist.plot([], [], lw=1.5, label=f'$e_{{{i+1} \\to {j+1}}}$')
                    self.dist_lines.append(line)
        
        self.ax_dist.legend(loc='upper right', fontsize='x-small', ncol=2)

        # Telemetry Text.
        # Create a Text Box to show the time and agent's velocity.
        self.telemetry_text = self.fig.text(0.02, 0.95, '', 
                                    verticalalignment='top', 
                                    family='monospace',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    #* Function to run and plot the animation that shows the robot's movement and formation process throughout time.
    def plot_animation(self, params, controller):
        self.animation = FuncAnimation(self.fig, controller.update_step, frames=params.steps, fargs=(params, self), interval=10, blit=False, repeat = False) # Run animation.
        #self.animation.save('rov_formation.gif', writer='pillow', fps=30) # Save animation.
        # Plot animation.
        plt.tight_layout()
        plt.show()
        #return self.animation # Uncomment when running on jupyter notebook.