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
        proj = '3d' if params.dim == 3 else None
        self.fig, self.ax = plt.subplots(figsize=(9, 9), subplot_kw={'projection': proj})

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
        # Create a Text Box to show the time and agent's velocity.
        self.telemetry_text = self.fig.text(0.02, 0.95, '', 
                                    verticalalignment='top', 
                                    family='monospace',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    #* Function to run and plot the animation that shows the robot's movement and formation process throughout time.
    def plot_animation(self, params, controller):
        self.animation = FuncAnimation(self.fig, controller.update_step, frames=params.steps, fargs=(params, self), interval=10, blit=False, repeat = False) # Run animation.
        #self.animation.save('rov_formation.gif', writer='pillow', fps=30) # Save animation.
        plt.show() # Plot animation.